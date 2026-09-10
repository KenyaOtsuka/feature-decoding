'''Benchmark one whole decoding pipeline, train -> predict, both ways.

The other benchmark (``bench_ridge_factorization``) isolates the Ridge solve and
the stored model size as the output dimension grows.  This one runs the actual
pipeline end to end so that the work the factorization *moves* is charged to the
side that pays it:

``legacy-equivalent``
    Training as the pre-PR script did it (base commit ``7322377``):
    ``(layer, subject, ROI)`` with ``get_multi_features`` inside the loop, the
    feature statistics, and ``bdpy.ml.ModelTraining`` writing a
    ``d_out x n_voxels`` model per (layer, subject, ROI).  Prediction reads that
    model back and predicts the features directly.
``factorized``
    Today's scripts: training is on the stimulus basis, reads no feature file
    and writes an ``M x n_voxels`` model per (subject, ROI); prediction loads
    that model, loads the training features of the layer, and contracts them.

Both variants predict through the **shipped** ``predict_feature.featdec_predict``,
which detects the decoder format, so the prediction rows are the real code path
rather than a reimplementation.

Three different quantities are measured and never added together:

logical I/O
    Exact and machine-independent: the summed ``nbytes`` of the arrays the run
    reads (feature files, model coefficients) and writes (model coefficients,
    normalization parameters, decoded features).  ``read + write`` is the
    headline number.
device I/O
    ``/proc/self/io`` around each phase.  This machine's local disk; a shared
    filesystem behaves differently.
stored size
    The decoder tree left on disk.  Storage, not traffic, so it is reported on
    its own and never folded into an I/O total.

    uv run python -m benchmarks.bench_pipeline

Deliberately small so it runs anywhere (~3 min, ~1 GB of temporary files); the
point is the ratio between the two variants, which does not depend on the size.
Not part of the test suite -- that training opens no feature file is pinned by
``tests/test_training_feature_loading.py``.
'''

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
from itertools import product
from time import perf_counter

import bdpy
import numpy as np
from bdpy import BData
from bdpy.bdata.utils import get_labels_multi_bdatas, select_data_multi_bdatas
from bdpy.dataform import Features, load_array, save_array
from bdpy.dataform.utils import get_multi_features
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.util import makedir_ifnot
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import bdpy.dataform.features as bdpy_features
import bdpy.ml.learning as bdpy_learning
import predict_feature
import ridge_factorization
import train_decoder_sklearn_ridge

LABEL_KEY = 'stimulus_name'
ALPHA = 100.0
DTYPE = np.float32
CHUNK_AXIS = 1

# Small stand-in for the real setting (M ~ 1200, 19 layers, 5 subjects, 9 ROIs).
N_STIMULI = 200
N_REPEATS = 5
N_TEST_STIMULI = 50
N_TEST_REPEATS = 2
N_VOXELS = 300
N_LAYERS = 2
N_SUBJECTS = 2
N_ROIS = 2
FEATURE_SIZES = (4096, 32768)

# DeepRecon VGG19 "allunits": units per stimulus, summed over the 16 conv and
# 3 fc layers, and the shape of a full run.
VGG19_LAYER_UNITS = (
    (2, 224 * 224 * 64), (2, 112 * 112 * 128), (4, 56 * 56 * 256),
    (4, 28 * 28 * 512), (4, 14 * 14 * 512),
    (1, 4096), (1, 4096), (1, 1000),
)
VGG19_UNITS = sum(count * units for count, units in VGG19_LAYER_UNITS)
VGG19_LAYERS = sum(count for count, _ in VGG19_LAYER_UNITS)
DEEPRECON_STIMULI = 1200
DEEPRECON_TEST_STIMULI = 50
DEEPRECON_VOXELS = 10 ** 4
DEEPRECON_SUBJECTS = 5
DEEPRECON_ROIS = 9


# Synthetic dataset ###########################################################

class Dataset(object):
    '''Paths and shape of one synthetic dataset on disk.'''

    def __init__(self, root, train_fmri, test_fmri, features_dir, layers, rois,
                 unique_labels, test_labels, n_trials, d_out):
        self.root = root
        self.train_fmri = train_fmri
        self.test_fmri = test_fmri
        self.features_dir = features_dir
        self.layers = layers
        self.rois = rois
        self.unique_labels = unique_labels
        self.test_labels = test_labels
        self.n_trials = n_trials
        self.d_out = d_out

    @property
    def feature_files(self):
        return [os.path.join(self.features_dir, layer, '%s.mat' % label)
                for layer in self.layers for label in self.unique_labels]

    @property
    def fmri_files(self):
        return [path for paths in list(self.train_fmri.values())
                + list(self.test_fmri.values()) for path in paths]

    def row_bytes(self):
        '''Logical size of one layer's features for the whole label set.'''
        return len(self.unique_labels) * self.d_out * np.dtype(DTYPE).itemsize


def _write_bdata(path, brain, labels, label_to_number, rois):
    bdata = BData()
    bdata.add(np.asarray(brain, dtype=float), 'VoxelData')
    bdata.add(np.array([[label_to_number[lb]] for lb in labels], dtype=float),
              LABEL_KEY)
    for roi in rois:
        bdata.add_metadata('ROI_%s' % roi, np.ones(brain.shape[1]),
                           where='VoxelData')
    bdata.add_vmap(LABEL_KEY, {v: k for k, v in label_to_number.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bdata.save(path)


def build_dataset(root, n_stimuli=N_STIMULI, n_repeats=N_REPEATS,
                  n_test=N_TEST_STIMULI, n_voxels=N_VOXELS, n_layers=N_LAYERS,
                  d_out=1024, n_subjects=N_SUBJECTS, n_rois=N_ROIS, seed=0):
    '''Write a feature store, training fMRI and test fMRI.

    Every ROI selects every voxel: the ROI changes the fit, never which feature
    files are read, so keeping them the same size keeps the rows of the table
    comparable.  **Every subject sees the same stimuli in the same order**,
    which is the ordinary case and the one in which prediction reads a layer's
    training features once for the whole run.  Features are 2-D (one vector per
    stimulus), the case ``bdpy`` does not chunk.
    '''
    rng = np.random.RandomState(seed)

    unique_labels = ['stim%05d' % (i + 1) for i in range(n_stimuli)]
    test_labels = ['test%05d' % (i + 1) for i in range(n_test)]
    label_to_number = {lb: i + 1 for i, lb in enumerate(unique_labels)}
    test_to_number = {lb: i + 1 for i, lb in enumerate(test_labels)}
    trial_labels = [lb for lb in unique_labels for _ in range(n_repeats)]
    test_trial_labels = [lb for lb in test_labels
                         for _ in range(N_TEST_REPEATS)]
    layers = ['layer%d' % (i + 1) for i in range(n_layers)]
    rois = {'roi%d' % (i + 1): 'ROI_roi%d = 1' % (i + 1)
            for i in range(n_rois)}

    features_dir = os.path.join(root, 'features')
    for layer in layers:
        layer_dir = os.path.join(features_dir, layer)
        os.makedirs(layer_dir, exist_ok=True)
        block = rng.randn(n_stimuli, d_out).astype(DTYPE)
        for i, label in enumerate(unique_labels):
            save_array(os.path.join(layer_dir, '%s.mat' % label),
                       block[i][np.newaxis], key='feat', dtype=DTYPE,
                       sparse=False)

    train_fmri, test_fmri = {}, {}
    for i in range(n_subjects):
        subject = 'sub-%02d' % (i + 1)
        train_path = os.path.join(root, 'fmri', '%s_train.h5' % subject)
        _write_bdata(train_path, rng.randn(len(trial_labels), n_voxels),
                     trial_labels, label_to_number, rois)
        train_fmri[subject] = [train_path]

        test_path = os.path.join(root, 'fmri', '%s_test.h5' % subject)
        _write_bdata(test_path, rng.randn(len(test_trial_labels), n_voxels),
                     test_trial_labels, test_to_number, rois)
        test_fmri[subject] = [test_path]

    return Dataset(root, train_fmri, test_fmri, features_dir, layers, rois,
                   unique_labels, test_labels, len(trial_labels), d_out)


# Instrumentation #############################################################

def _proc_io():
    '''``(read_bytes, write_bytes)`` of this process, or ``None``.'''
    values = {}
    try:
        with open('/proc/self/io') as f:
            for line in f:
                key, _, value = line.partition(':')
                if key in ('read_bytes', 'write_bytes'):
                    values[key] = int(value)
    except OSError:
        return None
    if len(values) != 2:
        return None
    return values['read_bytes'], values['write_bytes']


def drop_from_cache(paths):
    '''``POSIX_FADV_DONTNEED`` every path. Only evicts *clean* pages.'''
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def evict(paths):
    '''Drop ``paths`` from the page cache, syncing first.

    ``POSIX_FADV_DONTNEED`` leaves dirty pages alone, so for files that were
    just written the ``os.sync()`` is what makes the eviction take effect.
    '''
    os.sync()
    drop_from_cache(paths)


def tree_files(root):
    return [os.path.join(base, name)
            for base, _, names in os.walk(root) for name in names]


def tree_bytes(root):
    return sum(os.path.getsize(path) for path in tree_files(root))


class FeatureLoadProbe(object):
    '''Every per-stimulus feature-file load, wherever it happens.

    ``loads`` and ``logical_bytes`` are exact and machine-independent: how many
    times a feature file is loaded, and the summed ``nbytes`` of the arrays the
    loader returns.  Deliberately not the file sizes -- an hdf5storage file
    carries container overhead, and with a feature index the file holds more
    units than the selected array.
    '''

    def __init__(self, evict_after_load=False):
        self._evict_after_load = evict_after_load
        self.loads = 0
        self.logical_bytes = 0
        self.seconds = 0.0

    @contextlib.contextmanager
    def installed(self):
        load = bdpy_features._load_array_with_key

        def counted_load(key, path):
            start = perf_counter()
            array = load(key, path)
            self.seconds += perf_counter() - start
            self.loads += 1
            self.logical_bytes += int(np.asarray(array).nbytes)
            if self._evict_after_load:
                # A 15 GB layer never stays in the page cache across the
                # (subject, ROI) iterations, so neither may this store.
                drop_from_cache([path])
            return array

        bdpy_features._load_array_with_key = counted_load
        try:
            yield self
        finally:
            bdpy_features._load_array_with_key = load


class _PickleShim(object):
    '''``pickle`` with counted ``load``/``dump``, for the model files.'''

    def __init__(self, real, probe):
        self._real = real
        self._probe = probe

    def __getattr__(self, name):
        return getattr(self._real, name)

    def load(self, file_object):
        start = perf_counter()
        obj = self._real.load(file_object)
        self._probe.load_seconds += perf_counter() - start
        self._probe.loads += 1
        self._probe.load_bytes += _model_bytes(obj)
        return obj

    def dump(self, obj, file_object, **kwargs):
        self._probe.dumps += 1
        self._probe.dump_bytes += _model_bytes(obj)
        return self._real.dump(obj, file_object, **kwargs)


def _model_bytes(obj):
    '''Coefficient bytes of a stored model, however it is wrapped.'''
    model = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    total = 0
    for attribute in ('coef_', 'intercept_'):
        value = getattr(model, attribute, None)
        if value is not None:
            total += int(np.asarray(value).nbytes)
    return total


class ModelProbe(object):
    '''Model coefficients read from and written to the decoder.'''

    MODULES = (bdpy_learning, ridge_factorization)

    def __init__(self):
        self.loads = 0
        self.load_bytes = 0
        self.load_seconds = 0.0
        self.dumps = 0
        self.dump_bytes = 0

    @contextlib.contextmanager
    def installed(self):
        originals = [(module, module.pickle) for module in self.MODULES]
        for module, real in originals:
            module.pickle = _PickleShim(real, self)
        try:
            yield self
        finally:
            for module, real in originals:
                module.pickle = real


class ArrayWriteProbe(object):
    '''Every ``.mat`` array a run writes: normalization parameters, features.

    Counted the same way as the reads -- the ``nbytes`` of the array handed to
    ``save_array`` -- so that read and write can be added into one logical I/O
    total.
    '''

    MODULES = (train_decoder_sklearn_ridge, predict_feature,
               ridge_factorization)

    def __init__(self):
        self.writes = 0
        self.logical_bytes = 0
        self.decoded_bytes = 0  # the part both variants pay
        self.__decoded_key = 'feat'

    @contextlib.contextmanager
    def installed(self, extra_modules=()):
        modules = tuple(self.MODULES) + tuple(extra_modules)
        originals = [(module, module.save_array) for module in modules]

        def counted_save(save_file, array, key=None, **kwargs):
            n_bytes = int(np.asarray(array).nbytes)
            self.writes += 1
            self.logical_bytes += n_bytes
            if key == self.__decoded_key:
                self.decoded_bytes += n_bytes
            return original(save_file, array, key=key, **kwargs)

        # Every module imported the same function object.
        original = originals[0][1]
        for module, _ in originals:
            module.save_array = counted_save
        try:
            yield self
        finally:
            for module, real in originals:
                module.save_array = real


class CombineProbe(object):
    '''Time spent contracting the coefficients with the training features.'''

    def __init__(self):
        self.seconds = 0.0

    @contextlib.contextmanager
    def installed(self):
        original = predict_feature.combine_features

        def timed(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds += perf_counter() - start

        predict_feature.combine_features = timed
        try:
            yield self
        finally:
            predict_feature.combine_features = original


class InferenceProbe(object):
    '''Time inside ``ModelTest.run()``: the model load plus the inference.'''

    def __init__(self):
        self.seconds = 0.0

    @contextlib.contextmanager
    def installed(self):
        original = bdpy_learning.ModelTest.run

        def timed(instance):
            start = perf_counter()
            try:
                return original(instance)
            finally:
                self.seconds += perf_counter() - start

        bdpy_learning.ModelTest.run = timed
        try:
            yield self
        finally:
            bdpy_learning.ModelTest.run = original


class Phase(object):
    '''What one measured phase (training or prediction) cost.'''

    def __init__(self, seconds, features, models, writes, combine, inference,
                 device):
        self.seconds = seconds
        self.features = features
        self.models = models
        self.writes = writes
        self.combine = combine
        self.inference = inference
        self.device_read, self.device_write = device

    @property
    def logical_read(self):
        return self.features.logical_bytes + self.models.load_bytes

    @property
    def logical_write(self):
        return self.writes.logical_bytes + self.models.dump_bytes

    @property
    def logical_total(self):
        return self.logical_read + self.logical_write

    @property
    def counts(self):
        '''The deterministic part, compared across repeats.'''
        return (self.features.loads, self.features.logical_bytes,
                self.models.loads, self.models.load_bytes,
                self.models.dumps, self.models.dump_bytes,
                self.writes.writes, self.writes.logical_bytes)


@contextlib.contextmanager
def measured(cold, extra_write_modules=()):
    '''Run a phase under every probe and return a :class:`Phase`.'''
    features = FeatureLoadProbe(evict_after_load=cold)
    models = ModelProbe()
    writes = ArrayWriteProbe()
    combine = CombineProbe()
    inference = InferenceProbe()
    result = {}

    with features.installed(), models.installed(), \
            writes.installed(extra_write_modules), combine.installed(), \
            inference.installed():
        before = _proc_io()
        start = perf_counter()
        yield result
        seconds = perf_counter() - start
        after = _proc_io()

    device = ((0, 0) if before is None or after is None
              else (after[0] - before[0], after[1] - before[1]))
    result['phase'] = Phase(seconds, features, models, writes, combine,
                            inference, device)


# The two variants ############################################################

def legacy_equivalent_train(dataset, output_dir):
    '''Training as the pre-PR script did it (base commit ``7322377``).

    The loop, the per-(layer, subject, ROI) ``get_multi_features`` call, the
    feature statistics, and ``ModelTraining`` with the same normalization,
    ``Y_sort`` and ``chunk_axis`` -- so this really writes the
    ``d_out x n_voxels`` model that prediction then has to read back.  The one
    liberty taken is the iteration order: the original permutes it, which
    changes nothing that is measured here.
    '''
    data_brain = {subject: [bdpy.BData(f) for f in files]
                  for subject, files in dataset.train_fmri.items()}
    data_features = [Features(dataset.features_dir)]

    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')
    distcomp = DistComp(backend='sqlite3',
                        db_path=os.path.join('./tmp', 'legacy.db'))

    for layer, subject, roi in product(dataset.layers, dataset.train_fmri,
                                       dataset.rois):
        model_dir = os.path.join(output_dir, layer, subject, roi, 'model')
        makedir_ifnot(model_dir)

        brain = select_data_multi_bdatas(data_brain[subject],
                                         dataset.rois[roi])
        brain_labels = get_labels_multi_bdatas(data_brain[subject], LABEL_KEY)

        feat_labels = np.unique(brain_labels)
        feat = get_multi_features(data_features, layer, labels=feat_labels)

        brain = np.vstack([b for b, lb in zip(brain, brain_labels)
                           if lb in feat_labels])
        brain_labels = [lb for lb in brain_labels if lb in feat_labels]

        brain_mean = np.mean(brain, axis=0)[np.newaxis, :]
        brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]
        feat_mean = np.mean(feat, axis=0)[np.newaxis, :]
        feat_norm = np.std(feat, axis=0, ddof=1)[np.newaxis, :]

        feat_index = np.array([np.where(np.array(feat_labels) == lb)
                               for lb in brain_labels]).flatten()

        for key, value in (('x_mean', brain_mean), ('x_norm', brain_norm),
                           ('y_mean', feat_mean), ('y_norm', feat_norm)):
            save_array(os.path.join(model_dir, '%s.mat' % key), value, key=key,
                       dtype=DTYPE, sparse=False)

        train = ModelTraining(Ridge(alpha=ALPHA), brain, feat)
        train.id = 'legacy-%s-%s-%s' % (layer, subject, roi)
        train.X_normalize = {'mean': brain_mean, 'std': brain_norm}
        train.Y_normalize = {'mean': feat_mean, 'std': feat_norm}
        train.Y_sort = {'index': feat_index}
        train.dtype = DTYPE
        train.chunk_axis = CHUNK_AXIS
        train.save_format = 'pickle'
        train.save_path = model_dir
        train.distcomp = distcomp
        train.run()


def factorized_train(dataset, output_dir):
    '''The shipped training script, so the measured row is the real thing.'''
    train_decoder_sklearn_ridge.featdec_sklearn_ridge_train(
        dataset.train_fmri,
        output_dir=output_dir,
        rois=dict(dataset.rois),
        label_key=LABEL_KEY,
        alpha=ALPHA,
        analysis_name=os.path.basename(output_dir),
    )


def predict(dataset, decoder_dir, output_dir, training_features_paths):
    '''The shipped prediction script -- the same one for both variants.'''
    predict_feature.featdec_predict(
        dataset.test_fmri,
        decoder_dir,
        output_dir=output_dir,
        rois=dict(dataset.rois),
        label_key=LABEL_KEY,
        layers=list(dataset.layers),
        excluded_labels=[],
        average_sample=True,
        chunk_axis=CHUNK_AXIS,
        training_features_paths=training_features_paths,
        analysis_name=os.path.basename(output_dir),
    )


VARIANTS = ('legacy-equivalent', 'factorized')

THIS_MODULE = sys.modules[__name__]


# Measurement #################################################################

def run_variant(name, dataset, workdir, cold, repeats):
    '''Best-of-``repeats`` timed ``train`` then ``predict``.

    Each repeat trains into a fresh directory and predicts from a pristine copy
    of the decoder into a fresh output directory: the factorized prediction
    writes ``y_mean``/``y_norm`` on its first pass and skips them afterwards,
    and both scripts skip a decoded-feature directory that already exists.
    '''
    factorized = name == 'factorized'
    train = factorized_train if factorized else legacy_equivalent_train
    tag = name.replace('-', '_')
    pristine = os.path.join(workdir, 'decoder-%s' % tag)
    best = None

    for _ in range(repeats):
        shutil.rmtree(pristine, ignore_errors=True)
        if cold:
            evict(dataset.feature_files + dataset.fmri_files)

        # This module's own `save_array` calls are the legacy variant's
        # normalization parameters, so they belong in the write total too.
        with measured(cold, extra_write_modules=(THIS_MODULE,)) as result:
            with contextlib.redirect_stdout(io.StringIO()):
                train(dataset, pristine)
        training = result['phase']

        decoder_dir = os.path.join(workdir, 'predict-decoder-%s' % tag)
        decoded_dir = os.path.join(workdir, 'decoded-%s' % tag)
        shutil.rmtree(decoder_dir, ignore_errors=True)
        shutil.rmtree(decoded_dir, ignore_errors=True)
        shutil.copytree(pristine, decoder_dir)
        if cold:
            evict(dataset.feature_files + dataset.fmri_files
                  + tree_files(decoder_dir))

        training_features = [dataset.features_dir] if factorized else None
        with measured(cold) as result:
            with contextlib.redirect_stdout(io.StringIO()):
                predict(dataset, decoder_dir, decoded_dir, training_features)
        prediction = result['phase']

        stored = tree_bytes(pristine)
        candidate = (training, prediction, stored, decoder_dir, decoded_dir)
        if best is None:
            best = candidate
        else:
            for phase, previous in ((training, best[0]), (prediction, best[1])):
                if phase.counts != previous.counts:
                    raise AssertionError(
                        '%s is not deterministic: %r then %r'
                        % (name, previous.counts, phase.counts))
            if (training.seconds + prediction.seconds
                    < best[0].seconds + best[1].seconds):
                best = candidate

    return best


def check_expected_counts(name, training, prediction, dataset, n_stimuli):
    '''The load counts are the claim; assert them rather than only print.'''
    layers = len(dataset.layers)
    decoders = len(dataset.train_fmri) * len(dataset.rois)
    row = dataset.row_bytes()

    if name == 'legacy-equivalent':
        expected = {
            'training feature loads': (training.features.loads,
                                       layers * decoders * n_stimuli),
            'training feature bytes': (training.features.logical_bytes,
                                       layers * decoders * row),
            'training model writes': (training.models.dumps, layers * decoders),
            'prediction feature loads': (prediction.features.loads, 0),
            'prediction model loads': (prediction.models.loads,
                                       layers * decoders),
        }
    else:
        expected = {
            'training feature loads': (training.features.loads, 0),
            'training model writes': (training.models.dumps, decoders),
            # One read per layer: every subject shares the ordered label set,
            # and the layer is the outer loop of featdec_predict.
            'prediction feature loads': (prediction.features.loads,
                                         layers * n_stimuli),
            'prediction feature bytes': (prediction.features.logical_bytes,
                                         layers * row),
            # Stored once per (subject, ROI), but ModelTest.run() is called
            # once per (layer, subject, ROI).
            'prediction model loads': (prediction.models.loads,
                                       layers * decoders),
        }

    for what, (actual, wanted) in sorted(expected.items()):
        if actual != wanted:
            raise AssertionError('%s: %s is %d, expected %d'
                                 % (name, what, actual, wanted))


def cross_check(results, dataset):
    '''The two variants have to agree, or the rows are not comparable.'''
    legacy_decoder = results['legacy-equivalent'][3]
    factorized_decoder = results['factorized'][3]
    legacy_decoded = results['legacy-equivalent'][4]
    factorized_decoded = results['factorized'][4]

    for layer, subject, roi in product(dataset.layers, dataset.test_fmri,
                                       dataset.rois):
        for label in dataset.test_labels:
            expected = load_array(os.path.join(legacy_decoded, layer, subject,
                                               roi, '%s.mat' % label),
                                  key='feat')
            actual = load_array(os.path.join(factorized_decoded, layer, subject,
                                             roi, '%s.mat' % label), key='feat')
            np.testing.assert_allclose(
                actual, expected, rtol=1e-4, atol=1e-5,
                err_msg='decoded %s of %s/%s/%s differs'
                        % (label, layer, subject, roi))

        legacy_model = os.path.join(legacy_decoder, layer, subject, roi,
                                    'model')
        # The brain statistics live once per (subject, ROI) now; the feature
        # statistics are written into the per-layer directory by prediction.
        for key, directory in (
                ('x_mean', os.path.join(factorized_decoder, subject, roi,
                                        'model')),
                ('x_norm', os.path.join(factorized_decoder, subject, roi,
                                        'model')),
                ('y_mean', os.path.join(factorized_decoder, layer, subject,
                                        roi, 'model')),
                ('y_norm', os.path.join(factorized_decoder, layer, subject,
                                        roi, 'model'))):
            expected = load_array(os.path.join(legacy_model, '%s.mat' % key),
                                  key=key)
            actual = load_array(os.path.join(directory, '%s.mat' % key),
                                key=key)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-5, atol=1e-6,
                err_msg='%s of %s/%s/%s differs' % (key, layer, subject, roi))


# Reporting ###################################################################

def human_bytes(n_bytes):
    for unit, size in (('TB', 1024.0 ** 4), ('GB', 1024.0 ** 3),
                       ('MB', 1024.0 ** 2), ('KB', 1024.0)):
        if n_bytes >= size:
            return '%.1f %s' % (n_bytes / size, unit)
    return '%d B' % n_bytes


ROW = '%-7s | %-18s | %-10s | %8s | %10s | %10s | %10s | %10s'
HEADER = ROW % ('d_out', 'variant', 'phase', 'time [s]', 'log. read',
                'log. write', 'log. I/O', 'device I/O')


def print_rows(d_out, name, training, prediction):
    for phase_name, phase in (('training', training),
                              ('prediction', prediction)):
        print(ROW % (d_out if phase_name == 'training' else '', name,
                     phase_name, '%.2f' % phase.seconds,
                     human_bytes(phase.logical_read),
                     human_bytes(phase.logical_write),
                     human_bytes(phase.logical_total),
                     human_bytes(phase.device_read + phase.device_write)))
    print(ROW % ('', '', 'TOTAL', '%.2f' % (training.seconds
                                            + prediction.seconds),
                 human_bytes(training.logical_read + prediction.logical_read),
                 human_bytes(training.logical_write
                             + prediction.logical_write),
                 human_bytes(training.logical_total + prediction.logical_total),
                 human_bytes(training.device_read + training.device_write
                             + prediction.device_read
                             + prediction.device_write)))


def print_breakdown(results):
    print('')
    print('Prediction breakdown [s] (the same stages for both variants; the '
          'combination exists only in the factorized path):')
    columns = ('%-18s | %10s | %14s | %15s | %13s | %9s'
               % ('variant', 'model load', 'feature load', 'model inference',
                  'combination', 'remainder'))
    print(columns)
    print('-' * len(columns))
    for name in VARIANTS:
        prediction = results[name][1]
        inference = max(prediction.inference.seconds
                        - prediction.models.load_seconds, 0.0)
        remainder = max(prediction.seconds - prediction.inference.seconds
                        - prediction.features.seconds
                        - prediction.combine.seconds, 0.0)
        print('%-18s | %10.2f | %14.2f | %15.2f | %13.2f | %9.2f'
              % (name, prediction.models.load_seconds,
                 prediction.features.seconds, inference,
                 prediction.combine.seconds, remainder))
    print('"remainder" is reading the test fMRI, averaging it and writing the '
          'decoded features.')


def print_storage(results):
    print('')
    for name in VARIANTS:
        training, prediction, stored, _, _ = results[name]
        print('%-18s decoder tree on disk: %10s   (decoded-feature write, '
              'common to both: %s)'
              % (name, human_bytes(stored),
                 human_bytes(prediction.writes.decoded_bytes)))
    print('Stored size is what the decoder occupies, not traffic, so it is '
          'not part of any I/O total above.')


def print_extrapolation():
    '''DeepRecon arithmetic, with the load counts of the production path.'''
    item = np.dtype(DTYPE).itemsize
    layers, subjects, rois = VGG19_LAYERS, DEEPRECON_SUBJECTS, DEEPRECON_ROIS
    decoders = subjects * rois
    stimuli, voxels = DEEPRECON_STIMULI, DEEPRECON_VOXELS

    label_set = VGG19_UNITS * stimuli * item          # all layers, all stimuli
    legacy_models = VGG19_UNITS * voxels * item       # all layers, one decoder
    factorized_model = stimuli * voxels * item        # one decoder, any layer
    decoded = DEEPRECON_TEST_STIMULI * VGG19_UNITS * item
    # y_mean and y_norm are one row each per layer, so all the layers of one
    # decoder together come to two rows of the whole feature vector.
    sidecars = 2 * VGG19_UNITS * item

    rows = (
        ('training read (features)', decoders * label_set, 0),
        ('training write (models)', decoders * legacy_models,
         decoders * factorized_model),
        ('prediction read (models)', decoders * legacy_models,
         layers * decoders * factorized_model),
        ('prediction read (features)', 0, label_set),
        ('prediction write (decoded)', decoders * decoded, decoders * decoded),
        ('prediction write (y_mean/y_norm)', 0, decoders * sidecars),
    )

    print('')
    print('DeepRecon arithmetic on the same per-array sizes (VGG19 all units = '
          '%d units/stimulus over %d layers, %d training stimuli, %d test '
          'stimuli, %d voxels, %d subjects x %d ROIs, float32). Logical I/O '
          'with the load counts of the production path, not of what is stored.'
          % (VGG19_UNITS, layers, stimuli, DEEPRECON_TEST_STIMULI, voxels,
             subjects, rois))
    line = '%-34s | %12s | %12s' % ('', 'legacy', 'factorized')
    print(line)
    print('-' * len(line))
    legacy_total = factorized_total = 0
    for label, legacy, factorized in rows:
        legacy_total += legacy
        factorized_total += factorized
        print('%-34s | %12s | %12s'
              % (label, human_bytes(legacy), human_bytes(factorized)))
    print('-' * len(line))
    print('%-34s | %12s | %12s'
          % ('total logical I/O (read + write)', human_bytes(legacy_total),
             human_bytes(factorized_total)))
    print('The factorized model is stored once per (subject, ROI) but '
          'ModelTest.run() is called once per (layer, subject, ROI), so it is '
          'counted %d times; the training features are read once per layer '
          'because every subject shares the ordered training-label set. The '
          'page cache may serve some of those re-reads without touching the '
          'device, which is why these logical counts and the measured device '
          'bytes are kept apart.' % (layers * decoders))


# Entry point #################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--repeats', type=int, default=2,
                        help='timed train+predict runs per variant (best wins)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='BLAS threads; 1 keeps the comparison reproducible')
    parser.add_argument('--warm', action='store_true',
                        help='leave the page cache alone (the other bracket)')
    parser.add_argument('--quick', action='store_true',
                        help='tiny run: 20 stimuli, one layer, d_out = 1024')
    args = parser.parse_args()

    sizes = (1024,) if args.quick else FEATURE_SIZES
    n_stimuli = 20 if args.quick else N_STIMULI
    n_test = 10 if args.quick else N_TEST_STIMULI
    n_layers = 1 if args.quick else N_LAYERS
    repeats = 1 if args.quick else args.repeats
    cold = not args.warm

    print('Pipeline: %d layer(s) x %d subject(s) x %d ROI(s); training %d '
          'stimuli x %d repetitions, test %d stimuli x %d repetitions, %d '
          'voxels, alpha = %g'
          % (n_layers, N_SUBJECTS, N_ROIS, n_stimuli, N_REPEATS, n_test,
             N_TEST_REPEATS, N_VOXELS, ALPHA))
    print('%s page cache; best of %d; %d BLAS thread(s).'
          % ('Cold (feature and model files are evicted before each phase, and '
             'every feature load is evicted afterwards)' if cold
             else 'Warm (nothing is evicted)', repeats, args.threads))
    print('Logical I/O is the summed nbytes of the arrays read and written '
          '(features, model coefficients, normalization parameters, decoded '
          'features) and is exact. The fMRI files are identical for both '
          'variants and are not in it; they are in the device I/O, which is '
          '/proc/self/io read+write around the whole phase, on this machine\'s '
          'local disk.')
    print('')
    print(HEADER)
    print('-' * len(HEADER))

    root = tempfile.mkdtemp(prefix='featdec-pipeline-')
    cwd = os.getcwd()
    try:
        with threadpool_limits(limits=args.threads):
            for d_out in sizes:
                data_root = os.path.join(root, 'data-%d' % d_out)
                dataset = build_dataset(data_root, n_stimuli=n_stimuli,
                                        n_test=n_test, n_layers=n_layers,
                                        d_out=d_out, n_subjects=N_SUBJECTS,
                                        n_rois=N_ROIS)
                workdir = os.path.join(root, 'work-%d' % d_out)
                os.makedirs(workdir, exist_ok=True)
                os.chdir(workdir)  # the scripts write ./tmp/<analysis>.db

                results = {}
                for name in VARIANTS:
                    results[name] = run_variant(name, dataset, workdir, cold,
                                                repeats)
                    training, prediction = results[name][0], results[name][1]
                    check_expected_counts(name, training, prediction, dataset,
                                          n_stimuli)
                    print_rows(d_out, name, training, prediction)

                cross_check(results, dataset)
                if d_out == sizes[-1]:
                    print_breakdown(results)
                    print_storage(results)

                os.chdir(cwd)
                shutil.rmtree(data_root, ignore_errors=True)
                shutil.rmtree(workdir, ignore_errors=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)

    print_extrapolation()


if __name__ == '__main__':
    main()
