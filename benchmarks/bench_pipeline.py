'''What one whole decoding run costs, before and after the factorization.

Runs the same pipeline twice over the same synthetic data -- ``train`` then
``predict``, each in its own process, as they are run in practice -- and
reports what a user actually feels: wall time, peak memory, every byte read and
written, and the size of the decoder left on disk.

``legacy``
    The pre-PR scripts.  Training is the base-commit script itself, extracted
    with ``git show 7322377:train_decoder_sklearn_ridge.py``, so the baseline is
    not a reimplementation.  Prediction is today's ``predict_feature.py``, which
    detects the old format and takes the code path it always did.
``factorized``
    Today's ``train_decoder_sklearn_ridge.py`` and ``predict_feature.py``.

Every cost of the phase is counted, including the parts the two variants share
(the fMRI files, the decoded features, the Python interpreter itself): the
question is what a run costs, not where the two differ.  For reference the
script also measures an empty process that only imports the same modules.

    uv run python -m benchmarks.bench_pipeline                  # ~5 min
    uv run python -m benchmarks.bench_pipeline --profile conv   # ~5 min, ~6 GB RAM
    uv run python -m benchmarks.bench_pipeline --profile quick  # smoke test

Not part of the test suite -- that training opens no feature file is pinned by
``tests/test_training_feature_loading.py``.
'''

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from itertools import product
from time import perf_counter

import numpy as np
from bdpy import BData
from bdpy.dataform import load_array, save_array
from threadpoolctl import threadpool_limits

import bdpy.dataform.features as bdpy_features
import bdpy.ml.learning as bdpy_learning

LABEL_KEY = 'stimulus_name'
ALPHA = 100.0
DTYPE = np.float32
CHUNK_AXIS = 1
LEGACY_COMMIT = '7322377'
LEGACY_SCRIPT = 'train_decoder_sklearn_ridge.py'

# The measured settings.  'fc' is the shipped config's own layer list
# (deeprecon_sklearn_ridge_alpha100_vgg19_allunits.yaml enables fc6/fc7/fc8) at
# DeepRecon's stimulus count; 'conv' is one convolutional layer, where the
# decoder stops being small.
PROFILES = {
    'quick': dict(layers=(('fc8', 1000),), n_stimuli=50, n_repeats=5,
                  n_test=10, n_voxels=300, n_subjects=2, n_rois=2, repeats=1),
    'fc': dict(layers=(('fc6', 4096), ('fc7', 4096), ('fc8', 1000)),
               n_stimuli=1200, n_repeats=5, n_test=50, n_voxels=10000,
               n_subjects=2, n_rois=2, repeats=1),
    'conv': dict(layers=(('conv5_1', 14 * 14 * 512),), n_stimuli=1200,
                 n_repeats=5, n_test=50, n_voxels=10000, n_subjects=1,
                 n_rois=1, repeats=1),
}

# `conv` puts d_out >> d_in > n (100352 >> 10000 > 6000), which is the regime
# a convolutional layer of a real ROI is in.  If it does not fit in memory,
# --scale divides all three by the same factor instead of changing the layer,
# so the regime is preserved.  Voxel counts of real ROIs run to ~15000; 10000
# is what these runs measure.
N_TEST_REPEATS = 2

# DeepRecon VGG19 "allunits": units per stimulus over the 16 conv and 3 fc
# layers, and the shape of a full run.
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

def build_dataset(root, profile):
    '''Write a feature store, training fMRI and test fMRI.

    Every ROI selects every voxel: the ROI changes the fit, not which files are
    read.  Every subject sees the same stimuli in the same order, which is the
    ordinary case and the one in which prediction reads each layer's training
    features once for the whole run.  Features are 2-D (one vector per
    stimulus), the case ``bdpy`` does not chunk -- as ``fc6``/``fc7``/``fc8``
    are in the shipped config.
    '''
    rng = np.random.RandomState(0)
    layers = [name for name, _ in profile['layers']]

    unique_labels = ['stim%05d' % (i + 1) for i in range(profile['n_stimuli'])]
    test_labels = ['test%05d' % (i + 1) for i in range(profile['n_test'])]
    trial_labels = [lb for lb in unique_labels
                    for _ in range(profile['n_repeats'])]
    test_trial_labels = [lb for lb in test_labels
                         for _ in range(N_TEST_REPEATS)]
    rois = {'roi%d' % (i + 1): 'ROI_roi%d = 1' % (i + 1)
            for i in range(profile['n_rois'])}

    features_dir = os.path.join(root, 'features')
    for layer, d_out in profile['layers']:
        layer_dir = os.path.join(features_dir, layer)
        os.makedirs(layer_dir, exist_ok=True)
        for label in unique_labels:
            save_array(os.path.join(layer_dir, '%s.mat' % label),
                       rng.randn(1, d_out).astype(DTYPE), key='feat',
                       dtype=DTYPE, sparse=False)

    train_fmri, test_fmri = {}, {}
    for i in range(profile['n_subjects']):
        subject = 'sub-%02d' % (i + 1)
        train_fmri[subject] = [_write_bdata(
            os.path.join(root, 'fmri', '%s_train.h5' % subject),
            rng.randn(len(trial_labels), profile['n_voxels']), trial_labels,
            unique_labels, rois)]
        test_fmri[subject] = [_write_bdata(
            os.path.join(root, 'fmri', '%s_test.h5' % subject),
            rng.randn(len(test_trial_labels), profile['n_voxels']),
            test_trial_labels, test_labels, rois)]

    return {
        'features_dir': features_dir,
        'train_fmri': train_fmri,
        'test_fmri': test_fmri,
        'layers': layers,
        'rois': rois,
        'unique_labels': unique_labels,
        'test_labels': test_labels,
    }


def _write_bdata(path, brain, labels, unique_labels, rois):
    label_to_number = {label: i + 1 for i, label in enumerate(unique_labels)}
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
    return path


def input_files(dataset):
    '''Everything a phase may read, for the page-cache eviction.'''
    paths = [os.path.join(dataset['features_dir'], layer, '%s.mat' % label)
             for layer in dataset['layers']
             for label in dataset['unique_labels']]
    for group in ('train_fmri', 'test_fmri'):
        for files in dataset[group].values():
            paths.extend(files)
    return paths


# Process-level measurement ###################################################

def peak_rss():
    '''The kernel's own high-water mark for this process, in bytes.'''
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    return 0


def process_io():
    '''``rchar``/``wchar`` (syscall bytes) and the device counters.'''
    wanted = ('rchar', 'wchar', 'read_bytes', 'write_bytes')
    values = dict.fromkeys(wanted, 0)
    try:
        with open('/proc/self/io') as f:
            for line in f:
                key, _, value = line.partition(':')
                if key in values:
                    values[key] = int(value)
    except OSError:
        pass
    return values


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
    '''Drop ``paths`` from the page cache, syncing the dirty ones first.'''
    os.sync()
    drop_from_cache(paths)


def tree_files(root):
    return [os.path.join(base, name)
            for base, _, names in os.walk(root) for name in names]


def tree_bytes(root):
    return sum(os.path.getsize(path) for path in tree_files(root))


# In-process probes (sanity checks and the prediction breakdown) ##############

class FeatureLoadProbe(object):
    '''Counts and times the per-stimulus feature-file loads.'''

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
    model = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    total = 0
    for attribute in ('coef_', 'intercept_'):
        value = getattr(model, attribute, None)
        if value is not None:
            total += int(np.asarray(value).nbytes)
    return total


class ModelProbe(object):
    '''Model coefficients read from and written to the decoder.'''

    def __init__(self, modules):
        self._modules = modules
        self.loads = 0
        self.load_bytes = 0
        self.load_seconds = 0.0
        self.dumps = 0
        self.dump_bytes = 0

    @contextlib.contextmanager
    def installed(self):
        originals = [(module, module.pickle) for module in self._modules]
        for module, real in originals:
            module.pickle = _PickleShim(real, self)
        try:
            yield self
        finally:
            for module, real in originals:
                module.pickle = real


class TimeProbe(object):
    '''Time spent inside one patched callable.'''

    def __init__(self, owner, name):
        self._owner = owner
        self._name = name
        self.seconds = 0.0

    @contextlib.contextmanager
    def installed(self):
        original = getattr(self._owner, self._name)

        def timed(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds += perf_counter() - start

        setattr(self._owner, self._name, timed)
        try:
            yield self
        finally:
            setattr(self._owner, self._name, original)


# The phases, as they run in the worker process ###############################

def load_legacy_script(path):
    '''Import the extracted pre-PR training script.'''
    spec = importlib.util.spec_from_file_location('legacy_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training(spec, factorized_module):
    dataset = spec['dataset']
    if spec['variant'] == 'legacy':
        legacy = load_legacy_script(spec['legacy_script'])
        legacy.featdec_sklearn_ridge_train(
            dataset['train_fmri'],
            [dataset['features_dir']],
            output_dir=spec['decoder_dir'],
            rois=dataset['rois'],
            label_key=LABEL_KEY,
            layers=list(dataset['layers']),
            alpha=ALPHA,
            chunk_axis=CHUNK_AXIS,
            analysis_name='legacy_training',
        )
    else:
        factorized_module.featdec_sklearn_ridge_train(
            dataset['train_fmri'],
            output_dir=spec['decoder_dir'],
            rois=dataset['rois'],
            label_key=LABEL_KEY,
            alpha=ALPHA,
            analysis_name='factorized_training',
        )


def run_prediction(spec, predict_module):
    dataset = spec['dataset']
    predict_module.featdec_predict(
        dataset['test_fmri'],
        spec['decoder_dir'],
        output_dir=spec['decoded_dir'],
        rois=dataset['rois'],
        label_key=LABEL_KEY,
        layers=list(dataset['layers']),
        excluded_labels=[],
        average_sample=True,
        chunk_axis=CHUNK_AXIS,
        training_features_paths=([dataset['features_dir']]
                                 if spec['variant'] == 'factorized' else None),
        analysis_name='%s_prediction' % spec['variant'],
    )


def worker(spec_path):
    '''Run one phase and report what the whole process cost.'''
    start = perf_counter()
    with open(spec_path) as f:
        spec = json.load(f)

    os.makedirs(spec['workdir'], exist_ok=True)
    os.chdir(spec['workdir'])

    # Imported here rather than at module scope so that the baseline
    # process pays exactly the import cost every other phase pays.
    import predict_feature
    import ridge_factorization
    import train_decoder_sklearn_ridge

    features = FeatureLoadProbe(evict_after_load=spec['cold'])
    models = ModelProbe((bdpy_learning, ridge_factorization))
    inference = TimeProbe(bdpy_learning.ModelTest, 'run')
    combine = TimeProbe(predict_feature, 'combine_features')

    phase_seconds = 0.0
    if spec['phase'] != 'baseline':
        with threadpool_limits(limits=spec['threads']):
            with features.installed(), models.installed(), \
                    inference.installed(), combine.installed():
                phase_start = perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    if spec['phase'] == 'training':
                        run_training(spec, train_decoder_sklearn_ridge)
                    else:
                        run_prediction(spec, predict_feature)
                phase_seconds = perf_counter() - phase_start

    os.sync()  # so the writes are on the device counters before we read them
    report = dict(process_io())
    report.update({
        'phase_seconds': phase_seconds,
        'process_seconds': perf_counter() - start,
        'peak_rss': peak_rss(),
        'feature_loads': features.loads,
        'feature_logical_bytes': features.logical_bytes,
        'feature_seconds': features.seconds,
        'model_loads': models.loads,
        'model_logical_bytes': models.load_bytes,
        'model_dumps': models.dumps,
        'model_load_seconds': models.load_seconds,
        'inference_seconds': inference.seconds,
        'combine_seconds': combine.seconds,
    })
    with open(spec['report'], 'w') as f:
        json.dump(report, f)


# Driving the workers #########################################################

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_legacy_script(destination):
    '''The pre-PR training script, straight out of git history.'''
    try:
        source = subprocess.check_output(
            ['git', '-C', REPO_ROOT, 'show',
             '%s:%s' % (LEGACY_COMMIT, LEGACY_SCRIPT)],
            stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            'Cannot read %s:%s from git (%s). The legacy baseline is the '
            'pre-PR script itself, so this benchmark needs the repository '
            'history.' % (LEGACY_COMMIT, LEGACY_SCRIPT, error))
    with open(destination, 'wb') as f:
        f.write(source)
    return destination


def run_phase(phase, variant, dataset, workdir, decoder_dir, decoded_dir,
              legacy_script, threads, cold):
    '''Run one phase in a fresh process and return its report.'''
    spec_path = os.path.join(workdir, 'spec.json')
    report_path = os.path.join(workdir, 'report.json')
    spec = {
        'phase': phase,
        'variant': variant,
        'dataset': dataset,
        'workdir': workdir,
        'decoder_dir': decoder_dir,
        'decoded_dir': decoded_dir,
        'legacy_script': legacy_script,
        'threads': threads,
        'cold': cold,
        'report': report_path,
    }
    with open(spec_path, 'w') as f:
        json.dump(spec, f)

    start = perf_counter()
    try:
        subprocess.run([sys.executable, '-m', 'benchmarks.bench_pipeline',
                        '--worker', spec_path],
                       cwd=REPO_ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=None)
    except subprocess.CalledProcessError as error:
        if error.returncode == -signal.SIGKILL:
            # Not a crash but a result: the phase did not fit in this
            # machine's memory.  Scaling keeps d_out : d_in : n, so the
            # smaller run is still the same regime.
            raise SystemExit(
                '%s %s was killed (SIGKILL): it did not fit in this machine\'s '
                'memory. Re-run with --scale (e.g. --scale 2), which divides '
                'trials, voxels and output units by one common factor and so '
                'keeps the profile\'s d_out : d_in : n.'
                % (variant, phase))
        raise
    wall = perf_counter() - start

    with open(report_path) as f:
        report = json.load(f)
    report['wall_seconds'] = wall
    return report


def run_variant(variant, dataset, workdir, legacy_script, threads, cold,
                repeats):
    '''Best-of-``repeats`` ``train`` then ``predict``, each a fresh process.

    Each repeat trains into an empty directory and predicts from a pristine
    copy of the decoder into an empty output directory: the factorized
    prediction writes ``y_mean``/``y_norm`` on its first pass and skips them
    afterwards, and both scripts skip a decoded-feature directory that exists.
    '''
    pristine = os.path.join(workdir, 'decoder-%s' % variant)
    decoder_dir = os.path.join(workdir, 'predict-decoder-%s' % variant)
    decoded_dir = os.path.join(workdir, 'decoded-%s' % variant)
    best = None

    for _ in range(repeats):
        for path in (pristine, decoder_dir, decoded_dir):
            shutil.rmtree(path, ignore_errors=True)
        if cold:
            evict(input_files(dataset))
        training = run_phase('training', variant, dataset, workdir, pristine,
                             decoded_dir, legacy_script, threads, cold)

        shutil.copytree(pristine, decoder_dir)
        if cold:
            evict(input_files(dataset) + tree_files(decoder_dir))
        prediction = run_phase('prediction', variant, dataset, workdir,
                               decoder_dir, decoded_dir, legacy_script,
                               threads, cold)

        stored = tree_bytes(pristine)
        candidate = (training, prediction, stored, decoder_dir, decoded_dir)
        if best is None or (training['wall_seconds']
                            + prediction['wall_seconds']
                            < best[0]['wall_seconds']
                            + best[1]['wall_seconds']):
            best = candidate

    return best


def check_expected_counts(variant, training, prediction, dataset):
    '''The load counts are part of the claim; assert them.'''
    layers = len(dataset['layers'])
    decoders = len(dataset['train_fmri']) * len(dataset['rois'])
    stimuli = len(dataset['unique_labels'])

    if variant == 'legacy':
        expected = {
            'training feature loads': (training['feature_loads'],
                                       layers * decoders * stimuli),
            'training model writes': (training['model_dumps'],
                                      layers * decoders),
            'prediction feature loads': (prediction['feature_loads'], 0),
            'prediction model loads': (prediction['model_loads'],
                                       layers * decoders),
        }
    else:
        expected = {
            'training feature loads': (training['feature_loads'], 0),
            'training model writes': (training['model_dumps'], decoders),
            # One read per layer: the subjects share the ordered label set and
            # the layer is the outer loop of featdec_predict.
            'prediction feature loads': (prediction['feature_loads'],
                                         layers * stimuli),
            # Stored once per (subject, ROI), but ModelTest.run() is called
            # once per (layer, subject, ROI).
            'prediction model loads': (prediction['model_loads'],
                                       layers * decoders),
        }

    for what, (actual, wanted) in sorted(expected.items()):
        if actual != wanted:
            raise AssertionError('%s: %s is %d, expected %d'
                                 % (variant, what, actual, wanted))


def cross_check(results, dataset):
    '''The two variants have to agree, or the rows are not comparable.'''
    legacy_decoder, legacy_decoded = results['legacy'][3], results['legacy'][4]
    new_decoder, new_decoded = (results['factorized'][3],
                                results['factorized'][4])

    for layer, subject, roi in product(dataset['layers'], dataset['test_fmri'],
                                       dataset['rois']):
        for label in dataset['test_labels']:
            expected = load_array(os.path.join(legacy_decoded, layer, subject,
                                               roi, '%s.mat' % label),
                                  key='feat')
            actual = load_array(os.path.join(new_decoded, layer, subject, roi,
                                             '%s.mat' % label), key='feat')
            np.testing.assert_allclose(
                actual, expected, rtol=1e-4, atol=1e-5,
                err_msg='decoded %s of %s/%s/%s differs'
                        % (label, layer, subject, roi))

        legacy_model = os.path.join(legacy_decoder, layer, subject, roi,
                                    'model')
        shared = os.path.join(new_decoder, subject, roi, 'model')
        per_layer = os.path.join(new_decoder, layer, subject, roi, 'model')
        for key, directory in (('x_mean', shared), ('x_norm', shared),
                               ('y_mean', per_layer), ('y_norm', per_layer)):
            np.testing.assert_allclose(
                load_array(os.path.join(directory, '%s.mat' % key), key=key),
                load_array(os.path.join(legacy_model, '%s.mat' % key), key=key),
                rtol=1e-5, atol=1e-6,
                err_msg='%s of %s/%s/%s differs' % (key, layer, subject, roi))


# Reporting ###################################################################

def human_bytes(n_bytes):
    for unit, size in (('TB', 1024.0 ** 4), ('GB', 1024.0 ** 3),
                       ('MB', 1024.0 ** 2), ('KB', 1024.0)):
        if n_bytes >= size:
            return '%.1f %s' % (n_bytes / size, unit)
    return '%d B' % n_bytes


def change(legacy, factorized):
    '''How the factorized run compares, as a human would say it.'''
    if factorized <= 0:
        return 'n/a'
    ratio = float(legacy) / float(factorized)
    if ratio >= 1.05:
        return '%.1fx less' % ratio
    if ratio <= 0.95:
        return '%.1fx more' % (1.0 / ratio)
    return 'about the same'


def print_headline(results):
    legacy_train, legacy_predict, legacy_stored = results['legacy'][:3]
    new_train, new_predict, new_stored = results['factorized'][:3]

    rows = (
        ('training time', legacy_train['wall_seconds'],
         new_train['wall_seconds'], 'seconds'),
        ('prediction time', legacy_predict['wall_seconds'],
         new_predict['wall_seconds'], 'seconds'),
        ('total time', legacy_train['wall_seconds']
         + legacy_predict['wall_seconds'],
         new_train['wall_seconds'] + new_predict['wall_seconds'], 'seconds'),
        ('peak memory (training)', legacy_train['peak_rss'],
         new_train['peak_rss'], 'bytes'),
        ('peak memory (prediction)', legacy_predict['peak_rss'],
         new_predict['peak_rss'], 'bytes'),
        ('total disk read', legacy_train['rchar'] + legacy_predict['rchar'],
         new_train['rchar'] + new_predict['rchar'], 'bytes'),
        ('total disk write', legacy_train['wchar'] + legacy_predict['wchar'],
         new_train['wchar'] + new_predict['wchar'], 'bytes'),
        ('decoder size', legacy_stored, new_stored, 'bytes'),
    )

    header = '%-26s | %12s | %12s | %14s' % ('', 'legacy', 'factorized',
                                             'change')
    print(header)
    print('-' * len(header))
    for label, legacy, factorized, unit in rows:
        if unit == 'seconds':
            shown = ('%.1f s' % legacy, '%.1f s' % factorized)
        else:
            shown = (human_bytes(legacy), human_bytes(factorized))
        print('%-26s | %12s | %12s | %14s'
              % (label, shown[0], shown[1], change(legacy, factorized)))


def print_phases(results, baseline):
    header = ('%-12s | %-10s | %8s | %10s | %10s | %10s | %11s | %11s'
              % ('variant', 'phase', 'time [s]', 'peak RSS', 'read',
                 'written', 'device read', 'device wr.'))
    print('')
    print('Per phase (whole process, interpreter and imports included):')
    print(header)
    print('-' * len(header))
    for variant in ('legacy', 'factorized'):
        for name, report in (('training', results[variant][0]),
                             ('prediction', results[variant][1])):
            print('%-12s | %-10s | %8.1f | %10s | %10s | %10s | %11s | %11s'
                  % (variant, name, report['wall_seconds'],
                     human_bytes(report['peak_rss']),
                     human_bytes(report['rchar']),
                     human_bytes(report['wchar']),
                     human_bytes(report['read_bytes']),
                     human_bytes(report['write_bytes'])))
    print('%-12s | %-10s | %8.1f | %10s | %10s | %10s | %11s | %11s'
          % ('(reference)', 'imports', baseline['wall_seconds'],
             human_bytes(baseline['peak_rss']), human_bytes(baseline['rchar']),
             human_bytes(baseline['wchar']),
             human_bytes(baseline['read_bytes']),
             human_bytes(baseline['write_bytes'])))
    print('"imports" is a process that imports the same modules and exits: the '
          'floor under every row above.')


def print_breakdown(results):
    print('')
    print('Inside prediction [s] (the same stages for both; only the '
          'factorized path has a combination):')
    header = ('%-12s | %10s | %12s | %15s | %11s | %9s'
              % ('variant', 'model load', 'feature load', 'model inference',
                 'combination', 'the rest'))
    print(header)
    print('-' * len(header))
    for variant in ('legacy', 'factorized'):
        report = results[variant][1]
        inference = max(report['inference_seconds']
                        - report['model_load_seconds'], 0.0)
        rest = max(report['phase_seconds'] - report['inference_seconds']
                   - report['feature_seconds'] - report['combine_seconds'], 0.0)
        print('%-12s | %10.1f | %12.1f | %15.1f | %11.1f | %9.1f'
              % (variant, report['model_load_seconds'],
                 report['feature_seconds'], inference,
                 report['combine_seconds'], rest))
    print('"the rest" is reading the test fMRI, averaging it and writing the '
          'decoded features -- work both variants do identically.')


def print_extrapolation():
    '''DeepRecon arithmetic, with the load counts of the production path.'''
    item = np.dtype(DTYPE).itemsize
    layers, subjects, rois = VGG19_LAYERS, DEEPRECON_SUBJECTS, DEEPRECON_ROIS
    decoders = subjects * rois
    stimuli, voxels = DEEPRECON_STIMULI, DEEPRECON_VOXELS

    label_set = VGG19_UNITS * stimuli * item      # all layers, all stimuli
    legacy_models = VGG19_UNITS * voxels * item   # all layers, one decoder
    factorized_model = stimuli * voxels * item    # one decoder, any layer
    decoded = DEEPRECON_TEST_STIMULI * VGG19_UNITS * item
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
    print('DeepRecon arithmetic -- not a measurement. VGG19 all units '
          '(%d units/stimulus over %d layers), %d training and %d test '
          'stimuli, %d voxels, %d subjects x %d ROIs, float32. Load counts as '
          'the production path performs them; the fMRI is left out because it '
          'is the same for both.'
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
          % ('total', human_bytes(legacy_total),
             human_bytes(factorized_total)))
    print('The factorized model is stored once per (subject, ROI) but '
          'ModelTest.run() is called once per (layer, subject, ROI), so it is '
          'read back %d times; the training features are read once per layer '
          'because every subject shares the ordered training-label set.'
          % (layers * decoders))


def print_sanity(results):
    print('')
    print('Sanity checks (asserted, not just printed): feature-file loads and '
          'model loads/writes')
    for variant in ('legacy', 'factorized'):
        training, prediction = results[variant][0], results[variant][1]
        print('  %-11s training: %5d feature loads, %2d models written; '
              'prediction: %5d feature loads, %3d models read'
              % (variant, training['feature_loads'], training['model_dumps'],
                 prediction['feature_loads'], prediction['model_loads']))


# Entry point #################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('-p', '--profile', default='fc',
                        choices=sorted(PROFILES),
                        help='fc: the shipped config\'s fc6/fc7/fc8 at '
                             'DeepRecon\'s stimulus count (default); conv: one '
                             'convolutional layer (needs ~6 GB RAM); quick: a '
                             'smoke test')
    parser.add_argument('-r', '--repeats', type=int, default=None,
                        help='timed train+predict runs per variant (best wins)')
    parser.add_argument('--voxels', type=int, default=None,
                        help='override the profile\'s voxel count (real ROIs '
                             'run to ~15000; the profiles measure 10000)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='divide trials, voxels and output units by this '
                             'common factor, keeping their ratios, when the '
                             'profile does not fit in memory')
    parser.add_argument('-t', '--threads', type=int, default=os.cpu_count(),
                        help='BLAS threads (default: every core, as a real run '
                             'would use)')
    parser.add_argument('--warm', action='store_true',
                        help='leave the page cache alone (the other bracket)')
    args = parser.parse_args()

    if args.worker:
        worker(args.worker)
        return

    profile = dict(PROFILES[args.profile])
    if args.voxels:
        profile['n_voxels'] = args.voxels
    if args.scale != 1.0:
        # One factor for all three dimensions, so d_out : d_in : n is what the
        # profile says it is, only smaller.
        profile['n_stimuli'] = max(int(profile['n_stimuli'] / args.scale), 2)
        profile['n_voxels'] = max(int(profile['n_voxels'] / args.scale), 1)
        profile['layers'] = tuple(
            (name, max(int(units / args.scale), 1))
            for name, units in profile['layers'])
    repeats = args.repeats or profile['repeats']
    cold = not args.warm

    print('Profile %r: %s; %d training stimuli x %d repetitions, %d test '
          'stimuli x %d repetitions, %d voxels, %d subject(s) x %d ROI(s), '
          'alpha = %g'
          % (args.profile,
             ', '.join('%s (%d units)' % pair for pair in profile['layers']),
             profile['n_stimuli'], profile['n_repeats'], profile['n_test'],
             N_TEST_REPEATS, profile['n_voxels'], profile['n_subjects'],
             profile['n_rois'], ALPHA))
    if args.scale != 1.0:
        print('Scaled down by %g: trials, voxels and output units divided by '
              'the same factor, so their ratios are the profile\'s.'
              % args.scale)
    print('%s page cache; best of %d; %d BLAS thread(s); training and '
          'prediction each in their own process.'
          % ('Cold (inputs evicted before every phase)' if cold
             else 'Warm (nothing is evicted)', repeats, args.threads))
    print('')

    root = tempfile.mkdtemp(prefix='featdec-pipeline-')
    try:
        dataset = build_dataset(os.path.join(root, 'data'), profile)
        workdir = os.path.join(root, 'work')
        os.makedirs(workdir)
        legacy_script = extract_legacy_script(
            os.path.join(root, 'legacy_train_decoder.py'))

        baseline = run_phase('baseline', 'factorized', dataset, workdir, '', '',
                             legacy_script, args.threads, cold)

        results = {}
        for variant in ('legacy', 'factorized'):
            results[variant] = run_variant(variant, dataset, workdir,
                                           legacy_script, args.threads, cold,
                                           repeats)
            check_expected_counts(variant, results[variant][0],
                                  results[variant][1], dataset)
        cross_check(results, dataset)

        print_headline(results)
        print_phases(results, baseline)
        print_breakdown(results)
        print_sanity(results)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print_extrapolation()


if __name__ == '__main__':
    main()
