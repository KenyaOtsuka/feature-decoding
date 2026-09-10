'''Benchmark the feature-store I/O of one decoder training run.

The other benchmark (``bench_ridge_factorization``) measures the compute and the
stored model size.  This one measures what training *reads*, because the
factorized fit targets the training stimulus basis and therefore needs the trial
labels and no feature values at all.  Two variants of the same training run
are compared:

``legacy-equivalent``
    A reimplementation of the pre-PR loading pattern (base commit ``7322377``):
    ``(layer, subject, ROI)`` with ``get_multi_features`` inside the loop, so
    every layer is read once per (subject, ROI).
``factorized``
    Today's script: the fit is on the stimulus basis, so it takes no
    feature-side argument and never opens a feature file.

``bdpy``'s labelled ``Features.get`` does not cache (only the label-less
``get_features`` does, and ``get_multi_features`` never calls it), so the
``L x S x R`` factor of the first variant is real rather than assumed.

    uv run python -m benchmarks.bench_training_io

Deliberately small so it runs anywhere (~90 s, ~60 MB of temporary files); the
point is the *ratio* of the reads, which does not depend on the size.  Not part of the test suite -- that training
opens no feature file is pinned by
``tests/test_training_feature_loading.py``.
'''

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import tempfile
from itertools import product
from time import perf_counter

import bdpy
import numpy as np
from bdpy import BData
from bdpy.bdata.utils import get_labels_multi_bdatas, select_data_multi_bdatas
from bdpy.dataform import Features, load_array, save_array
from bdpy.dataform.utils import get_multi_features
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import bdpy.dataform.features as bdpy_features
import train_decoder_sklearn_ridge

LABEL_KEY = 'stimulus_name'
ALPHA = 100.0
DTYPE = np.float32

# Small stand-in for the real setting (M ~ 1200, 9 layers, 5 subjects, 9 ROIs).
N_STIMULI = 200
N_REPEATS = 5
N_VOXELS = 300
N_LAYERS = 2
N_SUBJECTS = 2
N_ROIS = 2
FEATURE_SIZES = (4096, 32768)

# DeepRecon VGG19 "allunits": units per stimulus, summed over the 16 conv and
# 3 fc layers, and the shape of a full training run.
VGG19_LAYER_UNITS = (
    (2, 224 * 224 * 64), (2, 112 * 112 * 128), (4, 56 * 56 * 256),
    (4, 28 * 28 * 512), (4, 14 * 14 * 512), (1, 4096 + 4096 + 1000),
)
VGG19_UNITS = sum(count * units for count, units in VGG19_LAYER_UNITS)
DEEPRECON_STIMULI = 1200
DEEPRECON_SUBJECTS = 5
DEEPRECON_ROIS = 9


# Synthetic dataset ###########################################################

class Dataset(object):
    '''Paths and shape of one synthetic training dataset on disk.'''

    def __init__(self, root, fmri, features_dir, layers, rois, unique_labels,
                 n_trials, d_out):
        self.root = root
        self.fmri = fmri
        self.features_dir = features_dir
        self.layers = layers
        self.rois = rois
        self.unique_labels = unique_labels
        self.n_trials = n_trials
        self.d_out = d_out

    @property
    def feature_files(self):
        return [os.path.join(self.features_dir, layer, '%s.mat' % label)
                for layer in self.layers for label in self.unique_labels]

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
                  n_voxels=N_VOXELS, n_layers=N_LAYERS, d_out=1024,
                  n_subjects=N_SUBJECTS, n_rois=N_ROIS, seed=0):
    '''Write a feature store and one ``BData`` file per subject.

    Every ROI selects every voxel: the ROI changes the brain-side fit, never
    which feature files are read, so keeping them the same size keeps the rows
    of the table comparable.  Features are 2-D (one vector per stimulus), which
    is the case ``bdpy`` does not chunk -- the same setting the other benchmark
    uses.
    '''
    rng = np.random.RandomState(seed)

    unique_labels = ['stim%05d' % (i + 1) for i in range(n_stimuli)]
    label_to_number = {lb: i + 1 for i, lb in enumerate(unique_labels)}
    trial_labels = [lb for lb in unique_labels for _ in range(n_repeats)]
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

    fmri = {}
    for i in range(n_subjects):
        subject = 'sub-%02d' % (i + 1)
        path = os.path.join(root, 'fmri', '%s.h5' % subject)
        _write_bdata(path, rng.randn(len(trial_labels), n_voxels),
                     trial_labels, label_to_number, rois)
        fmri[subject] = [path]

    return Dataset(root, fmri, features_dir, layers, rois, unique_labels,
                   len(trial_labels), d_out)


# Instrumentation #############################################################

def _device_read_bytes():
    '''Bytes this process has fetched from the block device, or None.'''
    try:
        with open('/proc/self/io') as f:
            for line in f:
                if line.startswith('read_bytes:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


class FeatureLoadProbe(object):
    '''Measure every per-stimulus feature-file load of a training run.

    ``loads`` and ``logical_bytes`` are exact and machine-independent: the
    number of times a feature file is loaded, and the summed ``nbytes`` of the
    arrays the loader returns.  Deliberately not the sum of the file sizes -- an
    hdf5storage file carries container overhead, and with a feature index the
    file holds more units than the selected array, so the file size is neither
    what is loaded nor what is transferred.

    ``device_bytes`` is read from ``/proc/self/io`` around each load only, so
    the ``BData`` files, the DistComp database and the decoder writes cannot
    contaminate it.  It is evidence about the page cache, not a primary metric.
    '''

    def __init__(self, evict_after_load=False):
        self._evict_after_load = evict_after_load
        self.loads = 0
        self.logical_bytes = 0
        self.seconds = 0.0
        self.device_bytes = 0

    @property
    def counts(self):
        '''The deterministic part, for comparing repetitions.'''
        return (self.loads, self.logical_bytes)

    @contextlib.contextmanager
    def installed(self):
        load = bdpy_features._load_array_with_key

        def counted_load(key, path):
            before = _device_read_bytes()
            start = perf_counter()
            array = load(key, path)
            elapsed = perf_counter() - start
            after = _device_read_bytes()
            self.loads += 1
            self.logical_bytes += int(np.asarray(array).nbytes)
            self.seconds += elapsed
            if before is not None and after is not None:
                self.device_bytes += after - before
            if self._evict_after_load:
                # A 15 GB layer never stays in the page cache across the
                # (subject, ROI) iterations, so neither may this store: without
                # this, only the first pass over a small store reaches the disk
                # and every repeated read looks free.
                drop_from_cache([path])
            return array

        bdpy_features._load_array_with_key = counted_load
        try:
            yield self
        finally:
            bdpy_features._load_array_with_key = load


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

    The store has just been written, and ``POSIX_FADV_DONTNEED`` leaves dirty
    pages alone, so the ``os.sync()`` is what makes the eviction take effect.
    '''
    os.sync()
    drop_from_cache(paths)


# The three variants ##########################################################

def legacy_equivalent_train(dataset, output_dir):
    '''The pre-PR loading pattern, reimplemented.

    This is *not* the base-commit script -- that no longer exists in the tree.
    What it reproduces faithfully is the loop and the per-(layer, subject, ROI)
    ``get_multi_features`` call, which is what this benchmark measures, plus the
    feature statistics and the Ridge fit on the expanded, normalized target.
    The model pickle is not written: its size is the storage table's subject,
    and writing tens of MB per decoder would only add noise here.  The "total"
    column therefore understates this variant.
    '''
    data_brain = {subject: [bdpy.BData(f) for f in files]
                  for subject, files in dataset.fmri.items()}
    stores = [Features(dataset.features_dir)]

    for layer, subject, roi in product(dataset.layers, dataset.fmri,
                                       dataset.rois):
        model_dir = os.path.join(output_dir, layer, subject, roi, 'model')
        os.makedirs(model_dir, exist_ok=True)

        brain = select_data_multi_bdatas(data_brain[subject], dataset.rois[roi])
        brain_labels = get_labels_multi_bdatas(data_brain[subject], LABEL_KEY)

        feat_labels = np.unique(brain_labels)
        feat = get_multi_features(stores, layer, labels=feat_labels)

        brain = np.vstack([b for b, lb in zip(brain, brain_labels)
                           if lb in feat_labels])
        brain_labels = [lb for lb in brain_labels if lb in feat_labels]

        brain_mean = np.mean(brain, axis=0)[np.newaxis, :]
        brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]
        feat_mean = np.mean(feat, axis=0)[np.newaxis, :]
        feat_norm = np.std(feat, axis=0, ddof=1)[np.newaxis, :]

        index = np.array([np.where(np.array(feat_labels) == lb)
                          for lb in brain_labels]).flatten()

        x = ((brain - brain_mean) / brain_norm).astype(DTYPE)
        x[np.isinf(x)] = 0
        y = ((feat[index] - feat_mean) / feat_norm).astype(DTYPE)
        y[np.isinf(y)] = 0

        Ridge(alpha=ALPHA).fit(x, y)

        for key, value in (('x_mean', brain_mean), ('x_norm', brain_norm),
                           ('y_mean', feat_mean), ('y_norm', feat_norm)):
            save_array(os.path.join(model_dir, '%s.mat' % key), value, key=key,
                       dtype=DTYPE, sparse=False)


def factorized_train(dataset, output_dir):
    '''The shipped training script, so the measured row is the real thing.'''
    train_decoder_sklearn_ridge.featdec_sklearn_ridge_train(
        dataset.fmri,
        output_dir=output_dir,
        rois=dict(dataset.rois),
        label_key=LABEL_KEY,
        alpha=ALPHA,
        analysis_name=os.path.basename(output_dir),
    )


VARIANTS = (
    ('legacy-equivalent', legacy_equivalent_train, 'L*S*R'),
    ('factorized', factorized_train, '0'),
)


# Measurement #################################################################

def run_variant(train, dataset, workdir, tag, cold, repeats):
    '''Best-of-``repeats`` timed runs, each into a fresh decoder directory.'''
    best = None
    output_dir = os.path.join(workdir, 'decoders-%s' % tag)

    for _ in range(repeats):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if cold:
            evict(dataset.feature_files)

        probe = FeatureLoadProbe(evict_after_load=cold)
        with probe.installed():
            start = perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                train(dataset, output_dir)
            total = perf_counter() - start

        if best is None:
            best = (probe, total)
        else:
            if probe.counts != best[0].counts:
                raise AssertionError(
                    '%s is not deterministic: %r then %r'
                    % (tag, best[0].counts, probe.counts))
            if total < best[1]:
                best = (probe, total)

    return best[0], best[1], output_dir


def cross_check(legacy_dir, factorized_dir, dataset):
    """The variants have to be running on the same data to be comparable.

    Only the brain-side parameters, and against the *shared* model directory:
    the factorized decoder stores one model per (subject, ROI) rather than a
    copy under every layer, and it stores no feature statistics (prediction
    writes those). The numerical equivalence of the two decoders is covered by
    the test suite.
    """
    for layer, subject, roi in product(dataset.layers, dataset.fmri,
                                       dataset.rois):
        for key in ('x_mean', 'x_norm'):
            expected = load_array(
                os.path.join(legacy_dir, layer, subject, roi, 'model',
                             '%s.mat' % key), key=key)
            actual = load_array(
                os.path.join(factorized_dir, subject, roi, 'model',
                             '%s.mat' % key), key=key)
            np.testing.assert_allclose(actual, expected, rtol=1e-6,
                                       err_msg='%s of %s/%s/%s differs'
                                       % (key, layer, subject, roi))


def human_bytes(n_bytes):
    for unit, size in (('TB', 1024.0 ** 4), ('GB', 1024.0 ** 3),
                       ('MB', 1024.0 ** 2), ('KB', 1024.0)):
        if n_bytes >= size:
            return '%.1f %s' % (n_bytes / size, unit)
    return '%d B' % n_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--repeats', type=int, default=3,
                        help='timed runs per variant (best is reported)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='BLAS threads; 1 keeps the comparison reproducible')
    parser.add_argument('--warm', action='store_true',
                        help='leave the page cache alone (the other bracket)')
    parser.add_argument('--quick', action='store_true',
                        help='tiny run: 20 stimuli, one layer, d_out = 1024')
    args = parser.parse_args()

    sizes = (1024,) if args.quick else FEATURE_SIZES
    n_stimuli = 20 if args.quick else N_STIMULI
    n_layers = 1 if args.quick else N_LAYERS
    repeats = 1 if args.quick else args.repeats
    cold = not args.warm

    n_subjects, n_rois = N_SUBJECTS, N_ROIS
    print('Training run: %d layer(s) x %d subject(s) x %d ROI(s), %d stimuli '
          'x %d repetitions, %d voxels, alpha = %g'
          % (n_layers, n_subjects, n_rois, n_stimuli, N_REPEATS, N_VOXELS,
             ALPHA))
    print('%s page cache; best of %d; %d BLAS thread(s).'
          % ('Cold (every feature load is evicted afterwards, so no read is '
             'served from cache)' if cold else 'Warm (nothing is evicted)',
             repeats, args.threads))
    print('"loads" and "logical" are exact and machine-independent. "device" is '
          '/proc/self/io read_bytes accumulated around the feature loads only, '
          'and differs from "logical" because the .mat container and the '
          'block alignment are larger than the array inside, while some pages '
          'may survive an eviction. The times are this '
          'machine\'s local disk; a shared filesystem is slower, so the '
          'seconds understate the difference.')
    print('')

    header = ('%-7s | %-20s | %6s | %9s | %9s | %10s | %9s'
              % ('d_out', 'variant', 'loads', 'logical', 'device',
                 'loader [s]', 'total [s]'))
    print(header)
    print('-' * len(header))

    root = tempfile.mkdtemp(prefix='featdec-io-')
    cwd = os.getcwd()
    try:
        with threadpool_limits(limits=args.threads):
            for d_out in sizes:
                data_root = os.path.join(root, 'data-%d' % d_out)
                dataset = build_dataset(data_root, n_stimuli=n_stimuli,
                                        n_layers=n_layers, d_out=d_out,
                                        n_subjects=n_subjects, n_rois=n_rois)
                workdir = os.path.join(root, 'work-%d' % d_out)
                os.makedirs(workdir, exist_ok=True)
                os.chdir(workdir)  # the scripts write ./tmp/<analysis>.db

                expected_loads = {
                    'L*S*R': n_layers * n_subjects * n_rois * n_stimuli,
                    '0': 0,
                }
                outputs = {}
                probes = {}
                for name, train, scaling in VARIANTS:
                    tag = name.split()[0].replace('-', '_') + (
                        '_stats' if 'stats' in name else '')
                    probe, total, output_dir = run_variant(
                        train, dataset, workdir, tag, cold, repeats)
                    outputs[name] = output_dir
                    probes[name] = probe

                    if probe.loads != expected_loads[scaling]:
                        raise AssertionError(
                            '%s loaded %d feature files, expected %s = %d'
                            % (name, probe.loads, scaling,
                               expected_loads[scaling]))
                    layer_loads = probe.loads // max(n_stimuli, 1)
                    if probe.logical_bytes != layer_loads * dataset.row_bytes():
                        raise AssertionError(
                            '%s: %d logical bytes disagree with %d layer loads'
                            % (name, probe.logical_bytes, layer_loads))

                    print('%-7d | %-20s | %6d | %9s | %9s | %10.2f | %9.2f'
                          % (d_out, name, probe.loads,
                             human_bytes(probe.logical_bytes),
                             human_bytes(probe.device_bytes),
                             probe.seconds, total))

                cross_check(outputs['legacy-equivalent'],
                            outputs['factorized'], dataset)

                if cold and probes['legacy-equivalent'].device_bytes < (
                        dataset.row_bytes() // 2):
                    print('%-7s | NOTE: little came from the device, so '
                            'posix_fadvise did not evict this filesystem\'s '
                            'cache and the times above are warm.' % '')

                os.chdir(cwd)
                shutil.rmtree(data_root, ignore_errors=True)
                shutil.rmtree(workdir, ignore_errors=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)

    print('')
    per_label_set = VGG19_UNITS * DEEPRECON_STIMULI * np.dtype(DTYPE).itemsize
    runs = DEEPRECON_SUBJECTS * DEEPRECON_ROIS
    print('Arithmetic on the same per-array sizes, at DeepRecon scale (VGG19 '
          'all layers = %d units/stimulus, %d stimuli, %d subjects x %d ROIs, '
          'float32):' % (VGG19_UNITS, DEEPRECON_STIMULI, DEEPRECON_SUBJECTS,
                         DEEPRECON_ROIS))
    print('  legacy-equivalent     %2d label-set reads   %s' % (
        runs, human_bytes(per_label_set * runs)))
    print('  factorized             none                0 B')
    print('Prediction is not measured here and is not free: it reads one copy '
          'of the features per layer, shared by every (subject, ROI), and '
          'writes their statistics into the decoder for evaluation.py. The '
          'legacy decoder reads no features at prediction but loads a '
          'd_out x n_voxels model instead -- see bench_ridge_factorization for '
          'that side.')


if __name__ == '__main__':
    main()
