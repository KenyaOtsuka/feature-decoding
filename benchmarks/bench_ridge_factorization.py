'''Benchmark the direct and factorized sklearn Ridge feature decoders.

Deliberately small so it runs anywhere: the point is the *scaling* with the
output feature dimension, which is visible from a short sweep, not the absolute
numbers of a full DeepRecon run.  Peak memory stays in the tens of MB.

    uv run python -m benchmarks.bench_ridge_factorization

This is not part of the test suite; correctness is covered by
``tests/test_ridge_factorization.py``.
'''

from __future__ import annotations

import argparse
import pickle
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from ridge_factorization import combine_features, one_hot_assignment

# Small stand-in for the real setting (N ~ 6000, M ~ 1200, p ~ 10^4).
N_STIMULI = 120
N_REPEATS = 5
N_VOXELS = 300
N_TEST = 50
ALPHA = 100.0
DTYPE = np.float32
FEATURE_SIZES = (1000, 4000, 16000)


def make_problem(n_features, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(N_STIMULI), N_REPEATS)
    assignment = one_hot_assignment(labels, range(N_STIMULI))

    features = rng.randn(N_STIMULI, n_features).astype(DTYPE)
    brain = rng.randn(len(labels), N_VOXELS).astype(DTYPE)
    brain_test = rng.randn(N_TEST, N_VOXELS).astype(DTYPE)
    return brain, brain_test, assignment.astype(DTYPE), features


def pickled_size(model, n_train_stimuli=None):
    payload = {'model': model, 'y_shape': (n_train_stimuli,)}
    return len(pickle.dumps(payload, protocol=4))


def best_of(fn, repeats):
    """Best wall time of ``repeats`` calls, after one untimed warm-up.

    The warm-up matters: the first BLAS call in a process pays for thread-pool
    setup, which is larger than the operations being measured here.
    """
    fn()
    best = None
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        elapsed = perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return best, result


def run_direct(brain, brain_test, assignment, features, repeats):
    """Direct decoder: regress brain activity onto the features themselves."""
    mean = np.mean(features, axis=0)[np.newaxis, :]
    std = np.std(features, axis=0, ddof=1)[np.newaxis, :]
    target = ((assignment @ features - mean) / std).astype(DTYPE)

    fit_time, model = best_of(
        lambda: Ridge(alpha=ALPHA).fit(brain, target), repeats)
    predict_time, prediction = best_of(
        lambda: model.predict(brain_test) * std + mean, repeats)

    return {
        'fit_time': fit_time,
        'predict_time': predict_time,
        'size': pickled_size(model, features.shape[1]),
        'prediction': prediction,
    }


def run_factorized(brain, brain_test, assignment, features, repeats):
    """Factorized decoder: regress onto the stimulus basis, then combine."""
    fit_time, model = best_of(
        lambda: Ridge(alpha=ALPHA).fit(brain, assignment), repeats)
    predict_time, prediction = best_of(
        lambda: combine_features(model.predict(brain_test), features), repeats)

    return {
        'fit_time': fit_time,
        'predict_time': predict_time,
        'size': pickled_size(model, assignment.shape[1]),
        'prediction': prediction,
    }


def megabytes(n_bytes):
    return n_bytes / (1024.0 ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--repeats', type=int, default=7,
                        help='number of timing repetitions (best is reported)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='BLAS threads; 1 keeps the comparison reproducible')
    args = parser.parse_args()

    print('N trials = %d (%d stimuli x %d repetitions), n_voxels = %d, '
          'n_test = %d, alpha = %g, dtype = %s'
          % (N_STIMULI * N_REPEATS, N_STIMULI, N_REPEATS, N_VOXELS, N_TEST,
             ALPHA, np.dtype(DTYPE).name))
    print('Best of %d runs, %d BLAS thread(s).' % (args.repeats, args.threads))
    print('Times cover the Ridge solve and the in-memory prediction only: '
          'reading the training features from disk and computing their '
          'statistics are outside the measured region.')
    print('')
    header = ('%-9s | %-24s | %-24s | %-21s'
              % ('d_out', 'Ridge solve [ms]', 'predict, in memory [ms]',
                 'model size [MB]'))
    print(header)
    print('%-9s | %-11s %-12s | %-11s %-12s | %-10s %-10s'
          % ('', 'direct', 'factorized', 'direct', 'factorized', 'direct',
             'factorized'))
    print('-' * len(header))

    with threadpool_limits(limits=args.threads):
        for n_features in FEATURE_SIZES:
            problem = make_problem(n_features)

            direct = run_direct(*problem, repeats=args.repeats)
            factorized = run_factorized(*problem, repeats=args.repeats)

            # Sanity check: the two formulations must agree.
            np.testing.assert_allclose(factorized['prediction'],
                                       direct['prediction'],
                                       rtol=1e-3, atol=1e-4)

            print('%-9d | %11.1f %12.1f | %11.1f %12.1f | %10.2f %10.2f'
                  % (n_features,
                     direct['fit_time'] * 1e3, factorized['fit_time'] * 1e3,
                     direct['predict_time'] * 1e3,
                     factorized['predict_time'] * 1e3,
                     megabytes(direct['size']), megabytes(factorized['size'])))

    print('')
    print('Stored coefficients scale as d_out x n_voxels (direct) versus '
          'n_train_stimuli x n_voxels (factorized), a factor of '
          'd_out / n_train_stimuli.')
    print('At DeepRecon scale (n_train_stimuli = 1200, n_voxels = 10^4, '
          'float32):')
    for name, d_out in (('VGG19 conv1_1', 64 * 224 * 224),
                        ('VGG19 conv5_1', 512 * 14 * 14),
                        ('VGG19 fc6', 4096)):
        direct_bytes = d_out * 10 ** 4 * 4
        factorized_bytes = 1200 * 10 ** 4 * 4
        print('  %-14s d_out = %9d  %10.1f GB -> %6.1f MB  (%.0fx)'
              % (name, d_out, direct_bytes / 1024.0 ** 3,
                 megabytes(factorized_bytes),
                 direct_bytes / float(factorized_bytes)))
    print('The Ridge solve is also shared by every layer, so one fit per '
          '(subject, ROI) replaces one fit per (layer, feature chunk).')


if __name__ == '__main__':
    main()
