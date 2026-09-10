# Benchmarks for KamitaniLab/feature-decoding PR #7

The two scripts behind the numbers quoted in
[KamitaniLab/feature-decoding#7](https://github.com/KamitaniLab/feature-decoding/pull/7)
("Factorize the sklearn Ridge decoder"). They live on this branch rather than
in the pull request because they change no production code and are evidence,
not product.

This branch is the PR's factorization commit plus this directory, and nothing
else, so the scripts run against exactly the code the numbers describe.

- **Target commit**: `8c47da0` on
  `KenyaOtsuka/feature-decoding:refactor/factorized-sklearn-ridge-decoder`
  (the PR's factorization commit, on top of `6938115`), which is the parent of
  this branch's only extra commit.
- `bench_ridge_factorization.py` — the direct against the factorized decoder:
  Ridge solve time, in-memory prediction time and stored model size as the
  output feature dimension grows. Pure in-memory, no files.
- `bench_training_io.py` — what one *training run* reads from the feature
  store: the pre-PR loading pattern against the factorized one. Builds a small
  synthetic feature store and fMRI files in a temporary directory.

## Running them

From the repository root, so that `ridge_factorization` and
`train_decoder_sklearn_ridge` are importable (the scripts do not touch
`sys.path`):

```shell
git clone https://github.com/KenyaOtsuka/feature-decoding
cd feature-decoding
git checkout bench/pr7-benchmarks
uv sync --group dev

uv run python -m benchmarks.bench_ridge_factorization   # ~10 s
uv run python -m benchmarks.bench_training_io           # ~90 s, ~60 MB of temp files
uv run python -m benchmarks.bench_training_io --quick   # a few seconds
uv run python -m benchmarks.bench_training_io --warm    # without cache eviction
```

Both take `-r/--repeats` and `-t/--threads` (default: 1 BLAS thread, so the
comparison is reproducible). `bench_training_io` writes its synthetic data to
a temporary directory and removes it afterwards.

## Results quoted in the PR

Machine: 4 vCPU, 15 GB RAM, local virtio disk, Python 3.11, 1 BLAS thread.

### Compute and stored size (`bench_ridge_factorization.py`)

600 trials (120 stimuli x 5 repetitions), 300 voxels, 50 test samples,
alpha = 100, float32, best of 7.

| `d_out` | Ridge solve [ms] direct | factorized | predict [ms] direct | factorized | model [MB] direct | factorized |
| --- | --- | --- | --- | --- | --- | --- |
| 1 000 | 14.3 | 4.5 | 0.6 | 0.4 | 1.15 | 0.14 |
| 4 000 | 49.8 | 4.4 | 2.6 | 1.0 | 4.59 | 0.14 |
| 16 000 | 196.2 | 4.4 | 8.5 | 3.2 | 18.37 | 0.14 |

The factorized solve time does not depend on `d_out`; the stored model is
`n_train_stimuli x n_voxels` instead of `d_out x n_voxels`.

At DeepRecon scale (1200 training stimuli, 10^4 voxels, float32), stored
coefficients go from `d_out x n_voxels` to `n_train_stimuli x n_voxels`:
VGG19 `conv1_1` 119.6 GB -> 45.8 MB, `conv5_1` 3.7 GB -> 45.8 MB,
`fc6` 0.2 GB -> 45.8 MB.

### Training-time feature-store I/O (`bench_training_io.py`)

2 layers x 2 subjects x 2 ROIs, 200 stimuli x 5 repetitions, 300 voxels, cold
page cache (every feature file is evicted right after it is read, because a
15 GB layer never stays cached across the (subject, ROI) iterations either),
best of 3.

| `d_out` | variant | load calls | logical | device | in loader [s] | run total [s] |
| --- | --- | --- | --- | --- | --- | --- |
| 4 096 | legacy-equivalent | 1600 | 25.0 MB | 31.2 MB | 2.75 | 4.49 |
| 4 096 | factorized | 0 | 0 B | 0 B | 0.00 | 0.28 |
| 32 768 | legacy-equivalent | 1600 | 200.0 MB | 193.8 MB | 4.38 | 19.67 |
| 32 768 | factorized | 0 | 0 B | 0 B | 0.00 | 0.28 |

"load calls" and "logical" (summed `nbytes` of the arrays the loader returns)
are exact and machine-independent; the `/proc/self/io` device bytes and the
seconds are this machine's local disk and understate a shared filesystem.

Multiplying the same per-array sizes out to DeepRecon scale (VGG19 all layers =
14 861 288 units per stimulus, 1200 stimuli, 5 subjects x 9 ROIs, float32):
**2.9 TB against nothing**. That extrapolation is arithmetic, not a
measurement.

`bench_training_io` asserts the load counts rather than only printing them,
and cross-checks that both variants produced the same `x_mean`/`x_norm`.
