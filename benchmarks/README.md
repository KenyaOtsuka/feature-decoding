# Benchmarks for KamitaniLab/feature-decoding PR #7

What one whole decoding run costs before and after the factorization
([KamitaniLab/feature-decoding#7](https://github.com/KamitaniLab/feature-decoding/pull/7),
"Factorize the sklearn Ridge decoder"): `train → predict`, both ways, over the
same synthetic data, measuring wall time, peak memory, every byte read and
written, and the decoder left on disk.

These scripts live on this branch rather than in the pull request because they
change no production code: they are evidence, not product. The branch is the
PR's factorization commit (`8c47da0`) plus this directory.

Machine: 4 vCPU, 15 GB RAM, local virtio disk, Python 3.11, 4 BLAS threads,
cold page cache (inputs evicted before every phase). Both profiles use
**10 000 voxels**, the scale of a real ROI (which runs to ~15 000) and the same
number the DeepRecon extrapolation below assumes.

## Results

### `fc` profile — the shipped config's own layers

`fc6`, `fc7`, `fc8` (4096 + 4096 + 1000 units), 1200 training stimuli x 5
repetitions = 6000 trials, 10 000 voxels, 50 test stimuli x 2 repetitions,
2 subjects x 2 ROIs.

| | legacy | factorized | change |
| --- | --- | --- | --- |
| training time | 248.5 s | 57.9 s | **4.3x less** |
| prediction time | 10.3 s | 11.3 s | 1.1x more |
| total time | 258.8 s | 69.2 s | **3.7x less** |
| peak memory (training) | 2.9 GB | 3.0 GB | about the same |
| peak memory (prediction) | 362.1 MB | 289.9 MB | 1.2x less |
| total disk read | 2.6 GB | 1.6 GB | 1.7x less |
| total disk write | 1.4 GB | 193.8 MB | **7.3x less** |
| decoder size | 1.4 GB | 183.5 MB | **7.7x less** |

### `conv` profile — one convolutional layer

`conv5_1` (100 352 units), 1200 training stimuli x 5 repetitions, 10 000
voxels, 50 test stimuli x 2 repetitions, 1 subject x 1 ROI.

*Being re-measured at 10 000 voxels; the numbers land in the next commit. The
figures previously here were taken at 1000 voxels and have been removed rather
than left to mislead.*

### What the tables say (`fc`)

- **Training is 4.3x faster.** Both variants now solve the same
  6000 x 6000 kernel (`n_features > n_samples`, so scikit-learn takes the dual
  path), so the win is no longer the solve itself: it is that the fit does not
  depend on the layer, 4 fits instead of 12, and no feature file is read.
- **Prediction is a wash** (10.3 s against 11.3 s): the legacy run spends 6.1 s
  loading 1.4 GB of models, the factorized run 6.1 s loading 3600 feature
  files. The costs are different in kind and nearly equal in size.
- **Peak memory is a wash too** at this size — 2.9 GB against 3.0 GB — because
  both are dominated by the fMRI matrix and the kernel, not by the target.
- **Disk is where it shows**: 7.3x less written and a 7.7x smaller decoder,
  and 1.7x less read.

### Per phase (whole process, interpreter and imports included)

`fc`:

| variant | phase | time [s] | peak RSS | read | written | device read | device written |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | training | 248.5 | 2.9 GB | 1.2 GB | 1.4 GB | 1.1 GB | 1.4 GB |
| legacy | prediction | 10.3 | 362.1 MB | 1.4 GB | 9.7 MB | 1.4 GB | 10.5 MB |
| factorized | training | 57.9 | 3.0 GB | 952.3 MB | 183.7 MB | 917.5 MB | 184.0 MB |
| factorized | prediction | 11.3 | 289.9 MB | 671.1 MB | 10.1 MB | 256.0 MB | 10.9 MB |
| (reference) | imports only | 3.5 | 160.5 MB | 34.2 MB | 3.2 KB | 2.1 MB | 4.0 KB |

`conv`: being re-measured.

"imports only" is a process that imports the same modules and exits: the floor
under every row, and the reason no row reads less than ~34 MB or peaks below
~160 MB.

### Inside prediction [s]

| profile | variant | model load | feature load | model inference | combination | the rest |
| --- | --- | --- | --- | --- | --- | --- |
| fc | legacy | 6.1 | 0.0 | 0.6 | 0.0 | 2.2 |
| fc | factorized | 1.1 | 6.1 | 0.2 | 0.1 | 2.4 |

"the rest" is reading the test fMRI, averaging it and writing the decoded
features — identical work for both. The factorized run's cost is the feature
load (3600 per-stimulus `.mat` files); the legacy run's is the model load
(1.4 GB of coefficients).

### Compute and stored size against `d_out` (`bench_ridge_factorization.py`)

A separate, much smaller in-memory experiment -- 600 trials (120 stimuli x 5
repetitions), **300 voxels**, 50 test samples, alpha = 100, float32, best of 7
-- included because it isolates how the solve and the stored model scale with
the output dimension. Its absolute numbers are not comparable with the tables
above:

| `d_out` | Ridge solve [ms] direct | factorized | predict [ms] direct | factorized | model [MB] direct | factorized |
| --- | --- | --- | --- | --- | --- | --- |
| 1 000 | 14.3 | 4.5 | 0.6 | 0.4 | 1.15 | 0.14 |
| 4 000 | 49.8 | 4.4 | 2.6 | 1.0 | 4.59 | 0.14 |
| 16 000 | 196.2 | 4.4 | 8.5 | 3.2 | 18.37 | 0.14 |

The factorized solve time does not depend on `d_out`; the stored model is
`n_train_stimuli x n_voxels` instead of `d_out x n_voxels`.

### DeepRecon extrapolation — arithmetic, not measurement

VGG19 all units (14 861 288 units/stimulus over 19 layers), 1200 training and
50 test stimuli, 10^4 voxels, 5 subjects x 9 ROIs, float32. Bytes of the arrays
the production path reads and writes; the fMRI is left out because it is
identical for both:

| | legacy | factorized |
| --- | --- | --- |
| training read (features) | 2.9 TB | 0 B |
| training write (models) | 24.3 TB | 2.0 GB |
| prediction read (models) | 24.3 TB | 38.2 GB |
| prediction read (features) | 0 B | 66.4 GB |
| prediction write (decoded) | 124.6 GB | 124.6 GB |
| prediction write (`y_mean`/`y_norm`) | 0 B | 5.0 GB |
| **total** | **51.7 TB** | **236.2 GB** |

Two counts are stated rather than assumed: the factorized model is stored once
per (subject, ROI) but `ModelTest.run()` is called once per (layer, subject,
ROI), so it is read back 855 times, not 45; and the training features are read
once per layer because every subject shares the same ordered training-label set
(as they do in the synthetic dataset). With per-subject label sets it would be
once per layer per distinct set.

## Running them

From the repository root, so that `ridge_factorization` and
`train_decoder_sklearn_ridge` are importable (the scripts do not touch
`sys.path`):

```shell
git clone https://github.com/KenyaOtsuka/feature-decoding
cd feature-decoding
git checkout bench/pr7-benchmarks
uv sync --group dev

uv run python -m benchmarks.bench_pipeline                  # fc, ~5 min, ~1 GB temp
uv run python -m benchmarks.bench_pipeline --profile conv   # ~5 min, needs ~7 GB RAM
uv run python -m benchmarks.bench_pipeline --profile quick  # smoke test, seconds
uv run python -m benchmarks.bench_pipeline --warm           # the warm-cache bracket
uv run python -m benchmarks.bench_ridge_factorization       # the d_out scaling, ~10 s
```

`-r/--repeats` sets the timed runs per variant (the best wins), `-t/--threads`
the BLAS threads (default: every core). Temporary data goes to a temporary
directory and is removed afterwards.

## How it is measured

- **The legacy baseline is the pre-PR script itself.** `bench_pipeline`
  extracts it with `git show 7322377:train_decoder_sklearn_ridge.py` and
  imports it, so nothing about the baseline is a reimplementation. Both
  variants predict through today's `predict_feature.py`, which detects the
  decoder format and takes the corresponding path.
- **Each phase is its own process**, as `train`/`predict` are in real use.
  That is also what makes the memory figure clean: peak RSS is the kernel's own
  `VmHWM` for that process, not a sampled estimate.
- **Everything the phase touches is counted**: the byte columns are `rchar` /
  `wchar` from `/proc/self/io`, i.e. every byte the process read or wrote —
  fMRI, feature store, model pickles, `.mat` statistics, decoded features, the
  DistComp database, the Python interpreter's own reads. Nothing is excluded
  for being common to both variants. The device columns (`read_bytes` /
  `write_bytes`) are the same run seen from the block device.
- **Cold cache by default**: inputs are evicted before every phase, and each
  feature file is evicted right after it is read, because a 15 GB layer never
  stays cached across the (subject, ROI) iterations either. `--warm` gives the
  other bracket.
- **The rows are checked, not assumed.** The script asserts that both variants
  decode the same features (`rtol=1e-4`), that their `x_mean`/`x_norm`/
  `y_mean`/`y_norm` agree, and that the feature and model load counts are what
  the two code paths should produce (`fc`: 14400 feature loads and 12 models
  written by legacy training against none and 4; 0 against 3600 feature loads
  at prediction, 12 model loads either way).
