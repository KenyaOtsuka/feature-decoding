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
cold page cache (inputs evicted before every phase).

## Results

### `fc` profile — the shipped config's own layers

`fc6`, `fc7`, `fc8` (4096 + 4096 + 1000 units), 1200 training stimuli x 5
repetitions, 1000 voxels, 50 test stimuli x 2 repetitions, 2 subjects x 2 ROIs.
Best of 2.

| | legacy | factorized | change |
| --- | --- | --- | --- |
| training time | 54.2 s | 6.7 s | **8.1x less** |
| prediction time | 4.5 s | 10.2 s | **2.3x more** |
| total time | 58.6 s | 16.9 s | **3.5x less** |
| peak memory (training) | 656.4 MB | 552.5 MB | 1.2x less |
| peak memory (prediction) | 190.8 MB | 228.4 MB | 1.2x more |
| total disk read | 579.6 MB | 288.9 MB | 2.0x less |
| total disk write | 151.3 MB | 28.8 MB | 5.3x less |
| decoder size | 140.9 MB | 18.5 MB | 7.6x less |

### `conv` profile — one convolutional layer

`conv5_1` (100 352 units), 1200 training stimuli x 5 repetitions, 1000 voxels,
50 test stimuli x 2 repetitions, 1 subject x 1 ROI. Best of 1, so the times
carry ~15% run-to-run noise; the memory and byte figures are stable.

| | legacy | factorized | change |
| --- | --- | --- | --- |
| training time | 62.7 s | 3.1 s | **20.5x less** |
| prediction time | 4.0 s | 15.8 s | **4.0x more** |
| total time | 66.7 s | 18.8 s | **3.5x less** |
| peak memory (training) | 6.5 GB | 469.3 MB | **14.2x less** |
| peak memory (prediction) | 591.0 MB | 1.1 GB | 1.8x more |
| total disk read | 925.2 MB | 545.3 MB | 1.7x less |
| total disk write | 401.6 MB | 23.0 MB | 17.5x less |
| decoder size | 383.8 MB | 4.6 MB | 83.2x less |

### What the tables say

- **Training gets much cheaper, and the wider the layer the more so**: it no
  longer reads the features at all, and its memory stops scaling with `d_out`
  (6.5 GB against 469 MB on one conv layer — that is the difference between
  needing a big machine and not).
- **Prediction gets more expensive**, by 2-4x: it now reads the training
  features (1200 small `.mat` files per layer) and holds one layer in memory.
  That is the cost the factorization moves, and it does not disappear with a
  warm cache — see below.
- **The total still improves ~3.5x**, and the decoder shrinks by 8-80x.
- Prediction's extra memory is the resident training features: 1.1 GB against
  591 MB on the conv layer. Prediction is the phase where the factorized
  pipeline now needs *more* RAM than the legacy one, though far less than
  legacy training needed.

### Per phase (whole process, interpreter and imports included)

`fc`:

| variant | phase | time [s] | peak RSS | read | written | device read | device written |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | training | 54.2 | 656.4 MB | 401.2 MB | 141.5 MB | 317.9 MB | 142.1 MB |
| legacy | prediction | 4.5 | 190.8 MB | 178.4 MB | 9.8 MB | 142.9 MB | 10.4 MB |
| factorized | training | 6.7 | 552.5 MB | 127.8 MB | 18.7 MB | 92.9 MB | 18.9 MB |
| factorized | prediction | 10.2 | 228.4 MB | 161.1 MB | 10.1 MB | 76.7 MB | 10.9 MB |
| (reference) | imports only | 2.2 | 160.3 MB | 34.2 MB | 3.2 KB | 0 B | 0 B |

`conv`:

| variant | phase | time [s] | peak RSS | read | written | device read | device written |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | training | 62.7 | 6.5 GB | 505.4 MB | 383.9 MB | 468.3 MB | 384.5 MB |
| legacy | prediction | 4.0 | 591.0 MB | 419.7 MB | 17.6 MB | 384.8 MB | 17.8 MB |
| factorized | training | 3.1 | 469.3 MB | 81.0 MB | 4.7 MB | 46.5 MB | 4.8 MB |
| factorized | prediction | 15.8 | 1.1 GB | 464.4 MB | 18.3 MB | 427.4 MB | 18.4 MB |
| (reference) | imports only | 3.3 | 160.8 MB | 34.2 MB | 3.2 KB | 0 B | 0 B |

"imports only" is a process that imports the same modules and exits: the floor
under every row, and the reason no row reads less than ~34 MB or peaks below
~160 MB.

### Inside prediction [s]

| profile | variant | model load | feature load | model inference | combination | the rest |
| --- | --- | --- | --- | --- | --- | --- |
| fc | legacy | 0.4 | 0.0 | 0.1 | 0.0 | 2.1 |
| fc | factorized | 0.1 | 6.1 | 0.0 | 0.1 | 2.3 |
| conv | legacy | 0.9 | 0.0 | 0.2 | 0.0 | 1.0 |
| conv | factorized | 0.0 | 7.4 | 0.0 | 0.2 | 6.4 |

"the rest" is reading the test fMRI, averaging it and writing the decoded
features — identical work for both. The whole of the factorized penalty is the
feature load: 3600 (`fc`) or 1200 (`conv`) per-stimulus `.mat` files.

### Warm page cache (the other bracket)

Re-running with `--warm` changes the picture very little, so the prediction
penalty is per-file overhead rather than device traffic:

| | legacy | factorized |
| --- | --- | --- |
| `fc` total time | 47.1 s | 16.0 s |
| `fc` prediction time | 4.4 s | 9.3 s |
| `conv` total time | 77.0 s | 19.7 s |
| `conv` prediction time | 4.5 s | 15.9 s |

### Compute and stored size against `d_out` (`bench_ridge_factorization.py`)

600 trials (120 stimuli x 5 repetitions), 300 voxels, 50 test samples,
alpha = 100, float32, best of 7, in memory only:

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
