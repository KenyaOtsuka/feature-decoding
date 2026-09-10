# Benchmarks for KamitaniLab/feature-decoding PR #7

The scripts behind the numbers quoted in
[KamitaniLab/feature-decoding#7](https://github.com/KamitaniLab/feature-decoding/pull/7)
("Factorize the sklearn Ridge decoder"). They live on this branch rather than
in the pull request because they change no production code: they are evidence,
not product.

This branch is the PR's factorization commit (`8c47da0`, on top of `6938115`)
plus this directory, and nothing else, so the scripts run against exactly the
code the numbers describe.

- `bench_pipeline.py` — **the main comparison**: one whole `train → predict`
  run each way, with time and I/O for training, for prediction and for the
  total. Both variants predict through the shipped
  `predict_feature.featdec_predict`, which detects the decoder format, so the
  prediction rows are the real code path.
- `bench_ridge_factorization.py` — the micro view: Ridge solve time, in-memory
  prediction time and stored model size as the output dimension `d_out` grows.
  Pure in-memory, no files.

## Running them

From the repository root, so that `ridge_factorization` and
`train_decoder_sklearn_ridge` are importable (the scripts do not touch
`sys.path`):

```shell
git clone https://github.com/KenyaOtsuka/feature-decoding
cd feature-decoding
git checkout bench/pr7-benchmarks
uv sync --group dev

uv run python -m benchmarks.bench_pipeline             # ~3 min, ~1 GB of temp files
uv run python -m benchmarks.bench_pipeline --quick     # a few seconds
uv run python -m benchmarks.bench_pipeline --warm      # without cache eviction
uv run python -m benchmarks.bench_ridge_factorization  # ~10 s
```

Both take `-r/--repeats` and `-t/--threads` (default: 1 BLAS thread, so the
comparison is reproducible). Temporary data goes to a temporary directory and
is removed afterwards.

## What the two variants are

| | legacy-equivalent | factorized |
| --- | --- | --- |
| training | the pre-PR script's loop (base commit `7322377`): `(layer, subject, ROI)` with `get_multi_features` inside it, the feature statistics, and `bdpy.ml.ModelTraining` writing a `d_out x n_voxels` model per (layer, subject, ROI) | the shipped `train_decoder_sklearn_ridge.py`: the fit is on the stimulus basis, so it reads no feature file and writes an `M x n_voxels` model per (subject, ROI) |
| prediction | the shipped `predict_feature.py`, legacy path: load the big model, predict the features directly, un-normalize | the shipped `predict_feature.py`, factorized path: load the small model, predict the coefficients, load the training features of the layer, contract |

`bench_pipeline` asserts, rather than only prints: the feature and model load
counts (`L*S*R` versus none, and so on), that they are identical across
repeats, that the two variants decode the same features (`rtol=1e-4`), and that
their `x_mean`/`x_norm`/`y_mean`/`y_norm` agree.

## Three quantities, never added together

- **logical I/O** — the summed `nbytes` of the arrays a run reads and writes:
  feature files, model coefficients, normalization parameters, decoded
  features. Exact and machine-independent, and `read + write` is the headline
  number. The fMRI files are identical for both variants and are not in it.
- **device I/O** — `/proc/self/io` `read_bytes + write_bytes` around each
  phase. This machine's local disk; a shared filesystem behaves differently.
- **stored size** — what the decoder tree occupies afterwards. Storage, not
  traffic, so it is never folded into an I/O total.

The decoded-feature write is the same work for both variants; it is counted in
both (this is an end-to-end benchmark) and also printed on its own so it can be
subtracted.

## Results quoted in the PR

Machine: 4 vCPU, 15 GB RAM, local virtio disk, Python 3.11, 1 BLAS thread.
2 layers x 2 subjects x 2 ROIs; training 200 stimuli x 5 repetitions, test 50
stimuli x 2 repetitions, 300 voxels, alpha = 100; cold page cache (feature and
model files evicted before each phase, every feature load evicted after it);
best of 2.

### End to end (`bench_pipeline.py`)

| `d_out` | variant | phase | time [s] | logical read | logical write | logical I/O | device I/O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 096 | legacy-equivalent | training | 4.46 | 25.0 MB | 37.9 MB | 62.9 MB | 75.1 MB |
| | | prediction | 1.39 | 37.6 MB | 6.2 MB | 43.9 MB | 47.5 MB |
| | | **total** | **5.86** | 62.6 MB | 44.2 MB | **106.8 MB** | 122.6 MB |
| 4 096 | factorized | training | 0.30 | 0 B | 959.4 KB | 959.4 KB | 6.5 MB |
| | | prediction | 2.01 | 8.1 MB | 6.5 MB | 14.6 MB | 18.6 MB |
| | | **total** | **2.31** | 8.1 MB | 7.4 MB | **15.5 MB** | 25.1 MB |
| 32 768 | legacy-equivalent | training | 20.47 | 200.0 MB | 303.0 MB | 503.0 MB | 502.5 MB |
| | | prediction | 4.38 | 301.0 MB | 50.0 MB | 351.0 MB | 353.2 MB |
| | | **total** | **24.85** | 501.0 MB | 353.0 MB | **854.0 MB** | 855.7 MB |
| 32 768 | factorized | training | 0.29 | 0 B | 959.4 KB | 959.4 KB | 6.5 MB |
| | | prediction | 4.08 | 51.8 MB | 52.0 MB | 103.8 MB | 101.3 MB |
| | | **total** | **4.37** | 51.8 MB | 52.9 MB | **104.8 MB** | 107.8 MB |

Of the prediction write, 6.2 MB (`d_out` = 4 096) and 50.0 MB (32 768) is the
decoded features, which both variants write; the factorized side additionally
writes the `y_mean`/`y_norm` sidecars for `evaluation.py`.

Stored decoder tree: 302.8 MB against 990.7 KB at `d_out` = 32 768.

**Prediction is not free for the factorized decoder**, and the table shows it:
at `d_out` = 4 096 factorized prediction is *slower* than legacy prediction
(2.01 s against 1.39 s), because loading the training features costs more than
loading a model that is still small. It turns around as `d_out` grows
(4.08 s against 4.38 s at 32 768, with a third of the I/O), and the training
side dominates the total either way.

Prediction breakdown at `d_out` = 32 768 [s]:

| variant | model load | feature load | model inference | combination | remainder |
| --- | --- | --- | --- | --- | --- |
| legacy-equivalent | 1.06 | 0.00 | 0.33 | 0.00 | 2.98 |
| factorized | 0.00 | 1.04 | 0.00 | 0.18 | 2.85 |

"remainder" is reading the test fMRI, averaging it and writing the decoded
features — common to both.

### Compute and stored size against `d_out` (`bench_ridge_factorization.py`)

600 trials (120 stimuli x 5 repetitions), 300 voxels, 50 test samples,
alpha = 100, float32, best of 7. In-memory only.

| `d_out` | Ridge solve [ms] direct | factorized | predict [ms] direct | factorized | model [MB] direct | factorized |
| --- | --- | --- | --- | --- | --- | --- |
| 1 000 | 14.3 | 4.5 | 0.6 | 0.4 | 1.15 | 0.14 |
| 4 000 | 49.8 | 4.4 | 2.6 | 1.0 | 4.59 | 0.14 |
| 16 000 | 196.2 | 4.4 | 8.5 | 3.2 | 18.37 | 0.14 |

The factorized solve time does not depend on `d_out`; the stored model is
`n_train_stimuli x n_voxels` instead of `d_out x n_voxels`.

### DeepRecon arithmetic

Same definition, extrapolated on the per-array sizes (VGG19 all units =
14 861 288 units/stimulus over 19 layers, 1200 training stimuli, 50 test
stimuli, 10^4 voxels, 5 subjects x 9 ROIs, float32). The load counts are those
of the production path, not of what is stored:

| | legacy | factorized |
| --- | --- | --- |
| training read (features) | 2.9 TB | 0 B |
| training write (models) | 24.3 TB | 2.0 GB |
| prediction read (models) | 24.3 TB | 38.2 GB |
| prediction read (features) | 0 B | 66.4 GB |
| prediction write (decoded) | 124.6 GB | 124.6 GB |
| prediction write (`y_mean`/`y_norm`) | 0 B | 5.0 GB |
| **total logical I/O (read + write)** | **51.7 TB** | **236.2 GB** |

Two assumptions are stated rather than hidden. The factorized model is stored
once per (subject, ROI), but `featdec_predict` calls `ModelTest.run()` once per
(layer, subject, ROI), so it is counted 855 times, not 45. The training
features are read once per layer because every subject shares the same ordered
training-label set — which is what the synthetic dataset does too; with
per-subject label sets it would be once per layer per distinct set. This is
arithmetic, not a measurement, and the page cache may serve some of those
re-reads without touching the device, which is why the logical counts and the
measured device bytes are kept in separate columns.
