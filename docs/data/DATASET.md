# Dataset Documentation

## Overview

The master research dataset is a unified time series of retail FX positioning, price data, and derived behavioral features. It is built deterministically from raw input files and versioned for reproducibility.

---

## Dataset Structure

Datasets are stored under `data/output/<version>/`:

```
data/output/
└── <version>/              # e.g. 1.1.0
    ├── master_research_dataset.csv           # full dataset
    ├── master_research_dataset_core.csv      # high-coverage pairs only (≥ 95%)
    ├── master_research_dataset_extended.csv  # broader coverage (≥ 90%)
    ├── master_research_dataset_reactive_jpy_v1.csv
    ├── master_research_dataset_reactive_jpy_v1_core.csv
    ├── master_research_dataset_reactive_jpy_v1_extended.csv
    ├── DATASET_MANIFEST.json                 # build metadata
    ├── DATASET_MANIFEST_reactive_jpy_v1.json # behavioral augmentation provenance
    └── dl/                                   # deep learning outputs
        ├── predictions_<feature_set>.csv
        └── metrics_<feature_set>.json
```

**Variants:**

| Variant    | Coverage threshold | Recommended for       |
|------------|-------------------|-----------------------|
| `core`     | ≥ 95%             | ML training, evaluation |
| `extended` | ≥ 90%             | Exploratory analysis  |
| `full`     | all pairs         | Diagnostics only      |

---

## Versioning Scheme

Dataset versions follow **semantic versioning** (`MAJOR.MINOR.PATCH`):

- **MAJOR** — breaking schema change (columns renamed, removed, or redefined)
- **MINOR** — new columns or feature additions (backward-compatible)
- **PATCH** — bug fixes, filter adjustments, or data corrections

Every build writes a `DATASET_MANIFEST.json` alongside the CSV files so that any experiment can be traced back to the exact dataset version used.

---

## Manifest Fields

`DATASET_MANIFEST.json` contains:

| Field              | Description                                      |
|--------------------|--------------------------------------------------|
| `version`          | Dataset version string (e.g. `"1.1.0"`)          |
| `schema_version`   | Schema format version                            |
| `build_timestamp`  | ISO-8601 UTC timestamp of the build              |
| `rows`             | Total row count                                  |
| `pairs`            | List of FX pairs included                        |
| `date_range`       | Start and end dates of the dataset               |
| `merge_tolerance`  | Sentiment–price merge tolerance (e.g. `"90min"`) |
| `excluded_pairs`   | Pairs excluded from the build                    |
| `horizons`         | Forward-return horizons computed                 |

---

## How to Rebuild the Dataset

Dataset generation and Behavioral Surface integration are a **two-stage workflow**:

1. Build canonical datasets.
2. Generate Behavioral Surface (BSVE).
3. Augment existing canonical datasets.

Canonical datasets are never modified during augmentation.

```bash
# 1) Build canonical datasets (fails fast if canonical files already exist).
python scripts/build_dataset.py --version 1.5.1

# If canonical files already exist and you explicitly want to rebuild them:
python scripts/build_dataset.py --version 1.5.1 --force

# 2) Generate Behavioral Surface with BSVE (example command in bsve/docs/CLI.md)
#    Produces a frozen parquet artifact + behavioral_surface_manifest.json.

# 3) Augment existing canonical datasets only (never rebuilds canonical files).
python scripts/build_dataset.py \
  --version 1.5.1 \
  --behavioral-surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
  --augment-only
```

The build script reads raw data from `data/input/` and writes versioned outputs to `data/output/<version>/`.

---

## Using the Dataset in ML Experiments

Training datasets are identified by two independent dimensions:

| Dimension         | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `dataset_version` | Semantic version string, e.g. `"1.5.1"`                       |
| `dataset_variant` | Variant identifier, e.g. `"core"` or `"reactive_jpy_v1_core"` |

The dataset loader (`research/deep_learning/dataset_loader.py`) is the sole
owner of filename resolution.  Training scripts must never construct dataset
filenames directly.

```python
from research.deep_learning.dataset_loader import (
    load_dataset,
    get_features,
    train_test_split,
    to_tensors,
)

# Canonical dataset (core variant — default)
df = load_dataset("1.5.1")

# Behavioral Surface dataset variant
df = load_dataset("1.5.1", variant="reactive_jpy_v1_core")

X, y, df_clean = get_features(df, "price_sentiment")
(X_train, y_train), (X_test, y_test) = train_test_split(X, y, df_clean)
X_train_t, y_train_t = to_tensors(X_train, y_train)
```

### CLI usage

Both training pipelines accept `--dataset-variant` (default: `core`):

```bash
# Canonical training
python research/deep_learning/train.py \
  --dataset-version 1.5.1 \
  --dataset-variant core \
  --regime LVTF

# Behavioral Surface training
python research/deep_learning/train.py \
  --dataset-version 1.5.1 \
  --dataset-variant reactive_jpy_v1_core \
  --surface reactive_jpy \
  --state JPY_CONSENSUS_YOUNG
```

Existing commands that omit `--dataset-variant` continue loading
`master_research_dataset_core.csv` unchanged.

---

## Feature Sets

Two canonical feature sets are defined in `research/deep_learning/feature_sets.py`:

| Feature set      | Columns                                            |
|------------------|----------------------------------------------------|
| `price_only`     | Trend returns, direction, strength, volume         |
| `price_sentiment`| All price features + sentiment features            |

All features are **causal** (backward-looking only). Forward-return columns are the target, never features.

---

## Reproducibility

To reproduce any experiment exactly:

1. Note the `dataset_version` **and** `dataset_variant` from the experiment log, artifact manifest, or metrics JSON.
2. Build canonical datasets: `python scripts/build_dataset.py --version <version>`.
3. Generate Behavioral Surface and augment with `--augment-only` if the experiment uses behavioral variants.
4. Re-run the experiment script with the same `--dataset-version` and `--dataset-variant` arguments.
