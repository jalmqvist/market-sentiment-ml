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
    ├── DATASET_MANIFEST.json                 # build metadata
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

```bash
# Build dataset version 1.1.0
python scripts/build_dataset.py --version 1.1.0

# Build with a custom variant
python scripts/build_dataset.py --version 1.1.0 --variant core
```

The build script reads raw data from `data/input/` and writes versioned outputs to `data/output/<version>/`.

---

## Using the Dataset in ML Experiments

```python
from research.deep_learning.dataset_loader import (
    load_dataset,
    get_features,
    train_test_split,
    to_tensors,
)

df = load_dataset("1.1.0")                         # load versioned dataset
X, y = get_features(df, "price_sentiment")         # select feature set
(X_train, y_train), (X_test, y_test) = train_test_split(X, y, df)
X_train_t, y_train_t = to_tensors(X_train, y_train)
```

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

1. Note the `dataset_version` from the experiment log or metrics JSON.
2. Rebuild the dataset: `python scripts/build_dataset.py --version <version>`.
3. Re-run the experiment script with the same `--dataset-version` argument.
