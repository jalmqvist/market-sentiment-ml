# Pipeline README

This document describes the canonical data flow for the `market-sentiment-ml`
research pipeline, explains which configuration variables control input/output
paths, and shows how to run each stage independently.

---

## Overview

```
build_dataset  →  [attach_regimes]  →  generate_signal  →  evaluate  →  portfolio
                        ↑
                  (optional step)
```

| Stage | Script | Dataset used |
|---|---|---|
| Build canonical dataset | `pipeline/build_dataset.py` | raw inputs → `DATA_PATH` |
| Attach regime labels | `attach_regimes_to_h1_dataset.py` | `DATA_PATH` → `DATA_PATH_REGIME` |
| Signal discovery | `experiments/discovery.py` | `DATA_PATH` (canonical) |
| Parametric sweep | `experiments/sweep.py` | `DATA_PATH` (canonical) |
| Walk-forward evaluation | `evaluation/walk_forward.py` (library) | `DATA_PATH` (canonical) |
| Portfolio construction | `portfolio/portfolio_builder.py` | `DATA_PATH` (canonical) |
| Regime experiments | `experiments/regime_v2.py` | `DATA_PATH_REGIME` (regime only) |

---

## Config variables that control paths

All paths are defined in **`config.py`** and can be overridden by passing
explicit `--input` / `--output` CLI arguments to individual scripts.

| Variable | Default value | Purpose |
|---|---|---|
| `DATA_PATH` | `data/output/master_research_dataset.csv` | **Canonical base dataset.** Used by all non-regime scripts. |
| `DATA_PATH_REGIME` | `data/output/master_research_dataset_with_regime.csv` | Regime-enriched dataset. Used **only** by regime scripts. |
| `SENTIMENT_DIR` | `data/input/sentiment` | Input directory for sentiment snapshot CSVs. |
| `PRICE_DIR` | `data/input/fx` | Input directory for hourly FX price CSVs. |
| `REGIME_PARQUET_DEFAULT` | `../market-phase-ml/data/output/regimes/phase_labels_d1.parquet` | D1 regime labels from the companion repo. Override via `REGIME_PARQUET_PATH` env var. |
| `OUTPUT_DIR` | `data/output/` | Root output directory. |
| `HORIZONS` | `[1, 2, 4, 6, 12, 24, 48]` | Forward-return horizons (hourly bars). |
| `EVAL_HORIZONS` | `[12, 48]` | Horizons used in evaluation experiments. |

> **Tip:** set the `LOG_LEVEL` environment variable (e.g. `export LOG_LEVEL=DEBUG`)
> to enable verbose output across all scripts.

---

## Canonical vs regime dataset

### Canonical dataset (`DATA_PATH`)

`data/output/master_research_dataset.csv`

Built by `pipeline/build_dataset.py` from raw sentiment snapshots and FX price
files.  It is the **single source of truth** for:

- signal discovery
- walk-forward (WF) evaluation
- portfolio construction
- sanity checks

Scripts that use the canonical dataset do **not** require regime columns and
will work correctly even if the regime-enriched file does not exist yet.

### Regime-enriched dataset (`DATA_PATH_REGIME`)

`data/output/master_research_dataset_with_regime.csv`

Produced by running `attach_regimes_to_h1_dataset.py` on top of the canonical
dataset.  It is an **optional filter layer** used **only** for:

- regime experiments (`experiments/regime_v2.py`)
- conditioning tests
- feature engineering that requires regime columns

Required regime columns: `phase`, `is_trending`, `is_high_vol`.

Regime-specific scripts validate that these columns exist and raise a clear
`ValueError` if they are absent (e.g. if the wrong file is passed as `--input`).

---

## Running each stage

### 0. One-time environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: set log verbosity
export LOG_LEVEL=INFO   # or DEBUG
```

### 1. Build the canonical dataset

Reads raw sentiment and price files, joins them, and writes
`data/output/master_research_dataset.csv`.

```bash
python -m pipeline.build_dataset

# Explicit paths (if your data lives elsewhere):
python -m pipeline.build_dataset \
  --sentiment-dir data/input/sentiment \
  --price-dir     data/input/fx \
  --output        data/output/master_research_dataset.csv \
  --log-level     INFO
```

**Output:** `DATA_PATH` (`data/output/master_research_dataset.csv`)

---

### 2. Attach regime labels (optional)

Joins D1 regime labels from the companion `market-phase-ml` repo onto the
canonical H1 dataset.  Run this step only when you need regime-conditioned
experiments.

```bash
python attach_regimes_to_h1_dataset.py

# Explicit paths:
python attach_regimes_to_h1_dataset.py \
  --h1      data/output/master_research_dataset.csv \
  --regimes /path/to/phase_labels_d1.parquet \
  --out     data/output/master_research_dataset_with_regime.csv
```

The regime parquet path defaults to `config.REGIME_PARQUET_DEFAULT`, which
can be overridden via the `REGIME_PARQUET_PATH` environment variable.

**Output:** `DATA_PATH_REGIME` (`data/output/master_research_dataset_with_regime.csv`)

---

### 3. Signal discovery

Per-pair behavioral signal discovery using the canonical dataset.

```bash
python -m experiments.discovery

# Override input:
python -m experiments.discovery \
  --input data/output/master_research_dataset.csv \
  --log-level DEBUG
```

**Uses:** `DATA_PATH` (canonical)

---

### 4. Walk-forward evaluation

`evaluation/walk_forward.py` is a **library module** — it does not have a
standalone CLI entry-point, but is used internally by `experiments/regime_v2.py`,
`experiments/sweep.py`, and `portfolio/portfolio_builder.py`.

To run walk-forward evaluation as part of the portfolio pipeline:

```bash
python -m portfolio.portfolio_builder
```

---

### 5. Parametric sweep

Sweeps streak threshold × persistence flag combinations and reports
walk-forward results.

```bash
python -m experiments.sweep

# Override input:
python -m experiments.sweep \
  --input data/output/master_research_dataset.csv \
  --log-level INFO
```

**Uses:** `DATA_PATH` (canonical)

---

### 6. Portfolio construction

Applies the canonical behavioral signal, selects survivor pairs, and
evaluates the portfolio walk-forward and on holdout.

```bash
python -m portfolio.portfolio_builder

# Override input:
python -m portfolio.portfolio_builder \
  --input data/output/master_research_dataset.csv \
  --log-level INFO
```

**Uses:** `DATA_PATH` (canonical)

---

### 7. Regime experiments

Walk-forward evaluation conditioned on market regimes.  Requires the
regime-enriched dataset (`DATA_PATH_REGIME`).  Will fail with a clear error
if regime columns (`phase`, `is_trending`, `is_high_vol`) are missing.

```bash
python -m experiments.regime_v2

# Override input:
python -m experiments.regime_v2 \
  --input data/output/master_research_dataset_with_regime.csv \
  --log-level INFO
```

**Uses:** `DATA_PATH_REGIME` (regime only)

---

## Timestamp semantics

- All timestamps in the canonical dataset are stored as strings (CSV) and
  parsed via `utils/validation.parse_timestamps()`, which uses `format="mixed"`.
- `entry_time` is the bar-open UTC timestamp for each FX price bar.
- `snapshot_time` is the UTC time of the corresponding sentiment snapshot.
- `attach_regimes_to_h1_dataset.py` floors `entry_time` to UTC midnight to
  join with D1 regime labels — **no forward-fill, no shifting**.
- All scripts normalize timestamps to pandas datetime objects; tz-naive series
  are localized to UTC automatically.

---

## Column validation behavior

All pipeline scripts call `utils/validation.require_columns()` before
processing data.  If a required column is missing the script raises
`ValueError` with a clear message listing the missing columns and the
call-site context.

| Script | Required columns |
|---|---|
| All scripts | `pair`, `time` |
| `experiments/regime_v2.py` | `phase`, `is_trending`, `is_high_vol` (regime columns) |
| `portfolio/portfolio_builder.py` | `extreme_streak_70`, `crowd_persistence_bucket_70` (signal columns) |

Non-regime scripts do **not** require regime columns.

---

## Logging and debug hooks

Set `LOG_LEVEL=DEBUG` to enable verbose pipeline logging:

```bash
export LOG_LEVEL=DEBUG
python -m experiments.discovery
```

At `DEBUG` level you will see:

- Row counts after each filter stage
- Min/max timestamps after normalization
- Missing column errors with call-site context
- Walk-forward fold statistics

At `WARNING` level (default) you will see:

- Empty DataFrame warnings (with the stage name)
- NaT counts from timestamp parsing
- Regime match-rate warnings (if < 99 %)

To change the log level for a single run without editing config:

```bash
python -m portfolio.portfolio_builder --log-level DEBUG
```
