# Pipeline README

This document describes the canonical data flow for the `market-sentiment-ml`
research pipeline, explains which configuration variables control input/output
paths, and shows how to run each stage independently.

---

## Overview

```
build_dataset  →  [attach_regimes]  →  discovery  →  portfolio  →  [regime_v2]  →  validate
                        ↑
                  (optional step)
```

| Stage                   | Script                                     | Dataset used                     | Required     |
| ----------------------- | ------------------------------------------ | -------------------------------- | ------------ |
| Build canonical dataset | `pipeline/build_dataset.py`                | raw inputs → `DATA_PATH`         | Yes          |
| Attach regime labels    | `attach_regimes_to_h1_dataset.py`          | `DATA_PATH` → `DATA_PATH_REGIME` | Optional     |
| Signal discovery        | `experiments/discovery.py`                 | `DATA_PATH` (canonical)          | Yes          |
| Parametric sweep        | `experiments/sweep.py`                     | `DATA_PATH` (canonical)          | Optional     |
| Walk-forward evaluation | `evaluation/walk_forward.py` (library)     | `DATA_PATH` (canonical)          | Library only |
| Portfolio construction  | `portfolio/portfolio_builder.py`           | `DATA_PATH` (canonical)          | Yes          |
| Regime experiments      | `experiments/regime_v2.py`                 | `DATA_PATH_REGIME` (regime only) | Optional     |
| Regime filter pipeline  | `experiments/regime_filter_pipeline.py`    | `DATA_PATH` (canonical)          | Optional     |
| Regime V4 (weighted)    | `experiments/regime_v4.py`                 | `DATA_PATH` (canonical)          | Optional     |
| Validation              | `validation/validate_pipeline_extended.py` | both datasets                    | Automatic    |

> **Important:** `--data` is the ONLY accepted dataset argument for all stage
> scripts.  It is **required** — no default path is used.  This prevents
> silent mistakes from wrong-dataset fallbacks.

---

## Config variables that control paths

All paths are defined in **`config.py`**.  Stage scripts do **not** fall back
to config paths; you must pass the dataset explicitly via `--data`.

| Variable                 | Default value                                                | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `DATA_PATH`              | `data/output/master_research_dataset.csv`                    | **Canonical base dataset.** Used by all non-regime scripts.  |
| `DATA_PATH_REGIME`       | `data/output/master_research_dataset_with_regime.csv`        | Regime-enriched dataset. Used **only** by regime scripts.    |
| `SENTIMENT_DIR`          | `data/input/sentiment`                                       | Input directory for sentiment snapshot CSVs.                 |
| `PRICE_DIR`              | `data/input/fx`                                              | Input directory for hourly FX price CSVs.                    |
| `REGIME_PARQUET_DEFAULT` | `../market-phase-ml/data/output/regimes/phase_labels_d1.parquet` | D1 regime labels from the companion repo. Override via `REGIME_PARQUET_PATH` env var. |
| `OUTPUT_DIR`             | `data/output/`                                               | Root output directory.                                       |
| `HORIZONS`               | `[1, 2, 4, 6, 12, 24, 48]`                                   | Forward-return horizons (hourly bars).                       |
| `EVAL_HORIZONS`          | `[12, 48]`                                                   | Horizons used in evaluation experiments.                     |

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
`ValueError` if they are absent (e.g. if the wrong file is passed as `--data`).

---

## Structured file logging

Every pipeline run writes logs to **both stdout and a timestamped file**:

```
logs/pipeline_YYYYMMDD_HHMMSS.log
```

The log directory is created automatically under the current working directory.
All subprocess stage outputs (stdout + stderr) are captured and written to the
log at `INFO` level with a `[subprocess]` prefix.

The final log summary includes **deterministic fingerprints**:

- **Dataset hash** — MD5 of `canonical_dataset` file contents.
- **Discovery artifact hash** — MD5 of `data/output/discovery_results.json`
  (if it was produced during the run).

Example final log lines:

```
PIPELINE COMPLETE
  canonical dataset : data/output/master_research_dataset.csv
  dataset hash      : a1b2c3d4e5f6...
  discovery artifact: data/output/discovery_results.json  hash=f9e8d7c6b5a4...
```

---

## Discovery artifact

The signal discovery stage writes a JSON artifact:

```
data/output/discovery_results.json
```

The artifact contains:

```json
{
  "thresholds": { ... },
  "raw_signal_count": 12345,
  "horizons": {
    "12": {
      "selected_pairs": ["usd-jpy", "eur-jpy"],
      "non_overlapping_count": 4321,
      "selection_metrics": { "usd-jpy": { "n": 200, "sharpe": 0.15, ... }, ... }
    },
    "48": { ... }
  }
}
```

The portfolio stage can consume this artifact via `--discovery-artifact` to
avoid redundant pair-selection computation:

```bash
python -m portfolio.portfolio_builder \
  --data             data/output/master_research_dataset.csv \
  --discovery-artifact data/output/discovery_results.json
```

If the artifact is not provided or the file is missing, the portfolio stage
falls back to recomputing pair selection from the data (current behaviour).

---

## Improved regime diagnostics

`attach_regimes_to_h1_dataset.py` now uses Python `logging` (not `print`) and
logs the following diagnostics at `INFO` level:

- **Pair coverage ratio**: `regime_pairs / canonical_pairs`
- **Row reduction**: `rows_before → rows_after`
- **Match rate**: matched rows / total rows after join
- **Missing regime rate**: overall + per pair

`WARNING` is logged (but execution is NOT stopped) when:

| Metric                   | Threshold |
| ------------------------ | --------- |
| `match_rate`             | `< 0.95`  |
| `missing_rate` (overall) | `> 0.10`  |
| pair coverage            | `< 0.50`  |

---

## Strict orchestrator (recommended)

`run_pipeline_strict.py` enforces the canonical execution order and dataset
separation automatically.  It is the recommended way to run the full pipeline.

### Canonical pipeline (discovery + portfolio only)

```bash
python run_pipeline_strict.py \
  --canonical-dataset data/output/master_research_dataset.csv
```

### Full pipeline (with regime stages)

```bash
python run_pipeline_strict.py \
  --canonical-dataset data/output/master_research_dataset.csv \
  --regime-dataset    data/output/master_research_dataset_with_regime.csv \
  --regimes-parquet   /path/to/phase_labels_d1.parquet
```

### Skip build (data files already exist)

```bash
python run_pipeline_strict.py \
  --canonical-dataset data/output/master_research_dataset.csv \
  --skip build
```

### Skip validation

```bash
python run_pipeline_strict.py \
  --canonical-dataset data/output/master_research_dataset.csv \
  --skip validate
```

The strict orchestrator:
- Writes logs to `logs/pipeline_YYYYMMDD_HHMMSS.log` AND stdout simultaneously
- Captures all subprocess output (stdout + stderr) and logs it
- Passes `--data` explicitly to every stage (no config fallbacks)
- Runs discovery and portfolio as **required** stages
- Passes `--discovery-artifact data/output/discovery_results.json` to the
  portfolio stage automatically (if the artifact was produced by discovery)
- Runs attach_regimes and regime_v2 **only** when `--regime-dataset` is provided
- **Never** calls walk_forward via subprocess (it is a library used internally)
- Logs a compact dataset summary (rows, pairs, date range) after each stage
- Logs deterministic fingerprints (dataset hash + discovery artifact hash) in
  the final summary
- Runs `validation/validate_pipeline_extended.py` at the end and fails if
  validation fails

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
python attach_regimes_to_h1_dataset.py \
  --data    data/output/master_research_dataset.csv \
  --regimes /path/to/phase_labels_d1.parquet \
  --out     data/output/master_research_dataset_with_regime.csv
```

`--data` is **required** (no default path).  The regime parquet path defaults
to `config.REGIME_PARQUET_DEFAULT`, which can be overridden via the
`REGIME_PARQUET_PATH` environment variable.

Diagnostics logged include pair coverage ratio, row reduction, match rate, and
per-pair missing rate.  Warnings are emitted (but do not stop execution) when
thresholds are violated (see [Improved regime diagnostics](#improved-regime-diagnostics)).

**Output:** `DATA_PATH_REGIME` (`data/output/master_research_dataset_with_regime.csv`)

---

### 3. Signal discovery

Per-pair behavioral signal discovery using the canonical dataset.  Writes a
JSON artifact with selected pairs and selection metrics.

```bash
python -m experiments.discovery \
  --data         data/output/master_research_dataset.csv \
  --out-artifact data/output/discovery_results.json \
  --log-level    DEBUG

# Direct execution (equivalent):
python experiments/discovery.py \
  --data data/output/master_research_dataset.csv
```

`--data` is **required**.  **Uses:** `DATA_PATH` (canonical)

**Output artifact:** `data/output/discovery_results.json`

---

### 4. Walk-forward evaluation

`evaluation/walk_forward.py` is a **library module** — it does not have a
standalone CLI entry-point.  It is used internally by `experiments/regime_v2.py`,
`experiments/sweep.py`, and `portfolio/portfolio_builder.py`.

To run walk-forward evaluation as part of the portfolio pipeline:

```bash
python -m portfolio.portfolio_builder \
  --data data/output/master_research_dataset.csv
```

---

### 5. Parametric sweep

Sweeps streak threshold × persistence flag combinations and reports
walk-forward results.

```bash
python -m experiments.sweep \
  --data      data/output/master_research_dataset.csv \
  --log-level INFO
```

**Uses:** `DATA_PATH` (canonical)

---

### 6. Portfolio construction

Applies the canonical behavioral signal, selects survivor pairs, and
evaluates the portfolio walk-forward and on holdout.

```bash
# Without discovery artifact (recomputes selection):
python -m portfolio.portfolio_builder \
  --data      data/output/master_research_dataset.csv \
  --log-level INFO

# With discovery artifact (faster; uses pre-computed selection):
python -m portfolio.portfolio_builder \
  --data               data/output/master_research_dataset.csv \
  --discovery-artifact data/output/discovery_results.json \
  --log-level          INFO
```

`--data` is **required**.  `--discovery-artifact` is optional; if provided and
valid, pair selection is loaded from the artifact instead of being recomputed.
**Uses:** `DATA_PATH` (canonical)

---

### 7. Regime experiments

Walk-forward evaluation conditioned on market regimes.  Requires the
regime-enriched dataset (`DATA_PATH_REGIME`).  Will fail with a clear error
if regime columns (`phase`, `is_trending`, `is_high_vol`) are missing.

```bash
python -m experiments.regime_v2 \
  --data      data/output/master_research_dataset_with_regime.csv \
  --log-level INFO

# Direct execution (equivalent):
python experiments/regime_v2.py \
  --data data/output/master_research_dataset_with_regime.csv
```

`--data` is **required**.  **Uses:** `DATA_PATH_REGIME` (regime only)

---

### 7b. Regime V3 (LightGBM walk-forward on canonical dataset)

`experiments/regime_v3.py` is a LightGBM regression walk-forward experiment that
predicts `ret_48b` from **sentiment + trend + a causal volatility feature**
(`vol_24b`, computed from `entry_close` with a past-only rolling window), plus a
small set of causal interaction terms.

It runs an **expanding-window walk-forward by year** (train on all prior years,
test on the current year) and prints per-fold metrics such as:

- Spearman IC between predictions and realized `ret_48b`
- a simple directional “signal Sharpe” using `sign(pred) * ret_48b`
- hit rate, and test-set R²

Run it on the **canonical dataset**:

```bash
python -m experiments.regime_v3 \
  --data data/output/master_research_dataset.csv \
  --log-level INFO

# Direct execution (equivalent):
python experiments/regime_v3.py \
  --data data/output/master_research_dataset.csv \
  --log-level DEBUG
```

If some candidate feature columns are missing in the dataset, they are skipped
(with a warning); the experiment still runs as long as at least one feature and
the target column `ret_48b` are present.

---

### 7c. Regime Filter Pipeline (no model, filter-only)

`experiments/regime_filter_pipeline.py` is a **model-free** regime-as-filter
pipeline that decides *when* to trade using discrete regime labels — it does
**not** predict return magnitudes.

#### Design philosophy

Unlike `regime_v3` (which uses regimes to condition a LightGBM model), the
Regime Filter Pipeline selects trades purely by regime membership:

1. Four causal discrete regime features are built per fold from **training
   data cut points only** (no lookahead):

   | Feature              | Source column            | Discretization                        |
   | -------------------- | ------------------------ | ------------------------------------- |
   | `vol_regime`         | `vol_24b`                | Tertiles from training data           |
   | `trend_dir`          | `trend_strength_48b`     | `sign()` → down / flat / up          |
   | `trend_strength_bin` | `\|trend_strength_48b\|` | Tertiles from training data           |
   | `sent_regime`        | `abs_sentiment`          | Fixed bins: [0–50) / [50–70) / [70+) |

   The four features are concatenated into a single `regime_key` string.

2. **Walk-forward (strict, no leakage)**: for each test year, regime
   statistics (mean, std, Sharpe) are computed on training years only.

3. **Regime selection**: regimes must satisfy `n >= min_n` (default 100)
   **and** `sharpe >= min_sharpe` (default 0.05) on training data.

4. **Filter**: only test-set rows whose `regime_key` is in the selected
   set are traded.

5. **Optional direction logic**: if a regime's training mean is positive →
   follow (keep return sign); if negative → fade (invert return sign).

6. **Metrics per fold**: mean return, Sharpe, hit rate, coverage
   (fraction of test signals kept).

#### Output schema

| DataFrame | Schema                                                        |
| --------- | ------------------------------------------------------------- |
| `fold_df` | `["year", "n", "mean", "sharpe", "hit_rate", "coverage"]`    |
| `pooled`  | `{"n_folds", "mean_return", "mean_sharpe", "mean_hit_rate", "mean_coverage"}` |

#### Running

Use the thin launcher at the repo root:

```bash
# Default thresholds (min_n=100, min_sharpe=0.05, direction enabled)
python run_regime_filter_pipeline.py \
  --data data/output/master_research_dataset.csv

# Log to stdout + file
python run_regime_filter_pipeline.py \
  --data     data/output/master_research_dataset.csv \
  --log-file logs/regime_filter.log

# Custom thresholds
python run_regime_filter_pipeline.py \
  --data       data/output/master_research_dataset.csv \
  --min-n      150 \
  --min-sharpe 0.08

# No direction logic (raw returns for all filtered regimes)
python run_regime_filter_pipeline.py \
  --data         data/output/master_research_dataset.csv \
  --no-direction

# Direct module execution (equivalent):
python -m experiments.regime_filter_pipeline \
  --data data/output/master_research_dataset.csv \
  --log-level DEBUG
```

`--data` is **required**.  **Uses:** `DATA_PATH` (canonical dataset).
No regime-enriched dataset or companion repo is needed.

> **Note:** This pipeline lives *alongside* `experiments/regime_v3.py` and
> does **not** replace it.  Both can be run independently.

---

### 7d. Regime V4 (continuous regime-conditioned signal)

`experiments/regime_v4.py` converts regime filtering into a **continuous
regime-weighted signal pipeline**.  Instead of selecting a discrete set of
"good" regimes and producing trades only for those, it assigns a smooth weight
to *every* regime based on its historical Sharpe ratio and applies that weight
multiplicatively to a base sentiment signal.  Every test row receives a
position — the weight may be zero, but no explicit filtering step exists.

#### Design philosophy

| Aspect                  | Regime Filter Pipeline (`7c`)        | Regime V4 (`7d`)                          |
| ----------------------- | ------------------------------------ | ----------------------------------------- |
| Signal generation       | Trade only selected regimes (top-k)  | Trade all regimes, scale by weight        |
| Weight form             | Binary (0 or 1)                      | Continuous in `(-1, +1)` via tanh         |
| Base signal             | Raw return sign                      | `sign(net_sentiment)`                     |
| Regime key components   | 3 (vol + trend_dir + sent)           | 4 (vol + trend_dir + trend_strength + sent) |
| Leakage guarantee       | Train-only cuts and stats            | Train-only cuts and stats                 |

#### Regime key (4 components)

All four features are built from **training-data cut points only**:

| Feature              | Source column            | Discretization                        |
| -------------------- | ------------------------ | ------------------------------------- |
| `vol_regime`         | `vol_24b`                | Tertiles from training data           |
| `trend_dir`          | `trend_strength_48b`     | `sign()` → down / flat / up          |
| `trend_strength_bin` | `\|trend_strength_48b\|` | Tertiles from training data           |
| `sent_regime`        | `abs_sentiment`          | Fixed bins: [0–50) / [50–70) / [70+) |

```
regime_key = f"{vol_regime}__{trend_dir}__{trend_strength_bin}__{sent_regime}"
```

#### Weight computation (train only)

For each eligible regime (`n >= min_n` in the training fold):

```
weight = tanh(sharpe / std_sharpe)      # default
weight = sharpe / max_abs_sharpe        # --normalize-weights
```

Regimes absent from the training map or with `n < min_n` receive `weight = 0`.

#### Signal application

```
base_signal = sign(net_sentiment)
position    = base_signal * weight
```

#### Output schema

| DataFrame | Schema                                                                          |
| --------- | ------------------------------------------------------------------------------- |
| `fold_df` | `["year", "n", "mean", "sharpe", "hit_rate", "coverage", "avg_weight"]`         |
| `pooled`  | `{"n_folds", "mean_sharpe", "mean_hit_rate", "mean_coverage", "mean_avg_weight"}` |

Metrics are computed on non-zero-position rows only.  `coverage` is the
fraction of test rows with a non-zero weight.  `avg_weight` is the mean
absolute weight across all test rows (including zero-weight rows).

#### Per-fold logging

Each fold logs:

- Number of eligible regimes in the training fold (`n >= min_n`)
- `std_sharpe` (or `max_abs_sharpe` in normalize mode)
- Top-5 regimes by `|weight|` with their Sharpe and weight
- Weight distribution: min / max / mean across eligible regimes

#### Running

Use the thin launcher at the repo root:

```bash
# Default settings (min_n=100, tanh weighting, file-only logging)
python run_regime_v4.py \
  --data data/output/master_research_dataset.csv

# Log to a specific file
python run_regime_v4.py \
  --data     data/output/master_research_dataset.csv \
  --log-file logs/regime_v4.log

# Alternative weight normalization
python run_regime_v4.py \
  --data              data/output/master_research_dataset.csv \
  --normalize-weights

# Custom min-n and verbose logging
python run_regime_v4.py \
  --data      data/output/master_research_dataset.csv \
  --min-n     150 \
  --log-level DEBUG \
  --log-file  logs/regime_v4_n150.log

# Direct module execution (equivalent):
python -m experiments.regime_v4 \
  --data data/output/master_research_dataset.csv
```

**CLI arguments:**

| Argument              | Default | Description                                           |
| --------------------- | ------- | ----------------------------------------------------- |
| `--data`              | —       | Path to master research dataset CSV (**required**)    |
| `--min-n`             | `100`   | Min training observations per regime for non-zero weight |
| `--log-level`         | `INFO`  | Logging verbosity (DEBUG / INFO / WARNING / ERROR)    |
| `--log-file`          | auto    | Log file path; auto-timestamped in `logs/` if omitted |
| `--normalize-weights` | off     | Use `sharpe / max_abs_sharpe` instead of `tanh` scaling |
| `--top-n-log`         | `5`     | Top-N regimes by `\|weight\|` to log per fold          |

`--data` is **required**.  **Uses:** `DATA_PATH` (canonical dataset).
No regime-enriched dataset or companion repo is needed.

> **Logging rule:** file logging is **on by default**; a timestamped log file
> is created in `logs/` automatically.  When a log file is used, no output is
> written to stdout.

> **Note:** Regime V4 is a standalone alternative signal pipeline.  It lives
> *alongside* `regime_filter_pipeline` and `regime_v3` and does **not** replace
> them.

---

### 7e. Signal V2.1 (causal price-momentum divergence)

`experiments/signal_v2.py` implements a composite sentiment-divergence signal
(referred to as **Signal V2.1** from this version onward).

#### Key change from V2.0

The divergence feature previously used `ret_48b` (a **forward** return) to
compute `price_mom_z`.  This introduced future leakage.  The corrected version
uses a **causal, past-only** 48-bar cumulative return (`mom_48b`):

```python
# 1. Per-bar return from closing price
grp["ret_1b"] = grp["price"].pct_change()

# 2. Causal 48-bar momentum (past only, no lookahead)
grp["mom_48b"] = grp["ret_1b"].rolling(48, min_periods=48).sum()

# 3. Z-score price momentum using the rolling window
grp["price_mom_z"] = rolling_zscore(grp["mom_48b"], window)

# 4. Divergence (unchanged)
grp["divergence"] = grp["sentiment_z"] - grp["price_mom_z"]
```

`ret_48b` (the forward return) is **never** used inside `build_features`; it
is only used as the PnL target in `_fold_metrics`.

#### Additional required column

Because `mom_48b` is derived from the `price` column, `price` is now a
required input column (previously it was not required by `signal_v2`).

| Column    | Role                                                  |
| --------- | ----------------------------------------------------- |
| `price`   | Closing price used to compute `ret_1b` and `mom_48b`  |
| `ret_48b` | Forward return — PnL target only, **not** a feature   |

#### Running

```bash
python run_signal_v2.py \
  --data data/output/master_research_dataset.csv

# With a custom window and threshold:
python run_signal_v2.py \
  --data      data/output/master_research_dataset.csv \
  --window    96 \
  --threshold 0.5 \
  --log-level DEBUG
```

`--data` is **required**.  **Uses:** `DATA_PATH` (canonical dataset).

---

### 8. Extended validation

Validates dataset integrity, signal parity, performance, and regime isolation.
Exits non-zero on failure.  Issues WARNING-level log messages (without failing)
for borderline regime coverage:

- regime row count < 30% of canonical
- regime pair count < 30% of canonical
- missing regime rate > 20%

```bash
python validation/validate_pipeline_extended.py \
  --data        data/output/master_research_dataset.csv \
  --data-regime data/output/master_research_dataset_with_regime.csv \
  --reference   data/reference/master_research_dataset.csv
```

Both `--data` and `--data-regime` are **required**.  `--reference` is optional
(hash parity checks are skipped when omitted).

---

## Repository directory structure

The repo root contains only **pipeline-critical** scripts:

| Script / Entry-point                 | Purpose                                       |
| ------------------------------------ | --------------------------------------------- |
| `run_pipeline.py`                    | Simple pipeline orchestrator                  |
| `run_pipeline_strict.py`             | Strict deterministic pipeline orchestrator    |
| `run_regime_v3.py`                   | Launcher for `experiments/regime_v3`          |
| `run_regime_filter_pipeline.py`      | Launcher for `experiments/regime_filter_pipeline` |
| `run_regime_v4.py`                   | Launcher for `experiments/regime_v4`          |
| `build_fx_sentiment_dataset.py`      | Build canonical dataset from raw inputs       |
| `build_sentiment_feature_contract.py`| Feature contract builder                      |
| `attach_regimes_to_h1_dataset.py`    | Attach D1 regime labels to H1 dataset         |
| `config.py`                          | Centralised pipeline configuration            |

Standalone analysis and investigation scripts (JPY effect validation,
one-off walk-forward analyses, pair-quality audits, etc.) live in
**`analysis/`**:

```
analysis/
  analyze_by_pair_group.py
  analyze_cross_pair_persistence.py
  analyze_jpy_cluster_permutation.py
  analyze_outliers.py
  analyze_pair_quality.py
  analyze_persistence.py
  analyze_regime_signal_interaction.py
  analyze_thresholds.py
  analyze_trend_alignment.py
  analyze_trend_behavior.py
  analyze_trend_strength_results.py
  discover_behavioral_signal.py
  evaluate_regime_holdout.py
  evaluate_signal_regime_aware.py
  experiment_regime_v2_sweep.py
  portfolio_behavioral_signal.py
  validate_jpy_effect_preregistered.py
  validate_jpy_effect_time_split.py
  validate_jpy_effect_walkforward.py
  validate_pipeline_extended.py
  walk_forward_jpy_hypothesis.py
  walk_forward_jpy_regime_signal.py
  walk_forward_regime_v2.py
```

These scripts are **not part of the canonical pipeline** and are not
invoked by `run_pipeline_strict.py` or any other orchestrator.  They
can be run directly for ad-hoc analysis:

```bash
python analysis/analyze_pair_quality.py --data data/output/master_research_dataset.csv
```

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

| Script                           | Required columns                                             |
| -------------------------------- | ------------------------------------------------------------ |
| All scripts                      | `pair`, `time`                                               |
| `experiments/regime_v2.py`       | `phase`, `is_trending`, `is_high_vol` (regime columns)       |
| `portfolio/portfolio_builder.py` | `extreme_streak_70`, `crowd_persistence_bucket_70` (signal columns) |

Non-regime scripts do **not** require regime columns.

---

## Logging and debug hooks

Set `LOG_LEVEL=DEBUG` to enable verbose pipeline logging:

```bash
export LOG_LEVEL=DEBUG
python -m experiments.discovery --data data/output/master_research
