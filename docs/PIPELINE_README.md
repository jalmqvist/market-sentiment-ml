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

| Stage                           | Script                                           | Dataset used                     | Required     |
| ------------------------------- | ------------------------------------------------ | -------------------------------- | ------------ |
| Build canonical dataset         | `pipeline/build_dataset.py`                      | raw inputs → `DATA_PATH`         | Yes          |
| Attach regime labels            | `attach_regimes_to_h1_dataset.py`                | `DATA_PATH` → `DATA_PATH_REGIME` | Optional     |
| Signal discovery                | `experiments/discovery.py`                       | `DATA_PATH` (canonical)          | Yes          |
| Parametric sweep                | `experiments/sweep.py`                           | `DATA_PATH` (canonical)          | Optional     |
| Walk-forward evaluation         | `evaluation/walk_forward.py` (library)           | `DATA_PATH` (canonical)          | Library only |
| Portfolio construction          | `portfolio/portfolio_builder.py`                 | `DATA_PATH` (canonical)          | Yes          |
| Regime experiments              | `experiments/regime_v2.py`                       | `DATA_PATH_REGIME` (regime only) | Optional     |
| Regime filter pipeline          | `experiments/regime_filter_pipeline.py`          | `DATA_PATH` (canonical)          | Optional     |
| Regime V4 (weighted)            | `experiments/regime_v4.py`                       | `DATA_PATH` (canonical)          | Optional     |
| Signal V2 × Regime Filter       | `experiments/regime_v4_signal_filter.py`         | `DATA_PATH` (canonical)          | Optional     |
| Regime V5 (blended signal)      | `experiments/regime_v5.py`                       | `DATA_PATH` (canonical)          | Optional     |
| Regime V6 (filtered + blended)  | `experiments/regime_v6.py`                       | `DATA_PATH` (canonical)          | Optional     |
| Regime V7 (event-based)         | `experiments/regime_v7.py`                       | `DATA_PATH` (canonical)          | Optional     |
| Regime V7.1 (continuous scores) | `experiments/regime_v7_1.py`                     | `DATA_PATH` (canonical)          | Optional     |
| Regime V7.2 (interaction scores)| `experiments/regime_v7_2.py`                     | `DATA_PATH` (canonical)          | Optional     |
| Regime V8 (model-based)         | `experiments/regime_v8.py`                       | `DATA_PATH` (canonical)          | Optional     |
| Regime V8.1 (top-k selection)   | `experiments/regime_v8.py` (`--top-frac`)         | `DATA_PATH` (canonical)          | Optional     |
| Validation                      | `validation/validate_pipeline_extended.py`       | both datasets                    | Automatic    |

> **Important:** `--data` is the ONLY accepted dataset argument for all stage
> scripts.  It is **required** — no default path is used.  This prevents
> silent mistakes from wrong-dataset fallbacks.

---

## System architecture (Signal × Regime pipelines)

The current research pipeline extends beyond discovery and portfolio construction.
It includes a **modular signal + regime conditioning system** that determines *when*
to trade.

### High-level flow

Canonical dataset
 ↓
 Signal V2 (base signal)
 ↓
 Regime construction (vol × trend × sentiment)
 ↓
 Train-only regime evaluation (walk-forward)
 ↓
 Regime selection (Sharpe + persistence filters)
 ↓
 Signal conditioning
 ├─ Filter (binary)
 ├─ Direction (follow / fade)
 └─ Weighting (continuous, optional)
 ↓
 Final trading signal

---

### Component mapping to code

| Layer                    | Component                 | Script                                   |
| ------------------------ | ------------------------- | ---------------------------------------- |
| Data                     | Canonical dataset         | `pipeline/build_dataset.py`              |
| Signal                   | Base signal (Signal V2)   | `experiments/signal_v2.py`               |
| Regime discovery         | Feature interactions / ML | `experiments/regime_v3.py`               |
| Regime filter (binary)   | Model-free filtering      | `experiments/regime_filter_pipeline.py`  |
| Regime weighting         | Continuous signal scaling | `experiments/regime_v4.py`               |
| Signal × regime (hybrid) | **Production candidate**  | `experiments/regime_v4_signal_filter.py` |
| Continuous blending      | Signal V2 × regime × behavior | `experiments/regime_v5.py`          |
| Filtered + blended       | Filter threshold + continuous weighting | `experiments/regime_v6.py`    |
| Event-based signal       | Discrete event detection + Signal V2 | `experiments/regime_v7.py`       |
| Continuous event scoring | Ranked continuous event scores + threshold | `experiments/regime_v7_1.py`  |
| Interaction scoring      | Multiplicative event scores + row normalisation | `experiments/regime_v7_2.py` |
| Model-based signal       | LightGBM learns alpha function from features | `experiments/regime_v8.py`  |
| Top-k signal selection   | Prediction ranking + top-frac filtering (V8.1) | `experiments/regime_v8.py` (`--top-frac`) |

---

### Three regime paradigms

The project currently supports three distinct ways of using regimes:

#### 1. Regime as model input (Regime V3)

- Uses LightGBM
- Regimes act as features
- Output: predicted returns
- features → model → prediction

---

#### 2. Regime as filter (Regime Filter Pipeline)

- No model
- Trade only selected regimes
- regime ∈ selected → trade
- else → skip

---

#### 3. Regime as weighting function (Regime V4)

- Continuous scaling of signal
- All observations receive a position
- position = base_signal × regime_weight

---

#### 4. Signal × Regime (V4 Signal Filter — current best)

- Combines Signal V2 with regime filtering
- Optional direction adjustment
- Best empirical performance so far
- Signal V2 → regime filter → (optional direction) → final signal

---

#### 5. Continuous signal blending (Regime V5)

- Multiplicative blending of three continuous layers — no filtering or
  discrete selection
- **Base signal**: Signal V2 raw composite passed through `tanh`
- **Regime score**: 4-component regime key → train-only Sharpe →
  `tanh(sharpe / std_sharpe)`
- **Behavioral score**: `tanh(0.5 * zscore(extreme_streak_70) + 0.5 * zscore(abs_sentiment))`
  — z-score parameters derived from training data only
- Final position: `base_signal × regime_score × behavior_score`
- Runner: `run_regime_v5.py --data <path> [--min-n 100] [--window 96]`

---

#### 6. Filtered + continuous signal blending (Regime V6)

- Combines regime filtering (V4) with continuous weighting (V5)
- **Base signal**: Signal V2 raw composite passed through `tanh`
- **Eligible regimes**: ``n >= min_n`` (same as V4/V5)
- **Selected regimes**: eligible regimes with ``sharpe >= filter_sharpe``
  (default 0.05) — regimes below the threshold receive zero regime score
- **Regime score**: weight map built from *selected* regimes only:
  `tanh(sharpe / std_sharpe_selected)`; filtered-out regimes → 0
- **Behavioral score**: identical to V5
- Final position: `base_signal × regime_score × behavior_score`
- **Coverage** reflects filtering: `mean(abs(position) > 1e-12)`
- **Safety fallback**: if all positions are zero (all filtered), reverts to
  V5 continuous weighting using all eligible regimes
- Runner: `python experiments/regime_v6.py --data <path> [--min-n 100] [--filter-sharpe 0.05] [--window 96]`

---

#### 7. Event-based signal pipeline (Regime V7)

- Replaces regime-based averaging with **discrete event detection**
- Three event types, each a boolean mask validated on training data only:

  * **SATURATION_EVENT** – `abs_sentiment > 70` AND `extreme_streak_70 >= streak_threshold`
    AND `abs(trend_strength_48b) > trend_threshold`
  * **DIVERGENCE_EVENT** – `abs(divergence) > divergence_threshold`
  * **EXHAUSTION_EVENT** – `extreme_streak_70 >= streak_threshold`
    AND `abs(trend_strength_48b) <= trend_threshold`

- **Train phase**: compute n, mean, Sharpe, hit rate per event on training data;
  keep events where `n >= min_n` AND `sharpe >= min_sharpe`; store direction
  (sign of mean return)
- **Test phase**: if a validated event fires → `position = tanh(signal_v2_raw) * direction`;
  if multiple fire → highest train-set Sharpe wins; if none → `position = 0`
- **Coverage** = fraction of test rows where `abs(position) > 0`
- No forward leakage: all thresholds and event directions set on train only
- Runner: `python run_regime_v7.py --data <path> [--min-n 50] [--min-sharpe 0.02] [--streak-threshold 3] [--trend-threshold 0.5] [--divergence-threshold 1.0]`

---

#### 8. Continuous event scoring pipeline (Regime V7.1)

- Upgrades V7 by replacing **boolean event masks** with **continuous event scores**,
  allowing graded signal strength rather than binary fire/no-fire logic
- Three continuous scores per row (z-scores fitted on the training split only):

  * **SATURATION_SCORE**  = `tanh(z(abs_sentiment) + z(extreme_streak_70) + z(trend_strength_48b))`
  * **DIVERGENCE_SCORE**  = `tanh(|z(divergence)|)`
  * **EXHAUSTION_SCORE**  = `tanh(z(extreme_streak_70) − z(trend_strength_48b))`

- **Selection**: the score with the greatest absolute magnitude is chosen per row
- **Train phase**: compute `mean(score × ret_48b)`, Sharpe, and correlation per score
  type; logged for diagnostics, no hard filter applied
- **Test phase**: if `max(|scores|) < score_threshold` → `position = 0`;
  otherwise `position = tanh(signal_v2_raw) * sign(best_score)`
  (or `* best_score` with `--use-score-weighting`)
- **Coverage** = fraction of test rows where `abs(position) > 0`
- No forward leakage: all z-score normalization parameters derived from training data only
- Runner: `python run_regime_v7_1.py --data <path> [--min-n 50] [--score-threshold 0.5] [--use-score-weighting]`

---

#### 9. Interaction-based scoring pipeline (Regime V7.2)

- Upgrades V7.1 by replacing **additive** score definitions with **multiplicative interactions**
  to correctly model nonlinear behavioural effects
- Three interaction-based continuous scores per row (z-scores fitted on the training
  split only):

  * **SATURATION_SCORE**  = `tanh(z(abs_sentiment) × z(extreme_streak_70) × z(trend_strength_48b))`
  * **DIVERGENCE_SCORE**  = `tanh(z(divergence) × z(abs_sentiment))`
  * **EXHAUSTION_SCORE**  = `tanh(z(extreme_streak_70) × (−z(trend_strength_48b)))`

- **Row-wise normalisation**: `norm = sum(abs(scores)) + 1e-6`; `scores = scores / norm`
- **Selection**: `best_idx = argmax(|scores|)`; `best_score = scores[best_idx]`
- **Train phase**: evaluate each score type — `mean(score × ret_48b)`, Sharpe, correlation;
  logged for diagnostics, no hard filter applied
- **Test phase**: if `|best_score| < score_threshold` → `position = 0`;
  otherwise `position = tanh(signal_v2_raw) * sign(best_score)`
  (or `* best_score` with `--use-score-weighting`)
- **Coverage** = fraction of test rows where `abs(position) > 0`
- Per-fold logging: per-score stats, score-type selection frequency, score magnitude distribution
- No forward leakage: all z-score stats derived from training data only
- Runner: `python run_regime_v7_2.py --data <path> [--min-n 50] [--score-threshold 0.3] [--use-score-weighting]`

---

#### 10. Model-based signal pipeline (Regime V8)

- Replaces handcrafted scoring entirely with a **learned alpha function**
- A LightGBM regressor is trained end-to-end to predict `ret_48b` from six
  sentiment and market features:
  `net_sentiment`, `abs_sentiment`, `extreme_streak_70`,
  `trend_strength_48b`, `divergence`, `signal_v2_raw`
- **Walk-forward**: for each test year the model is re-trained from scratch on
  all prior years only (strict expanding window, minimum 3 years)
- **Signal**: `position = sign(model.predict(X_test))`
- **PnL per row**: `position × ret_48b`
- **Metrics per fold**: n, mean return, Sharpe, hit_rate, IC
  (Spearman correlation between predictions and realized returns)
- No forward leakage: the model never sees any test-year data during training
- Model: LightGBM (`n_estimators=200`, `learning_rate=0.05`,
  `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`)
- Runner: `python run_regime_v8.py --data <path> [--log-level DEBUG]`

---

### Key principle

> **All regime decisions are made using training data only (strict walk-forward)**

This guarantees:

- no forward-looking bias
- realistic out-of-sample performance
- reproducibility across folds

---

### Why this matters

The pipeline evolved from:

Find signal → test signal

to:

Find signal → identify conditions → trade only under those conditions

This shift is the core reason for performance improvement.

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

#### Column mapping: `price_end` → `price`

The canonical dataset stores the closing price as `price_end`, not `price`.
Signal V2.1 internally requires a `price` column (to compute `mom_48b`).

The `main()` function in `experiments/signal_v2.py` automatically maps
`price_end` → `price` right after loading the dataset.  **You do not need to
add a `price` column to any dataset file on disk.**

| Dataset column | Signal V2 column | Role                                                 |
| -------------- | ---------------- | ---------------------------------------------------- |
| `price_end`    | `price`          | Closing price used to compute `ret_1b` and `mom_48b` |
| `ret_48b`      | `ret_48b`        | Forward return — PnL target only, **not** a feature  |

The mapping raises a `ValueError` if neither `price` nor `price_end` is
present in the loaded DataFrame.

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

### 7f. Signal V2 × Regime Filter (Regime V4 Signal Filter)

`experiments/regime_v4_signal_filter.py` implements the **Signal V2 × Regime
Filter** hybrid pipeline.  It uses **Signal V2** as the base signal and
applies **regime keys** as a conditional filter and optional direction modifier.

#### Design philosophy

This experiment answers the question: *do market regimes improve Signal V2
performance via filtering?*

Unlike Regime V4 (`7d`) which assigns a continuous weight to every regime,
this pipeline applies a binary filter — a regime is either **selected** (trade)
or **not selected** (no trade).  Unlike Regime Filter Pipeline (`7c`) which
uses `sign(net_sentiment)` as the base signal, this pipeline uses the richer
**Signal V2** composite signal (divergence + shock + exhaustion).

| Aspect                  | Regime Filter Pipeline (`7c`)        | Regime V4 (`7d`)                         | Signal V2 × Regime Filter (`7f`)            |
| ----------------------- | ------------------------------------ | ---------------------------------------- | ------------------------------------------- |
| Base signal             | Raw return sign                      | `sign(net_sentiment)`                    | Signal V2 (divergence + shock + exhaustion) |
| Weight form             | Binary (0 or 1)                      | Continuous in `(-1, +1)` via tanh        | Binary (0 or 1)                             |
| Regime key components   | 3 (vol + trend_dir + sent)           | 4 (vol + trend_dir + trend_strength + sent) | 4 (vol + trend_dir + trend_strength + sent) |
| Regime stats target     | Raw return (`ret_48b`)               | `sign(net_sentiment) × ret_48b`          | `position × ret_48b`                        |
| Leakage guarantee       | Train-only cuts and stats            | Train-only cuts and stats                | Train-only cuts and stats                   |

#### Regime key (4 components)

Same 4-component key as Regime V4, built from **training-data cut points only**:

| Feature              | Source column            | Discretization                        |
| -------------------- | ------------------------ | ------------------------------------- |
| `vol_regime`         | `vol_24b`                | Tertiles from training data           |
| `trend_dir`          | `trend_strength_48b`     | `sign()` → down / flat / up          |
| `trend_strength_bin` | `\|trend_strength_48b\|` | Tertiles from training data           |
| `sent_regime`        | `abs_sentiment`          | Fixed bins: [0–50) / [50–70) / [70+) |

#### Regime selection (train only)

Per-regime signal-weighted statistics are computed from the training fold:

```
signal_ret  = position × ret_48b
mean_return = mean(signal_ret)
sharpe      = mean_return / std(signal_ret)
```

A regime is **selected** when both conditions hold:

- `n >= min_n`          (default: 100)
- `sharpe >= min_sharpe` (default: 0.05)

#### Signal application

Filter (always applied):

```
if regime_key ∉ selected:
    position = 0
```

Direction modification (optional, `--with-direction`):

```
if mean_return >  direction_threshold  →  keep position
if mean_return < -direction_threshold  →  flip position (×−1)
else                                   →  position = 0
```

`direction_threshold` defaults to `0.0002`.

#### Output schema

| DataFrame | Schema                                                                              |
| --------- | ----------------------------------------------------------------------------------- |
| `fold_df` | `["year", "n", "mean", "sharpe", "hit_rate", "coverage", "n_selected_regimes"]`    |
| summary   | `{"folds", "mean_sharpe", "mean_coverage", "mean_hit_rate", "mean_n_selected"}`     |

#### Running

Use the thin launcher at the repo root:

```bash
# Default settings (min_n=100, min_sharpe=0.05, no direction)
python run_regime_v4_signal_filter.py \
  --data data/output/master_research_dataset.csv

# Custom thresholds
python run_regime_v4_signal_filter.py \
  --data       data/output/master_research_dataset.csv \
  --min-n      150 \
  --min-sharpe 0.1

# Enable direction modification
python run_regime_v4_signal_filter.py \
  --data                data/output/master_research_dataset.csv \
  --with-direction \
  --direction-threshold 0.0003

# Signal V2 with threshold + verbose logging
python run_regime_v4_signal_filter.py \
  --data      data/output/master_research_dataset.csv \
  --threshold 0.5 \
  --log-level DEBUG \
  --log-file  logs/regime_v4_signal_filter_debug.log

# Direct module execution (equivalent):
python -m experiments.regime_v4_signal_filter \
  --data data/output/master_research_dataset.csv
```

**CLI arguments:**

| Argument                | Default   | Description                                                             |
| ----------------------- | --------- | ----------------------------------------------------------------------- |
| `--data`                | —         | Path to master research dataset CSV (**required**)                      |
| `--window`              | `96`      | Rolling z-score window for Signal V2 features                           |
| `--threshold`           | `None`    | Optional position threshold for Signal V2 (`\|signal_v2_raw\| <= T → 0`) |
| `--min-n`               | `100`     | Min training observations per regime for selection                      |
| `--min-sharpe`          | `0.05`    | Min training Sharpe to select a regime                                  |
| `--direction-threshold` | `0.0002`  | Mean-return threshold for direction logic (requires `--with-direction`) |
| `--with-direction`      | off       | Enable direction modification                                           |
| `--no-direction`        | (default) | Disable direction modification                                          |
| `--log-level`           | `INFO`    | Logging verbosity (DEBUG / INFO / WARNING / ERROR)                      |
| `--log-file`            | auto      | Log file path; auto-timestamped in `logs/` if omitted                  |

`--data` is **required**.  **Uses:** `DATA_PATH` (canonical dataset).
No regime-enriched dataset or companion repo is needed.

> **Logging rule:** file logging is **on by default**; a timestamped log file
> named `regime_v4_signal_filter_YYYYMMDD_HHMMSS.log` is created in `logs/`
> automatically.  No output is written to stdout.

> **No leakage guarantee:** all regime cuts and statistics are derived
> exclusively from training data for each fold.  Signal V2 itself is left
> completely unmodified.

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

| Script / Entry-point                    | Purpose                                       |
| --------------------------------------- | --------------------------------------------- |
| `run_pipeline.py`                       | Simple pipeline orchestrator                  |
| `run_pipeline_strict.py`                | Strict deterministic pipeline orchestrator    |
| `run_regime_v3.py`                      | Launcher for `experiments/regime_v3`          |
| `run_regime_filter_pipeline.py`         | Launcher for `experiments/regime_filter_pipeline` |
| `run_regime_v4.py`                      | Launcher for `experiments/regime_v4`          |
| `run_regime_v4_signal_filter.py`        | Launcher for `experiments/regime_v4_signal_filter` |
| `build_fx_sentiment_dataset.py`         | Build canonical dataset from raw inputs       |
| `build_sentiment_feature_contract.py`   | Feature contract builder                      |
| `attach_regimes_to_h1_dataset.py`       | Attach D1 regime labels to H1 dataset         |
| `config.py`                             | Centralised pipeline configuration            |

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
