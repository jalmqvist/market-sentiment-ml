# FX Retail Sentiment — Behavioral Signal Research

A quantitative research project studying whether retail FX sentiment contains predictive signal.

> [!NOTE]
>
> **Current status (integration): v1 MSML → MPML DL artifact export is live (proof-of-concept complete).**  
> This repo can export per-run H1 DL prediction artifacts consumed by `market-phase-ml`, where they are aggregated to D1 and joined as an optional feature layer. Sparse/partial coverage is expected in v1.

---

## Executive Summary

**Current status: no standalone or additive alpha.**  
**However, a weak, regime-dependent predictive signal exists.**

Extensive experimentation with strict validation leads to the following:

- Raw retail sentiment = noise
- Pipeline-based signals (V20–V21) were invalidated (artifacts)
- Price-based signals show small but stable predictive power (~0.14 Sharpe)``
- Sentiment provides **no additive value in static models**

### Updated Finding (DL v3 — Controlled Experiments)

Using controlled (non-grid) experiments:

> A **weak but consistent predictive signal** exists when:
> - using temporal models (MLP/LSTM with lagged features)
> - conditioning on **market regime**
> - predicting medium horizons (~24 bars)

Observed performance:

- F1 ≈ 0.25–0.50 (depending on pair/regime)
- Consistent across MLP and LSTM

---

## Key Insight

> **Predictability is regime-dependent, not universal**

### Regime hierarchy (empirical)

| Regime                | Signal strength                                              |
| --------------------- | ------------------------------------------------------------ |
| LVTF (low-vol trend)  | ✅ strongest, stable                                          |
| HVR (high-vol range)  | ⚠ moderate — **volatility/stability is the missing gating variable** |
| LVR (low-vol range)   | ⚠ unstable / sparse                                          |
| HVTF (high-vol trend) | ❌ weak / near-random                                         |

---

## Interpretation

The signal exists when:

- markets are **stable**
- directional structure persists
- sentiment can **accumulate over time**

The signal breaks when:

- volatility is high
- price is dominated by macro/flow dynamics
- sentiment becomes reactive rather than predictive

---

## Key Conclusion

Retail sentiment is:

- not predictive in isolation
- not additive to price
- but contains **weak conditional structure**

> Predictability emerges only under specific **regime conditions**

---

## Agent-Based Modeling (ABM)

### Behavioral Insight

Retail sentiment behaves as a **path-dependent accumulation process**:

- accumulation (position building)
- inertia (resistance to switching)
- asymmetric reinforcement

This reproduces:

- persistent crowd imbalance
- clustered regimes
- realistic sentiment magnitude

---

### Multi-Pair Findings

ABM matches data in:

- EUR, GBP, NZD pairs

Fails in:

- JPY, CHF, some CAD pairs

---

### Interpretation

Markets differ structurally:

- Trend-dominated → accumulation works
- Macro/flow-driven → accumulation breaks

---

## DL ↔ ABM Synthesis (Updated)

DL results refine ABM:

Previous assumption:

trend → accumulation → signal

Updated:

trend + stability → accumulation → signal  
trend + high volatility → breakdown

> **Volatility/stability is the missing gating variable**

---

## Current ABM State (Stage‑2 “Decay/Release”)

The ABM now includes an **optional, volatility-conditioned decay/release mechanism** (defaults OFF) to prevent absorbing saturation states and to model “escape” behavior under volatility.

Recent work added a **Stage‑2 beta sensitivity harness**:

- `abm_experiments/decay_beta_sensitivity.py`

It reports:

- `pct_time_saturated` (|net_sentiment| ≥ 90)
- `sign_flips`
- `autocorr_lag1`
- plus verbose-only diagnostics for “near-boundary” behavior:
  - `pct_time_abs_le_20` (|net_sentiment| ≤ 20)
  - `pct_time_negative`

This supports explaining why DL can show regime-conditional behavior even when ABM outputs remain sign-stable.

---

## What’s Next (Clear Strategy)

The project is shifting from **parameter sweeps** to **mechanism matching**: aligning ABM dynamics with the *stylized behaviors* implied by DL results.

### 1) Make ABM↔DL comparisons metric-aligned

Add/compute comparable “distance-to-boundary” and regime-transition metrics for both ABM and DL outputs:

- time near boundary (already: `pct_time_abs_le_20`)
- run-length / dwell-time in regimes (e.g., time between transitions)
- saturation frequency vs regime

### 2) Target the structural knobs (not more escape tuning)

Use a small, controlled factorial design on a few representative pairs:

- **USD-JPY** (positive-locking)
- **EUR-JPY** (near-boundary)
- **GBP-JPY** (often negative)

Vary only a few core parameters at a time:

- `trend_ratio` (introduce more contrarians)
- herding/crowd weight (reduce lock-in)
- aggregation (consider magnitude-weighted voting vs pure sign voting)

### 3) Explain pair differences as regimes, not errors

Some pairs (e.g. GBP-JPY) naturally live in negative regimes for long stretches. Treat `pct_time_negative` as a **diagnostic** (what regime is the market in?) rather than a universal failure condition[...]

---

## Deep Learning Results

### Grouped Pair-Family Findings (NEW)

Recent controlled multi-pair experiments suggest that FX pairs may separate into
distinct behavioral “families” with different sentiment dynamics.

#### Persistent / accumulation-oriented family
Examples:
- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

Observed behavior:
- weaker directional generalization
- lower F1 scores
- smoother / more persistent sentiment structure
- consistent with ABM accumulation dynamics

#### Reactive / release-oriented family
Examples:
- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

Observed behavior:
- stronger directional generalization
- higher F1 scores
- more state-dependent behavior
- consistent with volatility-conditioned “release” dynamics in the ABM

Importantly, these differences emerge under:
- the same architecture
- same features
- same training procedure
- same target horizon
- same regime filter

This suggests that retail sentiment dynamics may not be universal across FX pairs.

Current hypothesis:

- EUR/GBP/NZD-type pairs are more persistence/accumulation dominated
- JPY/CHF-type pairs are more reactive / macro-flow dominated

This is now an active research direction.

---

### Cross-Family Transfer Experiments (NEW)

The DL export pipeline now supports:

- training on one FX pair family
- exporting inference predictions on another family

This enables explicit testing of whether learned sentiment structure:

- generalizes universally
- or is family-specific.

Example:

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --train-pairs EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD \
  --predict-pairs USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend
```

This produces predictions for the reactive family using a model trained only on the persistent family.

### Working hypothesis

Current working hypothesis:

- persistent-family pairs encode slower accumulation/inertia dynamics
- reactive-family pairs encode more volatility-conditioned release dynamics

If true:

- within-family transfer should remain partially stable
- cross-family transfer should degrade materially

This is now a primary research direction.

---

### Current Research Direction (May 2026)

Recent experiments suggest that FX sentiment dynamics may separate into
distinct behavioral “families”:

- persistence / accumulation-dominated pairs
- reactive / release-dominated pairs

This structure appears:

- across grouped DL experiments
- in downstream MPML integration
- and increasingly aligns with ABM persistence/release dynamics.

The project focus has therefore shifted from:

    "does sentiment predict returns?"

toward:

    "what behavioral mechanisms generate conditional predictability?"

---

### MLP (static + lagged features)

- Extracts weak signal under regime conditioning
- No additive value from sentiment alone
- Confirms price dominates

### LSTM (sequence model)

- Recovers similar structure
- No major advantage over MLP
- Confirms signal is not architecture-dependent
- Export pipeline now has parity with MLP
- Supports cross-family transfer (`--train-pairs` / `--predict-pairs`)
- Uses pair-safe grouped sequence construction with metadata-aligned export rows

---

## Core Findings

| Component          | Status              |
| ------------------ | ------------------- |
| Raw sentiment      | ❌ noise             |
| Additive sentiment | ❌ no value          |
| Price signal       | ✅ stable            |
| DL signal          | ⚠ weak, conditional |
| Regime dependence  | ✅ strong            |
| Pair dependence    | ⚠ secondary         |

---

## Research Direction

The project has shifted from:

> signal discovery

to:

> understanding **when and why signal exists**

### Active directions:

1. **ABM refinement**
   - add volatility/stability dynamics
   - reproduce regime-dependent signal

2. **Regime analysis**
   - persistence / autocorrelation
   - structural differences across regimes

3. **Controlled DL experiments**
   - no further grid search
   - isolate causal structure

---

## Philosophy

- correctness over results
- causality over complexity
- negative results are valuable

---

## MSML → MPML integration (DL prediction artifacts, v1)

**Status:** live (proof-of-concept complete).  
This repo exports per-run H1 DL prediction artifacts consumed by `market-phase-ml`, where they are aggregated (H1→D1) and joined as numeric features.

### What works (v1)
- Per-run artifact export (`data/output/dl_predictions/*.parquet` + `*.manifest.json`)
- Surface identity embedded in parquet rows: `model`, `dl_regime`, `target_horizon`, `feature_set`
- Export window controls to produce MPML-overlapping artifacts:
  - `--export-split {test,all}` (default `test`)
  - `--export-after-year YEAR` / `--export-before-year YEAR` (export-only filtering)
- Artifact diagnostics printed at export time (row count, unique pairs, entry_time range)

### Known limitations (v1)
- Coverage can be sparse depending on regime/pair/date overlap; this is expected in v1
- Single-surface artifacts (no multi-surface ensembles here; MPML selects a single surface per run)
- Regime filtering is explicit (`--regime LVTF|HVTF|HVR|LVR`); no “all-regime” export mode in v1

### A) Train + export predictions

Example (produces an artifact that overlaps MPML’s typical 2005–2024 D1 window):

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend \
  --export-split all \
  --export-after-year 2019 \
  --export-before-year 2024
```

LSTM export parity example (same artifact contract, sequence-safe export path):

```bash
python -m research.deep_learning.train_lstm \
  --dataset-version 1.3.2 \
  --train-pairs EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD \
  --predict-pairs USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend \
  --epochs 50 \
  --seq-len 24 \
  --export-split all \
  --export-after-year 2019 \
  --export-before-year 2024
```

Cross-family transfer example (train on one family, export inference on another):

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --train-pairs EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD \
  --predict-pairs USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend
```

In this mode, exported predictions can originate from a model trained on a
different pair family; row schema stays unchanged and provenance is recorded in
the run manifest.

Artifacts are written under:
- `data/output/dl_predictions/<run_id>.parquet`
- `data/output/dl_predictions/<run_id>.manifest.json`

### B) Use artifact in market-phase-ml

```bash
export DL_SIGNALS_ENABLED=true
export DL_PREDICTION_ARTIFACT_PATH=../market-sentiment-ml/data/output/dl_predictions/<run_id>.parquet
python -u main.py
```

Notes:
- MPML performs H1→D1 aggregation internally.
- Surface config must match the artifact identity (`model`, `dl_regime`, `target_horizon`, `feature_set`).
- Sparse/partial DL coverage is acceptable in v1 (MPML falls back per pair when coverage is zero).

---

## DL Signal Artifact (DL → market-phase-ml Integration)

The DL inference outputs are exported as versioned per-run artifacts and
consolidated into an operational cube for consumption by `market-phase-ml`.

### v1 Architecture (two-step)

**Step 1 — Per-run artifact** (written after each DL training / inference run):

```bash
python scripts/write_dl_prediction_artifact.py \
    --input-csv path/to/predictions.csv \
    --model MLP \
    --dl-regime LVTF \
    --target-horizon 24 \
    --feature-set price_vol_sentiment \
    [--output-dir data/output/dl_predictions]
```

Produces for each run:

| Path                                                | Description                                                  |
| --------------------------------------------------- | ------------------------------------------------------------ |
| `data/output/dl_predictions/{run_id}.parquet`       | Time-series payload (pair, entry_time, pred_prob_up, signal_strength, …) |
| `data/output/dl_predictions/{run_id}.manifest.json` | Identity + provenance (model, dl_regime, target_horizon, feature_set, calibration, …) |

**Step 2 — Consolidation** (builds the operational cube from all per-run artifacts):

```bash
python scripts/consolidate_dl_predictions.py \
    [--input-dir data/output/dl_predictions] \
    [--output-dir data/output/dl_signals]
```

Produces:

| Path                                                   | Description                 |
| ------------------------------------------------------ | --------------------------- |
| `data/output/dl_signals/dl_signals_h1_v1.parquet`      | Consolidated DL signal cube |
| `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` | Cube manifest and metadata  |

### Schema summary

| Column                 | Type     | Description                                      |
| ---------------------- | -------- | ------------------------------------------------ |
| `pair`                 | string   | Normalised FX pair (`xxx-yyy`)                   |
| `entry_time`           | datetime | H1 bar open timestamp (UTC, tz-naive)            |
| `pred_prob_up`         | float64  | P(price moves up) ∈ [0, 1]                       |
| `signal_strength`      | float64  | `2 * pred_prob_up − 1` ∈ [−1, 1]                 |
| `pred_direction`       | Int64    | Tri-state: +1 (>0.5), −1 (<0.5), 0 (==0.5)       |
| `prediction_timestamp` | datetime | Per-row inference timestamp (optional)           |
| `model`                | string   | Model identifier (e.g. `MLP`, `LSTM`)            |
| `dl_regime`            | string   | Producer regime: `HVTF` / `LVTF` / `HVR` / `LVR` |
| `target_horizon`       | Int64    | Prediction horizon in bars (numeric)             |
| `feature_set`          | string   | Feature set identifier                           |

Unique key: `(pair, entry_time, model, dl_regime, target_horizon, feature_set)`

> **Deprecated:** `scripts/build_dl_signal_artifact.py` (CSV consolidator) is
> kept for backward compatibility but will be removed in a future release.

See `docs/DL_SIGNAL_SCHEMA.md` for the complete schema, per-run artifact
format, and integration notes.

---

## Further Reading

- `docs/ABM_RUNBOOK.md`
- `docs/RESEARCH_STRATEGY.md`
- `docs/abm.md`
- `docs/DL_SIGNAL_SCHEMA.md`
