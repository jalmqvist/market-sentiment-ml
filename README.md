# FX Retail Sentiment — Behavioral Signal Research

A quantitative research project studying whether retail FX sentiment contains predictive signal.

---

## Executive Summary

**Current status: no standalone or additive alpha.**  
**However, a weak, regime-dependent predictive signal exists.**

Extensive experimentation with strict validation leads to the following:

- Raw retail sentiment = noise
- Pipeline-based signals (V20–V21) were invalidated (artifacts)
- Price-based signals show small but stable predictive power (~0.14 Sharpe)
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

| Regime | Signal strength |
| --- | --- |
| LVTF (low-vol trend) | ✅ strongest, stable |
| HVR (high-vol range) | ⚠ moderate — **volatility/stability is the missing gating variable** |
| LVR (low-vol range) | ⚠ unstable / sparse |
| HVTF (high-vol trend) | ❌ weak / near-random |

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

Some pairs (e.g. GBP-JPY) naturally live in negative regimes for long stretches. Treat `pct_time_negative` as a **diagnostic** (what regime is the market in?) rather than a universal failure condition.

---

## Deep Learning Results

### MLP (static + lagged features)

- Extracts weak signal under regime conditioning
- No additive value from sentiment alone
- Confirms price dominates

### LSTM (sequence model)

- Recovers similar structure
- No major advantage over MLP
- Confirms signal is not architecture-dependent

---

## Core Findings

| Component | Status |
| --- | --- |
| Raw sentiment | ❌ noise |
| Additive sentiment | ❌ no value |
| Price signal | ✅ stable |
| DL signal | ⚠ weak, conditional |
| Regime dependence | ✅ strong |
| Pair dependence | ⚠ secondary |

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

## Export DL predictions for market-phase-ml

### A) Train + export predictions

`train.py` exports a per-run DL prediction artifact (parquet + manifest) that
`market-phase-ml` can consume.

Example:

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend \
  --export-after-year 2010 \
  --export-split all \
  --export-after-year 2019 \
  --export-before-year 2024
```

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
- Sparse/partial DL coverage is acceptable in v1 (MPML falls back to baseline per pair when coverage is zero).

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

| Path | Description |
|---|---|
| `data/output/dl_predictions/{run_id}.parquet` | Time-series payload (pair, entry_time, pred_prob_up, signal_strength, …) |
| `data/output/dl_predictions/{run_id}.manifest.json` | Identity + provenance (model, dl_regime, target_horizon, feature_set, calibration, …) |

**Step 2 — Consolidation** (builds the operational cube from all per-run artifacts):

```bash
python scripts/consolidate_dl_predictions.py \
    [--input-dir data/output/dl_predictions] \
    [--output-dir data/output/dl_signals]
```

Produces:

| Path | Description |
|---|---|
| `data/output/dl_signals/dl_signals_h1_v1.parquet` | Consolidated DL signal cube |
| `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` | Cube manifest and metadata |

### Schema summary

| Column | Type | Description |
|---|---|---|
| `pair` | string | Normalised FX pair (`xxx-yyy`) |
| `entry_time` | datetime | H1 bar open timestamp (UTC, tz-naive) |
| `pred_prob_up` | float64 | P(price moves up) ∈ [0, 1] |
| `signal_strength` | float64 | `2 * pred_prob_up − 1` ∈ [−1, 1] |
| `pred_direction` | Int64 | Tri-state: +1 (>0.5), −1 (<0.5), 0 (==0.5) |
| `prediction_timestamp` | datetime | Per-row inference timestamp (optional) |
| `model` | string | Model identifier (e.g. `MLP`, `LSTM`) |
| `dl_regime` | string | Producer regime: `HVTF` / `LVTF` / `HVR` / `LVR` |
| `target_horizon` | Int64 | Prediction horizon in bars (numeric) |
| `feature_set` | string | Feature set identifier |

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
