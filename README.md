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

## Further Reading

- `docs/ABM_RUNBOOK.md`
- `docs/RESEARCH_STRATEGY.md`
- `docs/abm.md`
