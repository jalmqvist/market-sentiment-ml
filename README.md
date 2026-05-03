# FX Retail Sentiment — Behavioral Signal Research

A quantitative research project studying whether retail FX sentiment contains predictive signal.

---

## Executive Summary

**Current status: no standalone or additive alpha.  
However, a weak, regime-dependent predictive signal exists.**

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

| Regime                | Signal strength                                              |
| --------------------- | ------------------------------------------------------------ |
| LVTF (low-vol trend)  | ✅ strongest, stable                                          |
| HVR (high-vol range)  | ⚠ moderate> **Volatility/stability is the missing gating variable** |
| LVR (low-vol range)   | ## Deep Learning Results⚠ unstable / sparse                  |
| HVTF (high-vol trend) | ### MLP (static + lagged features)❌ weak / near-random       |

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

## Conclusion

> Retail sentiment does not provide global predictive signal.

However:

> A **weak, regime-dependent signal exists**,  
> revealing structural constraints on market behavior.

---

## Further Reading

- `RESEARCH_STATE.md`
- `docs/RESEARCH_STRATEGY.md`
