# Research State Summary — Market Sentiment ML

## Objective

Determine whether retail FX sentiment contains **causal, exploitable predictive signal**.

---

## Current Status

| Area                 | Status                          |
| -------------------- | ------------------------------- |
| Data quality         | ✅ verified                      |
| Pipeline correctness | ✅ verified                      |
| Validation framework | ✅ strong                        |
| Price signal         | ✅ stable (~0.14 Sharpe)         |
| Sentiment signal     | ❌ no standalone/additive signal |
| DL signal            | ⚠ weak, regime-dependent        |

---

## Core Finding

> Retail sentiment has **no standalone or additive predictive value**,  
> but exhibits **weak conditional signal under regime filtering**.

---

## Deep Learning — Controlled Signal Cartography

### Setup

- Fixed configuration:
  - horizon = 24
  - quantile = 0.50
- Models:
  - MLP (baseline)
  - LSTM (sequence)
- Evaluation:
  - per pair × regime
  - weighted F1

---

### Key Results

- Weak signal exists (F1 ≈ 0.25–0.50)
- Structure is **stable across models**
- Signal is **strongly regime-dependent**

---

### Regime Hierarchy (Empirical)

| Regime | Interpretation    |
| ------ | ----------------- |
| LVTF   | strongest, stable |
| HVR    | moderate          |
| LVR    | unstable / sparse |
| HVTF   | weak / noise      |

---

### Pair Effects

- Secondary to regime effects
- Stronger structure in:
  - USDJPY
  - USDCHF
  - EURJPY
- Weak / flat:
  - EURGBP
  - GBPJPY

---

### Interpretation

Predictability requires:

- directional structure (trend)
- **and stability (low volatility)**

Signal fails when:

- volatility dominates
- market driven by macro/flows

---

## Agent-Based Modeling (ABM)

### Current Model

Captures:

- accumulation
- inertia
- asymmetric reinforcement

Produces:

- persistence
- clustering
- path dependence

---

### Limitation

Fails in:

- JPY
- CHF
- some CAD pairs

---

### Updated Understanding

ABM currently models:

trend → accumulation → sentiment structure

DL results imply:

trend + stability → predictive signal

trend + high volatility → breakdown

---

### Required Update

ABM must incorporate:

- volatility dynamics
- regime switching / persistence

---

## DL ↔ ABM Relationship

DL is not validating ABM directly.

Instead:

> DL provides empirical constraints on ABM

ABM must reproduce:

- LVTF > HVTF signal difference
- regime-conditioned predictability
- pair-dependent variation

---

## What Has Been Ruled Out

- raw sentiment signal
- additive sentiment signal
- simple conditional filters
- nonlinear static interactions

---

## What Remains Open

- regime-driven predictability mechanisms
- persistence / autocorrelation structure
- behavioral explanation of volatility gating
- cross-pair structural differences

---

## Next Steps

### 1. ABM Refinement (primary)

- introduce volatility/stability dynamics
- reproduce DL regime map

### 2. Regime Diagnostics

- feature persistence
- sentiment autocorrelation
- return predictability decay

### 3. Controlled DL (minimal)

- no further grid search
- confirm robustness under fixed config

---

## Research Phase

The project has transitioned to:

> **structure discovery and model reconciliation**

---

## Conclusion

- No global predictive signal
- No additive value from sentiment
- Weak signal exists **conditionally**

> Predictability is governed by **regime structure**, not sentiment alone.