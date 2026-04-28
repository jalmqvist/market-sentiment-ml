# Research State Summary — Market Sentiment ML

## Objective

Determine whether retail FX sentiment contains **causal, exploitable predictive signal**.

------

## Current Status

| Area                 | Status                                |
| -------------------- | ------------------------------------- |
| Data quality         | ✅ Verified                            |
| Pipeline correctness | ✅ Verified                            |
| Validation framework | ✅ Strong                              |
| Price signal         | ✅ Stable (~0.14 Sharpe)               |
| Sentiment signal     | ❌ No standalone or incremental signal |

------

## Core Finding

> **Retail sentiment provides no predictive value for returns under tested conditions.**

This holds across:

- standalone signals
- additive combinations with price
- conditional / regime-based models

------

## Incremental Value Tests (V28–V29)

### V28 — Additive Model

Test:

> Does sentiment improve a price-based signal when added linearly?

Result:

- Price Sharpe ≈ 0.14
- Price + sentiment → Sharpe ≈ 0

Conclusion:

> Sentiment introduces noise and degrades signal quality.

------

### V29 — Conditional Model

Test:

> Does sentiment improve performance in extreme regimes?

Result:

- Identical performance to price-only signal

Conclusion:

> No measurable conditional benefit under this formulation.

------

## Interpretation

The combined evidence supports:

### 1. No standalone signal

[
E[r_{t+1} | S_t] \approx 0
]

------

### 2. No incremental contribution

[
E[r_{t+1} | P_t, S_t] \approx E[r_{t+1} | P_t]
]

------

### 3. Informational redundancy

> Sentiment does not add new information beyond price.

Likely explanation:

- sentiment reflects price dynamics
- but does not lead them
- and may introduce noise when used directly

------

## Key Lessons

### 1. Strong validation is essential

Pipeline-based signals (V19–V21):

- appeared robust
- failed under independent validation

------

### 2. Simpler models reveal truth faster

Minimal, causal implementations:

- exposed artifacts
- eliminated false positives

------

### 3. Negative results define the search space

We have now ruled out:

- linear relationships
- additive models
- simple conditional filters

This significantly narrows future research directions.

------

## Hypotheses Tested (Falsified)

- Raw sentiment predicts returns
- Extremes produce contrarian alpha
- Additive sentiment improves price signals (V28)
- Conditional sentiment improves regimes (V29)
- Pipeline-derived signals are valid

------

## What remains open

Not ruled out:

- nonlinear / higher-order interactions
- alternative targets (volatility, drawdowns)
- sequence-based effects
- structural behavioral dynamics

------

## Active Research Directions

### 1. Hypothesis Testing

Continue systematic falsification of simple behavioral models.

------

### 2. Deep Learning

Test whether sequence models can capture nonlinear dependencies.

------

### 3. Agent-Based Modeling

Understand how sentiment emerges from trader behavior.

------

## Conclusion

> The project has transitioned from **signal discovery** to **structured exploration of model space**

The absence of signal is:

- consistent
- reproducible
- informative

It provides a solid foundation for future work.