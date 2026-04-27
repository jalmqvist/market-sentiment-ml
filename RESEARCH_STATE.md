# Research State Summary — Market Sentiment ML

## Objective

Determine whether retail FX sentiment contains **causal, exploitable predictive signal**.

---

## Current Status

| Area                 | Status          |
| -------------------- | --------------- |
| Data quality         | ✅ Verified      |
| Pipeline correctness | ✅ Verified      |
| Validation framework | ✅ Strong        |
| Signal existence     | ❌ Not confirmed |

---

## Core Finding

> **No standalone predictive signal has been validated.**

Across all tested variants:

- Raw signal → Sharpe ≈ 0
- Shifted signal → Sharpe ≈ 0
- Shuffled signal → Sharpe ≈ 0

This holds consistently across years and configurations.

---

## Key Lessons

### 1. Apparent alpha is fragile

Previously observed signals (e.g. V19–V21):

- showed strong Sharpe (~0.20)
- appeared stable in pipeline

But failed under independent validation.

---

### 2. Pipeline complexity creates false positives

Identified failure modes include:

- index misalignment
- groupby/apply artifacts
- implicit selection bias

Conclusion:

> complex pipelines can generate convincing but invalid signals

---

### 3. Sentiment is reactive, not predictive

Empirically:

- sentiment tracks price behavior
- does not lead it
- adds little to no incremental information

---

## Hypotheses Tested (Falsified)

- Raw sentiment predicts returns
- Extremes produce contrarian alpha
- Persistence (streaks) contains signal
- Simple transformations (z-score, tanh) add edge
- Regime conditioning (volatility, trend) stabilizes signal
- Pipeline-generated signals (V19–V21) are valid

---

## Validation Standard (Current)

All new hypotheses must:

1. Work in **raw form** (no pipeline dependence)
2. Survive:
   - shift test
   - shuffle test
   - time split validation
3. Only then be considered for further modeling

---

## Interpretation

> The absence of signal is a **robust, repeatable result**

This suggests:

- retail sentiment may not encode exploitable information
- or signal exists only in highly conditional / nonlinear form

---

## Active Research Directions

### 1. Hypothesis Testing (statistical)

- continue systematic testing of behavioral features
- focus on falsifiable, minimal assumptions

---

### 2. Deep Learning (predictive)

- sequence models on sentiment + price
- goal: detect nonlinear or interaction effects

---

### 3. Agent-Based Modeling (explanatory)

- simulate trader behavior
- reproduce observed sentiment dynamics
- understand structural limitations of the data

---

## Conclusion

> The project has transitioned from **signal discovery** to **signal validation and falsification**

This provides a reliable foundation for:

- future modeling work
- methodological rigor
- credible research conclusions