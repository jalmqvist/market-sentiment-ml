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

## Agent-Based Modeling (ABM)

### Objective

Identify the minimal set of behavioral rules required to reproduce the statistical
properties of retail FX sentiment.

---

### Empirical Targets

The ABM is evaluated against:

- mean absolute sentiment (|S|)
- standard deviation
- autocorrelation
- frequency of extreme regimes

No predictive objective is used.

---

### Working Model

The following mechanisms are jointly sufficient:

- **Position accumulation**  
  Agents increase position size when aligned with price signal.

- **Inertia (anchoring)**  
  Agents resist switching positions.

- **Asymmetric reinforcement**  
  Agents strengthen positions when aligned with price.

The system operates without:

- decay
- forced release
- explicit equilibrium mechanisms

---

### Observed Dynamics

The model exhibits:

- strong persistence (high autocorrelation)
- clustered sentiment regimes
- heavy-tailed positioning
- non-mean-reverting behavior

The system is **path-dependent**.

---

### Stability Properties

The ABM displays phase-transition behavior:

- narrow parameter regime → stable, realistic output
- outside regime → saturation or collapse

This indicates sensitivity to behavioral balance rather than parameter tuning.

---

### Interpretation

The results imply:

- sentiment is driven by **position accumulation dynamics**
- there is no strong endogenous mechanism enforcing reversion
- crowd imbalance can persist without correction

---

### Relationship to Predictive Results

This explains earlier findings:

- sentiment contains no predictive signal
- sentiment does not improve price-based models

Because:

> sentiment encodes **current positioning state**, not future returns

---

### Status

- EUR/USD: validated (structural match achieved)
- Other pairs: not yet tested

---

## Deep Learning (LSTM)

Sequence models were tested to evaluate whether temporal dependencies
or path-dependent interactions contain predictive signal.

Result:

- No predictive signal recovered
- No improvement from sentiment
- No temporal structure detected

Conclusion:

> The absence of signal persists across sequence models.

---

## Deep Learning Experiments (MLP)

A structured set of deep learning experiments was conducted to test whether nonlinear models can extract predictive signal from sentiment.

### Models Tested

- price_only
- price + sentiment
- price + volatility
- price + volatility + sentiment

### Results

- price_only: small but stable predictive signal
- price + sentiment: no improvement
- price + volatility: performance degrades
- price + volatility + sentiment: no recovery

### Conclusion

> No predictive signal is recovered by nonlinear models.

This extends previous findings:

- no standalone signal
- no additive signal
- no conditional signal
- no nonlinear interaction signal

### Implication

Sentiment does not contain exploitable directional information under current representation.

### Remaining Hypothesis

Not yet falsified:

- sequence-dependent effects
- delayed or path-dependent interactions

→ Sequence models (LSTM) tested — no predictive signal identified

---

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

\[
E[r_{t+1} \mid S_t] \approx 0
\]

------

### 2. No incremental contribution

\[
E[r_{t+1} | P_t, S_t] \approx E[r_{t+1} | P_t]
\]

------

### 3. Informational redundancy

> Sentiment does not add new information beyond price.

Likely explanation:

- sentiment reflects accumulated trader positioning responding to price dynamics
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