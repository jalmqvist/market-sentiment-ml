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

Identify the minimal behavioral rules required to reproduce the statistical
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

- **Accumulation**  
  Agents increase position size when aligned with price signal.

- **Inertia (anchoring)**  
  Agents resist switching positions.

- **Asymmetric reinforcement**  
  Agents strengthen positions when aligned with price.

No decay or release mechanism is required.

---

### Observed Dynamics

The model produces:

- strong persistence (high autocorrelation)
- clustered sentiment regimes
- heavy-tailed positioning
- non-mean-reverting behavior

The system is **path-dependent**.

---

### Multi-Pair Validation

Results show clear regime dependence.

#### Group A — Model matches data

- EUR/USD, GBP/USD, NZD/USD
- AUD/NZD

Characteristics:
- low abs_mean_diff
- stable autocorrelation
- realistic extreme frequency

#### Group B — Model fails

- All JPY pairs
- CHF pairs
- Some CAD pairs

Failure mode:
- excessive accumulation
- distorted magnitude (high abs_mean_diff)
- unstable persistence

---

### Interpretation

The ABM captures sentiment structure only in **trend-dominated regimes**.

It fails in markets characterized by:

- macro-driven flows
- carry dynamics
- regime switching

---

### Relationship to Predictive Results

This explains earlier findings:

- sentiment has no predictive signal globally
- sentiment does not improve price-based models

Because:

> sentiment reflects **current positioning state**, not future returns

However, the regime dependence suggests:

> predictive signal may exist **within specific market subsets**

---

### Conclusion

Retail sentiment is:

- path-dependent
- regime-dependent
- structurally non-equilibrium

---

### Status

- EUR/USD: validated
- Multi-pair: validated with regime segmentation
- Generalization: conditional, not universal

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

### 4. Regime-Aware Experiments

#### Motivation

ABM multi-pair validation showed that the behavioral accumulation model
reproduces sentiment structure only in **trend-dominated markets**
(EUR/USD, GBP/USD, NZD/USD) and fails in macro/carry-driven markets
(JPY, CHF pairs). This regime dependence is the empirical basis for
testing whether predictive signal exists conditionally within specific regimes.

#### Regime Definitions (Dataset v1.3.0)

Regimes are derived from two backward-looking, per-pair binary flags:

| Flag | Computation |
|---|---|
| `is_trending` | `trend_strength > 1.0`, where `trend_strength = abs(trend_12b) / (vol_12b + 1e-8)` |
| `is_high_vol` | `vol_12b > vol_12b.median()` (per pair) |

| Label | Condition |
|---|---|
| HVTF | High-vol & trending |
| LVTF | Low-vol & trending |
| HVR  | High-vol & ranging |
| LVR  | Low-vol & ranging |

All features are strictly causal (backward-looking only, grouped by pair).

#### How to Run

Build dataset v1.3.0 (includes vol + regime features):

```bash
python scripts/build_dataset.py --version 1.3.0
```

Run regime experiment (MLP + LSTM):

```bash
./scripts/run_dl_regime_experiment.sh 1.3.0 HVTF EURUSD,GBPUSD,NZDUSD 50
```

Filtering is applied before feature extraction and sequence building to
ensure deterministic train/test splits and leak-free normalization.

#### Status

- Dataset v1.3.0: defined (requires build)
- Regime feature sets: `price_trend`, `price_trend_sentiment`
- HVTF / LVTF experiments: pending

------

## Conclusion

> The project has transitioned from **signal discovery** to **structured exploration of model space**

The absence of signal is:

- consistent
- reproducible
- informative

It provides a solid foundation for future work.