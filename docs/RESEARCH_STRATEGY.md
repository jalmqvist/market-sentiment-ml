# Research Strategy

## Why Retail FX Sentiment is Difficult

Retail FX sentiment data reflects the positioning of retail traders — a group that is
systematically on the wrong side of major moves. This creates a plausible contrarian signal:
when retail traders are heavily long, prices often fall.

However, this effect is:

- **Noisy**: The raw contrarian relationship is weak and inconsistent.
- **Non-stationary**: The relationship changes over time as market structure evolves.
- **Regime-dependent**: The signal only appears (weakly) under specific combinations of
  trend direction, volatility, and sentiment extremes.
- **Pipeline-sensitive**: Many apparent edges disappear when validated outside the pipeline
  that produced them. Pipeline complexity introduces subtle biases and data leakage.

The core challenge is distinguishing **genuine conditional predictability** from
**pipeline artifacts and in-sample overfitting**.

---

## Predictive Modeling vs Behavioral Modeling

This project pursues two fundamentally different modeling paradigms:

### Predictive Modeling

**Goal**: Learn a mapping from features (sentiment, price, regime indicators) to future returns.

**Methods**: Linear models (Ridge), gradient boosting (LightGBM), deep learning (LSTM, Transformer).

**Limitations**:
- Prone to overfitting on small, noisy datasets.
- Requires strict walk-forward validation to avoid look-ahead bias.
- Results are fragile: small changes in feature construction or evaluation methodology
  can produce large swings in apparent performance.

**Current status**: No validated predictive model exists. All apparent edges have been
attributed to pipeline artifacts after strict out-of-sample testing.

---

### Behavioral Modeling

**Goal**: Understand *why* retail sentiment might be informative, by simulating the
mechanisms that generate it.

**Methods**: Agent-based models (ABM) that simulate retail trader behavior — trend-following,
loss aversion, herding — and study when these behaviors create exploitable crowd dynamics.

**Advantages**:
- Does not require training data in the traditional sense.
- Can generate synthetic data for hypothesis testing.
- Provides interpretable, mechanistic explanations.
- Less prone to overfitting (no direct fitting to price data).

**Current status**: Experimental. ABM framework in `research/abm/` is under development.

---

## Deep Learning Approach

Sequence models (LSTM, Transformer) applied to sentiment and price time series may detect
nonlinear temporal patterns that linear models miss.

**Motivation**:
- Retail positioning dynamics may have complex temporal structure.
- Sequence models can learn variable-length dependencies.
- Attention mechanisms may identify which market regimes are most informative.

**Risks**:
- High overfitting risk on limited data.
- Requires careful temporal cross-validation (no shuffling).
- Interpretability is low.

**Current status**: Experimental. Framework in `research/deep_learning/` is under development.

---

## Validation Framework

All hypotheses must pass the following validation ladder before being considered credible:

1. **Raw validation** (`research/raw_validation/`): Test the core hypothesis using a minimal,
   standalone script with no pipeline dependencies. If the signal does not appear here, it
   does not exist.

2. **Shift test**: Confirm that shifting the signal by one or more periods eliminates any
   apparent edge. If shifting does not change results, the signal is likely spurious.

3. **Shuffle test**: Confirm that randomly shuffling the signal destroys performance. If
   shuffling does not change results, the apparent edge is not driven by the signal.

4. **Walk-forward validation**: Evaluate on rolling out-of-sample windows using only
   information available at each decision point.

5. **Hold-out test**: Final evaluation on a held-out period never seen during development.

Only hypotheses that survive all five stages are considered for further development.

---

## Terminology

| Preferred term | Replaces |
| -------------- | -------- |
| Research experiment | Production pipeline |
| Validation framework | Pipeline |
| Hypothesis testing | Signal engineering |
| Out-of-sample validation | Backtesting |
| Conditional predictability | Alpha |

---

## On Negative Results

Every experiment that fails to find a validated signal is a contribution. It rules out a
hypothesis, narrows the search space, and prevents wasted effort on dead ends.

The discipline of recording, archiving, and properly labeling negative results is central
to this project's research integrity.
