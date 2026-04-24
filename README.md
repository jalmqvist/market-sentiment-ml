# FX Retail Sentiment Research & Signal Pipeline

A quantitative research and signal engineering pipeline for extracting **conditional alpha from retail FX sentiment**.

---

## Executive summary

This project investigates whether retail FX sentiment contains predictive information — and more importantly:

> **When it becomes predictive**

The final outcome is not just a research conclusion, but a working system:

> **A regime-conditioned trading pipeline that improves signal quality by filtering for favorable market + behavioral conditions**

---

## Main result (current)

After extensive validation, debiasing, and multiple model iterations:

> **Retail sentiment contains no unconditional edge — but a measurable, conditional edge exists**

Using regime filtering:

- **Baseline signal (always-on)** → Sharpe ≈ 0.01
- **Regime-filtered signal** → Sharpe ≈ **0.04–0.05**
- Coverage ≈ **10% of opportunities**
- Hit rate ≈ **52–53%**

---

## Core insight

> **Retail crowd behavior is only exploitable under specific market conditions**

In particular:

> **Crowd extremes become predictive when they occur within certain trend and volatility contexts**

This transforms the problem from:

- ❌ “Is sentiment predictive?”

to:

- ✅ “When is sentiment predictive?”

---

## Evolution of the project

### Phase 1 — False discovery (invalidated)

Initial findings suggested:

- strong pair-specific effects (JPY)
- regime dependence
- large Sharpe ratios

These were invalidated due to:

- overlapping signals
- in-sample bias
- flawed validation

---

### Phase 2 — Strict validation (negative result)

After enforcing:

- non-overlapping samples
- walk-forward validation
- holdout testing

Result:

> **No unconditional signal**

---

### Phase 3 — Behavioral insight

Shift from price regimes → **crowd behavior**

Discovery:

> Signal depends on interaction between **sentiment extremes and trend context**

---

### Phase 4 — Regime discovery (Regime V3)

Introduced:

- regime definitions (volatility × trend × sentiment)
- interaction features
- walk-forward modeling (Ridge, LightGBM)

Result:

> Weak predictive power, but **clear regime-dependent structure**

---

### Phase 5 — Regime filtering (Regime V4 — current)

Key breakthrough:

> Use regimes **not to predict**, but to **filter trades**

Pipeline:

1. Discover regimes on training data
2. Select regimes with:
   - sufficient sample size
   - positive Sharpe
3. Apply filter to test data
4. (Optional) apply directional logic

Result:

> **Significant improvement in signal quality via selective execution**

---

## System architecture

### Layer 1 — Base signal (Signal V2)

- Derived from sentiment features
- Uses price-based momentum (causal, no leakage)
- Always-on baseline

---

### Layer 2 — Regime filter (Regime V4)

Defines regimes using:

| Feature          | Method           |
| ---------------- | ---------------- |
| Volatility       | Quantile buckets |
| Trend direction  | Sign of trend    |
| Trend strength   | Quantiles        |
| Sentiment regime | Fixed bins       |

Each observation is mapped to a **regime key**.

---

### Layer 3 — Regime selection

Per walk-forward fold:

- compute regime statistics on training data
- select regimes satisfying:
  - minimum sample size
  - minimum Sharpe
  - persistence across folds

---

### Layer 4 — Signal execution

Only trade when:

- observation belongs to selected regime

Optional:

- follow or fade based on regime direction

---

## Key findings

### 1. No unconditional edge

- Raw sentiment ≈ noise

---

### 2. Conditional edge exists

- Appears only under specific regimes
- Requires filtering

---

### 3. Regime filtering improves Sharpe

- ~4× improvement vs baseline
- reduces coverage significantly

---

### 4. Trade-off: quality vs capacity

| Metric   | Baseline | Filtered |
| -------- | -------- | -------- |
| Sharpe   | ~0.01    | ~0.05    |
| Coverage | 100%     | ~10%     |
| Hit rate | ~50%     | ~52–53%  |

---

### 5. Instability remains

- performance varies across years
- some regimes degrade or disappear

This is expected for behavioral signals.

---

## Validation methodology

Strict validation throughout:

- walk-forward evaluation
- expanding window training
- no forward-looking features
- regime selection using **training data only**
- out-of-sample performance tracking

---

## Current limitations

- regime instability across time
- low coverage (capacity constraints)
- sensitivity to thresholds (min_n, Sharpe)
- early-period cold start (no regimes available)

---

## Research directions

### 1. Regime stabilization

- persistence constraints
- smoothing regime definitions

---

### 2. Signal + regime interaction

- thresholding base signal
- combining strength + regime filter

---

### 3. Portfolio construction

- cross-pair aggregation
- regime-aware weighting

---

### 4. Behavioral modeling

- crowd saturation
- positioning pressure
- trend exhaustion

---

## Running the project

### Build dataset

```bash
python build_fx_sentiment_dataset.py
```

------

### Signal pipeline

```
python run_signal_v2.py
```

------

### Regime-filtered pipeline (recommended)

```
python run_regime_v4_signal_filter.py \
  --data data/output/master_research_dataset.csv
```

------

### Example variations

```
# With direction logic
python run_regime_v4_signal_filter.py --with-direction

# Stricter regime selection
python run_regime_v4_signal_filter.py --min-n 150 --min-sharpe 0.1

# Signal thresholding
python run_regime_v4_signal_filter.py --threshold 0.5
```

------

## Output schema (core)

Per fold:

```
["year", "n", "mean", "sharpe", "hit_rate", "coverage"]
```

------

## Project structure

```
build → signal → regime → filter → evaluate
```

- `signal_v2.py` → base signal
- `regime_v3.py` → regime discovery
- `regime_filter_pipeline.py` → filtering
- `regime_v4_signal_filter.py` → combined pipeline

------

## Final takeaway

This project demonstrates:

> **Alpha is not static — it is conditional**

Retail sentiment is not broadly predictive.

But:

> **Under the right conditions, it becomes exploitable**

------

## Meta insight

The most important result is methodological:

> Fixing validation did not destroy the signal —
>  it revealed **where it actually lives**

---
