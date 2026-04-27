# Research State Summary — Market Sentiment ML

## Objective

Determine whether retail FX sentiment contains exploitable predictive signal.

------

## Phase Summary

### Phase 1 — Early discovery (invalidated)

Findings:

- Strong Sharpe
- Pair-specific effects (JPY)
- Regime dependence

Outcome:
❌ Invalidated due to:

- overlapping samples
- in-sample bias
- flawed validation

------

### Phase 2 — Strict validation

Methods introduced:

- walk-forward evaluation
- non-overlapping samples
- holdout testing

Result:

> ❌ No unconditional signal

------

### Phase 3 — Regime exploration

Approach:

- volatility × trend × sentiment regimes
- filtering + weighting

Result:

> ⚠️ Weak, unstable conditional edge (~0.04–0.05 Sharpe)

------

### Phase 4 — Signal engineering (V19–V21)

Approach:

- direct signal extraction from `net_sentiment`
- minimal transformations
- walk-forward validation

Initial result:

> 🚀 Sharpe ≈ 0.21 (appeared strong)

------

### Phase 5 — Independent validation

New tools:

- validate_signal_raw.py
- pipeline_sanity_check.py
- pipeline_leakage_diagnosis.py

Result:

| Test             | Sharpe |
| ---------------- | ------ |
| Raw signal       | ~0.00  |
| Shifted          | ~0.00  |
| Shuffled         | ~0.00  |
| Pipeline (clean) | ~0.00  |

Conclusion:

> ❌ V19–V21 signal is an artifact

------

## Confirmed Negative Results

The following hypotheses are **falsified**:

- Raw sentiment predicts returns
- Simple transformations (tanh, z-score) produce alpha
- Cross-sectional ranking produces alpha
- Time-local signal exists in `net_sentiment`
- Pipeline-generated Sharpe (V19–V21) is real

------

## Confirmed Positive Results

The following are **true and reliable**:

- Dataset is correct (scraping verified)
- No forward leakage in clean pipeline
- Validation framework is sound
- Failure-mode tests reproduce leakage Sharpe (~0.12)

------

## Key Insight

> **Most apparent alpha was produced by implementation artifacts, not data**

------

## Current State

| Area                 | Status     |
| -------------------- | ---------- |
| Data quality         | ✅ Verified |
| Pipeline correctness | ✅ Verified |
| Signal existence     | ❌ None     |
| Validation framework | ✅ Strong   |

------

## Implications

- There is currently **no trading signal**
- Pipeline is now **trustworthy**
- Research must restart from **hypothesis generation**

------

## Recommended Research Strategy

### Step 1 — Signal-first validation

All new ideas must:

1. Work in raw form (no pipeline)
2. Survive:
   - shift test
   - shuffle test
   - time split

------

### Step 2 — Only then integrate

Pipeline becomes:

> evaluation tool, not discovery engine

------

### Step 3 — New hypothesis directions

Focus areas:

- event-based signals (rare extremes)
- cross-asset relationships
- latency / reaction effects
- structural features (time-of-day, clustering)

------

## Final Conclusion

> The system is now **scientifically valid but alpha-free**

This is the correct foundation for real discovery.