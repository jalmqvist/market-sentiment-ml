# Validation Protocol

## Purpose

Define rules for detecting data leakage, testing causal validity, and
ensuring that reported model performance reflects genuine out-of-sample
predictive ability.

---

## 1. Shift Tests

Shift tests verify that a model relies on the correct causal relationship
between features and targets.

### t+1 shift
Shift the target forward by one bar:

```
target_shifted = target.shift(-1)
```

If model performance drops to chance, the model was not exploiting look-ahead
leakage from the target itself.

### t+5 shift
Shift target forward by five bars. Used to confirm that any marginal signal at
t+1 does not originate from within-bar autocorrelation.

**Rule**: A model that shows no edge at t+5 but claims edge at t+0 must be
scrutinized for leakage.

---

## 2. Shuffle Tests

### Row shuffle
Randomly permute all rows (breaking time alignment):

```python
df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
```

Expected result: performance collapses to chance. If not, the model is
exploiting a distributional artifact, not temporal signal.

### Label shuffle
Keep features intact; randomly permute only the target column. Expected
performance: chance level. If performance remains above chance, the evaluation
framework itself is leaking.

---

## 3. Leakage Rules

The following are **prohibited** during model development:

| Rule | Rationale |
|---|---|
| No normalization using full-dataset statistics | Future data would contaminate training |
| No feature engineering that uses future bars | Causal violation |
| No reusing test data for threshold tuning | Implicit overfitting to test set |
| No deduplication across train/test boundary | Can artificially inflate similarity |
| No target encoding on the training set before splitting | Target leakage |

---

## 4. Causal Constraints

All features must be available strictly before the bar being predicted.

| Feature type | Constraint |
|---|---|
| `net_sentiment` at time T | Available at T (broker snapshot at bar close) |
| `ret_1b` at time T | Return from T-1 to T — available at T |
| `ret_48b` at time T | Return from T to T+48 — **target only, never feature** |
| Regime labels | Must be assigned using only data up to T |

**Sentiment snapshot timing**: raw sentiment data is assumed to reflect
positions at bar close. The pipeline applies a `–1h` snapshot shift to account
for reporting delay. Do not remove this shift without re-validating all
downstream signal metrics.

---

## 5. Walk-Forward Evaluation

All final performance estimates use an expanding-window walk-forward:

1. Train on bars 0…N.
2. Evaluate on bars N+1…N+K (fixed test window).
3. Advance N by K, repeat.
4. Aggregate metrics across folds (mean ± std of AUC / accuracy).

No hyperparameter search is performed on the test fold at any step.
