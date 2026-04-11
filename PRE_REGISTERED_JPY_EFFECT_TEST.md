# Pre-Registered JPY Effect Validation Test

## Purpose

This document locks a final validation test for the JPY-cross persistence effect discovered during earlier exploratory analysis.

The goal is to reduce researcher degrees of freedom by fixing:

- the subgroup
- the sentiment condition
- the horizons
- the comparison group
- the evaluation metrics
- the untouched validation period


before running the final test.

## Background

Earlier exploratory analysis suggested that a simple broad contrarian sentiment rule was not robust across the cleaned FX universe.

However, a more specific pattern emerged:

- within the cleaned cross universe
- under persistent extreme sentiment
- JPY crosses appeared to exhibit stronger contrarian returns than non-JPY crosses


This pattern survived:

- pair-level outlier filtering
- pair-group analysis
- persistence analysis
- pair-cluster permutation testing
- time-based holdout testing
- expanding-window walk-forward validation


This document defines a stricter locked test of that effect.

---

## Locked hypothesis

Within the cleaned cross universe, the JPY-cross subgroup will show stronger contrarian returns than the non-JPY cross subgroup when retail sentiment is both extreme and persistent.

---

## Locked dataset

Canonical dataset:

```text
data/output/master_research_dataset_core.csv
```

If a cleaned core variant is used, it must be explicitly documented and fixed before running the test.
For the current validation phase, the intended input is:

```
data/output/analysis/master_research_dataset_core_cleaned.csv
```

------

## Locked universe

### Pair universe

Cross pairs only.

### JPY-cross subgroup

The subgroup is fixed as:

- `aud-jpy`
- `cad-jpy`
- `chf-jpy`
- `eur-jpy`
- `gbp-jpy`
- `nzd-jpy`

### Comparison group

All remaining non-JPY pairs within the cross universe.

------

## Locked sentiment condition

The condition is fixed as:

- `abs_sentiment >= 70`
- `extreme_streak_70 >= 3`

No threshold, persistence, or subgroup changes are allowed inside this validation step.

------

## Locked horizons

The test horizons are fixed as:

- `12b`
- `48b`

These correspond to trading-bar forward contrarian returns:

- `contrarian_ret_12b`
- `contrarian_ret_48b`

------

## Locked evaluation period

The final untouched validation period is defined as the most recent period not used for subgroup selection or further tuning.

For the current version of this test, the intended untouched period is:

- all rows with `snapshot_time` in calendar year **2026**

If the available data does not cover a full 2026 calendar year, the test should still use the latest available 2026 segment and explicitly report the observed date range.

No further subgroup or threshold tuning may be performed after this choice.

------

## Locked metrics

For each horizon (`12b`, `48b`), report:

### Point estimates

- JPY-cross mean contrarian return
- non-JPY-cross mean contrarian return
- difference in mean contrarian return
- JPY-cross hit rate
- non-JPY-cross hit rate
- difference in hit rate

### Uncertainty

Report 95% bootstrap confidence intervals for:

- difference in mean contrarian return
- difference in hit rate

Bootstrap should use **date-block resampling**, not iid row resampling.

### Stability

Report the same metrics across subwindows inside the untouched period, preferably by:

- quarter, if sample size allows
- otherwise by month

------

## Interpretation rules

### Supportive result

The result is considered supportive if:

- the JPY-minus-non-JPY mean difference remains positive
- the hit-rate difference remains positive
- bootstrap intervals are meaningfully informative
- and subwindow results do not indicate complete instability

### Non-supportive result

The result is considered non-supportive if:

- the JPY-minus-non-JPY difference is near zero or negative
- bootstrap intervals are wide and centered near zero
- or the effect collapses across subwindows

A non-supportive result does **not** imply that the sentiment dataset is useless. It may indicate that:

- the discovered effect was partly exploratory
- the signal is regime-dependent
- or the current forward-return target does not fully capture the underlying retail-trader behavior

------

## Change control

This document defines a locked validation step.

Any later change to:

- subgroup definition
- threshold
- persistence condition
- horizon
- untouched period
- metric choice

must be treated as a **new exploratory or validation phase**, not as part of this pre-registered test.