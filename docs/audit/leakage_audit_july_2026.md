# Volatility Feature Leakage Fix — Summary (L-11)

**Date:** 2026-07-10
**Scope:** market-sentiment-ml (MSML) dataset builder
**Related:** `leakage_audit.md` (2026-05-18), Sections 1.7, 1.8, 7.1

## Summary

The original leakage audit rated `vol_12b`, `vol_48b`, `vol_median`, and the derived `regime` label as safe. This was incorrect. All four are affected by a transitive one-bar forward leak, traced to their dependency on `ret_1b`, which is itself a forward return by design.

## Root cause

`ret_1b[T] = close[T+1] / close[T] - 1` (intentional forward return, used as a label elsewhere in the pipeline).

`vol_12b`/`vol_48b` were computed as a rolling standard deviation of `ret_1b`. Since `ret_1b[T]` already contains `close[T+1]`, any rolling window over it inherits that one-bar-ahead information, even though the window's row-index alignment is backward-looking.

## Why the original audit misclassified this

The audit checked that the rolling window's **row index** (`[T-11, T]`) never reaches into future rows, and concluded the feature was safe on that basis. It did not check whether the **values inside the window** were themselves already computed using future data. `ret_1b` is a forward return; a rolling statistic of a forward return is not automatically backward-looking just because its row-index window is. The audit applied the correct rule for raw `ret_*` columns (excluded from feature sets) but did not extend that rule to statistics derived from them.

## Measured impact

Diagnostic run against dataset 1.5.1 (core variant, 145,043 rows):

| Metric | Value |
|---|---|
| Correlation, old vs. corrected `vol_12b` | 0.437 |
| Regime label agreement (old vs. corrected) | 60.86% |
| Regime labels flipped | 56,775 / 145,043 (39.1%) |
| Flips involving HVTF or LVTF | 52,821 (93% of all flips) |

This is a material effect, not a rounding-level discrepancy.

## Features changed

| Feature | Before | After | Status |
|---|---|---|---|
| `vol_12b` | Rolling std of `ret_1b` (forward return) | Rolling std of `entry_close.pct_change(1)` (pure backward return) | Fixed |
| `vol_48b` | Rolling std of `ret_1b` | Rolling std of `entry_close.pct_change(1)` | Fixed |
| `vol_median` | Expanding median of leaky `vol_12b` | Expanding median of corrected `vol_12b` | Fixed (inherits fix) |
| `regime` (HVTF/LVTF/HVR/LVR) | Derived from leaky `vol_12b` | Derived from corrected `vol_12b` | Fixed (inherits fix) |
| `target_cls` | `ret_48b > 0.1 * vol_48b` (leaky threshold) | `ret_48b > 0.1 * vol_48b` (corrected threshold) | Fixed (inherits fix) |
| `trend_12b`, `trend_48b` | `entry_close.pct_change(h)` | Unchanged | Not affected — never depended on `vol_*` |
| Sentiment/streak/persistence features | Built from `crowd_side`/`abs_sentiment` | Unchanged | Not affected — independent of volatility |

Column names are unchanged. The fix replaces the computation, it does not add parallel `_v2` columns.

## Not covered by this fix

Two previously identified findings remain open and are unrelated to this issue:

- **L-02**: `acceleration_bucket` quantiles computed on the full dataset rather than fold-local.
- **L-03**: `trend_strength_bucket_{12b,48b}` via `pd.qcut()` on the full dataset rather than fold-local.

## Consequence

Any model trained on dataset versions ≤ 1.5.1 used `regime` and `target_cls` values that were partially derived from future price information. These datasets and any models trained on them should be treated as contaminated for regime-conditioned or `target_cls`-based work. A new dataset build (1.6.0 or later, versioning pending) using the corrected `build_dataset_vol.py` and `build_dataset.py` is required before continuing regime-dependent analysis.
