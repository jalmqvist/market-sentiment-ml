# Leakage Fixes — Resolution Log

Canonical record of leakage findings from `leakage_audit.md` (2026-05-18)
that have been investigated and resolved. Each entry documents the finding,
what was measured, what changed, and why.

---

## L-11 — Volatility features transitively leak via ret_1b

**Status:** Fixed — dataset 1.6.0+
**See:** `docs/audit/leakage_audit_july_2026.md` for full writeup.

`vol_12b`/`vol_48b` were rolling std of `ret_1b`, a forward return. Fixed to
use rolling std of `entry_close.pct_change(1)` (pure backward return).
Measured impact: 39.1% of regime labels changed on dataset 1.5.1.

---

## L-02 — Acceleration bucket quantiles computed globally

**Status:** Resolved by removal — dataset 1.6.0+ (post-L-11 branch)
**Original severity:** MEDIUM
**Files affected:** `scripts/build_fx_sentiment_dataset.py`, `pipeline/features.py`

### Finding

`acceleration_bucket` was derived from `sentiment_change_6h` using quantile
boundaries (`quantile(0.33)`, `quantile(0.66)`) computed on the entire
dataset, including rows that would later fall in any test/holdout split.
Every row's bucket label was therefore influenced by the full-dataset
distribution, not just data available up to that row.

### Investigation

Repo-wide search confirmed `acceleration_bucket`:
- Does not appear in `research/deep_learning/feature_sets.py` (not a DL
  model input)
- Does not appear anywhere in MPML
- Does not appear in any active `evaluation/` or `research/analysis/`
  script (two references exist in `research/analysis/analyze_regime_signal_interaction.py`
  and `analyze_trend_behavior.py`, both confirmed as old local analysis
  scripts, not part of any current or frozen result)
- `pipeline/features.py` contains an identical implementation but is not
  imported anywhere in the active build or inference path (confirmed via
  repo-wide `import pipeline` grep — only self-reference in its own
  docstring)

No frozen or validated result (including `VALIDATION_SPEC_JPY.md`) sliced
by, filtered by, or trained on this column.

### Resolution

Removed from `scripts/build_fx_sentiment_dataset.py`. The underlying
`sentiment_change_6h` column (a causal `diff(6)`) is retained — only the
global-quantile discretization was removed. `add_acceleration_bucket()`
was also removed from `pipeline/features.py` (orphaned code, not
imported).

There was no active consumer at time of removal. This is a correctness
fix with zero measured downstream impact.

---

## L-03 — Trend-strength buckets via global pd.qcut()

**Status:** Resolved by removal — dataset 1.6.0+ (post-L-11 branch)
**Original severity:** MEDIUM
**Files affected:** `scripts/build_fx_sentiment_dataset.py`, `pipeline/features.py`

### Finding

`trend_strength_bucket_12b` and `trend_strength_bucket_48b` were derived
from `trend_strength_12b`/`48b` (= `|trend_12b|`/`|trend_48b|`, themselves
strictly causal) using `pd.qcut(q=4)` computed on the entire dataset.
Quartile boundaries therefore incorporated the full-dataset return
distribution, including future rows.

### Investigation

Same repo-wide search as L-02. `trend_strength_bucket_12b` appears only in
`research/analysis/analyze_trend_behavior.py` and
`analyze_regime_signal_interaction.py` (both confirmed old, non-production,
not part of any frozen result). Not present in `feature_sets.py`, MPML, or
any active evaluation script. `pipeline/features.py`'s
`add_trend_strength_buckets()` is likewise unused (orphaned).

### Resolution

Removed from `scripts/build_fx_sentiment_dataset.py`. The underlying
`trend_strength_12b`/`48b` columns (computed in `add_trend_features()`,
strictly causal `pct_change(h).abs()`) are retained and unaffected —
only the global-quartile discretization was removed. Removed from
`pipeline/features.py` as orphaned code.

No active consumer at time of removal.

---

## Clarification — bucket columns that were never affected

Two other columns share the word "bucket" with L-02/L-03 but use **fixed,
hardcoded thresholds**, not data-derived quantiles. They were never part
of L-02 or L-03 and require no fix. Recorded here explicitly to prevent
future confusion from the naming overlap.

| Column                        | Boundary definition                                          | Data-dependent?          |
| ----------------------------- | ------------------------------------------------------------ | ------------------------ |
| `saturation_bucket`           | Fixed cutoffs on `abs_sentiment`: `<60` normal, `<75` elevated, `<85` extreme, else panic | No — hardcoded constants |
| `crowd_persistence_bucket_70` | Fixed integer cutoffs on `extreme_streak_70`: `0` none, `≤2` low, `≤5` medium, `>5` high | No — hardcoded constants |

Both remain in the dataset builder unchanged. Their boundaries do not
depend on the sample they're computed over, so there is no train/test
contamination path, adding more rows to the dataset cannot shift where
an existing row's bucket label falls.

**Rule of thumb for any future "bucket" column:** if the boundary is
computed with `.quantile()`, `pd.qcut()`, `.median()` over the *current*
column's own values, check whether that computation spans the full
dataset (leak) or is fold-local/expanding (safe). If the boundary is a
fixed number written directly in the code, it is not data-dependent and
cannot leak by construction.

---

## Summary table

| ID   | Column(s)                                                  | Root cause                                          | Resolution                            | Status          |
| ---- | ---------------------------------------------------------- | --------------------------------------------------- | ------------------------------------- | --------------- |
| L-11 | `vol_12b`, `vol_48b`, `vol_median`, `regime`, `target_cls` | Rolling stat of a forward-shifted column (`ret_1b`) | Recomputed from pure backward returns | Fixed, 1.6.0+   |
| L-02 | `acceleration_bucket`                                      | Global `quantile()` boundary                        | Removed (zero active consumers)       | Removed, 1.6.0+ |
| L-03 | `trend_strength_bucket_12b/48b`                            | Global `pd.qcut()` boundary                         | Removed (zero active consumers)       | Removed, 1.6.0+ |
| —    | `saturation_bucket`                                        | N/A — fixed thresholds                              | No action needed                      | Not affected    |
| —    | `crowd_persistence_bucket_70`                              | N/A — fixed thresholds                              | No action needed                      | Not affected    |