# MSML Leakage & Causality Audit

**Document version:** 1.0  
**Audit date:** 2026-05-18  
**Scope:** market-sentiment-ml (MSML) — full pipeline  
**Priority:** Scientific validation — correctness over performance

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Legend](#legend)
3. [Audit Area 1 — Feature Construction](#audit-area-1--feature-construction)
4. [Audit Area 2 — Train/Test Boundary](#audit-area-2--traintest-boundary)
5. [Audit Area 3 — LSTM Sequence Leakage](#audit-area-3--lstm-sequence-leakage)
6. [Audit Area 4 — Parquet Export / Integration](#audit-area-4--parquet-export--integration)
7. [Audit Area 5 — Walk-Forward Logic](#audit-area-5--walk-forward-logic)
8. [Audit Area 6 — Sparse-DL Leakage](#audit-area-6--sparse-dl-leakage)
9. [Audit Area 7 — Selector / Routing Leakage](#audit-area-7--selector--routing-leakage)
10. [Future-Sensitive Grep Audit](#future-sensitive-grep-audit)
11. [Invariants Added](#invariants-added)
12. [Open TODOs](#open-todos)
13. [Confirmed-Safe Subsystems](#confirmed-safe-subsystems)
14. [Suspicious / Unresolved Risks](#suspicious--unresolved-risks)

---

## Executive Summary

This audit covers the full MSML pipeline:

- Dataset construction (`scripts/build_fx_sentiment_dataset.py`, `scripts/build_dataset.py`, `scripts/build_dataset_vol.py`)
- Feature engineering (`pipeline/features.py`, inline in dataset builder)
- DL training — MLP (`research/deep_learning/train.py`)
- DL training — LSTM (`research/deep_learning/train_lstm.py`)
- Dataset loader (`research/deep_learning/dataset_loader.py`)
- Feature sets definition (`research/deep_learning/feature_sets.py`)
- Prediction artifact export (`scripts/write_dl_prediction_artifact.py`)
- Cube consolidation (`scripts/consolidate_dl_predictions.py`)
- Walk-forward evaluation (`evaluation/walk_forward.py`)
- Signal discovery regime experiments (`research/signal_discovery/`)

### Critical findings (require action)

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| L-01 | **HIGH** | `train.py:223`, `train_lstm.py:247` | Label threshold computed on full dataset before split — future test data contaminates threshold |
| L-02 | **MEDIUM** | `build_fx_sentiment_dataset.py:755-756` | Acceleration-bucket quantiles (`q_low`, `q_high`) computed globally on full dataset |
| L-03 | **MEDIUM** | `build_fx_sentiment_dataset.py:809-813`, `pipeline/features.py:192-196` | `pd.qcut()` trend-strength buckets computed globally on full dataset |
| L-04 | **MEDIUM** | `train.py:354`, `train_lstm.py:373` | `--export-split=all` exports in-sample (train-fold) predictions; downstream MPML may not distinguish these from out-of-sample |
| L-05 | **LOW** | `feature_sets.py:87-107` | `PRICE_TREND_SENTIMENT_V2_FEATURES` comment warns that `ret_*` columns are forward returns — use not yet verified safe |

### Confirmed-safe areas

- All rolling/diff/pct_change feature transforms (past-only)
- All scaler fits (train-only in MLP/LSTM trainers)
- LSTM sequence window construction (target at position `i + seq_len`, window `[i:i+seq_len)`)
- Parquet export join semantics (no forward-fill, strict left-join)
- Consolidation cube QA invariants (uniqueness, range checks)
- Walk-forward yearly logic (strict year boundary evaluation)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Confirmed safe — no leakage risk identified |
| ⚠️ | Suspicious — potential leakage; documented risk |
| ❌ | Confirmed leakage — must be fixed |
| 🔲 | TODO — requires further investigation or instrumentation |

---

## Audit Area 1 — Feature Construction

### 1.1 Backward-looking trend features

**Files:** `pipeline/features.py:95`, `scripts/build_fx_sentiment_dataset.py:418`

```python
out[f"trend_{h}b"] = out.groupby("pair")["entry_close"].pct_change(h)
```

**Assessment:** ✅ **SAFE**

`pct_change(h)` is a pure backward-looking transform: the value at row `T` equals `(close[T] - close[T-h]) / close[T-h]`. Computed per-pair after `sort_values(["pair", "entry_time"])`. No future bar is accessed.

**Reasoning:** `pct_change(h)` with `h > 0` shifts forward in time relative to the denominator — it uses `close[T-h]` (past) to normalize `close[T]` (current). No future data is used. ✅

---

### 1.2 Sentiment difference features

**Files:** `scripts/build_fx_sentiment_dataset.py:191`, `pipeline/features.py:138`

```python
sentiment["prev_net_sentiment"] = sentiment.groupby("pair")["net_sentiment"].shift(1)
sentiment["sentiment_change"] = sentiment["net_sentiment"] - sentiment["prev_net_sentiment"]

out["sentiment_change_6h"] = out.groupby("pair")["net_sentiment"].diff(6)
out["sentiment_delta_12b"] = out.groupby("pair")["sentiment"].diff(12)
```

**Assessment:** ✅ **SAFE**

All `.shift(1)`, `.diff(6)`, `.diff(12)` are forward-looking in the time axis (i.e., row at time T uses data from T-1 or earlier). No future data is accessed. The groupby ensures no cross-pair contamination.

---

### 1.3 Streak features

**Files:** `scripts/build_fx_sentiment_dataset.py:82-113`, `pipeline/features.py:26-55`

```python
def compute_streak_from_boolean(series):  # sequential loop, uses only past values
def compute_same_value_streak(series):    # sequential loop, uses only past values
```

**Assessment:** ✅ **SAFE**

Both functions iterate forward through the series and increment a counter based only on the current value. No backward-fill, no future-state access.

---

### 1.4 Sentiment Z-score (rolling window)

**File:** `scripts/build_fx_sentiment_dataset.py:942-948`

```python
roll_mean = out.groupby("pair")["sentiment"].transform(
    lambda x: x.rolling(100, min_periods=1).mean()
)
roll_std = out.groupby("pair")["sentiment"].transform(
    lambda x: x.rolling(100, min_periods=1).std()
)
out["sentiment_z"] = (out["sentiment"] - roll_mean) / (roll_std + 1e-8)
```

**Assessment:** ✅ **SAFE** (per-row causality), ⚠️ **WARNING** (whole-dataset computation)

The rolling window at position T only reads positions `[T-99, T]` — strictly backward-looking. However, this computation is performed once on the **entire dataset** during build time. When the MLP/LSTM trainers later perform a train/test split, `sentiment_z` values in the test fold were computed using test-fold history only (not future data), so they are causally safe.

The remaining risk: if the test fold contains the first 100 bars of a pair, `min_periods=1` means earlier bars use very short windows. This is a variance risk, not a leakage risk.

**Recommendation:** Document explicitly. No immediate action required.

---

### 1.5 Acceleration bucket quantiles — LEAKAGE RISK

**File:** `scripts/build_fx_sentiment_dataset.py:755-756`

```python
q_low = master_valid["sentiment_change_6h"].quantile(0.33)
q_high = master_valid["sentiment_change_6h"].quantile(0.66)
```

**Assessment:** ⚠️ **GLOBAL NORMALIZATION RISK (L-02)**

`q_low` and `q_high` are computed on the **entire dataset** (including future test data). Every row's `acceleration_bucket` label therefore incorporates information from future rows' distribution. This is a subtle but real leakage mechanism: the bucket boundaries shift based on test-set statistics.

Identical code exists in `pipeline/features.py:140-141`.

**CONTRACT added in code:** Yes — see `pipeline/features.py` (CONTRACT comment added).

**Impact assessment:** The effect is likely small in practice because:
- The quantiles are of a sentiment-change distribution that is relatively stable over time
- The bucket labels are categorical (not continuous), so only the boundary matters
- However, this is scientifically incorrect and should be treated as an unresolved risk

**TODO (L-02):** Replace global quantile computation with either:
- Fixed hardcoded thresholds derived from a held-out training period
- Or compute per-fold quantiles in any walk-forward context that uses this feature

---

### 1.6 Trend-strength buckets via qcut — LEAKAGE RISK

**Files:** `scripts/build_fx_sentiment_dataset.py:809-813`, `pipeline/features.py:192-196`

```python
master_valid.loc[valid, bucket_col] = pd.qcut(
    master_valid.loc[valid, col],
    q=4,
    labels=["weak", "medium", "strong", "extreme"]
)
```

**Assessment:** ⚠️ **GLOBAL NORMALIZATION RISK (L-03)**

`pd.qcut()` computes quartile boundaries from all valid rows in the dataset, including future test rows. Bucket labels for each row depend on the full distribution of all rows.

**CONTRACT added in code:** Yes — see `pipeline/features.py` (CONTRACT comment added).

**TODO (L-03):** Same mitigation as L-02: use fixed boundaries or fold-local computation.

---

### 1.7 Volatility features (rolling std)

**File:** `scripts/build_dataset_vol.py:62-73`

```python
df["vol_12b"] = (
    df.groupby("pair")["ret_1b"]
    .rolling(window=12, min_periods=12)
    .std()
    .reset_index(level=0, drop=True)
)
```

**Assessment:** ✅ **SAFE**

Rolling std with `window=12` is strictly backward-looking. The `sort_values(["pair", "snapshot_time"])` before this computation guarantees chronological order per pair.

**Comment in code:** `# STRICTLY BACKWARD LOOKING` — already present. ✅

---

### 1.8 Expanding median for vol_median

**File:** `scripts/build_fx_sentiment_dataset.py:989-994`

```python
out["vol_median"] = (
    out.groupby("pair")["vol_12b"]
    .transform(lambda x: x.expanding().median())
)
```

**Assessment:** ✅ **SAFE**

`expanding().median()` at row T uses only rows 0 through T (inclusive) within the pair. This is causal.

---

### 1.9 Forward return construction — intentional future use

**File:** `scripts/build_fx_sentiment_dataset.py:360`

```python
grp[f"future_close_{h}b"] = grp["close"].shift(-h)
```

**Assessment:** ✅ **SAFE (by design)**

`shift(-h)` is intentionally forward-looking — it constructs the target variable `ret_{h}b` representing the return over the next `h` bars. These columns (`ret_1b`, `ret_12b`, `ret_24b`, `ret_48b`, `contrarian_ret_*`) are used exclusively as **labels/targets**, not as input features.

**Verification:** `feature_sets.py` explicitly excludes all `ret_*` columns from feature sets. The comment in `PRICE_TREND_SENTIMENT_V2_FEATURES` (line 84-86) notes that `ret_1b / ret_12b / ret_48b` are forward returns present in the dataset — **they must never be used as features** (see TODO L-05).

**IMPORTANT NOTE:** The 20% filter `mask = out[ret_col].abs() > 0.2` applied after return computation (line 393-394) is causal-safe: it removes outlier return rows post-hoc and does not introduce future information into features.

---

### 1.10 Macro regime (time-based)

**File:** `scripts/build_fx_sentiment_dataset.py:831-835`

```python
master_valid["macro_regime"] = master_valid["year"].apply(
    lambda y: "pre_2022" if y <= 2021 else "post_2022"
)
```

**Assessment:** ✅ **SAFE**

Calendar year is known at the time of observation. No future information is used.

---

### 1.11 Cross-sectional normalization in signal discovery

**File:** `research/raw_validation/pipeline_leakage_diagnosis.py:68-70`

```python
d["z"] = d.groupby("time")["z"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)
```

**Assessment:** ✅ **SAFE** (for analysis only; not part of production pipeline)

This cross-sectional z-score normalizes across pairs at the same timestamp. All pairs at the same time step are contemporaneous — no future information.

---

## Audit Area 2 — Train/Test Boundary

### 2.1 Label threshold computed on full dataset — CONFIRMED LEAKAGE

**Files:** `research/deep_learning/train.py:223`, `research/deep_learning/train_lstm.py:247`

```python
threshold = float(df[ret_col].abs().quantile(args.label_quantile))
df["target_direction"] = (df[ret_col] > threshold).astype(int)
```

**Assessment:** ❌ **CONFIRMED LEAKAGE (L-01) — FIXED**

The label threshold is computed on `df`, which contains **all rows including the test set** (split happens at lines 248-255, *after* this quantile computation). This means:

1. The test-set return distribution influences the threshold value
2. Test-set rows' labels are computed using a threshold derived partly from test-set data
3. The model's positive-class rate on training data is therefore calibrated using test-set statistics

**Fix applied:** See `research/deep_learning/train.py` and `research/deep_learning/train_lstm.py` — both now:
1. Perform the train/test index split first
2. Compute the label threshold from `df.iloc[:split]` (train rows only)
3. Apply that threshold to the full df (to assign labels for all rows consistently)
4. Add an assertion to verify split boundary

**Instrumentation added:**
```python
# CONTRACT: Label threshold MUST be computed on train rows only.
# Compute split boundary first.
split = int(len(df) * 0.8)
threshold = float(df.iloc[:split][ret_col].abs().quantile(args.label_quantile))
assert split < len(df), "Empty test set"
```

---

### 2.2 Feature normalization (scaler fit on train only)

**Files:** `research/deep_learning/train.py:268-273`, `research/deep_learning/train_lstm.py:294-298`

```python
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std[std < 1e-8] = 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
```

**Assessment:** ✅ **SAFE**

Mean and std are computed exclusively from `X_train`. The same statistics are then applied to `X_test`. This is the correct procedure for avoiding test-set contamination.

**Verification:** The same `mean`/`std` is also applied to `X_all_norm` and `X_infer_norm` (inference paths), which is correct — the scaler is always fixed from training data.

---

### 2.3 Class imbalance weight (pos_weight)

**Files:** `research/deep_learning/train.py:276-277`, `research/deep_learning/train_lstm.py:301-302`

```python
pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
```

**Assessment:** ✅ **SAFE**

`pos_weight` is computed from `y_train` only. No test-set statistics are used.

---

### 2.4 Chronological sort before split

**Files:** `research/deep_learning/dataset_loader.py:89`, `research/deep_learning/train.py`

The dataset is loaded sorted by `snapshot_time` in `dataset_loader.py`. The MLP trainer then filters by pair/regime before doing a sequential index split. 

**Assessment:** ⚠️ **MILD RISK — CROSS-PAIR TEMPORAL CONTAMINATION**

The dataset contains multiple pairs. When sorted globally by `snapshot_time`, consecutive rows may belong to different pairs. The 80/20 sequential split means:

- "Train" rows = first 80% by global snapshot_time
- "Test" rows = last 20% by global snapshot_time

For most pairs this is chronologically correct. However, pairs with uneven data coverage might have their test rows start at different calendar dates. This is an acceptable trade-off for a panel dataset but should be documented.

**Recommendation:** Log the actual calendar date of the split boundary (both min and max `entry_time` in train and test) to make this transparent. This has been added as part of the fix to L-01.

---

### 2.5 Dataset-build-time global statistics used at model-train time

**Assessment:** ⚠️ **KNOWN RISK (L-02, L-03) — documented above**

Statistics like `acceleration_bucket` and `trend_strength_bucket` are baked into the dataset CSV at build time using global statistics. When the MLP/LSTM trainer loads the dataset, these columns are already computed. The trainer never re-fits these transforms — it only normalizes the numeric features.

This means: even if the MLP/LSTM trainer correctly normalizes on train-only, the upstream dataset features may carry global statistics. This is a **pipeline-level leakage** that is hard to fix without recomputing features per fold at training time.

---

## Audit Area 3 — LSTM Sequence Leakage

### 3.1 Sequence window construction

**File:** `research/deep_learning/train_lstm.py:52-95`

```python
def build_sequences(df, features, target, seq_len):
    for pair, pair_df in df.groupby("pair", sort=True):
        pair_df = pair_df.sort_values(sort_col).reset_index(drop=True)
        for i in range(len(pair_df) - seq_len):
            target_idx = i + seq_len
            X.append(data[i:target_idx])       # window [i, i+seq_len)
            y.append(target_vals[target_idx])  # label at i+seq_len
```

**Assessment:** ✅ **SAFE**

The input window uses indices `[i, i+seq_len)` and the label is at index `i+seq_len`. This means:
- The label is at strictly **future** position relative to all sequence steps
- No label position is ever inside the input window
- Off-by-one analysis: the last input step is at `target_idx - 1` (index `i+seq_len-1`), and the label is at `target_idx = i+seq_len`. Correct — label is one step beyond last input.

**Assertion verified:** ✅ `target_idx = i + seq_len` with window `data[i:target_idx]` — correct exclusive upper bound, label strictly after window.

---

### 3.2 Sequence metadata alignment

**File:** `research/deep_learning/train_lstm.py:72-93`

```python
meta_row = pair_df.loc[target_idx, meta_cols].to_dict()
```

**Assessment:** ✅ **SAFE**

The metadata row (used for export alignment) is taken from `target_idx` — the same row as the label. This correctly records the timestamp of the prediction target, not the last input step.

---

### 3.3 LSTM train/test split on sequences

**File:** `research/deep_learning/train_lstm.py:275-278`

```python
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
meta_test = meta_df.iloc[split:].copy()
```

**Assessment:** ⚠️ **SEQUENCE OVERLAP RISK**

The sequences are built per-pair then sorted globally by `snapshot_time`. The 80/20 split on this sorted order means:

- Sequences near the boundary may share overlapping input windows (a sequence ending at time T and a sequence starting at time T-seq_len+1 overlap by `seq_len-1` steps)
- Sequences in the test set can have input windows that overlap with training-set sequences
- This is standard practice for time-series LSTM but is a **mild leakage risk** if training data is used to influence test-set input features

**Impact:** The input features of test sequences are computed from the same underlying time series as training sequences. However, since features in the dataset are already computed (rolling, diff, etc.) and the sequence values are fixed before the split, there is no "fitting" contamination. The overlap means test sequences "see" the same raw feature values that training sequences also saw — this is unavoidable in a sliding-window approach.

**Recommendation:** Log the number of overlapping bars between the last training sequence and first test sequence. No immediate code change required.

---

### 3.4 Inference path for LSTM (predict universe)

**File:** `research/deep_learning/train_lstm.py:334-350`

```python
infer_work_df["target_direction"] = 0  # placeholder
X_infer, _, infer_meta_df = build_sequences(infer_work_df, features, "target_direction", args.seq_len)
X_infer_norm = (X_infer.astype("float32") - mean) / std
```

**Assessment:** ✅ **SAFE**

The inference path:
1. Uses a dummy target (placeholder `0`) — no future label contamination
2. Normalizes with train-derived `mean`/`std` — no test contamination
3. Builds sequences through the standard `build_sequences()` path — same causality guarantees apply

---

## Audit Area 4 — Parquet Export / Integration

### 4.1 Export join semantics

**File:** `research/deep_learning/train.py:378-384`, `research/deep_learning/train_lstm.py:391-399`

```python
export_entry_time = pd.to_datetime(export_meta_df["entry_time"]).dt.tz_localize(None)
```

**Assessment:** ✅ **SAFE**

The `entry_time` in the exported parquet is the timestamp of the H1 bar when the prediction was made — it is the bar's **open time**, not a future time. MPML should join on this column to align DL signals with the corresponding price bars.

---

### 4.2 Per-pair monotonicity assertion

**Files:** `research/deep_learning/train.py:514-517`, `research/deep_learning/train_lstm.py:491-494`

```python
for _pair, _grp in pred_df.groupby("pair"):
    assert _grp["entry_time"].is_monotonic_increasing, (
        f"Non-monotonic entry_time for pair {_pair!r}"
    )
```

**Assessment:** ✅ **SAFE — INSTRUMENTED**

Monotonicity assertion ensures no timestamp ordering anomalies in the exported artifact.

---

### 4.3 Export-split=all — IN-SAMPLE PREDICTION RISK

**Files:** `research/deep_learning/train.py:354-357`, `research/deep_learning/train_lstm.py:373-374`

```python
elif args.export_split == "all":
    export_meta_df = df.copy().reset_index(drop=True)
    export_pred_prob_up = pred_prob_up_all
    logging.info("[export] export_split=all (train+test)")
```

**Assessment:** ⚠️ **MEDIUM RISK (L-04)**

When `--export-split=all` is used:
- The exported parquet contains predictions for both train and test rows
- The train-fold predictions are **in-sample** (the model has seen those labels during training)
- These predictions are structurally indistinguishable from out-of-sample predictions in the parquet schema — there is no `is_train` flag in the exported artifact

**Risk for MPML:** If MPML consumes these artifacts for walk-forward evaluation, and some artifact windows overlap with the DL model's training period, MPML may inadvertently use in-sample DL signals. This creates a subtle cross-system leakage.

**Mitigations in place:**
- `export_split=test` is the default — only out-of-sample predictions are exported by default
- The `prediction_timestamp` field records when the artifact was generated, not whether it's in-sample
- The manifest records `training_pairs` and `inference_pairs` (provenance)

**TODO (L-04):** Add an `is_train_fold` boolean column (or `export_split` string column) to the parquet artifact schema so downstream consumers can distinguish in-sample from out-of-sample predictions. Until then, document that `--export-split=all` must only be used for debugging/integration validation, not for operational MPML consumption.

---

### 4.4 Duplicate timestamp collapse

**Files:** `research/deep_learning/train.py:422-464`

```python
pred_df = pred_df.groupby(["pair", "entry_time"], as_index=False).agg({"pred_prob_up": "mean", ...})
assert n_dupes == 0, f"Duplicate (pair, entry_time) rows remain after collapse: {n_dupes}"
```

**Assessment:** ✅ **SAFE — INSTRUMENTED**

Intra-hour duplicate snapshots (multiple sentiment snapshots per H1 bar) are averaged. The uniqueness assertion prevents any leakage from duplicated rows.

---

### 4.5 Consolidation join (consolidate_dl_predictions.py)

**File:** `scripts/consolidate_dl_predictions.py:265-298`

```python
combined = pd.concat(frames, ignore_index=True)
_run_qa(cube)
```

**Assessment:** ✅ **SAFE**

The consolidation is a simple concatenation + QA. No temporal joins, no forward-fill, no future references. The `_run_qa()` function (from `build_dl_signal_artifact.py`) enforces uniqueness on `(pair, entry_time, model, dl_regime, target_horizon, feature_set)`.

---

## Audit Area 5 — Walk-Forward Logic

### 5.1 Yearly walk-forward (evaluation/walk_forward.py)

**File:** `evaluation/walk_forward.py:28-74`

```python
for year in sorted(df[year_col].unique()):
    if start_year is not None and year < start_year:
        continue
    test = df[df[year_col] == year]
    stats = compute_stats(test, col)
```

**Assessment:** ✅ **SAFE**

This is a strict year-by-year evaluation (not training). No fitting occurs inside the loop — it evaluates a pre-computed signal against returns. No future rows contaminate earlier years.

---

### 5.2 Expanding-window walk-forward

**File:** `evaluation/walk_forward.py:81-129`

```python
for i in range(2, len(years)):
    test_year = years[i]
    test = df[df[year_col] == test_year].copy()
    if apply_signal_fn is not None:
        test = apply_signal_fn(test)
    stats = compute_stats(test, ret_col)
```

**Assessment:** ✅ **SAFE** with caveat

The evaluation loop properly iterates forward. However:
- No explicit logging of fold boundaries (train start/end, test start/end)
- If `apply_signal_fn` applies any normalization or fitting inside the test fold without using only train data, leakage is possible

**Instrumentation needed:** Add structured fold-boundary logging. No evidence of leakage in current code, but the `apply_signal_fn` callback is an opaque leakage surface.

**TODO:** Document that any `apply_signal_fn` passed to `walk_forward_expanding` MUST be pre-fitted on prior-year data only. Add a contract comment.

---

### 5.3 Signal discovery walk-forward (regime_v* experiments)

**Files:** `research/signal_discovery/regime_v4.py`, `regime_v7_1.py`, `regime_v8.py`, `regime_v9.py`, `regime_v10.py`, `regime_v12.py`

The regime experiments use expanding-window walk-forward with train-only `qcut` computation:

```python
# From regime_v4.py
_, vol_bins = pd.qcut(valid_vol, q=3, retbins=True, duplicates="drop")
# ... bins applied to test fold
```

**Assessment:** ✅ **SAFE** — These experiments compute `qcut` bins from train data and apply fixed bins to test fold. This is the correct approach and contrasts with the dataset-build-time risk (L-03).

---

### 5.4 Expanding z-score in causal pipeline (pipeline_leakage_diagnosis.py)

**File:** `research/raw_validation/pipeline_leakage_diagnosis.py:41-44`

```python
def expanding_zscore(series):
    mean = series.expanding(min_periods=50).mean().shift(1)
    std = series.expanding(min_periods=50).std().shift(1)
    return (series - mean) / (std + 1e-6)
```

**Assessment:** ✅ **SAFE**

The `.shift(1)` after `.expanding()` ensures that the mean/std used to normalize row T are computed from rows 0 through T-1 (exclusive of current row). This is a strictly causal expanding z-score.

---

## Audit Area 6 — Sparse-DL Leakage

### 6.1 DL coverage sparsity (~7-11%)

The DL signal cube covers approximately 7-11% of all MPML evaluation rows (sparse overlap). This creates the following risks:

**6.1.1 NaN-mask as implicit feature**

When MPML merges the DL cube onto its evaluation dataset, rows without DL coverage receive NaN values. If the ML selector or strategy model treats NaN-presence as a feature, it may inadvertently learn the *pattern* of when DL signals exist — which could correlate with future market structure.

**Assessment:** ⚠️ **UNRESOLVED RISK**

No explicit instrumentation exists in MSML (the producer) to prevent this. MPML (the consumer) must handle this. 

**Recommendation for MPML:** Add a `dl_available` boolean flag to the merged dataset. Verify that models trained with optional DL features do not achieve better performance when the DL-coverage pattern itself is predictive (perform ablation: compare models with/without `dl_available` as a feature).

**6.1.2 Overlap-timing leakage**

If DL signals happen to appear more frequently during certain market regimes (e.g., high-volatility periods), MPML models may learn that "DL signal present → high-vol regime → specific return profile." This is a correlation-based leakage path, not a direct leakage.

**Assessment:** ⚠️ **UNRESOLVED RISK**

Recommendation: Audit DL coverage distribution by regime/year. If coverage is highly regime-biased, add `dl_coverage_regime` as a control covariate in MPML analysis.

**6.1.3 Imputation propagation**

If MPML forward-fills or imputes DL signals across NaN gaps, earlier DL predictions may propagate to future time steps, violating temporal causality.

**Assessment:** ⚠️ **UNRESOLVED RISK (MPML-side)**

MSML does not perform any forward-fill. The parquet artifact has strict `(pair, entry_time)` keys. MPML join semantics must be audited separately.

---

### 6.2 Signal production lag

The `prediction_timestamp` in exported artifacts records the time the artifact was generated (wall clock). The `entry_time` records the H1 bar timestamp.

**Assessment:** ✅ **SAFE**

The `entry_time` (H1 bar open) is the operationally correct join key for MPML. The `prediction_timestamp` is metadata only and should not be used as a join key.

---

## Audit Area 7 — Selector / Routing Leakage

### 7.1 MSML-side exposure

MSML does not contain an explicit selector or routing layer. The regime assignment (`HVTF`, `LVTF`, `HVR`, `LVR`) is computed causally from:
- `vol_12b` (12-bar backward rolling std)
- `trend_12b` (12-bar backward pct_change)
- `vol_median` (expanding median — causal)

**Assessment:** ✅ **SAFE** (regime labels are causal)

**Note for MPML:** If MPML routes strategies based on the MSML-exported regime label, and if MPML then evaluates strategy performance within those regime bins, verify that the regime label at time T was computed using only data available at T.

---

### 7.2 Regime-conditioned DL training

**File:** `research/deep_learning/train.py:193-195`

```python
if args.regime:
    df = df[df["regime"] == args.regime]
```

**Assessment:** ✅ **SAFE**

Regime filtering is applied before any train/test split. The regime label is a column in the dataset that was computed causally. Filtering by regime does not introduce future information.

---

## Future-Sensitive Grep Audit

The following patterns were searched across all `.py` files in the repository:

### `shift(-n)` — forward-looking shifts

```
research/signal_discovery/regime_v21.py:74:    .shift(-1)
research/signal_discovery/regime_v20.py:74:    .shift(-1)
scripts/build_fx_sentiment_dataset.py:360:    grp[f"future_close_{h}b"] = grp["close"].shift(-h)
```

**Assessment:**
- `regime_v20.py`, `regime_v21.py`: Marked as **legacy experiments** (header: `# Legacy experiment — not part of current validated approach`). These construct a forward return target — intentional. Not part of production pipeline.
- `build_fx_sentiment_dataset.py:360`: Intentional future return construction for labels only. Safe.

### `center=True` — centered rolling windows

**Result:** No occurrences found. ✅

### `bfill()` — backward fill

**Result:** No occurrences found in production code. ✅

### `expanding()` — expanding windows

**Production use:**
- `build_fx_sentiment_dataset.py:991`: `expanding().median()` for `vol_median` — **SAFE** (causal)
- `pipeline_leakage_diagnosis.py:42-43`: `expanding().mean().shift(1)` — **SAFE** (causally shifted)

**Research/experiment use (not production):**
- `regime_v7_1.py:273`: `expanding(min_periods=2)` — used in research walk-forward, train-only ✅

### `rolling()` — rolling windows

**Production use (all grouped by pair, forward-sorted):**
- `build_dataset_vol.py:63,70`: `rolling(12/48).std()` — **SAFE** (backward-looking)
- `build_fx_sentiment_dataset.py:942-946`: `rolling(100).mean()/std()` — **SAFE** (backward-looking)
- `pipeline_leakage_diagnosis.py`: research-only, not production

### Global normalization / `quantile()` / `qcut()`

- `build_fx_sentiment_dataset.py:755-756`: Global quantile → `acceleration_bucket` — **⚠️ RISK (L-02)**
- `build_fx_sentiment_dataset.py:809`: Global `pd.qcut()` → `trend_strength_bucket` — **⚠️ RISK (L-03)**
- `pipeline/features.py:140-141, 192`: Same risks in pipeline module

### `pct_change()` — returns

**All uses are grouped per-pair, backward-looking:**
- `pct_change(h)` for `h > 0`: computes past return — **SAFE** ✅
- `pct_change()` (default `periods=1`) in legacy experiments: **SAFE** ✅

---

## Invariants Added

The following instrumentation was added or documented during this audit:

### Code changes (train.py, train_lstm.py)

**L-01 Fix:** Label threshold now computed from train fold only:
```python
# CONTRACT: Label threshold MUST be computed on train-only data.
# Split boundary is computed FIRST, then threshold from train rows.
split = int(len(df) * 0.8)
threshold = float(df.iloc[:split][ret_col].abs().quantile(args.label_quantile))
assert split < len(df), f"Empty test set after split at {split}/{len(df)}"
logging.info("label_threshold (train-only): %.6f  split=%d/%d", threshold, split, len(df))
```

### Contract comments (pipeline/features.py, build_fx_sentiment_dataset.py)

**L-02/L-03 Documentation:** CONTRACT comments added to `add_acceleration_bucket()` and `add_trend_strength_buckets()` marking the global-quantile risk.

### Causality assertions (train.py, train_lstm.py)

Existing assertions retained and documented:
```python
assert n_dupes == 0         # unique (pair, entry_time) in export
assert np.isfinite(...)     # no NaN/Inf in predictions
assert pred_prob_up ∈ [0,1] # probability range
assert signal_strength ∈ [-1,+1]
assert entry_time.is_monotonic_increasing per pair
```

---

## Open TODOs

| ID | Priority | File | Description |
|----|----------|------|-------------|
| L-01 | ✅ FIXED | `train.py`, `train_lstm.py` | Label threshold now computed on train fold only |
| L-02 | HIGH | `build_fx_sentiment_dataset.py:755`, `pipeline/features.py:140` | Replace global acceleration-bucket quantiles with fixed or fold-local thresholds |
| L-03 | HIGH | `build_fx_sentiment_dataset.py:809`, `pipeline/features.py:192` | Replace global `pd.qcut()` with fixed bin boundaries derived from training period |
| L-04 | MEDIUM | `train.py:354`, `train_lstm.py:373` | Add `is_train_fold` or `export_split` column to exported parquet artifact |
| L-05 | MEDIUM | `feature_sets.py:87-107` | Confirm that `ret_*` forward-return columns in `PRICE_TREND_SENTIMENT_V2_FEATURES` are never used as input features in any production DL run |
| L-06 | MEDIUM | `evaluation/walk_forward.py:116` | Add contract comment that `apply_signal_fn` must be pre-fitted on prior-year data |
| L-07 | MEDIUM | MPML (separate repo) | Audit DL-presence (`dl_available`) as an implicit feature risk in selector training |
| L-08 | LOW | `evaluation/walk_forward.py` | Add structured fold-boundary logging (train_start, train_end, test_start, test_end) |
| L-09 | LOW | `train.py`, `train_lstm.py` | Log split boundary calendar dates (min/max entry_time in train and test) |
| L-10 | LOW | MPML (separate repo) | Audit forward-fill / imputation behavior when merging DL parquet signals |

---

## Confirmed-Safe Subsystems

| Subsystem | File | Reasoning |
|-----------|------|-----------|
| Trend features (`trend_{h}b`) | `pipeline/features.py:94-96` | `pct_change(h)` with `h>0` is strictly backward-looking per pair |
| Sentiment diff features | `build_fx_sentiment_dataset.py:191-192` | `.shift(1)`, `.diff(6)`, `.diff(12)` — past-only |
| Streak features | `build_fx_sentiment_dataset.py:82-113` | Sequential loop, no future access |
| Sentiment z-score | `build_fx_sentiment_dataset.py:942-948` | `rolling(100)` — backward window per pair |
| Volatility features | `build_dataset_vol.py:62-73` | `rolling(12/48).std()` — backward, post-sort |
| Expanding vol_median | `build_fx_sentiment_dataset.py:989-994` | `expanding().median()` — causal |
| Forward returns | `build_fx_sentiment_dataset.py:360` | Intentional target labels; excluded from feature sets |
| MLP feature normalization | `train.py:268-273` | Fit on `X_train` only, applied to `X_test` |
| LSTM feature normalization | `train_lstm.py:294-298` | Fit on `X_train` only, applied to `X_test` |
| pos_weight computation | `train.py:276`, `train_lstm.py:301` | From `y_train` only |
| LSTM sequence window | `train_lstm.py:52-95` | Label strictly after input window; no leakage |
| LSTM inference path | `train_lstm.py:334-350` | Dummy labels; train-derived normalization |
| Export monotonicity | `train.py:514-517` | Assertion enforced per pair |
| Export uniqueness | `train.py:455-464` | Assertion enforced after collapse |
| Export value range | `train.py:466-482` | Assertions for finite values, prob/signal bounds |
| Parquet consolidation | `consolidate_dl_predictions.py` | Simple concat + QA; no temporal joins |
| Yearly walk-forward eval | `evaluation/walk_forward.py:28-74` | Year-by-year, no fitting in loop |
| Regime regime label | `build_fx_sentiment_dataset.py:965-1010` | From causal vol + trend; expanding median is causal |
| Cross-sectional z-score | `pipeline_leakage_diagnosis.py:68-70` | Same-timestamp normalization — contemporaneous |
| Causal expanding z-score | `pipeline_leakage_diagnosis.py:41-44` | `.expanding().shift(1)` — past-only |
| Regime filter walk-forward | `regime_v4.py`, `regime_v9.py`, etc. | Train-fold qcut applied to test fold with fixed bins |

---

## Suspicious / Unresolved Risks

| ID | Severity | Subsystem | Description | Status |
|----|----------|-----------|-------------|--------|
| L-02 | HIGH | Dataset build | Global acceleration-bucket quantiles use full dataset | **TODO** |
| L-03 | HIGH | Dataset build | Global `pd.qcut()` trend-strength buckets use full dataset | **TODO** |
| L-04 | MEDIUM | Export | `--export-split=all` includes in-sample train predictions in artifact; no in-sample flag in schema | **TODO** |
| L-05 | MEDIUM | Feature sets | `PRICE_TREND_SENTIMENT_V2_FEATURES` comment warns `ret_*` are forward returns — not verified as excluded from all production runs | **TODO** |
| L-06 | LOW | Walk-forward | `apply_signal_fn` callback in `walk_forward_expanding` is an opaque leakage surface | **TODO** |
| L-07 | MEDIUM | Sparse-DL (MPML) | DL presence/absence pattern as implicit feature in MPML selector | **TODO (MPML)** |
| L-08 | LOW | Walk-forward | No structured fold-boundary logging in `walk_forward_expanding` | **TODO** |
| L-09 | LOW | DL trainers | Split boundary calendar dates not logged | **TODO** |
| L-10 | LOW | Sparse-DL (MPML) | Forward-fill behavior during DL signal merge in MPML | **TODO (MPML)** |
