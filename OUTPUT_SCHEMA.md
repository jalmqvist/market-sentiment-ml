# Output Schema

This document defines the stable output artifact contract for the FX retail sentiment research pipeline.

The purpose of this contract is to make downstream usage predictable across repositories, including integration with external research workflows such as market-phase-ml.

Note:
This schema describes the research dataset used for analysis.
Not all fields are included in downstream feature contracts.

## Schema version

Current schema version: **1.0**

## Canonical dataset

The canonical research dataset is:

```text
data/output/master_research_dataset_core.csv
```

Unless explicitly stated otherwise, downstream analysis should use the **core** dataset by default.

## Dataset variants

The pipeline produces three compatible dataset variants:

- `data/output/master_research_dataset_core.csv`
- `data/output/master_research_dataset_extended.csv`
- `data/output/master_research_dataset.csv`

### Intended meaning

- **core**: canonical dataset; stricter coverage filter
- **extended**: looser coverage filter; broader universe
- **full**: all valid matched events before universe filtering; used mainly for diagnostics and robustness checks

## Required invariants

### Column parity

All three dataset variants must have identical columns and column meanings.
They may differ in row count, but not in schema.

The `full`, `core`, and `extended` artifacts are required to have identical columns and column order.

### Subset relationship

Using `(pair, snapshot_time, entry_time)` as row identity for subset checks:

- `core ⊆ extended`
- `extended ⊆ full(valid-entry)`

### Core definition

The current core universe is defined by:

- `eligible_match_ratio >= 0.95`

The current extended universe is defined by:

- `eligible_match_ratio >= 0.90`

These thresholds must be recorded in `DATASET_MANIFEST.json`.

------

## Grain

The dataset grain is:

- one row per `(pair, snapshot_time)` sentiment event
- aligned to the first valid hourly entry bar at or after the event time

### Event key

The sentiment event key is:

- `(pair, snapshot_time)`

### Aligned execution key

The aligned market-bar key is:

- `(pair, entry_time)`

Note: multiple sentiment events may map to the same `entry_time` in the full dataset.

------

## Time semantics

### Snapshot timestamp
- `snapshot_time` represents the **final corrected UTC timestamp** used in the dataset.
- Sentiment snapshots are originally recorded in UTC+2 and shifted by -1 hour to align with FX price data (UTC+1).
- The stored `snapshot_time` is the **canonical reference time** used for all joins and downstream analysis.

------

### Entry timestamp (H1 bars)
- `entry_time` is the **open time of the H1 bar** (start of the hour, UTC).
- Sentiment is aligned using a forward-asof rule:
  - each snapshot is matched to the **first bar with time ≥ snapshot_time** within a fixed tolerance (see DATASET_MANIFEST.json).
- Interpretation:
  - sentiment is assumed to be available **at or before `entry_time`**
  - and informs decisions for that bar

------

### Execution alignment (important)
- All features are defined at `entry_time` (bar open).
- Forward returns are computed relative to `entry_close`.
- Backtesting systems may apply execution delays (e.g., enter at next bar close), but:
  - this must be handled in the backtester
  - the dataset itself remains **unshifted and time-consistent**

------

### Causality guarantee
The dataset enforces the following ordering:

```
snapshot_time ≤ entry_time < future return horizon timestamps
```

This ensures that:
- no future information is used in feature construction
- all forward returns are strictly out-of-sample relative to the signal

------

### Column naming clarity
- `snapshot_time`: UTC timestamp of sentiment observation
- `entry_time`: UTC timestamp of bar open used for evaluation

------

## Pair normalization

Pairs must be normalized to lowercase 3-3 format with `-` separator.

Examples:

- `eur-usd`
- `usd-jpy`
- `gbp-chf`

This normalization must be consistent across all output artifacts.

------

## Required columns

## 1. Identity and time

- `pair`
   string; normalized pair identifier such as `eur-usd`
- `snapshot_time`
   datetime; sentiment event timestamp after timezone correction
- `entry_time`
   datetime; first valid hourly bar at or after `snapshot_time`
- `source_file`
   string; source sentiment snapshot filename

## 2. Sentiment features

- `net_sentiment`
   float; signed crowd positioning, where positive means crowd long and negative means crowd short
- `abs_sentiment`
   float; absolute value of `net_sentiment`
- `crowd_side`
   integer; `+1` for crowd long, `-1` for crowd short, `0` neutral (rare)
- `sentiment_change`
   float; change in `net_sentiment` vs previous snapshot within pair
- `side_streak`
   integer; consecutive same-side crowd positioning count within pair
- `extreme_70`
   boolean; whether `abs_sentiment >= 70`
- `extreme_80`
   boolean; whether `abs_sentiment >= 80`
- `extreme_streak_70`
   integer; consecutive `extreme_70` streak length within pair
- `extreme_streak_80`
   integer; consecutive `extreme_80` streak length within pair

## 3. Entry-bar market context

- `entry_open`
- `entry_high`
- `entry_low`
- `entry_close`
   float; OHLC values of the aligned entry bar
- `entry_tick_volume`
   float or integer; tick activity measure from the aligned entry bar

## 4. Forward returns

> ⚠️ **Target columns — forward-looking, NEVER use as features**
>
> The columns below are computed from future price information.  They are
> strictly out-of-sample relative to the signal at ``entry_time`` and must
> **never** be used as predictive inputs in any model or analysis.
> Specifically, **`ret_*` and `contrarian_ret_*` columns are prohibited as
> features** in all downstream modelling code.

For each horizon `h` in:

```
[1, 2, 4, 6, 12, 24, 48]
```

the dataset must include:

- `ret_{h}b`
   float; forward return defined as:

  ```
  future_close_h / entry_close - 1
  ```

- `contrarian_ret_{h}b`
   float; contrarian return defined as:

  ```
  -sign(net_sentiment) * ret_{h}b
  ```

### Return semantics

Returns are defined on **trading bars ahead**, not wall-clock hours.

Example:

- `ret_12b` means return after 12 future hourly trading bars within that pair’s price series

### Contrarian sign convention

A positive `contrarian_ret_{h}b` means:

- fading the retail crowd would have been profitable

A negative `contrarian_ret_{h}b` means:

- the crowd was correct over that horizon

## 5. Quality and filtering helpers

- `eligible`
   boolean; row qualifies for coverage-based universe evaluation
- `within_price_window`
   boolean; `snapshot_time` falls within available price history window for that pair
- `is_weekday`
   boolean; event falls on a weekday according to current pipeline logic
- `price_start`
   datetime; first available price timestamp for the pair
- `price_end`
   datetime; last available price timestamp for the pair
- `price_bars`
   integer; number of hourly price bars available for the pair

------

## 6. Trend features (analysis-only)

The dataset includes trend-related columns derived from **past returns**.

These features are intended for **behavioral analysis only** and must not be used as predictive inputs without careful leakage consideration.

### Definition

For horizons:

```
[12, 48]
```

the dataset includes:

- `past_ret_{h}b`
  float; return over the previous `h` trading bars:

  ```
  entry_close / entry_close_shifted_h - 1
  ```

- `trend_dir_{h}b`
  float; sign of `past_ret_{h}b`:
  - `+1` → uptrend
  - `-1` → downtrend
  - `0` → neutral

- `trend_alignment_{h}b`
  float; alignment between crowd and trend:

  ```
  crowd_side * trend_dir_{h}b
  ```

  Interpretation:
  - `+1` → crowd aligned with trend (trend-following / late chasing)
  - `-1` → crowd fighting trend (early reversal behavior)

- `trend_strength_{h}b`
  float; absolute magnitude of past trend:

  ```
  abs(past_ret_{h}b)
  ```

### Important constraint

These features are computed using **past price information only** and are therefore causally safe for conditional analysis.

They are intentionally separated from forward returns to avoid leakage.

------

### 7. Trend strength features

- `trend_strength_12b` (float)  
  Absolute magnitude of past return over 12 bars, used as a proxy for trend intensity.

- `trend_strength_48b` (float)  
  Absolute magnitude of past return over 48 bars.

- `trend_strength_bucket_12b` (string)  
  Discretized trend strength for 12-bar horizon.  
  Values: `weak`, `medium`, `strong`, `extreme` (quantile-based).

- `trend_strength_bucket_48b` (string)  
  Discretized trend strength for 48-bar horizon.  
  Values: `weak`, `medium`, `strong`, `extreme`.

  Notes:
- Buckets are computed cross-sectionally using quantiles.
- Used for regime-conditioned behavioral analysis (not part of feature contract v1).

---

Note:

Price-derived regime features (trend, volatility proxies) have been
empirically tested and found not to produce robust predictive conditioning.

They are retained for analysis and comparison, but are not considered
primary candidates for signal gating in the current research direction.

Future work focuses on behavioral regime features derived from sentiment.

---

## Column classification for downstream modeling

This section clarifies which columns are safe to use as model inputs and which
are prohibited due to forward-looking content.

### (A) Target columns — forward-looking, NOT usable as features

The following columns are computed from future price information and must
**never** be used as predictive inputs in any model or signal:

| Column pattern          | Reason                                      |
|-------------------------|---------------------------------------------|
| `ret_{h}b`              | Forward return — depends on future prices   |
| `contrarian_ret_{h}b`   | Derived from `ret_{h}b` — equally forbidden |

**Rule**: Any column whose name starts with `ret_` or matches
`contrarian_ret_*` is a leaking target column.  Downstream code must assert
that no such column appears in the feature list.

### (B) Causal feature columns — safe for modelling

The following columns are safe for use as model inputs because they depend
only on past information available at `entry_time`:

**Sentiment features** (available at `snapshot_time`, which precedes `entry_time`):

- `net_sentiment`
- `abs_sentiment`
- `sentiment_change`
- `side_streak`
- `extreme_streak_70`
- `extreme_streak_80`

**Trend features** (backward-looking past-price columns):

- `trend_strength_12b`
- `trend_strength_48b`
- `trend_dir_12b`
- `trend_dir_48b`

**Volatility** (rolling past-bar standard deviation):

- `vol_24b`

**Interaction features** (products of causal base columns):

- `abs_sent_x_trend12b`
- `abs_sent_x_trend48b`
- `abs_sent_x_vol24b`
- `extreme70_x_trend48b`

The canonical safe feature list is defined in `experiments/regime_v3.SAFE_FEATURES`.

### (C) Deprecated or ambiguous names

| Column                  | Status        | Notes                                           |
|-------------------------|---------------|-------------------------------------------------|
| `past_ret_{h}b`         | Causal        | Safe for analysis; not in primary feature set   |
| `trend_alignment_{h}b`  | Causal        | Safe for analysis; not in primary feature set   |
| `trend_strength_bucket_*` | Causal      | Discretized version; for regime diagnostics only|
| `vol_bucket`            | Causal        | Regime diagnostic only; not a model input       |
| `vol_regime`            | Causal        | Regime label; not a model input                 |
| `regime`                | Causal        | Combined regime label; not a model input        |

---

## 9. Regime-based signal weighting output artifacts

The following output artifacts are produced by the regime-weighted signal
pipeline in `experiments/regime_v3.py` and `run_regime_v3.py`.  They extend
the existing `regime_direction_performance` and `regime_direction_wf`
artifacts with continuous regime weights derived from training-only Sharpe.

### `regime_weighted_performance`

Aggregate performance of the **filter + direction + weighting** strategy over
the full dataset.  The Sharpe map is computed from all available data (not
leakage-free for the full-dataset view; use `regime_weighted_wf` for
leakage-free evaluation).

| Column     | Type  | Description                                                      |
|------------|-------|------------------------------------------------------------------|
| `n`        | int   | Number of active (non-zero) weighted-signal rows                 |
| `mean`     | float | Mean PnL: `mean(weighted_signal * ret_48b)` over active rows     |
| `std`      | float | Standard deviation of PnL over active rows                       |
| `sharpe`   | float | `mean / std` ratio (raw, not annualized)                         |
| `hit_rate` | float | Fraction of active rows with positive PnL                        |

### `regime_weighted_wf`

Per-year walk-forward performance of the **filter + direction + weighting**
strategy.  For each test year the regime Sharpe map is computed on **training
data only** (expanding window), making this artifact fully leakage-free.

| Column     | Type  | Description                                                      |
|------------|-------|------------------------------------------------------------------|
| `year`     | int   | Calendar test year                                               |
| `n`        | int   | Number of active weighted-signal rows in the test year           |
| `mean`     | float | Mean PnL in the test year                                        |
| `sharpe`   | float | Sharpe ratio in the test year                                    |
| `hit_rate` | float | Hit rate in the test year                                        |

### Comparison table (strategy ladder)

The four strategies are logged in order of increasing sophistication:

| Strategy                          | Logged section header                          |
|-----------------------------------|------------------------------------------------|
| Baseline (global contrarian)      | `FULL DATASET PERFORMANCE`                     |
| Filter only                       | `FILTERED PERFORMANCE`                         |
| Filter + direction                | `FILTER + DIRECTION (FINAL)`                   |
| Filter + direction + weighting    | `FILTER + DIRECTION + WEIGHTING (FINAL)`       |

Walk-forward (OOS) versions are logged under:

| Walk-forward strategy             | Logged section header                                   |
|-----------------------------------|---------------------------------------------------------|
| Filter only                       | `WALK-FORWARD FILTERED PERFORMANCE`                     |
| Filter + direction                | `WALK-FORWARD FILTER + DIRECTION`                       |
| Filter + direction + weighting    | `WALK-FORWARD FILTER + DIRECTION + WEIGHTING`           |

### Configuration

| Parameter           | Default | CLI flag              | Description                                              |
|---------------------|---------|-----------------------|----------------------------------------------------------|
| `WEIGHT_THRESHOLD`  | `0.05`  | `--weight-threshold`  | Minimum `abs(weight)` for a signal to be active          |
| `NORMALIZE_WEIGHTS` | `False` | `--normalize-weights` | Normalize weights by `max_abs_sharpe` instead of clipping|

---

## 8. Analysis filtering convention

All downstream analysis scripts are expected to apply the following return filter:


-0.1 < contrarian_ret_{h}b < 0.1


This removes extreme outliers caused by data issues or illiquid price jumps and ensures comparability across analyses.

---

## Missingness rules

### Entry alignment

If no valid `entry_time` can be assigned, then:

- `entry_time` is `NA`
- entry-bar OHLC fields are `NA`
- forward returns are `NA`

Rows without valid entry bars may be removed in filtered outputs.

### Forward returns

If the required future price bar does not exist for a given horizon, then:

- `ret_{h}b` is `NA`
- `contrarian_ret_{h}b` is `NA`

This is expected near the end of the available price series.

------

## Merge contract

The current merge contract is:

- merge type: forward alignment
- by pair
- align to first hourly bar at or after `snapshot_time`
- exact matches allowed
- tolerance: `90min`

These settings must be recorded in `DATASET_MANIFEST.json`.

------

## Timezone alignment contract

The current timezone alignment assumption is:

- sentiment timestamps correspond to `UTC+2`
- price timestamps correspond to `UTC+1`
- therefore `snapshot_time` is shifted by `-1h` before merging

This assumption must be recorded in `DATASET_MANIFEST.json`.

------

## Compatibility and evolution

### Immutability rule

After schema version `1.0` is published:

- existing columns must not be renamed
- existing column meanings must not change
- new columns may be added only in a backward-compatible way

### Backward compatibility rule

Downstream consumers should be able to rely on all required columns documented here remaining present and semantically stable across `1.x` versions.
