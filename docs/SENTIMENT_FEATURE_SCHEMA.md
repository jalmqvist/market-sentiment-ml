# Sentiment Feature Schema — `sentiment_features_h1_v1`

This document defines the stable schema contract for the hourly sentiment
feature artifact produced by `build_sentiment_feature_contract.py`.

Schema version: **`sentiment_features_h1_v1`**

Related documents:
- `OUTPUT_SCHEMA.md` — event-level research dataset contract
- `data/output/features/SENTIMENT_FEATURE_MANIFEST_h1_v1.json` — per-build
  data ranges and metadata

---

## Purpose

`sentiment_features_h1_v1` is an hourly, forward-filled state table that
surfaces the *current sentiment state* at every H1 bar open, keyed by
`(pair, entry_time)`.  It is designed for direct consumption by downstream
ML and backtesting workflows without requiring any joins back to the raw
snapshot data.

This contract is also designed to support integration with external regime
datasets (e.g. market-phase-ml), where daily or lower-frequency regime
labels can be joined to the hourly grid using `(pair, entry_time → date)` alignment.

---

## Grain

One row per `(pair, entry_time)` where `entry_time` is the H1 bar open
timestamp (UTC) taken directly from the FX price files.

---

## Time semantics

### `entry_time`

- The **open time of the H1 bar** (start of the hour, UTC).
- Derived directly from the FX price file timestamps.
  **All timestamps** present in the price files are used; no manual
  filtering, rounding, or invention of hours is performed.
- This is the primary time key for the contract.

### `snapshot_time`

- The **final corrected UTC timestamp** of the sentiment snapshot.
- Non-hour-aligned (reflects the original scrape time after the -1 h
  timezone correction applied in the research dataset).
- Nullable: null means no prior snapshot exists for this pair at or before
  this `entry_time`.

### Relationship between the two timestamps

```
snapshot_time  ≤  entry_time
```

This ordering is enforced as a hard contract invariant (see QA checks below).

See `OUTPUT_SCHEMA.md` § *Time semantics* for the full context on how
`snapshot_time` is derived from raw scrape files.

---

## As-of / forward-fill semantics

### Rule

At each `entry_time`, features reflect the **latest snapshot** with
`snapshot_time ≤ entry_time` for that pair.

This is equivalent to a *last-observation-carried-forward* (LOCF) join.

### Forward-fill per pair

Features are forward-filled **per pair** from the last observed snapshot.
Once a snapshot is observed at time *T*, its feature values are carried
forward to all subsequent `entry_time` values until a newer snapshot
arrives.

### No backward fill

If no prior snapshot exists for a pair at a given `entry_time`
(i.e., the grid row precedes the first snapshot for that pair), all feature
columns are **null** and `has_snapshot = false`.

Backward filling is explicitly prohibited.

---

## Contract columns

### Keys

| Column | Type | Description |
|---|---|---|
| `schema_version` | string | Constant `sentiment_features_h1_v1` |
| `pair` | string | Normalised FX pair, e.g. `eur-usd` |
| `entry_time` | datetime (UTC) | H1 bar open timestamp from price files |

### Provenance

| Column | Type | Nullable | Description |
|---|---|---|---|
| `snapshot_time` | datetime (UTC) | Yes | Latest `snapshot_time ≤ entry_time`; null if none |
| `snapshot_age_hours` | float | Yes | `(entry_time − snapshot_time)` in hours; null if no snapshot |
| `snapshot_age_hours_int` | int | Yes | `floor(snapshot_age_hours)`; convenience column; null if no snapshot |
| `has_snapshot` | bool | No | `(snapshot_time is not null) AND (snapshot_time ≤ entry_time)` |
| `is_stale` | bool | No | `has_snapshot AND snapshot_age_hours > stale_hours_threshold` |

`stale_hours_threshold` defaults to **24** and is recorded in the manifest.

### Core sentiment

| Column | Type | Nullable | Description |
|---|---|---|---|
| `net_sentiment` | float | Yes | Signed crowd positioning; positive = crowd net long |
| `abs_sentiment` | float | Yes | `abs(net_sentiment)` |
| `crowd_side` | int `{−1, 0, 1}` | Yes | `+1` crowd long, `−1` crowd short, `0` neutral |

### Persistence

| Column | Type | Nullable | Description |
|---|---|---|---|
| `extreme_70` | bool | Yes | `abs_sentiment >= 70` |
| `extreme_streak_70` | int | Yes | Consecutive `extreme_70` streak count within pair |
| `side_streak` | int | Yes | Consecutive same-side `crowd_side` streak count within pair |

### Dynamics

| Column | Type | Nullable | Description |
|---|---|---|---|
| `sentiment_change` | float | Yes | Event-based change: `net_sentiment(k) − net_sentiment(k−1)` between consecutive snapshots per pair.  Forward-filled to the hourly grid.  Null for the first snapshot per pair (no prior observation). |

`sentiment_change` is **event-based** (computed on snapshot events, not on
the hourly grid).  This avoids artefacts that would arise from differencing
a forward-filled series.

### Structure

| Column | Type | Description |
|---|---|---|
| `pair_group` | string | `JPY_cross` if pair ends with `-jpy`; otherwise `non_JPY` |
| `pair_group` | string | `JPY_cross` if pair ends with `-jpy`; otherwise `non_JPY`. Used in behavioral regime definitions. |

---

## `has_snapshot` and `is_stale` — exact definitions

```
has_snapshot = (snapshot_time IS NOT NULL) AND (snapshot_time <= entry_time)

is_stale     = has_snapshot AND (snapshot_age_hours > stale_hours_threshold)
             where stale_hours_threshold = 24   [default; see manifest]
```

Rows where `has_snapshot = false` have all feature columns set to null.
Rows where `is_stale = true` have valid feature values but the values are
potentially outdated; downstream models should decide whether to mask or
weight these rows.

---

## QA invariants

The build script enforces the following checks and will raise an error if
any are violated:

1. **Uniqueness**: `(pair, entry_time)` is unique.
2. **Causality**: For all rows where `has_snapshot = true`,
   `snapshot_time ≤ entry_time`.
3. **Null consistency**: Feature columns are null iff `has_snapshot = false`.

A staleness summary is printed per pair during the build.

---

## Output artifacts

| Path | Description |
|---|---|
| `data/output/features/sentiment_features_h1_v1.parquet` | Feature table (Parquet) |
| `data/output/features/SENTIMENT_FEATURE_MANIFEST_h1_v1.json` | Build manifest |

### Manifest fields

The manifest (`SENTIMENT_FEATURE_MANIFEST_h1_v1.json`) records:

- `schema_version`
- `generated_at_utc`
- `source_dataset` — path and schema version of the input research dataset
- `grid` — description of how the hourly grid is constructed from price files
- `semantics` — as-of rule, fill rule, and null behaviour for first snapshot
- `stale_hours_threshold`
- `git_commit`
- `total_rows`, `total_pairs`
- `overall_entry_time_min`, `overall_entry_time_max`
- `pair_stats` — per-pair row counts and `entry_time` min/max

---

## Building the artifact

```bash
python build_sentiment_feature_contract.py
```

Prerequisites:
- `data/input/fx/*.csv` — hourly FX price files (MT4 format)
- `data/output/master_research_dataset.csv` — full research dataset
  (produced by `python build_fx_sentiment_dataset.py`)

---

## Consistency with `OUTPUT_SCHEMA.md`

| Concept | `OUTPUT_SCHEMA.md` | This contract |
|---|---|---|
| `snapshot_time` | Corrected UTC timestamp of sentiment observation | Same |
| `entry_time` | First valid H1 bar at or after `snapshot_time` (event-level) | H1 bar open; derived from ALL price-file timestamps |
| Causality guarantee | `snapshot_time ≤ entry_time` | Same (enforced by QA) |
| `crowd_side` | `+1` long, `−1` short, `0` neutral | Same |
| `sentiment_change` | Change vs previous snapshot within pair | Same (event-based) |

The key difference is grain: `OUTPUT_SCHEMA.md` defines an **event-level**
table (one row per sentiment snapshot), whereas this contract defines an
**hourly state table** (one row per H1 bar per pair) via forward-filling.
