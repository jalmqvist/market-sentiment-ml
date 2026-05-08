# DL Signal Schema — `dl_signals_h1_v1`

This document defines the stable schema contract for the hourly DL inference
signal artifact produced by `scripts/build_dl_signal_artifact.py`.

Schema version: **`dl_signals_h1_v1`**

Related documents:
- `docs/SENTIMENT_FEATURE_SCHEMA.md` — hourly sentiment feature contract
- `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` — per-build metadata

---

## Purpose

`dl_signals_h1_v1` is a **multi-dimensional signal cube**: one row per
`(pair, entry_time, model, dl_regime, target_horizon, feature_set)`.

It consolidates DL inference outputs from potentially multiple models,
regimes, and horizons into a single versioned Parquet artifact for
downstream consumption by `market-phase-ml`.

This artifact is designed to be:
- **Stable**: schema changes always bump the version suffix.
- **Non-breaking**: `market-phase-ml` selects a specific *surface* (slice)
  via a config dict; it does not depend on having exactly one set of rows.
- **Causally safe**: `entry_time` represents the H1 bar open; all features
  used to generate predictions are backward-looking relative to `entry_time`.

---

## Grain

One row per `(pair, entry_time, model, dl_regime, target_horizon, feature_set)`.

Multiple "surfaces" (regime × horizon × model combinations) may share the
same `(pair, entry_time)`.  The consumer selects a single surface by
filtering on the provenance columns.

---

## Time semantics

### `entry_time`

- The **open time of the H1 bar** (UTC, tz-naive).
- The DL model's prediction covers the price move from `entry_time`
  over the next `target_horizon` bars.
- No forward-looking features are permitted.

---

## Signal semantics

### `signal_strength`

```
signal_strength = 2 * pred_prob_up - 1
```

This maps P(up) ∈ [0, 1] linearly to signal_strength ∈ [−1, +1]:

| pred_prob_up | signal_strength | Interpretation |
|---|---|---|
| 1.0 | +1.0 | Maximum bullish pressure |
| 0.75 | +0.5 | Moderate bullish pressure |
| 0.5 | 0.0 | Neutral |
| 0.25 | −0.5 | Moderate bearish pressure |
| 0.0 | −1.0 | Maximum bearish pressure |

**signal_strength is NOT standardised.**  Standardisation would destroy the
calibration meaning, comparability across runs, and downstream
interpretability.  Treat it like a probability or confidence value.

---

## DL regime taxonomy (producer-side)

The `dl_regime` column uses the **producer taxonomy** from
`market-sentiment-ml`:

| Label | Meaning |
|---|---|
| `HVTF` | High-volatility trend-following regime |
| `LVTF` | Low-volatility trend-following regime |
| `HVR` | High-volatility ranging regime |
| `LVR` | Low-volatility ranging regime |

The **consumer** (`market-phase-ml`) may optionally map these to its own
internal labels via `mpml_regime_equiv`:

| Producer (`dl_regime`) | Consumer equiv (`mpml_regime_equiv`) |
|---|---|
| `HVTF` | `HVTF` |
| `LVTF` | `LVTF` |
| `HVR` | `HVMR` |
| `LVR` | `LVMR` |

The `dl_regime` column is **never modified** in this artifact.

---

## Contract columns

### Keys / unique key

| Column | Type | Description |
|---|---|---|
| `pair` | string | Normalised FX pair, e.g. `eur-usd` (lowercase `xxx-yyy`) |
| `entry_time` | datetime (UTC) | H1 bar open timestamp (tz-naive UTC) |

### Core signal

| Column | Type | Nullable | Description |
|---|---|---|---|
| `pred_prob_up` | float64 | No | P(price moves up over `target_horizon` bars) ∈ [0, 1] |
| `signal_strength` | float64 | No | `2 * pred_prob_up − 1` ∈ [−1, 1] |

### Optional signal

| Column | Type | Nullable | Description |
|---|---|---|---|
| `pred_direction` | Int64 | Yes | `+1` (up) or `−1` (down); derived from `pred_prob_up > 0.5` if not supplied |
| `confidence` | float64 | Yes | Caller-supplied reliability estimate ∈ [0, 1]; null if not supplied |

### Provenance

| Column | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"unknown"` | Model architecture / identifier, e.g. `"MLP"`, `"LSTM"` |
| `dl_regime` | string | `"unknown"` | Producer-side regime label; see taxonomy table above |
| `target_horizon` | string | `"unknown"` | Prediction horizon in bars, e.g. `"24"` |
| `feature_set` | string | `"unknown"` | Feature set used, e.g. `"price_vol_sentiment"` |
| `dataset_version` | string | `"unknown"` | Dataset version string, e.g. `"1.1.0"` |
| `model_version` | string | `"unknown"` | Model version / run identifier |

### Schema

| Column | Type | Description |
|---|---|---|
| `schema_version` | string | Constant `dl_signals_h1_v1` |

---

## Unique key

```
(pair, entry_time, model, dl_regime, target_horizon, feature_set)
```

This key is enforced as a hard invariant.  Duplicate rows on this key will
cause the build script to abort with an error.

---

## QA invariants

The build script enforces the following checks:

1. **Uniqueness**: The unique key `(pair, entry_time, model, dl_regime,
   target_horizon, feature_set)` is unique.
2. **Non-null entry_time**: All rows have a valid `entry_time`.
3. **pred_prob_up range**: All values are in [0, 1].
4. **signal_strength range**: All values are in [−1, 1].
5. **Per-pair monotonic**: Within each pair (and surface), `entry_time` is
   monotonically non-decreasing.
6. **dl_regime taxonomy**: Values are checked against
   `{HVTF, LVTF, HVR, LVR}`; non-standard values produce a warning.

---

## Output artifacts

| Path | Description |
|---|---|
| `data/output/dl_signals/dl_signals_h1_v1.parquet` | Signal table (Parquet) |
| `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` | Build manifest |

### Manifest fields

| Field | Description |
|---|---|
| `schema_version` | Artifact schema version string |
| `generated_at_utc` | ISO 8601 timestamp of build |
| `export_frequency` | `"H1"` |
| `signal_definition` | Human-readable signal semantics |
| `unique_key` | Unique key column list and description |
| `dl_regime_taxonomy` | Valid producer regime labels |
| `time_semantics` | entry_time definition and as-of rule |
| `source_predictions` | Input directory path |
| `git_commit` | Git SHA of the build |
| `total_rows` | Total row count |
| `total_pairs` | Number of distinct pairs |
| `overall_entry_time_min/max` | Time range of the artifact |
| `dl_regimes_present` | Regime labels found in the data |
| `models_present` | Model identifiers found in the data |
| `feature_sets_present` | Feature set identifiers found |
| `target_horizons_present` | Target horizons found |
| `pair_stats` | Per-pair row count and time range |
| `signal_stats` | Aggregate statistics (mean, std of signal_strength, etc.) |

---

## Input CSV format

The build script consolidates per-pair/per-run prediction CSVs from an input
directory.  Each CSV must conform to the following:

### Required columns

| Column | Type | Description |
|---|---|---|
| `entry_time` | datetime | UTC timestamp; tz-naive or UTC-aware |
| `pair` | string | FX pair in any common format (normalised to `xxx-yyy`) |
| `pred_prob_up` | float | P(up) ∈ [0, 1] |

### Optional columns

| Column | Type | Description |
|---|---|---|
| `model` | string | Model identifier; defaults to `"unknown"` |
| `dl_regime` | string | Producer regime label; defaults to `"unknown"` |
| `target_horizon` | int/string | Horizon in bars; defaults to `"unknown"` |
| `feature_set` | string | Feature set name; defaults to `"unknown"` |
| `dataset_version` | string | Dataset version; defaults to `"unknown"` |
| `model_version` | string | Model run ID; defaults to `"unknown"` |
| `pred_direction` | int | `+1` or `−1`; derived from `pred_prob_up > 0.5` if absent |
| `confidence` | float | Reliability estimate ∈ [0, 1]; null if absent |

### Minimal example CSV

```csv
entry_time,pair,pred_prob_up,model,dl_regime,target_horizon,feature_set,dataset_version,model_version
2023-01-02 00:00:00,eur-usd,0.62,MLP,LVTF,24,price_vol_sentiment,1.1.0,v1.0
2023-01-02 01:00:00,eur-usd,0.48,MLP,LVTF,24,price_vol_sentiment,1.1.0,v1.0
2023-01-02 00:00:00,usd-jpy,0.55,MLP,HVR,24,price_vol_sentiment,1.1.0,v1.0
```

---

## Building the artifact

```bash
python scripts/build_dl_signal_artifact.py \
    --input-dir data/output/dl_predictions \
    [--output-dir data/output/dl_signals]
```

### Prerequisites

- `data/output/dl_predictions/*.csv` — per-pair/per-run prediction CSVs
  with at minimum: `entry_time`, `pair`, `pred_prob_up`

### Generating prediction CSVs from the DL pipeline

The existing DL training scripts (`research/deep_learning/train.py`,
`research/deep_learning/train_lstm.py`) should be extended to write
row-level predictions to `data/output/dl_predictions/` with the columns
described above.  Until that is done, prediction CSVs can be created
manually or by any script that produces the required columns.

---

## Integration with market-phase-ml

`market-phase-ml` consumes this artifact via its `src/dl_surface_loader.py`
module.  The consumer selects a specific signal **surface** (a slice of the
multi-dimensional signal cube) by specifying:

```python
surface = {
    "model": "MLP",
    "target_horizon": "24",
    "feature_set": "price_vol_sentiment",
    "dl_regime": "LVTF",   # optional; omit to select all regimes
}
```

The consumer renames `entry_time` → `timestamp` internally and optionally
adds a `mpml_regime_equiv` column mapping the producer regime labels to the
consumer's own taxonomy.

See `market-phase-ml/docs/DL_SIGNAL_INTEGRATION.md` for the consumer-side
documentation.
