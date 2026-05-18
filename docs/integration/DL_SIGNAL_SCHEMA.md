# DL Signal Schema — `dl_signals_h1_v1` / v2.0.0

This document defines the stable schema contract for the hourly DL inference
signal artifacts produced by `market-sentiment-ml`.

Current schema version: **`2.0.0`** (see `schemas/dl_artifact_schema.DL_SCHEMA_VERSION`)

Previous schema version: `dl_signals_h1_v1` (backward-compat constants kept)

Related documents:
- `docs/integration/dl_artifact_contract.md` — formal contract specification (v2)
- `docs/integration/dataset_semantics.md`   — dataset & export semantics
- `docs/SENTIMENT_FEATURE_SCHEMA.md`        — hourly sentiment feature contract
- `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` — per-build metadata

---

## Schema v2 Notes

Schema v2.0.0 adds **explicit timestamp semantics** to resolve a prior
ambiguity where MSML artifact-generation timestamps were misread by MPML as
causal prediction-availability timestamps.

### New parquet columns (v2)

| Column | Type | Meaning | Causal? |
|---|---|---|---|
| `prediction_available_timestamp` | datetime (UTC) | Earliest historical timestamp the prediction could have been known; **≤ entry_time** | ✓ (MPML causal boundary) |
| `prediction_generated_timestamp` | datetime | Wall-clock inference time | ✗ (diagnostics only) |
| `artifact_created_timestamp` | datetime | Wall-clock export time; same for all rows | ✗ (provenance only) |

### New manifest fields (v2)

- `schema_version` = `"2.0.0"` (was `"dl_signals_h1_v1"`)
- `artifact_created_timestamp` promoted to top-level and to
  `artifact_metadata.artifact_created_timestamp`
- `artifact_metadata.timestamp_semantics` block added for explicit
  per-column documentation

### Centralized constants (v2)

All column names and schema version are centralized in
`schemas/dl_artifact_schema.py`. Import from there; do not hardcode strings.

### Fail-fast validation (v2)

`write_dl_prediction_artifact()` now calls `validate_dl_artifact()` before
writing.  Contract violations raise `ValueError` immediately.

---

## Architecture Overview (v1)

The DL signal pipeline uses a **two-step** architecture:

```
DL training/inference
        │
        ▼
scripts/write_dl_prediction_artifact.py
        │
        ├── data/output/dl_predictions/{run_id}.parquet      (time-series payload)
        └── data/output/dl_predictions/{run_id}.manifest.json (identity + provenance)

        │
        ▼ (after all runs)
scripts/consolidate_dl_predictions.py
        │
        ├── data/output/dl_signals/dl_signals_h1_v1.parquet  (operational cube)
        └── data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json
```

### Why two steps?

- **Per-run artifacts** are lightweight, reproducible, and written immediately
  after each DL training / inference run — no consolidation needed at training
  time.
- **Consolidation** is a separate, auditable step: it reads all per-run
  artifacts, injects identity columns, enforces the cube uniqueness contract,
  and writes the final operational cube consumed by `market-phase-ml`.

---

## A) Per-run prediction artifacts

Written by `scripts/write_dl_prediction_artifact.py`.

### Per-run Parquet schema (v2)

One row per `(pair, entry_time)`.  Identity columns are **not** stored here —
they live in the companion manifest.

**Core columns (unchanged from v1)**

| Column | Type | Nullable | Description |
|---|---|---|---|
| `pair` | string | No | Normalised FX pair, e.g. `eur-usd` (lowercase `xxx-yyy`) |
| `entry_time` | datetime (UTC) | No | H1 bar open timestamp (tz-naive UTC) |
| `pred_prob_up` | float64 | No | P(price moves up over `target_horizon` bars) ∈ [0, 1] |
| `signal_strength` | float64 | No | `2 * pred_prob_up − 1` ∈ [−1, 1] |
| `pred_direction` | Int64 | No | Tri-state: `+1` (>0.5), `−1` (<0.5), `0` (==0.5) |
| `confidence` | float64 | Yes | Caller-supplied reliability estimate ∈ [0, 1]; null if absent |
| `prediction_timestamp` | datetime (UTC) | Yes | v1 legacy: per-row inference timestamp; null if absent |

**New timestamp columns (v2 — explicit semantics)**

| Column | Type | Nullable | v2? | Description |
|---|---|---|---|---|
| `prediction_available_timestamp` | datetime (UTC) | No | **NEW** | Causal boundary; earliest historical time prediction could be known; **must be ≤ entry_time** |
| `prediction_generated_timestamp` | datetime (UTC) | Yes | **NEW** | Wall-clock inference time (diagnostics only; must NOT be used for causality) |
| `artifact_created_timestamp` | datetime (UTC) | No | **NEW** | Wall-clock export time; same for all rows in artifact (provenance only) |

### Per-run manifest schema (identity + provenance)

The manifest is a JSON file with the **same stem** as the parquet.

#### Required blocks

```json
{
  "schema_version": "2.0.0",
  "artifact_created_timestamp": "2024-01-15T10:30:00+00:00",
  "export_frequency": "H1",
  "generated_at_utc": "2024-01-15T10:30:00+00:00",
  "run_id": "MLP__LVTF__24__price_vol_sentiment__20240115T103000Z",
  "parquet_file": "MLP__LVTF__24__price_vol_sentiment__20240115T103000Z.parquet",
  "signal_definition": {
    "formula": "signal_strength = 2 * pred_prob_up - 1",
    "range": "[-1, +1]",
    "semantics": "positive = behavioral upside pressure; negative = downside pressure"
  },
  "identity": {
    "model": "MLP",
    "dl_regime": "LVTF",
    "target_horizon": 24,
    "feature_set": "price_vol_sentiment"
  },
  "artifact_metadata": {
    "artifact_created_timestamp": "2024-01-15T10:30:00+00:00",
    "export_timestamp": "2024-01-15T10:30:00+00:00",
    "prediction_horizon_hours": 24,
    "feature_surface": "price_vol_sentiment",
    "dl_regime": "LVTF",
    "availability_semantics": "sparse_observed_only",
    "timestamp_semantics": {
      "entry_time": "H1 bar open timestamp (UTC tz-naive); the bar being predicted",
      "prediction_available_timestamp": "earliest historical timestamp the prediction could have been observed; used by MPML for causality checks; must be <= entry_time",
      "prediction_generated_timestamp": "wall-clock inference time (diagnostics only); MUST NOT be used for causality",
      "artifact_created_timestamp": "wall-clock artifact export time (provenance only); MUST NOT be used for causality"
    }
  },
  "provenance": {
    "dataset_version": "1.1.0",
    "model_version": "v1.0",
    "training_run_id": "run_20240115_abc",
    "training_pairs": ["eur-usd", "gbp-usd"],
    "inference_pairs": ["eur-usd", "gbp-usd"]
  },
  "calibration": {
    "method": "none",
    "notes": "raw model probability; no post-hoc calibration applied"
  },
  "train_period": {
    "start": "2018-01-01",
    "end": "2022-12-31"
  },
  "git_commit": "abc123",
  "row_count": 8760,
  "pairs": ["eur-usd", "usd-jpy"],
  "entry_time_min": "2023-01-02T00:00:00",
  "entry_time_max": "2023-12-31T23:00:00",
  "causal_assumptions": [
    "prediction_available_timestamp is the causal boundary for MPML; it is set to entry_time (bar open) by default.",
    "prediction_generated_timestamp is wall-clock only; MUST NOT be used for causality.",
    "artifact_created_timestamp is wall-clock only; MUST NOT be used for causality."
  ],
  "warnings": [],
  "missing_provenance_counts": {
    "dataset_version": 0,
    "model_version": 0,
    "training_run_id": 0
  }
}
```

#### `identity` block (required, operational)

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model architecture / identifier, e.g. `"MLP"`, `"LSTM"` |
| `dl_regime` | string | Yes | Producer-side regime label; see taxonomy table below |
| `target_horizon` | integer | Yes | Prediction horizon in bars (numeric) |
| `feature_set` | string | Yes | Feature set identifier, e.g. `"price_vol_sentiment"` |

#### `calibration` block (required placeholder)

| Field | Type | Description |
|---|---|---|
| `method` | string | `"none"` until calibration is applied |
| `notes` | string | Human-readable calibration notes |

#### `train_period` block (required placeholder)

| Field | Type | Description |
|---|---|---|
| `start` | string or null | Training period start date (ISO 8601 or null) |
| `end` | string or null | Training period end date (ISO 8601 or null) |

#### Optional cross-family provenance fields

For runs that train and infer on different pair families, producer manifests may
include:

- `provenance.training_pairs`
- `provenance.inference_pairs`

These are manifest-only provenance fields for reproducibility/debugging. They do
not add or change parquet row columns.

---

## B) Consolidated operational cube

Written by `scripts/consolidate_dl_predictions.py`.

### Grain

One row per `(pair, entry_time, model, dl_regime, target_horizon, feature_set)`.

Multiple "surfaces" (regime × horizon × model combinations) may share the
same `(pair, entry_time)`.

---

## Time semantics (v2)

### `entry_time`

- The **open time of the H1 bar** (UTC, tz-naive).
- The DL model's prediction covers the price move from `entry_time`
  over the next `target_horizon` bars.
- No forward-looking features are permitted.

### `prediction_available_timestamp` (v2, new)

- The **earliest simulated historical timestamp** the prediction could have
  been known.
- Used by MPML for causality checks.
- **Contract**: `prediction_available_timestamp <= entry_time`.
- Default value: `entry_time` (equality; the model uses only features
  available at or before bar open time).
- **Must NOT be a wall-clock `pd.Timestamp.now()` value**.

### `prediction_generated_timestamp` (v2, new)

- Wall-clock time the prediction was generated (diagnostics only).
- **Must NOT be used for causality checks**.
- Null is acceptable if not recorded.

### `artifact_created_timestamp` (v2, new)

- Wall-clock time the parquet artifact was exported.
- Same value for all rows within one artifact.
- **Must NOT be used for causality checks**.

### `prediction_timestamp` (v1 legacy)

- Kept for backward compatibility; same value as `prediction_generated_timestamp`.
- Historically ambiguous — use `prediction_available_timestamp` for all causal
  reasoning in MPML.

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

### `pred_direction` (tri-state)

| `pred_prob_up` | `pred_direction` | Interpretation |
|---|---|---|
| > 0.5 | +1 | Bullish |
| < 0.5 | −1 | Bearish |
| == 0.5 | 0 | Neutral (model is at the decision boundary) |

Type: `Int64` (nullable integer).

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

## Cube contract columns

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
| `pred_direction` | Int64 | No | Tri-state: `+1`, `−1`, `0`; see table above |
| `confidence` | float64 | Yes | Caller-supplied reliability estimate ∈ [0, 1]; null if not supplied |
| `prediction_timestamp` | datetime | Yes | Per-row inference timestamp (tz-naive UTC); null if not supplied |

### Provenance (injected from per-run manifests)

| Column | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model architecture / identifier, e.g. `"MLP"`, `"LSTM"` |
| `dl_regime` | string | — | Producer-side regime label; see taxonomy table above |
| `target_horizon` | Int64 | — | Prediction horizon in bars (numeric; not a string) |
| `feature_set` | string | — | Feature set used, e.g. `"price_vol_sentiment"` |
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

This key is enforced as a hard invariant in the consolidation step.  Duplicate
rows on this key will cause the consolidator to abort with an error.

---

## QA invariants

The consolidator enforces the following checks:

1. **Uniqueness**: The unique key is unique.
2. **Non-null entry_time**: All rows have a valid `entry_time`.
3. **pred_prob_up range**: All values are in [0, 1].
4. **signal_strength range**: All values are in [−1, 1].
5. **Per-surface monotonic**: Within each surface
   `(pair, model, dl_regime, target_horizon, feature_set)`, `entry_time` is
   monotonically non-decreasing.
6. **dl_regime taxonomy**: Values are checked against
   `{HVTF, LVTF, HVR, LVR}`; non-standard values produce a warning.

---

## Output artifacts

### Per-run (written by `write_dl_prediction_artifact.py`)

| Path | Description |
|---|---|
| `data/output/dl_predictions/{run_id}.parquet` | Time-series payload |
| `data/output/dl_predictions/{run_id}.manifest.json` | Identity + provenance |

### Consolidated cube (written by `consolidate_dl_predictions.py`)

| Path | Description |
|---|---|
| `data/output/dl_signals/dl_signals_h1_v1.parquet` | Signal cube (Parquet) |
| `data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json` | Cube build manifest |

---

## Usage

### Step 1: Write a per-run artifact after DL training/inference

```bash
python scripts/write_dl_prediction_artifact.py \
    --input-csv path/to/predictions.csv \
    --model MLP \
    --dl-regime LVTF \
    --target-horizon 24 \
    --feature-set price_vol_sentiment \
    --dataset-version 1.1.0 \
    --model-version v1.0 \
    --training-run-id run_20240115_abc \
    --train-period-start 2018-01-01 \
    --train-period-end 2022-12-31 \
    [--output-dir data/output/dl_predictions]
```

Or via Python API:

```python
from scripts.write_dl_prediction_artifact import write_dl_prediction_artifact

pq_path, mf_path = write_dl_prediction_artifact(
    df=predictions_df,
    identity={
        "model": "MLP",
        "dl_regime": "LVTF",
        "target_horizon": 24,
        "feature_set": "price_vol_sentiment",
    },
    provenance={
        "dataset_version": "1.1.0",
        "model_version": "v1.0",
        "training_run_id": "run_20240115_abc",
        "train_period_start": "2018-01-01",
        "train_period_end": "2022-12-31",
    },
)
```

Input CSV / DataFrame must contain at minimum: `entry_time`, `pair`, `pred_prob_up`.

### Step 2: Consolidate into the operational cube

```bash
python scripts/consolidate_dl_predictions.py \
    [--input-dir data/output/dl_predictions] \
    [--output-dir data/output/dl_signals]
```

### Deprecated: CSV consolidator

The legacy `scripts/build_dl_signal_artifact.py` (CSV consolidator) remains
functional but emits a `DeprecationWarning`.  It will be removed in a future
release.

---

## Integration with market-phase-ml

`market-phase-ml` consumes the consolidated cube via its `src/dl_surface_loader.py`
module.  The consumer selects a specific signal **surface** (a slice of the
multi-dimensional signal cube) by specifying:

```python
surface = {
    "model": "MLP",
    "target_horizon": 24,
    "feature_set": "price_vol_sentiment",
    "dl_regime": "LVTF",   # optional; omit to select all regimes
}
```

The consumer renames `entry_time` → `timestamp` internally and optionally
adds a `mpml_regime_equiv` column mapping the producer regime labels to the
consumer's own taxonomy.

See `market-phase-ml/docs/DL_SIGNAL_INTEGRATION.md` for the consumer-side
documentation.
