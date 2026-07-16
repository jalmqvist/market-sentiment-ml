# DL Artifact Contract — MSML Producer Side (v2)

This document is the **formal contract specification** for DL prediction
artifacts exported by `market-sentiment-ml` (MSML) and consumed by
`market-phase-ml` (MPML).

Schema version: **`2.0.0`** (see `schemas/dl_artifact_schema.py`)

Related documents:

- `docs/integration/DL_SIGNAL_SCHEMA.md` — column-level schema reference
- `docs/integration/dataset_semantics.md` — dataset & export semantics
- `schemas/dl_artifact_schema.py` — centralized constants and validation

---

## Background

A prior production failure exposed semantic ambiguity in timestamps exported
by MSML and interpreted by MPML.  MSML exported artifact-generation
timestamps, while MPML interpreted them as causal prediction-availability
timestamps.  This was not classic data leakage but a **contract ambiguity**.

This document formalizes the contract to eliminate that ambiguity.

---

## Timestamp Semantics (v2)

Four timestamp columns are exported with **exactly one meaning each**.
Import their names from `schemas/dl_artifact_schema.py` — do not hardcode
strings.

| Constant | Column name | Meaning | Causal use? |
|---|---|---|---|
| `DL_TIMESTAMP_COL` | `entry_time` | H1 bar open timestamp being predicted (UTC tz-naive) | ✓ (bar key) |
| `DL_AVAILABLE_TS_COL` | `prediction_available_timestamp` | Earliest historical timestamp the prediction **could have been known** | ✓ (causal boundary) |
| `DL_GENERATED_TS_COL` | `prediction_generated_timestamp` | Wall-clock time prediction was generated | ✗ (diagnostics only) |
| `DL_ARTIFACT_CREATED_COL` | `artifact_created_timestamp` | Wall-clock time parquet was exported | ✗ (provenance only) |

> **Note on `DL_TIMESTAMP_COL`**: The canonical logical name is `"timestamp"`.
> The physical parquet column is `"entry_time"` for backward compatibility.
> MPML consumers may alias `entry_time` → `timestamp`; a future migration may
> rename the physical column.

### Causal ordering contract

```
prediction_available_timestamp <= entry_time
```

This invariant is **enforced** by `validate_dl_artifact()` before any write.
Violations cause a hard `ValueError` (fail-fast).

### What to use for causality checks (MPML)

MPML **must** use `prediction_available_timestamp` for all causality checks.
`prediction_generated_timestamp` and `artifact_created_timestamp` are
wall-clock timestamps and **must not** be used for causality.

---

## Schema Version

```python
from schemas.dl_artifact_schema import DL_SCHEMA_VERSION
# DL_SCHEMA_VERSION = "2.0.0"
```

The `schema_version` field in manifests and parquet metadata **must** equal
`DL_SCHEMA_VERSION`.

---

## Required Parquet Columns (v2)

| Column | Type | Nullable | v2? | Description |
|---|---|---|---|---|
| `pair` | string | No | — | Normalized FX pair (lowercase `xxx-yyy`) |
| `entry_time` | datetime (UTC) | No | — | H1 bar open timestamp (tz-naive UTC) |
| `pred_prob_up` | float64 | No | — | P(up) ∈ [0, 1] |
| `signal_strength` | float64 | No | — | `2 * pred_prob_up − 1` ∈ [−1, 1] |
| `pred_direction` | Int64 | No | — | Tri-state: +1, −1, 0 |
| `confidence` | float64 | Yes | — | Reliability estimate ∈ [0, 1]; null if absent |
| `prediction_timestamp` | datetime | Yes | — | v1 legacy name for per-row inference time |
| `prediction_available_timestamp` | datetime (UTC) | No | **NEW** | Causal boundary for MPML; ≤ entry_time |
| `prediction_generated_timestamp` | datetime | Yes | **NEW** | Wall-clock inference time (diagnostics only) |
| `artifact_created_timestamp` | datetime | No | **NEW** | Wall-clock export time; same for all rows |
| `model` | string | No | — | Model identifier (e.g. `"MLP"`) |
| `surface_id` | string | No | — | Behavioral Surface identifier (e.g. `"trend_vol"`, `"reactive_jpy"`) |
| `surface_version` | string | No | — | Behavioral Surface version used during training |
| `state_id` | string | No | — | Behavioral State identifier (e.g. `"LVTF"`, `"JPY_CONSENSUS_YOUNG"`) |
| `dl_regime` | string | No | — | Deprecated compatibility alias for `state_id` (Trend/Vol surfaces) |
| `target_horizon` | Int64 | No | — | Prediction horizon in bars |
| `feature_set` | string | No | — | Feature set identifier |
| `dl_feature_available` | Int64 | No | — | Availability flag (1 = real, 0 = synthetic) |

---

## Required Manifest Fields (v2)

```json
{
  "schema_version": "2.0.0",
  "artifact_created_timestamp": "<ISO 8601 UTC>",
  "git_commit": "<sha>",
  "identity": {
    "model": "...",
    "surface_id": "...",
    "surface_version": "...",
    "state_id": "...",
    "dl_regime": "...",
    "target_horizon": 24,
    "feature_set": "..."
  },
  "artifact_metadata": {
    "artifact_created_timestamp": "<ISO 8601 UTC>",
    "feature_surface": "...",
    "availability_semantics": "...",
    "timestamp_semantics": { ... }
  },
  "provenance": {
    "dataset_version": "...",
    "training_pairs": [...],
    "inference_pairs": [...]
  }
}
```

`training_pairs` and `inference_pairs` are manifest-only provenance fields.
Do **not** add them as parquet row columns.

---

## Validation (`validate_dl_artifact`)

Import from `schemas/dl_artifact_schema`:

```python
from schemas.dl_artifact_schema import validate_dl_artifact

violations = validate_dl_artifact(df, metadata={"schema_version": DL_SCHEMA_VERSION})
```

Checks enforced:

1. Required columns present.
2. Schema version present and correct.
3. No null values in key columns (`pair`, `entry_time`,
   `prediction_available_timestamp`).
4. Pair normalization — must match `^[a-z]{3}-[a-z]{3}$`.
5. No duplicate `(pair, entry_time)` rows.
6. `entry_time` monotonically non-decreasing within each pair.
7. Timezone consistency — all timestamp columns must be tz-naive (UTC).
8. Causal ordering — `prediction_available_timestamp <= entry_time`.

`write_dl_prediction_artifact()` calls `validate_dl_artifact()` **before**
any disk write (fail-fast enforcement).

---

## Required Export Behavior (MSML)

- Export `prediction_available_timestamp` as the earliest simulated historical
  timestamp the prediction could have been observed.
- **Do NOT** use wall-clock `pd.Timestamp.now()` for `prediction_available_timestamp`.
- For the default MLP/LSTM train pipeline, set
  `prediction_available_timestamp = entry_time` (the model uses only features
  available at or before the bar open time; this satisfies the causal
  constraint as equality).
- `prediction_generated_timestamp` and `artifact_created_timestamp` **may**
  be wall-clock, but must be **explicitly named** and documented as
  non-causal.

---

## Control Mode Behavior

The special export control modes maintain the causal invariant:

- **`normal`**: `prediction_available_timestamp = entry_time` (default).
- **`constant_presence`**: expanded synthetic rows get
  `prediction_available_timestamp = entry_time` (their synthetic bar time).
- **`availability_shuffle`**: after shuffling entry_times,
  `prediction_available_timestamp` is reset to the new shuffled `entry_time`
  to maintain the invariant.  This is an ablation mode — causality is
  intentionally perturbed for experimental control.

---

## Centralized Constants

All column name strings and the schema version are in one place:

```python
from schemas.dl_artifact_schema import (
    DL_SCHEMA_VERSION,      # "2.0.0"
    DL_TIMESTAMP_COL,       # "timestamp" (logical; physical: "entry_time")
    DL_AVAILABLE_TS_COL,    # "prediction_available_timestamp"
    DL_GENERATED_TS_COL,    # "prediction_generated_timestamp"
    DL_ARTIFACT_CREATED_COL,# "artifact_created_timestamp"
    DL_PAIR_COL,            # "pair"
    validate_dl_artifact,
)
```

**Never hardcode these strings elsewhere in the codebase.**

---

## MPML Compatibility

This PR is MSML-only (producer side).  A follow-up PR in MPML will implement
strict consumer-side validation and use `prediction_available_timestamp`
exclusively for causality checks.

Until then:

- MPML continues to work with existing `entry_time`-based joins.
- The new timestamp columns are additive and do not break existing parquet
  consumers.
- `artifact_created_timestamp` and `prediction_generated_timestamp` should be
  ignored by MPML until the consumer-side contract is formalized.
