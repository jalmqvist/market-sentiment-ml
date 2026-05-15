# DL prediction artifacts (v1.1)

This document specifies the standardized row-level DL prediction export artifacts produced by `market-sentiment-ml`.

Artifact producers now include both:

* `research.deep_learning.train` (MLP)
* `research.deep_learning.train_lstm` (LSTM)

Both producers emit the same parquet schema and manifest identity contract.
For LSTM, sequence target-row metadata alignment is preserved during export so
that exported `(pair, entry_time)` rows match sequence prediction targets.

The exporter is designed as a stable operational boundary between:

* `market-sentiment-ml` (DL signal producer)
* `market-phase-ml` (DL signal consumer)

---

# Design goals

* **No log parsing**

  * Predictions are exported directly from the DL inference/evaluation pipeline.

* **Minimal and stable**

  * The artifact is an operational interface, not an experiment dump.

* **One artifact per surface/run**

  * Each DL run exports exactly one prediction surface.

* **Explicit semantics**

  * `pred_prob_up` and `signal_strength` are contractually defined.

* **Strict invariants**

  * Timestamp alignment, uniqueness, monotonicity, and probability ranges are enforced at write time.

* **Minimal downstream coupling**

  * Consumers should not need knowledge of training internals.

---

# Summary

A DL run exports:

## Parquet payload

```text
data/output/dl_predictions/<run_id>.parquet
```

Contains:

* time-indexed prediction rows
* canonical signal columns
* optional inference metadata

---

## Manifest

```text
data/output/dl_predictions/<run_id>.manifest.json
```

Contains:

* identity
* provenance
* schema metadata
* signal semantics
* diagnostics
* warnings

---

# Example artifact pair

```text
data/output/dl_predictions/mlp__HVR__24__price_trend__20260510T204302Z.parquet
data/output/dl_predictions/mlp__HVR__24__price_trend__20260510T204302Z.manifest.json
```

---

# Exporter usage

## Example command

From repository root:

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime HVR \
  --target-horizon 24
```

Expected output:

```text
artifact_parquet: ...
artifact_manifest: ...
```

---

# Integration point

Predictions are exported directly from the DL inference/evaluation path:

* after `pred_prob_up` exists,
* while `(pair, entry_time)` alignment still exists,
* before metadata is discarded.

The exporter implementation intentionally remains small and focused:

```text
scripts/write_dl_prediction_artifact.py
```

---

# Surface identity

A DL signal surface is identified by:

| field            | description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| `model`          | DL architecture (`mlp`, `lstm`, etc.)                                 |
| `target_horizon` | prediction horizon measured in bars                                   |
| `feature_set`    | feature-set identifier                                                |
| `dl_regime`      | producer-side regime taxonomy (`HVTF`, `LVTF`, `HVR`, `LVR`, `MIXED`) |

These fields define the operational signal stream.

They are distinct from:

* training provenance
* hyperparameters
* experiment metadata

---

# Parquet schema (v1.1)

Parquet path:

```text
data/output/dl_predictions/<run_id>.parquet
```

---

# Canonical column order

The exporter writes columns in canonical order:

| column                 | type     | required    | description                               |
| ---------------------- | -------- | ----------- | ----------------------------------------- |
| `pair`                 | string   | yes         | FX pair normalized to lowercase `xxx-yyy` |
| `entry_time`           | datetime | yes         | H1 timestamp, tz-naive UTC                |
| `pred_prob_up`         | float64  | yes         | DL probability in `[0,1]`                 |
| `signal_strength`      | float64  | yes         | Canonical signal in `[-1,+1]`             |
| `prediction_timestamp` | datetime | yes (v1.1+) | inference/export timestamp, tz-naive UTC  |

---

# Optional columns

Optional columns may be added in later versions without breaking compatibility:

| column           | type   | description                     |
| ---------------- | ------ | ------------------------------- |
| `pred_direction` | Int64  | tri-state direction (`+1/-1/0`) |
| `confidence`     | float  | optional confidence estimate    |
| `schema_version` | string | optional row-level schema tag   |

Consumers should ignore unknown columns.

---

# Semantic contract

## `pred_prob_up`

Definition:

```text
P(price moves up over target_horizon bars)
```

Range:

```text
[0, 1]
```

---

## `signal_strength`

Definition:

```text
signal_strength = 2 * pred_prob_up - 1
```

Range:

```text
[-1, +1]
```

Interpretation:

| condition              | meaning                      |
| ---------------------- | ---------------------------- |
| `signal_strength > 0`  | behavioral upside pressure   |
| `signal_strength < 0`  | behavioral downside pressure |
| `signal_strength == 0` | neutral                      |

Important:

* `signal_strength` is NOT standardized.
* Raw calibration semantics are intentionally preserved.

---

# Time semantics

## `entry_time`

* H1-aligned
* tz-naive UTC
* represents the timestamp associated with the prediction row

---

## `prediction_timestamp`

* tz-naive UTC
* represents when inference/export occurred
* all rows in a single artifact may share the same timestamp

This exists to support future:

* rolling inference
* asynchronous export
* delayed generation
* mixed retraining cadence

---

# Writer-enforced invariants

The exporter enforces:

---

## 1. Pair normalization

Pairs are normalized to:

```text
xxx-yyy
```

lowercase.

Example:

```text
eur-usd
```

---

## 2. Timestamp semantics

`entry_time` must:

* be tz-naive UTC
* be H1-aligned

---

## 3. Probability constraints

```text
pred_prob_up ∈ [0,1]
signal_strength ∈ [-1,+1]
```

---

## 4. Canonical signal identity

The exporter enforces:

```text
signal_strength = 2 * pred_prob_up - 1
```

---

## 5. Uniqueness

No duplicate:

```text
(pair, entry_time)
```

rows may exist within a single artifact.

---

## 6. Monotonicity

`entry_time` must be monotonically increasing per pair.

---

## 7. Finite values

No:

* NaN
* inf
* nested arrays
* object-dtype probability payloads

are allowed in signal columns.

---

# Manifest schema (v1.1)

Manifest path:

```text
data/output/dl_predictions/<run_id>.manifest.json
```

The manifest owns:

* identity
* provenance
* schema metadata
* diagnostics
* warnings

---

# Required manifest fields

## Core metadata

| field              | description                   |
| ------------------ | ----------------------------- |
| `schema_version`   | artifact schema identifier    |
| `export_frequency` | expected frequency (`H1`)     |
| `generated_at_utc` | manifest generation timestamp |
| `run_id`           | opaque artifact identifier    |
| `parquet_file`     | parquet filename              |

---

## Signal semantics

```json
"signal_definition": {
  "formula": "signal_strength = 2 * pred_prob_up - 1",
  "range": "[-1, +1]"
}
```

---

## Identity

```json
"identity": {
  "model": "mlp",
  "dl_regime": "HVR",
  "target_horizon": 24,
  "feature_set": "price_trend"
}
```

---

## Provenance

```json
"provenance": {
  "dataset_version": "1.3.2",
  "model_version": null,
  "training_run_id": null
}
```

---

## Calibration metadata

```json
"calibration": {
  "method": "none",
  "notes": "raw model probability; no post-hoc calibration applied"
}
```

---

## Train period placeholder

```json
"train_period": {
  "start": null,
  "end": null
}
```

---

## Statistics

The manifest includes:

* row counts
* pair coverage
* timestamp ranges
* signal distribution summaries

Example:

```json
"signal_stats": {
  "pred_prob_up_mean": 0.427,
  "pred_prob_up_std": 0.088,
  "signal_strength_mean": -0.145,
  "signal_strength_std": 0.176
}
```

---

## Warnings

Warnings are explicitly surfaced:

```json
"warnings": []
```

Missing provenance fields are tracked separately:

```json
"missing_provenance_counts": {}
```

---

# Run ID

`run_id` is an opaque identifier.

Current convention:

```text
<model>__<dl_regime>__<target_horizon>__<feature_set>__<timestamp>Z
```

Example:

```text
mlp__HVR__24__price_trend__20260510T204302Z
```

Consumers must treat `run_id` as opaque and rely on manifest/parquet contents for semantics.

---

# Relation to market-phase-ml

`market-phase-ml` consumes DL artifacts by selecting a surface identity:

```python
DL_SIGNAL_SURFACE = {
    "model": "mlp",
    "target_horizon": 24,
    "feature_set": "price_trend",
    "dl_regime": "HVR",
}
```

The selected surface is joined on:

```text
(pair, entry_time)
```

using:

* lowercase normalized pairs
* tz-naive UTC H1 timestamps

---

# Scope boundaries

This exporter intentionally does NOT handle:

* H1→D1 aggregation
* calibration
* multi-surface orchestration
* ensemble logic
* evaluation metrics
* target labels
* returns
* walk-forward metadata

Those belong to downstream systems.

---

# Version status

## v1.0

Initial parquet + manifest exporter.

## v1.1

Adds:

* canonical `signal_strength`
* `prediction_timestamp`
* stricter dtype enforcement
* canonical column ordering
* stronger invariant validation
* richer manifest diagnostics/statistics
