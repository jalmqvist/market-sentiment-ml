# Behavioral Surface Schema

The **Behavioral Surface** is the canonical downstream artifact produced by
BSVE (Behavioral State Validation Engine). It is the sole supported interface
consumed by downstream systems (MSML, MPML, and future research pipelines).

The Behavioral Surface Parquet is the canonical BSVE artifact intended for
downstream integration. It provides a stable, deterministic and
ontology-independent interface that can be joined onto compatible research
datasets using `(timestamp, pair)` and subsequently consumed by MSML, MPML
or future research pipelines.

---

## Schema

| Column             | Type     | Description                                                                                                                                                                                                                                                                |
| ------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timestamp`        | datetime | Observation timestamp (UTC, tz-naive).                                                                                                                                                                                                                                     |
| `pair`             | string   | Currency pair (e.g. `usd-jpy`).                                                                                                                                                                                                                                            |
| `surface_id`       | string   | Ontology family identifier (e.g. `reactive_jpy`). Constant within a surface.                                                                                                                                                                                              |
| `surface_version`  | string   | Frozen ontology version (e.g. `1.0.0`). Constant within a surface.                                                                                                                                                                                                        |
| `state_id`         | string   | Canonical behavioral state identifier. Values for Reactive-JPY: `JPY_NON_EXTREME`, `JPY_CONSENSUS_YOUNG`, `JPY_CONSENSUS_MATURING`, `JPY_CONSENSUS_MATURE`.                                                                                                               |
| `episode_id`       | string   | Deterministic episode identifier. Unique per consensus episode per pair. Format: `{pair}:{counter:08d}`.                                                                                                                                                                   |
| `maturity_bars`    | int      | Running episode maturity. Counts consecutive consensus bars within the current episode; resets to `0` on non-extreme observations and to `1` on a new consensus episode.                                                                                                   |
| `crowd_side`       | string   | Crowd direction during consensus. Canonical values: `LONG`, `SHORT`, or `""` (neutral/unknown). Integer encodings from the master research dataset (`1`, `-1`, `0`) are normalised to these strings before export.                                                         |
| `transition_event` | string   | **String enum.** Stable transition label produced by the ontology state engine. Current Reactive-JPY values are `entry`, `continuation`, `exit_reversal`, and `exit_unknown`. Semantics are defined below. This field must not introduce new transition labels in this PR. |

---

## Transition Event Semantics

| Value            | Condition                                                                           |
| ---------------- | ----------------------------------------------------------------------------------- |
| `entry`          | First bar of a new consensus episode (`consensus_active == True`, new episode).     |
| `continuation`   | Continuing within an active consensus episode (`consensus_active == True`, ongoing).|
| `exit_reversal`  | First non-extreme bar immediately following a consensus episode.                    |
| `exit_unknown`   | Any other non-extreme bar (no prior consensus context, or multiple bars after exit).|

---

## Key Properties

- **One row per `(timestamp, pair)`**: the Behavioral Surface has a unique key
  for every observation. No duplicate keys are permitted.
- **Deterministic ordering**: rows are sorted by `(pair, timestamp)` within a
  surface. Repeated generation from the same inputs produces bit-identical
  results (excluding the `generated_at` metadata field).
- **Deterministic state assignment**: given identical inputs and a frozen
  calibration artifact, `state_id` assignments are fully reproducible.
- **Immutable artifact**: the Behavioral Surface is a purely descriptive
  record of behavioral state assignment. It does not record calibration
  thresholds, hazard statistics, validation results, probabilities, or any
  other derived quantities.

---

## Joining onto the Master Research Dataset

The Behavioral Surface is designed to be joined onto compatible research
datasets using the composite key `(timestamp, pair)`:

```python
import pandas as pd

surface = pd.read_parquet("behavioral_surface_reactive_jpy_1.0.0.parquet")
dataset = pd.read_csv("master_research_dataset_core.csv")
dataset["timestamp"] = pd.to_datetime(dataset["entry_time"])

joined = surface.merge(dataset, on=["timestamp", "pair"], how="inner")
```

---

## Artifact Metadata

Provenance metadata is stored in the accompanying manifest JSON (not as
per-row columns) and in the Parquet file schema metadata. The manifest
exposes:

| Key                              | Description                                          |
| -------------------------------- | ---------------------------------------------------- |
| `ontology_id`                    | Ontology family identifier.                          |
| `ontology_version`               | Frozen ontology version.                             |
| `calibration_id`                 | Identifier of the calibration artifact used.         |
| `dataset_version`                | Version of the input dataset.                        |
| `behavioral_surface_schema_version` | Schema version of this Behavioral Surface.        |
| `generated_timestamp`            | UTC timestamp of surface generation (ISO 8601).      |
| `row_count`                      | Total number of observations in the surface.         |
| `pair_counts`                    | Observation count per pair.                          |
| `state_counts`                   | Observation count per `state_id` value.              |

---

## Schema Versioning

The Behavioral Surface schema is a versioned public interface. Future
evolution should be **append-only** wherever possible to preserve downstream
compatibility. The current schema version is `1.0.0`.

Quantities already present in the master research dataset or deterministically
derivable from exported fields are intentionally omitted from the Behavioral
Surface to keep it minimal and stable.

---

## Quick Start

The following commands generate the canonical Behavioral Surface artifact from
a clean checkout.

### 1. Run calibration

```bash
python -m bsve.calibration.jpy_maturity_calibration \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --dataset-version 1.5.1 \
    --output-dir bsve.test/
```

### 2. Generate Behavioral Surface

```bash
python -m bsve.state_machine.rule_based \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/ \
    --calibration-artifact bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --state-spec bsve/state_specs/reactive_jpy_v1.yaml \
    --dataset-version 1.5.1
```

### 3. Inspect output

```bash
python -m bsve.validation.inspect_surface \
    --surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
    --calibration bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --output-dir bsve.test/inspection
```

### 4. Join onto the master research dataset (future PR)

```python
import pandas as pd

surface = pd.read_parquet("bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet")
dataset = pd.read_csv("data/output/1.5.1/master_research_dataset_core.csv")
dataset["timestamp"] = pd.to_datetime(dataset["entry_time"])

joined = surface.merge(dataset, on=["timestamp", "pair"], how="inner")
print(joined[["timestamp", "pair", "state_id", "maturity_bars"]].head())
```
