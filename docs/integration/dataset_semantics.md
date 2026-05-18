# MSML Dataset Semantics (Canonical Reference)

This document is the canonical contract for dataset/export semantics in MSML.

## 1) Scope

Primary files:

- `config.py`
- `scripts/write_dl_prediction_artifact.py`
- `scripts/consolidate_dl_predictions.py`
- `scripts/build_dl_signal_artifact.py`
- `research/deep_learning/train.py`
- `research/deep_learning/train_lstm.py`

## 2) Feature Availability Semantics

### Core rule

`dl_feature_available` is the explicit availability flag for DL signal rows.

- `1` = model-produced value is available.
- `0` = row exists due to control-mode expansion/imputation.

This flag is produced in:

- `write_dl_prediction_artifact.py::_apply_control_mode`
- `build_dl_signal_artifact.py::_build_artifact` (deprecated CSV path)

## 3) Missingness Handling (Explicit + Configurable)

Centralized config in `config.py`:

- `DL_ADD_MISSING_INDICATORS`
- `DL_IMPUTE_OPTIONAL_FEATURES`
- `DL_IMPUTATION_VALUE`

Semantics:

- **Mode A (explicit indicators):**
  - `DL_ADD_MISSING_INDICATORS=True`
  - Adds `pred_prob_up_missing`, `signal_strength_missing`
- **Mode B (impute only):**
  - `DL_ADD_MISSING_INDICATORS=False`
  - No `*_missing` indicator columns are added

Imputation behavior:

- `DL_IMPUTE_OPTIONAL_FEATURES=True` enables deterministic neutral imputation.
- `DL_IMPUTATION_VALUE` is interpreted as neutral `pred_prob_up` (default `0.5`).
- `signal_strength` imputation is derived as `2 * DL_IMPUTATION_VALUE - 1`.

Implementation points:

- `write_dl_prediction_artifact.py::_resolve_semantics_config`
- `write_dl_prediction_artifact.py::_apply_control_mode`
- `write_dl_prediction_artifact.py::_apply_missing_indicators`

## 4) Deterministic Control / Export Modes

Control mode config:

- `DL_EXPORT_CONTROL_MODE` in `config.py`
- Optional per-run override via provenance `control_mode`

Supported modes:

1. `normal`
   - No availability transformation.
2. `constant_presence`
   - Per-pair hourly grid expansion (`min(entry_time)` to `max(entry_time)`).
   - Missing rows are explicitly marked with `dl_feature_available=0`.
   - Optional deterministic imputation applied per config.
3. `availability_shuffle`
   - Deterministic per-pair timestamp shuffle using `DL_AVAILABILITY_SHUFFLE_SEED`.
   - Preserves signal values while perturbing availability timing.

## 5) DL Export Contract

Per-run export function:

- `write_dl_prediction_artifact.py::write_dl_prediction_artifact`

Identity contract columns in parquet rows:

- `model`, `dl_regime`, `target_horizon`, `feature_set`

Required metadata fields (`artifact_metadata`) in manifest:

- `export_timestamp`
- `prediction_horizon_hours`
- `feature_surface`
- `dl_regime`
- `availability_semantics`
- `missing_indicator_mode`
- `imputation_mode`

Also persisted in parquet schema metadata (best effort) by:

- `write_dl_prediction_artifact.py::_write_parquet_with_metadata`

## 6) Timestamp & Visibility Semantics

Contracts:

- `entry_time` = H1 bar open timestamp for prediction alignment.
- `prediction_timestamp` = inference visibility timestamp.
- Predictions are visible only after `prediction_timestamp`.
- `target_horizon` is in H1 bars and interpreted as hours in export metadata.

## 7) Overlap Semantics

Operational overlap is explicit via:

- Sparse row presence (`dl_feature_available=1`) in normal mode.
- Controlled availability transformations in `constant_presence`/`availability_shuffle`.

Consolidation output:

- `dl_signals_h1_v1.parquet`
- `DL_SIGNAL_MANIFEST_h1_v1.json`

## 8) Provenance & Diagnostics

Per-run manifest includes:

- `git_commit`
- `identity`
- `artifact_metadata`
- `export_config`
- `missingness_config`
- `causal_assumptions`

Consolidation now writes deterministic diagnostics:

- `data/output/.../dataset_provenance.json`
- `data/output/.../feature_coverage.csv`
- `data/output/.../feature_missingness.csv`

Generated in:

- `consolidate_dl_predictions.py::_write_provenance_diagnostics`

Diagnostics include:

- per-pair first/last timestamp
- overlap %
- non-null % / missingness %
- feature sparsity
- effective sample counts
- global pair/feature counts and control/missingness mode summaries

## 9) Dataset Contracts (Code-Level)

Contracts encoded in code comments/logic:

- `write_dl_prediction_artifact.py`:
  - `dl_feature_available` is availability semantics, not value semantics.
- `config.py`:
  - missing indicators are optional experimental controls.

## 10) Example Semantics

### Example A: explicit missing indicators

- `DL_ADD_MISSING_INDICATORS=True`
- `DL_IMPUTE_OPTIONAL_FEATURES=True`
- `DL_IMPUTATION_VALUE=0.5`
- `DL_EXPORT_CONTROL_MODE=constant_presence`

Result: expanded hourly surface, explicit `*_missing` columns, deterministic neutral fill.

### Example B: no missing indicators

- `DL_ADD_MISSING_INDICATORS=False`
- `DL_IMPUTE_OPTIONAL_FEATURES=True`
- `DL_EXPORT_CONTROL_MODE=normal`

Result: no `*_missing` columns; imputation behavior remains explicit in manifest config.

## 11) Remaining Unresolved Ambiguities

1. Cross-repository consumer behavior (MPML) can still alter effective semantics if it applies additional post-join imputation/forward-fill.
2. `constant_presence` currently expands within observed per-pair export ranges; it does not infer external market calendars beyond those ranges.
3. `availability_shuffle` is deterministic and per-pair, but downstream experiments must keep seed/control mode fixed for rerun comparability.
