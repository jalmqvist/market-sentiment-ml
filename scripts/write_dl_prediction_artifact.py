"""
write_dl_prediction_artifact.py
================================
Write a single per-run DL prediction artifact: one Parquet file (time-series
payload) and one JSON manifest (identity + provenance) for a single DL
training / inference run on one surface.

This is the **producer API** that DL training/inference scripts call after
generating row-level predictions.  Each call produces:

    {output_dir}/{run_id}.parquet         — time-series payload
    {output_dir}/{run_id}.manifest.json   — identity + provenance

These per-run artifacts are later consolidated into the operational cube by
``scripts/consolidate_dl_predictions.py``.

Schema version:  2.0.0 (see schemas/dl_artifact_schema.py)
See also:        docs/integration/DL_SIGNAL_SCHEMA.md
                 docs/integration/dl_artifact_contract.md

Usage (CLI)::

    python scripts/write_dl_prediction_artifact.py \\
        --input-csv data/output/dl_predictions/raw/mlp_lvtf_24_preds.csv \\
        --model MLP \\
        --dl-regime LVTF \\
        --target-horizon 24 \\
        --feature-set price_vol_sentiment \\
        [--dataset-version 1.1.0] \\
        [--model-version v1.0] \\
        [--training-run-id run_20240115_abc] \\
        [--output-dir data/output/dl_predictions] \\
        [--run-id mlp_lvtf_24_20240115]

Python API::

    from scripts.write_dl_prediction_artifact import write_dl_prediction_artifact

    write_dl_prediction_artifact(
        df=predictions_df,           # DataFrame with entry_time, pair, pred_prob_up
        identity={
            "model": "MLP",
            "dl_regime": "LVTF",
            "target_horizon": 24,    # numeric: number of bars
            "feature_set": "price_vol_sentiment",
        },
        provenance={                 # all optional
            "dataset_version": "1.1.0",
            "model_version": "v1.0",
            "training_run_id": "run_20240115_abc",
        },
        output_dir=Path("data/output/dl_predictions"),
        run_id="mlp_lvtf_24_20240115",  # auto-generated if None
    )

Per-run Parquet schema (time-series payload only)
--------------------------------------------------
Required:
    pair             — normalized FX pair (xxx-yyy)
    entry_time       — H1 bar open timestamp (tz-naive UTC)
    pred_prob_up     — float in [0, 1]
    signal_strength  — float in [-1, 1] (= 2*pred_prob_up - 1)

Optional:
    pred_direction   — tri-state Int64: +1 (>0.5), -1 (<0.5), 0 (==0.5)
    confidence       — float in [0, 1]; null if not supplied
    prediction_timestamp — tz-naive UTC; per-row inference timestamp

Per-run Manifest schema
-----------------------
Required:
    schema_version, export_frequency, generated_at_utc, signal_definition,
    identity (model, dl_regime, target_horizon, feature_set),
    calibration, train_period, warnings, missing_provenance_counts,
    row_count, pairs, entry_time_min/max, git_commit
Optional provenance:
    dataset_version, model_version, training_run_id, seed, epochs, ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (shared constants re-imported to avoid circular deps)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from build_dl_signal_artifact import (  # noqa: E402
    EXPORT_FREQUENCY,
    SCHEMA_VERSION,
    VALID_DL_REGIMES,
    _get_git_commit_hash,
    _normalize_entry_time,
    _normalize_pair,
)
import config as cfg  # noqa: E402
from schemas.dl_artifact_schema import (  # noqa: E402
    DL_SCHEMA_VERSION,
    DL_AVAILABLE_TS_COL,
    DL_GENERATED_TS_COL,
    DL_ARTIFACT_CREATED_COL,
    validate_dl_artifact,
)

# ---------------------------------------------------------------------------
# Per-run parquet schema (v2)
#
# v2 adds explicit timestamp columns with one meaning each:
#   prediction_available_timestamp — causal boundary used by MPML
#   prediction_generated_timestamp — wall-clock inference time (diagnostics)
#   artifact_created_timestamp     — wall-clock export time (provenance)
#
# IMPORTANT: market-phase-ml filters surfaces directly from parquet row
# columns, so the identity columns MUST be physically present in the parquet.
# ---------------------------------------------------------------------------

REQUIRED_PARQUET_COLS = [
    "pair",
    "entry_time",
    "pred_prob_up",
    "signal_strength",
    "pred_direction",
    "confidence",
    "prediction_timestamp",        # v1 compat: per-row inference time (legacy name)
    DL_AVAILABLE_TS_COL,           # v2: causal boundary for MPML
    DL_GENERATED_TS_COL,           # v2: wall-clock inference time (diagnostics only)
    DL_ARTIFACT_CREATED_COL,       # v2: artifact export time (provenance only)
    "model",
    "dl_regime",
    "target_horizon",
    "feature_set",
    "dl_feature_available",
]

# Legacy payload-only schema (kept for backward compatibility in code paths
# that rely on the old constant name).
RUN_PARQUET_COLS = [
    "pair",
    "entry_time",
    "pred_prob_up",
    "signal_strength",
    "pred_direction",
    "confidence",
    "prediction_timestamp",
    "dl_feature_available",
]

PREDICTIONS_DIR_DEFAULT = Path("data/output/dl_predictions")
VALID_CONTROL_MODES = {"normal", "constant_presence", "availability_shuffle"}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _resolve_semantics_config(
    provenance: dict | None,
) -> dict[str, Any]:
    provenance = provenance or {}
    def _as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    control_mode = str(
        provenance.get("control_mode", cfg.DL_EXPORT_CONTROL_MODE)
    ).strip().lower()
    if control_mode not in VALID_CONTROL_MODES:
        raise ValueError(
            f"Unknown control_mode={control_mode!r}. Supported: {sorted(VALID_CONTROL_MODES)}"
        )

    add_missing_indicators = _as_bool(
        provenance.get("dl_add_missing_indicators", cfg.DL_ADD_MISSING_INDICATORS),
        cfg.DL_ADD_MISSING_INDICATORS,
    )
    impute_optional_features = _as_bool(
        provenance.get("dl_impute_optional_features", cfg.DL_IMPUTE_OPTIONAL_FEATURES),
        cfg.DL_IMPUTE_OPTIONAL_FEATURES,
    )
    try:
        imputation_value = float(
            provenance.get("dl_imputation_value", cfg.DL_IMPUTATION_VALUE)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DL imputation value must be numeric for pred_prob_up semantics."
        ) from exc
    shuffle_seed = int(
        provenance.get("availability_shuffle_seed", cfg.DL_AVAILABILITY_SHUFFLE_SEED)
    )

    if not (0.0 <= imputation_value <= 1.0):
        raise ValueError(
            f"DL imputation value must be in [0, 1] for pred_prob_up semantics. Got {imputation_value}"
        )

    return {
        "control_mode": control_mode,
        "add_missing_indicators": add_missing_indicators,
        "impute_optional_features": impute_optional_features,
        "imputation_value": imputation_value,
        "availability_shuffle_seed": shuffle_seed,
    }


def _apply_control_mode(payload: pd.DataFrame, semantics_config: dict[str, Any]) -> pd.DataFrame:
    mode = semantics_config["control_mode"]
    out = payload.copy()

    # CONTRACT:
    # dl_feature_available represents availability semantics (not value semantics).
    out["dl_feature_available"] = 1

    if mode == "normal":
        return out

    if mode == "availability_shuffle":
        seed = int(semantics_config["availability_shuffle_seed"])
        shuffled_frames: list[pd.DataFrame] = []
        for pair, grp in out.groupby("pair", sort=False):
            grp = grp.sort_values("entry_time").reset_index(drop=True)
            if len(grp) <= 1:
                shuffled_frames.append(grp)
                continue
            pair_seed = int(hashlib.sha256(str(pair).encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed + pair_seed)
            shuffled_times = grp["entry_time"].to_numpy().copy()
            rng.shuffle(shuffled_times)
            grp["entry_time"] = pd.to_datetime(shuffled_times)
            # After shuffling, prediction_available_timestamp must track the
            # new entry_time to maintain the causal invariant
            # (prediction_available_timestamp <= entry_time).
            # In shuffle mode this is an ablation — the timestamps are
            # deliberately permuted for experimental control, so equality is
            # the correct post-shuffle assignment.
            if DL_AVAILABLE_TS_COL in grp.columns:
                grp[DL_AVAILABLE_TS_COL] = grp["entry_time"].copy()
            shuffled_frames.append(grp)
        out = pd.concat(shuffled_frames, ignore_index=True)
        return out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    # mode == "constant_presence"
    expanded_frames: list[pd.DataFrame] = []
    for pair, grp in out.groupby("pair", sort=False):
        grp = grp.sort_values("entry_time").reset_index(drop=True)
        et_min = grp["entry_time"].min()
        et_max = grp["entry_time"].max()
        if pd.isna(et_min) or pd.isna(et_max):
            expanded_frames.append(grp)
            continue
        full_times = pd.date_range(et_min, et_max, freq="h")
        full = pd.DataFrame({"pair": pair, "entry_time": full_times})
        merged = full.merge(
            grp.drop(columns=["pair"]),
            on="entry_time",
            how="left",
        )
        merged["pair"] = pair
        merged["dl_feature_available"] = merged["pred_prob_up"].notna().astype(int)

        # Fill prediction_available_timestamp for synthetically expanded rows.
        # These rows have no real prediction (dl_feature_available=0), so
        # default to entry_time to satisfy the causal invariant.
        if DL_AVAILABLE_TS_COL in merged.columns:
            null_avail = merged[DL_AVAILABLE_TS_COL].isna()
            merged.loc[null_avail, DL_AVAILABLE_TS_COL] = merged.loc[null_avail, "entry_time"]

        unavailable_mask = merged["dl_feature_available"].eq(0)
        if semantics_config["impute_optional_features"]:
            imputed_prob = float(semantics_config["imputation_value"])
            merged.loc[unavailable_mask, "pred_prob_up"] = imputed_prob
            merged.loc[unavailable_mask, "signal_strength"] = (2.0 * imputed_prob) - 1.0
            merged.loc[unavailable_mask, "pred_direction"] = 0
            merged.loc[unavailable_mask, "confidence"] = 0.0

        expanded_frames.append(merged)

    out = pd.concat(expanded_frames, ignore_index=True)
    return out.sort_values(["pair", "entry_time"]).reset_index(drop=True)


def _apply_missing_indicators(payload: pd.DataFrame, semantics_config: dict[str, Any]) -> pd.DataFrame:
    out = payload.copy()
    if semantics_config["add_missing_indicators"]:
        out["pred_prob_up_missing"] = out["pred_prob_up"].isna().astype(int)
        out["signal_strength_missing"] = out["signal_strength"].isna().astype(int)
    return out


def _build_run_payload(df: pd.DataFrame, semantics_config: dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw prediction rows into the per-run time-series payload schema.

    Parameters
    ----------
    df:
        Raw DataFrame with at minimum: entry_time, pair, pred_prob_up.

    Returns
    -------
    pd.DataFrame
        Validated and normalized time-series payload.
    """
    result = df.copy()

    # pair
    result["pair"] = result["pair"].map(_normalize_pair)

    # entry_time
    result["entry_time"] = _normalize_entry_time(result["entry_time"])

    # pred_prob_up
    result["pred_prob_up"] = pd.to_numeric(result["pred_prob_up"], errors="coerce")
    invalid = (
        result["pred_prob_up"].isna()
        | (result["pred_prob_up"] < 0)
        | (result["pred_prob_up"] > 1)
    )
    if invalid.any():
        n = int(invalid.sum())
        raise ValueError(
            f"{n:,} row(s) have invalid pred_prob_up (must be float in [0, 1]).\n"
            f"Sample:\n{result.loc[invalid, ['pair', 'entry_time', 'pred_prob_up']].head()}"
        )

    # signal_strength
    result["signal_strength"] = 2.0 * result["pred_prob_up"] - 1.0

    # pred_direction: tri-state (+1, -1, 0)
    if "pred_direction" not in result.columns:
        result["pred_direction"] = pd.array(
            np.where(
                result["pred_prob_up"] > 0.5,
                1,
                np.where(result["pred_prob_up"] < 0.5, -1, 0),
            ),
            dtype="Int64",
        )
    else:
        result["pred_direction"] = pd.to_numeric(
            result["pred_direction"], errors="coerce"
        ).astype("Int64")

    # confidence
    if "confidence" not in result.columns:
        result["confidence"] = np.nan
    else:
        result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")

    # prediction_timestamp: optional per-row inference timestamp (v1 legacy name)
    if "prediction_timestamp" not in result.columns:
        result["prediction_timestamp"] = pd.NaT
    else:
        result["prediction_timestamp"] = _normalize_entry_time(
            result["prediction_timestamp"]
        )

    # prediction_available_timestamp (v2): causal boundary for MPML.
    # Defaults to entry_time — the prediction is available at the bar open time.
    # Callers may supply per-row values (must satisfy <= entry_time).
    if DL_AVAILABLE_TS_COL not in result.columns:
        result[DL_AVAILABLE_TS_COL] = result["entry_time"].copy()
    else:
        result[DL_AVAILABLE_TS_COL] = _normalize_entry_time(
            result[DL_AVAILABLE_TS_COL]
        )

    # prediction_generated_timestamp (v2): wall-clock inference time (diagnostics only).
    # Falls back to prediction_timestamp if not explicitly supplied.
    if DL_GENERATED_TS_COL not in result.columns:
        result[DL_GENERATED_TS_COL] = result["prediction_timestamp"].copy()
    else:
        result[DL_GENERATED_TS_COL] = _normalize_entry_time(
            result[DL_GENERATED_TS_COL]
        )

    # Add any missing columns as NaN (payload-only optional fields)
    for col in RUN_PARQUET_COLS:
        if col not in result.columns:
            result[col] = np.nan

    result["dl_feature_available"] = 1

    # v2 timestamp columns must be preserved through the slice.
    _v2_ts_cols = [DL_AVAILABLE_TS_COL, DL_GENERATED_TS_COL]
    _payload_cols = RUN_PARQUET_COLS + [c for c in _v2_ts_cols if c in result.columns]
    result = result[_payload_cols].sort_values(["pair", "entry_time"]).reset_index(
        drop=True
    )
    result = _apply_control_mode(result, semantics_config)
    result = _apply_missing_indicators(result, semantics_config)
    return result


def _validate_identity(identity: dict) -> None:
    """Validate the required identity fields."""
    required = {"model", "dl_regime", "target_horizon", "feature_set"}
    missing = required - set(identity.keys())
    if missing:
        raise ValueError(
            f"identity dict is missing required keys: {sorted(missing)}\n"
            f"Required: model, dl_regime, target_horizon (int), feature_set"
        )
    # target_horizon must be a non-negative integer
    th = identity["target_horizon"]
    if not isinstance(th, (int, np.integer)) or int(th) < 0:
        raise ValueError(
            f"identity['target_horizon'] must be a non-negative integer (bars). "
            f"Got: {th!r} (type={type(th).__name__})"
        )
    # dl_regime should be from the known taxonomy (warn only)
    if identity["dl_regime"] not in VALID_DL_REGIMES:
        import warnings

        warnings.warn(
            f"dl_regime={identity['dl_regime']!r} is not in the known producer taxonomy "
            f"{sorted(VALID_DL_REGIMES)}. This run artifact will still be written.",
            UserWarning,
            stacklevel=4,
        )


def _build_run_manifest(
    payload: pd.DataFrame,
    identity: dict,
    provenance: dict | None,
    run_id: str,
    parquet_filename: str,
    semantics_config: dict[str, Any],
) -> dict:
    """Build the per-run manifest dict."""
    provenance = provenance or {}

    # Per-pair stats
    pair_stats: dict = {}
    for pair, grp in payload.groupby("pair"):
        et = grp["entry_time"].dropna()
        pair_stats[str(pair)] = {
            "row_count": int(len(grp)),
            "entry_time_min": et.min().isoformat() if len(et) else None,
            "entry_time_max": et.max().isoformat() if len(et) else None,
        }

    # Warnings
    warnings_list: list[str] = []
    if identity.get("dl_regime") not in VALID_DL_REGIMES:
        warnings_list.append(
            f"dl_regime={identity.get('dl_regime')!r} is not in known producer taxonomy "
            f"{sorted(VALID_DL_REGIMES)}"
        )
    if payload["prediction_timestamp"].isna().all():
        warnings_list.append("prediction_timestamp not recorded (all null)")
    if semantics_config["control_mode"] != "normal":
        warnings_list.append(
            f"control_mode={semantics_config['control_mode']} modifies default availability behavior"
        )

    # Missing provenance counts
    missing_provenance_counts = {
        "dataset_version": 1 if not provenance.get("dataset_version") else 0,
        "model_version": 1 if not provenance.get("model_version") else 0,
        "training_run_id": 1 if not provenance.get("training_run_id") else 0,
    }
    if any(missing_provenance_counts.values()):
        warnings_list.append(
            "some optional provenance fields not supplied; "
            "see missing_provenance_counts"
        )

    prediction_horizon_hours = int(identity["target_horizon"])
    missing_indicator_mode = (
        "explicit_missing_indicators"
        if semantics_config["add_missing_indicators"]
        else "imputation_only"
    )
    imputation_mode = (
        "neutral_imputation"
        if semantics_config["impute_optional_features"]
        else "no_imputation"
    )
    availability_semantics = (
        "visibility_by_control_mode"
        if semantics_config["control_mode"] != "normal"
        else "sparse_observed_only"
    )

    artifact_created_ts = datetime.now(timezone.utc).isoformat()

    artifact_metadata = {
        "export_timestamp": artifact_created_ts,  # legacy key (kept for compat)
        DL_ARTIFACT_CREATED_COL: artifact_created_ts,  # v2 canonical key
        "prediction_horizon_hours": prediction_horizon_hours,
        "feature_surface": str(identity["feature_set"]),
        "dl_regime": str(identity["dl_regime"]),
        "availability_semantics": availability_semantics,
        "missing_indicator_mode": missing_indicator_mode,
        "imputation_mode": imputation_mode,
        "timestamp_semantics": {
            "entry_time": "H1 bar open timestamp (UTC tz-naive); the bar being predicted",
            DL_AVAILABLE_TS_COL: (
                "earliest historical timestamp the prediction could have been observed; "
                "used by MPML for causality checks; must be <= entry_time"
            ),
            DL_GENERATED_TS_COL: (
                "wall-clock inference time (diagnostics only); "
                "MUST NOT be used for causality"
            ),
            DL_ARTIFACT_CREATED_COL: (
                "wall-clock artifact export time (provenance only); "
                "MUST NOT be used for causality"
            ),
        },
    }

    return {
        "schema_version": DL_SCHEMA_VERSION,
        "export_frequency": EXPORT_FREQUENCY,
        "generated_at_utc": artifact_created_ts,
        DL_ARTIFACT_CREATED_COL: artifact_created_ts,
        "run_id": run_id,
        "parquet_file": parquet_filename,
        "signal_definition": {
            "formula": "signal_strength = 2 * pred_prob_up - 1",
            "range": "[-1, +1]",
            "semantics": (
                "positive = behavioral upside pressure; "
                "negative = downside pressure"
            ),
            "pred_prob_up": (
                "P(price moves up over target_horizon bars) from DL model"
            ),
            "note": (
                "signal_strength is NOT standardised; "
                "preserves calibration semantics"
            ),
        },
        "identity": {
            "model": str(identity["model"]),
            "dl_regime": str(identity["dl_regime"]),
            "target_horizon": int(identity["target_horizon"]),
            "feature_set": str(identity["feature_set"]),
        },
        "artifact_metadata": artifact_metadata,
        "export_config": {
            "control_mode": semantics_config["control_mode"],
            "availability_shuffle_seed": semantics_config["availability_shuffle_seed"],
        },
        "missingness_config": {
            "DL_ADD_MISSING_INDICATORS": semantics_config["add_missing_indicators"],
            "DL_IMPUTE_OPTIONAL_FEATURES": semantics_config["impute_optional_features"],
            "DL_IMPUTATION_VALUE": semantics_config["imputation_value"],
        },
        "causal_assumptions": [
            "prediction_available_timestamp is the causal boundary for MPML; "
            "it is set to entry_time (bar open) by default.",
            "prediction_generated_timestamp is wall-clock only; MUST NOT be used for causality.",
            "artifact_created_timestamp is wall-clock only; MUST NOT be used for causality.",
            "Missing indicators are optional experimental controls, not guaranteed behavioral features.",
        ],
        "provenance": {
            "dataset_version": provenance.get("dataset_version", None),
            "model_version": provenance.get("model_version", None),
            "training_run_id": provenance.get("training_run_id", None),
            **{
                k: v
                for k, v in provenance.items()
                if k not in {"dataset_version", "model_version", "training_run_id"}
            },
        },
        "calibration": {
            "method": "none",
            "notes": "raw model probability; no post-hoc calibration applied",
        },
        "train_period": {
            "start": provenance.get("train_period_start", None),
            "end": provenance.get("train_period_end", None),
        },
        "git_commit": _get_git_commit_hash(),
        "row_count": int(len(payload)),
        "pairs": sorted(payload["pair"].unique().tolist()),
        "entry_time_min": (
            payload["entry_time"].min().isoformat()
            if not payload["entry_time"].isna().all()
            else None
        ),
        "entry_time_max": (
            payload["entry_time"].max().isoformat()
            if not payload["entry_time"].isna().all()
            else None
        ),
        "pair_stats": pair_stats,
        "signal_stats": {
            "pred_prob_up_mean": float(payload["pred_prob_up"].mean()),
            "pred_prob_up_std": float(payload["pred_prob_up"].std()),
            "signal_strength_mean": float(payload["signal_strength"].mean()),
            "signal_strength_std": float(payload["signal_strength"].std()),
            "feature_available_frac": float(payload["dl_feature_available"].mean()),
            "feature_missingness_frac": float(1.0 - payload["dl_feature_available"].mean()),
        },
        "warnings": warnings_list,
        "missing_provenance_counts": missing_provenance_counts,
    }


def _make_run_id(identity: dict) -> str:
    """Generate a run_id from identity + timestamp if not user-supplied."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = str(identity.get("model", "unknown")).replace(" ", "_")
    regime = str(identity.get("dl_regime", "unknown"))
    horizon = str(identity.get("target_horizon", "unk"))
    fset = str(identity.get("feature_set", "unknown")).replace(" ", "_")[:20]
    return f"{model}__{regime}__{horizon}__{fset}__{ts}"


def _attach_identity_columns(payload: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """
    Attach surface identity columns to each payload row.

    This is required so downstream consumers (market-phase-ml) can filter
    the artifact by exact-match on these row-level identity columns.
    """
    out = payload.copy()
    out["model"] = str(identity["model"])
    out["dl_regime"] = str(identity["dl_regime"])
    out["feature_set"] = str(identity["feature_set"])

    # Ensure target_horizon is numeric Int64 (nullable integer) for stable parquet schema.
    out["target_horizon"] = pd.array([int(identity["target_horizon"])] * len(out), dtype="Int64")
    return out


def _coerce_required_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce required column dtypes to ensure a stable parquet schema."""
    out = df.copy()

    # entry_time must be datetime64[ns] (tz-naive UTC)
    out["entry_time"] = _normalize_entry_time(out["entry_time"])

    # prediction_timestamp must survive as datetime64[ns] (v1 legacy)
    out["prediction_timestamp"] = _normalize_entry_time(out["prediction_timestamp"])

    # v2 timestamp columns: normalize to tz-naive datetime64[ns]
    for ts_col in [DL_AVAILABLE_TS_COL, DL_GENERATED_TS_COL, DL_ARTIFACT_CREATED_COL]:
        if ts_col in out.columns:
            out[ts_col] = _normalize_entry_time(out[ts_col])

    # numeric coercions
    out["pred_prob_up"] = pd.to_numeric(out["pred_prob_up"], errors="raise").astype("float64")
    out["signal_strength"] = pd.to_numeric(out["signal_strength"], errors="raise").astype("float64")

    # target_horizon must be nullable Int64
    out["target_horizon"] = pd.to_numeric(out["target_horizon"], errors="raise").astype("Int64")
    out["dl_feature_available"] = pd.to_numeric(
        out["dl_feature_available"], errors="raise"
    ).astype("Int64")
    for col in ["pred_prob_up_missing", "signal_strength_missing"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("Int64")

    return out


def _write_parquet_with_metadata(
    artifact_df: pd.DataFrame,
    parquet_path: Path,
    artifact_metadata: dict[str, Any],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(artifact_df, preserve_index=False)
        existing = dict(table.schema.metadata or {})
        extra = {
            f"msml.{k}".encode("utf-8"): str(v).encode("utf-8")
            for k, v in artifact_metadata.items()
        }
        table = table.replace_schema_metadata({**existing, **extra})
        pq.write_table(table, parquet_path)
    except Exception:
        artifact_df.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_dl_prediction_artifact(
    df: pd.DataFrame,
    identity: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    output_dir: Path = PREDICTIONS_DIR_DEFAULT,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """
    Validate and write a per-run DL prediction artifact.

    Each call writes:
    - ``{output_dir}/{run_id}.parquet``       — time-series payload
    - ``{output_dir}/{run_id}.manifest.json`` — identity + provenance

    Parameters
    ----------
    df:
        DataFrame with at minimum: ``entry_time``, ``pair``, ``pred_prob_up``.
        Optional columns: ``pred_direction``, ``confidence``,
        ``prediction_timestamp``.

        IMPORTANT (v1 downstream constraint):
        The written parquet MUST contain surface identity columns
        (model, dl_regime, target_horizon, feature_set) so that consumers can
        filter surfaces directly from parquet rows.
    identity:
        Required operational signal identity::

            {
                "model": "MLP",              # string
                "dl_regime": "LVTF",         # HVTF | LVTF | HVR | LVR
                "target_horizon": 24,        # int: number of bars
                "feature_set": "price_vol_sentiment",
            }

    provenance:
        Optional provenance / debug fields::

            {
                "dataset_version": "1.1.0",
                "model_version": "v1.0",
                "training_run_id": "run_20240115_abc",
                "train_period_start": "2018-01-01",  # used in train_period block
                "train_period_end": "2022-12-31",
                # any extra fields are stored as-is under provenance
            }

    output_dir:
        Directory where per-run artifacts are written.
        Default: ``data/output/dl_predictions``.
    run_id:
        Filename stem for the artifact pair.  Auto-generated from identity +
        timestamp if ``None``.

    Returns
    -------
    tuple[Path, Path]
        ``(parquet_path, manifest_path)``

    Raises
    ------
    ValueError
        If required columns are missing, ``pred_prob_up`` is out of range,
        or identity is invalid.
    """
    required = {"entry_time", "pair", "pred_prob_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}\n"
            f"Required: {sorted(required)}"
        )

    _validate_identity(identity)
    semantics_config = _resolve_semantics_config(provenance)

    if run_id is None:
        run_id = _make_run_id(identity)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{run_id}.parquet"
    manifest_path = output_dir / f"{run_id}.manifest.json"

    # Build payload (time-series columns)
    payload = _build_run_payload(df, semantics_config)

    # Attach identity columns (required for downstream surface filtering)
    artifact_df = _attach_identity_columns(payload, identity)

    # Attach v2 timestamp columns that are derived per-artifact (not per-row).
    # artifact_created_timestamp: wall-clock export time (same for all rows).
    artifact_created_ts = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    artifact_df[DL_ARTIFACT_CREATED_COL] = artifact_created_ts

    # prediction_generated_timestamp and prediction_available_timestamp
    # were already populated in _build_run_payload; carry them through.

    # Enforce stable deterministic schema & order
    artifact_df = _coerce_required_dtypes(artifact_df)

    missing_cols = [c for c in REQUIRED_PARQUET_COLS if c not in artifact_df.columns]
    if missing_cols:
        raise ValueError(f"Internal error: artifact missing required parquet columns: {missing_cols}")

    optional_missing_indicator_cols = [
        c for c in ["pred_prob_up_missing", "signal_strength_missing"] if c in artifact_df.columns
    ]
    artifact_df = artifact_df[REQUIRED_PARQUET_COLS + optional_missing_indicator_cols]

    # Fail-fast contract validation before any disk write.
    validate_dl_artifact(
        artifact_df,
        metadata={"schema_version": DL_SCHEMA_VERSION},
        strict=True,
    )

    # Debug print required by MPML integration troubleshooting
    print("artifact_columns:", sorted(artifact_df.columns.tolist()))

    # Compact artifact diagnostics for overlap debugging
    et_min = artifact_df["entry_time"].min()
    et_max = artifact_df["entry_time"].max()
    pairs_n = int(artifact_df["pair"].nunique())
    print("artifact_entry_time_range:")
    print(f"{et_min} -> {et_max}")
    print(f"artifact_row_count: {len(artifact_df):,}")
    print(f"artifact_unique_pairs: {pairs_n:,}")

    # Write parquet with metadata when possible
    artifact_metadata = {
        "export_timestamp": artifact_created_ts.isoformat(),  # legacy key
        DL_ARTIFACT_CREATED_COL: artifact_created_ts.isoformat(),
        "prediction_horizon_hours": int(identity["target_horizon"]),
        "feature_surface": str(identity["feature_set"]),
        "dl_regime": str(identity["dl_regime"]),
        "availability_semantics": (
            "visibility_by_control_mode"
            if semantics_config["control_mode"] != "normal"
            else "sparse_observed_only"
        ),
        "missing_indicator_mode": (
            "explicit_missing_indicators"
            if semantics_config["add_missing_indicators"]
            else "imputation_only"
        ),
        "imputation_mode": (
            "neutral_imputation"
            if semantics_config["impute_optional_features"]
            else "no_imputation"
        ),
    }
    _write_parquet_with_metadata(artifact_df, parquet_path, artifact_metadata)
    print(f"Wrote parquet:  {parquet_path.resolve()}")

    # Build and write manifest (unchanged behavior; uses payload-only DF)
    manifest = _build_run_manifest(
        payload,
        identity,
        provenance,
        run_id=run_id,
        parquet_filename=parquet_path.name,
        semantics_config=semantics_config,
    )
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote manifest: {manifest_path.resolve()}")

    return parquet_path, manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Write a per-run DL prediction artifact (Parquet + manifest).\n\n"
            "This is the producer-side API: call after each DL training / "
            "inference run to register row-level predictions.\n\n"
            "Use scripts/consolidate_dl_predictions.py to build the "
            "operational cube from all per-run artifacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help=(
            "CSV with time-series predictions. "
            "Required columns: entry_time, pair, pred_prob_up. "
            "Optional: pred_direction, confidence, prediction_timestamp."
        ),
    )
    # Identity (required)
    p.add_argument("--model", required=True, help="Model architecture / identifier.")
    p.add_argument(
        "--dl-regime",
        required=True,
        choices=sorted(VALID_DL_REGIMES),
        help="Producer-side regime: HVTF | LVTF | HVR | LVR.",
    )
    p.add_argument(
        "--target-horizon",
        type=int,
        required=True,
        help="Prediction horizon in bars (integer).",
    )
    p.add_argument(
        "--feature-set",
        required=True,
        help="Feature set identifier, e.g. price_vol_sentiment.",
    )
    # Optional provenance
    p.add_argument("--dataset-version", default=None, help="Dataset version string.")
    p.add_argument("--model-version", default=None, help="Model version / run ID.")
    p.add_argument("--training-run-id", default=None, help="Training run identifier.")
    p.add_argument(
        "--train-period-start",
        default=None,
        help="Training period start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--train-period-end",
        default=None,
        help="Training period end date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--control-mode",
        default=cfg.DL_EXPORT_CONTROL_MODE,
        choices=sorted(VALID_CONTROL_MODES),
        help="Deterministic availability control mode.",
    )
    p.add_argument(
        "--dl-add-missing-indicators",
        type=int,
        choices=[0, 1],
        default=1 if cfg.DL_ADD_MISSING_INDICATORS else 0,
        help="Mode A/B toggle: 1 adds *_missing controls, 0 disables them.",
    )
    p.add_argument(
        "--dl-impute-optional-features",
        type=int,
        choices=[0, 1],
        default=1 if cfg.DL_IMPUTE_OPTIONAL_FEATURES else 0,
        help="Whether optional/synthetic gaps are imputed deterministically.",
    )
    p.add_argument(
        "--dl-imputation-value",
        type=float,
        default=cfg.DL_IMPUTATION_VALUE,
        help="Neutral pred_prob_up value used for imputation.",
    )
    p.add_argument(
        "--availability-shuffle-seed",
        type=int,
        default=cfg.DL_AVAILABILITY_SHUFFLE_SEED,
        help="Seed for deterministic availability shuffling.",
    )
    # Output
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PREDICTIONS_DIR_DEFAULT,
        help=f"Output directory. Default: {PREDICTIONS_DIR_DEFAULT}",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=(
            "Filename stem for the artifact pair. "
            "Auto-generated from identity + timestamp if not supplied."
        ),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"Loaded {len(df):,} rows from {args.input_csv}")

    identity = {
        "model": args.model,
        "dl_regime": args.dl_regime,
        "target_horizon": args.target_horizon,
        "feature_set": args.feature_set,
    }
    provenance = {
        k: v
        for k, v in {
            "dataset_version": args.dataset_version,
            "model_version": args.model_version,
            "training_run_id": args.training_run_id,
            "train_period_start": args.train_period_start,
            "train_period_end": args.train_period_end,
            "control_mode": args.control_mode,
            "dl_add_missing_indicators": bool(args.dl_add_missing_indicators),
            "dl_impute_optional_features": bool(args.dl_impute_optional_features),
            "dl_imputation_value": args.dl_imputation_value,
            "availability_shuffle_seed": args.availability_shuffle_seed,
        }.items()
        if v is not None
    }

    pq_path, mf_path = write_dl_prediction_artifact(
        df=df,
        identity=identity,
        provenance=provenance or None,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    print(f"\nDone. run_id={pq_path.stem}")
    print(f"  Parquet:  {pq_path}")
    print(f"  Manifest: {mf_path}")

# Backwards-compatible aliases kept (as present in original file).
PIP_PREDICTIONS_DIR_DEFAULT = PREDICTIONS_DIR_DEFAULT
pipPREDICTIONS_DIR_DEFAULT = PREDICTIONS_DIR_DEFAULT
