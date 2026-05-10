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

Schema version:  dl_signals_h1_v1
See also:        docs/DL_SIGNAL_SCHEMA.md

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

from build_dl_signal_artifact import (  # noqa: E402
    EXPORT_FREQUENCY,
    SCHEMA_VERSION,
    VALID_DL_REGIMES,
    _get_git_commit_hash,
    _normalize_entry_time,
    _normalize_pair,
)

# Per-run parquet: time-series payload only (no identity columns)
RUN_PARQUET_COLS = [
    "pair",
    "entry_time",
    "pred_prob_up",
    "signal_strength",
    "pred_direction",
    "confidence",
    "prediction_timestamp",
]

PREDICTIONS_DIR_DEFAULT = Path("data/output/dl_predictions")

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _build_run_payload(df: pd.DataFrame) -> pd.DataFrame:
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

    # prediction_timestamp: optional per-row inference timestamp
    if "prediction_timestamp" not in result.columns:
        result["prediction_timestamp"] = pd.NaT
    else:
        result["prediction_timestamp"] = _normalize_entry_time(
            result["prediction_timestamp"]
        )

    # Add any missing columns as NaN
    for col in RUN_PARQUET_COLS:
        if col not in result.columns:
            result[col] = np.nan

    return result[RUN_PARQUET_COLS].sort_values(["pair", "entry_time"]).reset_index(
        drop=True
    )


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

    return {
        "schema_version": SCHEMA_VERSION,
        "export_frequency": EXPORT_FREQUENCY,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
        Identity columns (``model``, ``dl_regime``, etc.) must NOT be present
        in ``df``; they are provided via ``identity`` and ``provenance``.
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

    if run_id is None:
        run_id = _make_run_id(identity)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{run_id}.parquet"
    manifest_path = output_dir / f"{run_id}.manifest.json"

    # Build payload
    payload = _build_run_payload(df)

    # Write parquet
    payload.to_parquet(parquet_path, index=False)
    print(f"Wrote parquet:  {parquet_path.resolve()}")

    # Build and write manifest
    manifest = _build_run_manifest(
        payload,
        identity,
        provenance,
        run_id=run_id,
        parquet_filename=parquet_path.name,
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

PIP_PREDICTIONS_DIR_DEFAULT = PREDICTIONS_DIR_DEFAULT
pipPREDICTIONS_DIR_DEFAULT = PREDICTIONS_DIR_DEFAULT