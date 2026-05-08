"""
consolidate_dl_predictions.py
==============================
Consolidate all per-run DL prediction artifacts into the versioned
``dl_signals_h1_v1`` operational cube (Parquet + JSON manifest).

This is the **recommended** replacement for the deprecated
``scripts/build_dl_signal_artifact.py`` CSV consolidator.

Workflow
--------
1. DL training / inference writes per-run artifacts via
   ``scripts/write_dl_prediction_artifact.py``:

       data/output/dl_predictions/{run_id}.parquet
       data/output/dl_predictions/{run_id}.manifest.json

2. This script reads all ``.parquet`` + ``.manifest.json`` pairs, injects
   identity columns from each manifest into the corresponding parquet rows,
   and validates the resulting cube.

3. Writes the consolidated cube to:

       data/output/dl_signals/dl_signals_h1_v1.parquet
       data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json

Schema version:  dl_signals_h1_v1
See also:        docs/DL_SIGNAL_SCHEMA.md

Usage::

    python scripts/consolidate_dl_predictions.py \\
        [--input-dir data/output/dl_predictions] \\
        [--output-dir data/output/dl_signals]

Cube column contract
--------------------
Time-series payload (from per-run parquet):
    pair, entry_time, pred_prob_up, signal_strength,
    pred_direction, confidence, prediction_timestamp

Identity / provenance (injected from per-run manifest):
    model, dl_regime, target_horizon (Int64, bars), feature_set,
    dataset_version, model_version

Schema constant:
    schema_version

Uniqueness contract (enforced, hard fail on violation):
    (pair, entry_time, model, dl_regime, target_horizon, feature_set)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_dl_signal_artifact import (  # noqa: E402
    EXPORT_FREQUENCY,
    OUTPUT_COLS,
    SCHEMA_VERSION,
    SURFACE_GRAIN_COLS,
    UNIQUE_KEY_COLS,
    VALID_DL_REGIMES,
    _get_git_commit_hash,
    _run_qa,
    _write_manifest,
)

PREDICTIONS_DIR_DEFAULT = Path("data/output/dl_predictions")
SIGNALS_DIR_DEFAULT = Path("data/output/dl_signals")
OUTPUT_PARQUET_DEFAULT = SIGNALS_DIR_DEFAULT / "dl_signals_h1_v1.parquet"
OUTPUT_MANIFEST_DEFAULT = SIGNALS_DIR_DEFAULT / "DL_SIGNAL_MANIFEST_h1_v1.json"

# ---------------------------------------------------------------------------
# Loading per-run artifacts
# ---------------------------------------------------------------------------


def _find_run_artifact_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find all (parquet, manifest) pairs in *input_dir*.

    A pair is matched when a ``.parquet`` file has a corresponding
    ``.manifest.json`` with the same stem.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.
    ValueError
        If no valid pairs are found.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory not found: {input_dir}\n"
            "Create per-run artifacts first with "
            "scripts/write_dl_prediction_artifact.py."
        )

    parquets = sorted(input_dir.glob("*.parquet"))
    pairs = []
    orphan_parquets = []
    for pq in parquets:
        mf = pq.with_suffix(".manifest.json")
        if mf.exists():
            pairs.append((pq, mf))
        else:
            orphan_parquets.append(pq.name)

    if orphan_parquets:
        print(
            f"  ⚠ {len(orphan_parquets)} parquet file(s) have no matching "
            f".manifest.json and will be skipped: {orphan_parquets[:5]}"
        )

    if not pairs:
        raise ValueError(
            f"No valid (parquet + manifest.json) pairs found in: {input_dir}\n"
            "Write per-run artifacts first with "
            "scripts/write_dl_prediction_artifact.py."
        )

    return pairs


def _load_run_artifact(parquet_path: Path, manifest_path: Path) -> pd.DataFrame:
    """
    Load one per-run parquet, read its manifest, and inject identity columns.

    Parameters
    ----------
    parquet_path:
        Path to the per-run ``.parquet`` file.
    manifest_path:
        Path to the corresponding ``.manifest.json`` file.

    Returns
    -------
    pd.DataFrame
        Per-run rows enriched with identity + provenance columns from the
        manifest, plus ``schema_version``.

    Raises
    ------
    ValueError
        If the manifest is missing required identity fields or the parquet
        is missing required time-series columns.
    """
    # Read manifest
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    identity = manifest.get("identity")
    if not identity:
        raise ValueError(
            f"Manifest {manifest_path.name} is missing the 'identity' block. "
            "Only artifacts written by write_dl_prediction_artifact.py are supported."
        )
    required_identity = {"model", "dl_regime", "target_horizon", "feature_set"}
    missing_id = required_identity - set(identity.keys())
    if missing_id:
        raise ValueError(
            f"Manifest {manifest_path.name} identity block is missing keys: "
            f"{sorted(missing_id)}"
        )

    provenance = manifest.get("provenance", {}) or {}

    # Read parquet
    df = pd.read_parquet(parquet_path)

    required_payload_cols = {"entry_time", "pair", "pred_prob_up", "signal_strength"}
    missing_cols = required_payload_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Per-run parquet {parquet_path.name} is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    # Inject identity columns
    df["model"] = str(identity["model"])
    df["dl_regime"] = str(identity["dl_regime"])
    df["target_horizon"] = pd.array(
        [int(identity["target_horizon"])] * len(df), dtype="Int64"
    )
    df["feature_set"] = str(identity["feature_set"])

    # Inject optional provenance
    df["dataset_version"] = str(provenance.get("dataset_version") or "unknown")
    df["model_version"] = str(provenance.get("model_version") or "unknown")

    # Schema version
    df["schema_version"] = SCHEMA_VERSION

    return df


# ---------------------------------------------------------------------------
# Main consolidation function
# ---------------------------------------------------------------------------


def consolidate_dl_predictions(
    input_dir: Path = PREDICTIONS_DIR_DEFAULT,
    output_parquet: Path = OUTPUT_PARQUET_DEFAULT,
    output_manifest: Path = OUTPUT_MANIFEST_DEFAULT,
) -> pd.DataFrame:
    """
    Consolidate all per-run DL prediction artifacts into the ``dl_signals_h1_v1``
    cube.

    Parameters
    ----------
    input_dir:
        Directory containing per-run ``.parquet`` + ``.manifest.json`` pairs.
        Default: ``data/output/dl_predictions``.
    output_parquet:
        Path to write the consolidated cube Parquet.
    output_manifest:
        Path to write the consolidated cube manifest JSON.

    Returns
    -------
    pd.DataFrame
        The consolidated cube DataFrame.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.
    ValueError
        If no valid per-run artifact pairs are found.
    AssertionError
        If any cube QA invariant is violated.
    """
    print(f"Input directory:  {input_dir.resolve()}")

    # 1. Find artifact pairs
    pairs = _find_run_artifact_pairs(input_dir)
    print(f"Found {len(pairs):,} per-run artifact pair(s).")

    # 2. Load each artifact pair
    frames = []
    for pq_path, mf_path in pairs:
        df = _load_run_artifact(pq_path, mf_path)
        frames.append(df)
        print(f"  Loaded: {pq_path.name}  ({len(df):,} rows)")

    # 3. Concatenate
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined):,} rows across {len(frames):,} run(s).")

    # 4. Ensure output columns are present and in the right order
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Enforce dtypes
    combined["entry_time"] = pd.to_datetime(combined["entry_time"])
    combined["pred_prob_up"] = combined["pred_prob_up"].astype("float64")
    combined["signal_strength"] = combined["signal_strength"].astype("float64")
    if "pred_direction" in combined.columns:
        combined["pred_direction"] = pd.to_numeric(
            combined["pred_direction"], errors="coerce"
        ).astype("Int64")
    if "target_horizon" in combined.columns:
        combined["target_horizon"] = pd.to_numeric(
            combined["target_horizon"], errors="coerce"
        ).astype("Int64")
    if "prediction_timestamp" in combined.columns:
        combined["prediction_timestamp"] = pd.to_datetime(
            combined["prediction_timestamp"], errors="coerce"
        )
    for col in ["pair", "model", "dl_regime", "feature_set",
                "dataset_version", "model_version", "schema_version"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(str)

    cube = (
        combined[OUTPUT_COLS]
        .sort_values(["pair", "entry_time"])
        .reset_index(drop=True)
    )

    # 5. QA
    print("\nBuilding artifact …")
    _run_qa(cube)

    # 6. Write outputs
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    cube.to_parquet(output_parquet, index=False)
    print(f"\nWrote parquet:  {output_parquet.resolve()}")

    _write_manifest(cube, input_dir, output_manifest)
    print(f"Wrote manifest: {output_manifest.resolve()}")

    return cube


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Consolidate per-run DL prediction artifacts into the "
            "dl_signals_h1_v1 cube.\n\n"
            "Reads all (*.parquet + *.manifest.json) pairs from --input-dir "
            "and produces the consolidated operational cube.\n\n"
            "Per-run artifacts are written by "
            "scripts/write_dl_prediction_artifact.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=PREDICTIONS_DIR_DEFAULT,
        help=f"Directory with per-run artifacts. Default: {PREDICTIONS_DIR_DEFAULT}",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=SIGNALS_DIR_DEFAULT,
        help=f"Output directory for the cube. Default: {SIGNALS_DIR_DEFAULT}",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out_pq = args.output_dir / "dl_signals_h1_v1.parquet"
    out_mf = args.output_dir / "DL_SIGNAL_MANIFEST_h1_v1.json"

    result = consolidate_dl_predictions(
        input_dir=args.input_dir,
        output_parquet=out_pq,
        output_manifest=out_mf,
    )

    print(f"\nDone.  {len(result):,} rows  |  {result['pair'].nunique():,} pairs")
    print(
        f"entry_time range: {result['entry_time'].min()} → {result['entry_time'].max()}"
    )
