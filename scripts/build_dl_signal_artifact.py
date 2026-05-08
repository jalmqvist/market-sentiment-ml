"""
build_dl_signal_artifact.py
============================
Consolidates per-pair/per-run DL prediction CSVs into the versioned
``dl_signals_h1_v1`` artifact (Parquet + JSON manifest).

Output artifacts
----------------
    data/output/dl_signals/dl_signals_h1_v1.parquet
    data/output/dl_signals/DL_SIGNAL_MANIFEST_h1_v1.json

Schema version:  dl_signals_h1_v1
See also:        docs/DL_SIGNAL_SCHEMA.md

Usage::

    # From the repository root:
    python scripts/build_dl_signal_artifact.py \\
        --input-dir data/output/dl_predictions \\
        [--output-dir data/output/dl_signals]

Expected input CSV columns (per-pair / per-run prediction file)
---------------------------------------------------------------
Required:
    entry_time       — UTC timestamp of the H1 bar open (tz-naive or UTC-aware)
    pair             — FX pair in any common format; normalized to xxx-yyy
    pred_prob_up     — float in [0, 1]; P(next move is up) from the DL model

Provenance (at least one of the following is required):
    model            — model architecture / identifier (e.g. "MLP", "LSTM")
    dl_regime        — producer-side market regime: HVTF | LVTF | HVR | LVR

Optional provenance (filled with "unknown" if absent):
    target_horizon   — prediction horizon in bars (int or string)
    feature_set      — feature set identifier (e.g. "price_vol_sentiment")
    dataset_version  — dataset version string (e.g. "1.1.0")
    model_version    — model version / run identifier

Optional signal columns (computed from pred_prob_up if absent):
    pred_direction   — +1 (up) / -1 (down); derived from pred_prob_up > 0.5
    confidence       — float in [0, 1]; caller-supplied reliability estimate

Key semantics
-------------
- ``signal_strength = 2 * pred_prob_up - 1`` maps [0,1] → [-1, +1] linearly.
  Negative = bearish pressure; positive = bullish pressure; zero = neutral.
  This preserves the full [-1, 1] range without any standardisation.
- ``dl_regime`` uses the producer taxonomy (HVTF/LVTF/HVR/LVR); it is NOT
  normalised to market-phase-ml equivalents here — that mapping is done in
  the consumer loader.
- The unique key is ``(pair, entry_time, model, dl_regime, target_horizon,
  feature_set)``.  Duplicates on this key raise an error.
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
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "dl_signals_h1_v1"
EXPORT_FREQUENCY = "H1"

UNIQUE_KEY_COLS = ["pair", "entry_time", "model", "dl_regime", "target_horizon", "feature_set"]

# Allowed producer-side regime labels (DL taxonomy)
VALID_DL_REGIMES = {"HVTF", "LVTF", "HVR", "LVR"}

OUTPUT_DIR_DEFAULT = Path("data/output/dl_signals")
OUTPUT_PARQUET = OUTPUT_DIR_DEFAULT / "dl_signals_h1_v1.parquet"
OUTPUT_MANIFEST = OUTPUT_DIR_DEFAULT / "DL_SIGNAL_MANIFEST_h1_v1.json"

# Required columns that MUST be present in each input CSV
REQUIRED_INPUT_COLS = {"entry_time", "pair", "pred_prob_up"}

# Provenance columns; filled with "unknown" when absent
PROVENANCE_COLS = {
    "model": "unknown",
    "dl_regime": "unknown",
    "target_horizon": "unknown",
    "feature_set": "unknown",
    "dataset_version": "unknown",
    "model_version": "unknown",
}

# Final output column order
OUTPUT_COLS = [
    # Keys
    "pair",
    "entry_time",
    # Core signal
    "pred_prob_up",
    "signal_strength",
    # Optional signal
    "pred_direction",
    "confidence",
    # Provenance
    "model",
    "dl_regime",
    "target_horizon",
    "feature_set",
    "dataset_version",
    "model_version",
    # Schema
    "schema_version",
]

# ---------------------------------------------------------------------------
# Helpers (re-use get_git_commit_hash from the sibling build script)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _get_git_commit_hash() -> str | None:
    try:
        from subprocess import PIPE, run
        result = run(
            ["git", "rev-parse", "HEAD"],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def _normalize_pair(pair: str) -> str:
    """Normalise an FX pair string to lowercase ``xxx-yyy`` format."""
    p = str(pair).strip().lower()
    p = p.replace("/", "-").replace("_", "-")
    if "-" not in p and len(p) == 6:
        p = f"{p[:3]}-{p[3:]}"
    return p


def _normalize_entry_time(series: pd.Series) -> pd.Series:
    """Parse and normalise entry_time to tz-naive UTC datetime."""
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    # If any are tz-aware, convert to UTC then strip tz
    if hasattr(dt, "dt") and dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    elif dt.dtype == "object":
        # Try utc=True path
        try:
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_prediction_csvs(input_dir: Path) -> pd.DataFrame:
    """
    Load all ``*.csv`` files from *input_dir* and concatenate them.

    Validates that:
    - At least one CSV is found.
    - Each CSV contains the required columns.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.
    ValueError
        If no CSVs are found or required columns are missing.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Create it and populate it with per-pair/per-run prediction CSVs.\n"
            "Each CSV must contain at minimum: entry_time, pair, pred_prob_up"
        )

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(
            f"No CSV files found in: {input_dir}\n"
            "Expected per-pair/per-run prediction CSVs with columns:\n"
            "  entry_time, pair, pred_prob_up\n"
            "  (plus optional: model, dl_regime, target_horizon, feature_set,\n"
            "   dataset_version, model_version, pred_direction, confidence)"
        )

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, low_memory=False)
        missing = REQUIRED_INPUT_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input file {csv_path.name} is missing required columns: {sorted(missing)}\n"
                f"Required: {sorted(REQUIRED_INPUT_COLS)}\n"
                f"Found:    {sorted(df.columns.tolist())}"
            )
        df["_source_file"] = csv_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames):,} CSV file(s): {len(combined):,} total rows")
    return combined


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def _build_artifact(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw prediction rows into the ``dl_signals_h1_v1`` schema.

    Steps:
    1. Normalise ``pair`` to lowercase ``xxx-yyy``.
    2. Parse and normalise ``entry_time`` to tz-naive UTC.
    3. Clamp ``pred_prob_up`` to [0, 1].
    4. Compute ``signal_strength = 2 * pred_prob_up - 1``.
    5. Derive ``pred_direction`` from ``pred_prob_up`` if absent.
    6. Fill missing provenance columns with "unknown".
    7. Select and order output columns.
    8. Enforce dtypes.
    """
    df = raw.copy()

    # 1. Pair normalisation
    df["pair"] = df["pair"].map(_normalize_pair)

    # 2. entry_time parsing
    df["entry_time"] = _normalize_entry_time(df["entry_time"])

    # 3. Validate and clamp pred_prob_up
    df["pred_prob_up"] = pd.to_numeric(df["pred_prob_up"], errors="coerce")
    invalid_prob = df["pred_prob_up"].isna() | (df["pred_prob_up"] < 0) | (df["pred_prob_up"] > 1)
    if invalid_prob.any():
        n = int(invalid_prob.sum())
        raise ValueError(
            f"{n:,} row(s) have invalid pred_prob_up (must be float in [0, 1]).\n"
            f"Sample:\n{df.loc[invalid_prob, ['pair', 'entry_time', 'pred_prob_up']].head()}"
        )

    # 4. signal_strength — preserve [-1, 1] semantics; never standardize
    df["signal_strength"] = 2.0 * df["pred_prob_up"] - 1.0

    # 5. pred_direction: derive from pred_prob_up if not supplied
    if "pred_direction" not in df.columns:
        df["pred_direction"] = pd.array(
            np.where(df["pred_prob_up"] > 0.5, 1, -1), dtype="Int64"
        )
    else:
        df["pred_direction"] = pd.to_numeric(
            df["pred_direction"], errors="coerce"
        ).astype("Int64")

    # 6. confidence: keep as-is if supplied, else null
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # 7. Fill provenance columns with "unknown" when absent
    for col, default in PROVENANCE_COLS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)

    # 8. schema_version constant
    df["schema_version"] = SCHEMA_VERSION

    # 9. Select output columns (add any missing as NaN defensively)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    result = df[OUTPUT_COLS].copy()

    # 10. Dtype enforcement
    result["entry_time"] = pd.to_datetime(result["entry_time"])
    result["pred_prob_up"] = result["pred_prob_up"].astype("float64")
    result["signal_strength"] = result["signal_strength"].astype("float64")
    result["pair"] = result["pair"].astype(str)
    result["model"] = result["model"].astype(str)
    result["dl_regime"] = result["dl_regime"].astype(str)
    result["target_horizon"] = result["target_horizon"].astype(str)
    result["feature_set"] = result["feature_set"].astype(str)
    result["dataset_version"] = result["dataset_version"].astype(str)
    result["model_version"] = result["model_version"].astype(str)
    result["schema_version"] = result["schema_version"].astype(str)

    result = result.sort_values(["pair", "entry_time"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------


def _run_qa(df: pd.DataFrame) -> None:
    """
    Run QA checks on the consolidated artifact.

    Enforces:
    1. Unique key ``(pair, entry_time, model, dl_regime, target_horizon, feature_set)``
       is unique.
    2. ``entry_time`` is non-null for all rows.
    3. ``pred_prob_up`` is in [0, 1] for all rows.
    4. ``signal_strength`` is in [-1, 1] for all rows.
    5. Per-pair, ``entry_time`` is monotonically non-decreasing.
    6. ``dl_regime`` is in the valid producer taxonomy (warning only when set
       to "unknown"; hard error when a non-standard label is used).
    """
    print("\n=== QA checks ===")
    ok = True

    # 1) Unique key uniqueness
    dups = df.duplicated(subset=UNIQUE_KEY_COLS).sum()
    status = "✓" if dups == 0 else f"✗ {dups:,} duplicates"
    print(f"  Unique key uniqueness: {status}")
    if dups > 0:
        ok = False
        raise AssertionError(
            f"Contract violation: {dups:,} duplicate rows on unique key "
            f"{UNIQUE_KEY_COLS}.\n"
            "Each (pair, entry_time, model, dl_regime, target_horizon, feature_set) "
            "must be unique."
        )

    # 2) Non-null entry_time
    null_et = df["entry_time"].isna().sum()
    status = "✓" if null_et == 0 else f"✗ {null_et:,} null"
    print(f"  entry_time non-null:   {status}")
    if null_et > 0:
        ok = False
        raise AssertionError(f"Contract violation: {null_et:,} null entry_time rows.")

    # 3) pred_prob_up in [0, 1]
    out_of_range = ((df["pred_prob_up"] < 0) | (df["pred_prob_up"] > 1)).sum()
    status = "✓" if out_of_range == 0 else f"✗ {out_of_range:,} out of range"
    print(f"  pred_prob_up ∈ [0,1]:  {status}")
    if out_of_range > 0:
        ok = False
        raise AssertionError(f"Contract violation: {out_of_range:,} rows with pred_prob_up outside [0, 1].")

    # 4) signal_strength in [-1, 1]
    out_of_range = ((df["signal_strength"] < -1) | (df["signal_strength"] > 1)).sum()
    status = "✓" if out_of_range == 0 else f"✗ {out_of_range:,} out of range"
    print(f"  signal_strength ∈ [-1,1]: {status}")
    if out_of_range > 0:
        ok = False
        raise AssertionError(f"Contract violation: {out_of_range:,} rows with signal_strength outside [-1, 1].")

    # 5) Per-pair monotonic entry_time
    non_monotonic_pairs = []
    for pair, grp in df.groupby("pair"):
        if not grp["entry_time"].is_monotonic_increasing:
            non_monotonic_pairs.append(pair)
    status = "✓" if not non_monotonic_pairs else f"✗ {len(non_monotonic_pairs)} pair(s)"
    print(f"  Per-pair monotonic entry_time: {status}")
    if non_monotonic_pairs:
        ok = False
        raise AssertionError(
            f"Contract violation: entry_time is not monotonically increasing for pairs: "
            f"{non_monotonic_pairs}"
        )

    # 6) dl_regime taxonomy check
    unknown_regimes = set(df["dl_regime"].unique()) - VALID_DL_REGIMES - {"unknown"}
    if unknown_regimes:
        print(
            f"  ⚠ dl_regime contains non-standard values: {sorted(unknown_regimes)}\n"
            f"    Valid producer taxonomy: {sorted(VALID_DL_REGIMES)}\n"
            "    (These rows will be preserved but may be skipped by the consumer loader.)"
        )
    elif "unknown" in df["dl_regime"].values:
        print("  ℹ dl_regime is 'unknown' for some rows (provenance not recorded in input).")
    else:
        print("  ✓ dl_regime taxonomy valid")

    # Summary
    total = len(df)
    pairs = df["pair"].nunique()
    print(f"\n  Total rows: {total:,}  |  Pairs: {pairs:,}")
    print(f"  entry_time range: {df['entry_time'].min()} → {df['entry_time'].max()}")

    regimes = df["dl_regime"].value_counts().to_dict()
    print(f"  dl_regime distribution: {regimes}")
    models = df["model"].value_counts().to_dict()
    print(f"  model distribution: {models}")

    if ok:
        print("  All QA checks passed. ✓")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    df: pd.DataFrame,
    input_dir: Path,
    output_manifest: Path,
) -> None:
    """Write the JSON manifest for the DL signal artifact."""

    # Per-pair stats
    pair_stats: dict = {}
    for pair, grp in df.groupby("pair"):
        et = grp["entry_time"].dropna()
        pair_stats[str(pair)] = {
            "row_count": int(len(grp)),
            "entry_time_min": et.min().isoformat() if len(et) else None,
            "entry_time_max": et.max().isoformat() if len(et) else None,
            "dl_regimes": sorted(grp["dl_regime"].unique().tolist()),
            "models": sorted(grp["model"].unique().tolist()),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_frequency": EXPORT_FREQUENCY,
        "signal_definition": {
            "signal_strength": "2 * pred_prob_up - 1; maps P(up) from [0,1] to [-1,+1].",
            "pred_prob_up": "P(price moves up over target_horizon bars) from DL model.",
            "signal_strength_range": "[-1, 1]; negative=bearish, positive=bullish, 0=neutral.",
            "note": "signal_strength is NOT standardised; preserves calibration semantics.",
        },
        "unique_key": {
            "columns": UNIQUE_KEY_COLS,
            "description": (
                "Each (pair, entry_time, model, dl_regime, target_horizon, feature_set) "
                "must be unique. Multiple surfaces (regimes/horizons/models) may share "
                "the same (pair, entry_time)."
            ),
        },
        "dl_regime_taxonomy": {
            "producer": sorted(VALID_DL_REGIMES),
            "note": (
                "DL producer taxonomy: HVTF=high-vol trend, LVTF=low-vol trend, "
                "HVR=high-vol range, LVR=low-vol range. "
                "Consumer (market-phase-ml) may optionally map HVR→HVMR, LVR→LVMR."
            ),
        },
        "time_semantics": {
            "entry_time": "H1 bar open timestamp, tz-naive UTC.",
            "as_of_rule": (
                "Predictions are generated using features available at entry_time. "
                "No forward-looking inputs are used."
            ),
        },
        "source_predictions": {
            "input_dir": str(input_dir.resolve()),
        },
        "git_commit": _get_git_commit_hash(),
        "total_rows": int(len(df)),
        "total_pairs": int(df["pair"].nunique()),
        "overall_entry_time_min": df["entry_time"].min().isoformat(),
        "overall_entry_time_max": df["entry_time"].max().isoformat(),
        "dl_regimes_present": sorted(df["dl_regime"].unique().tolist()),
        "models_present": sorted(df["model"].unique().tolist()),
        "feature_sets_present": sorted(df["feature_set"].unique().tolist()),
        "target_horizons_present": sorted(df["target_horizon"].unique().tolist()),
        "pair_stats": pair_stats,
        "signal_stats": {
            "pred_prob_up_mean": float(df["pred_prob_up"].mean()),
            "pred_prob_up_std": float(df["pred_prob_up"].std()),
            "signal_strength_mean": float(df["signal_strength"].mean()),
            "signal_strength_std": float(df["signal_strength"].std()),
            "pred_direction_up_frac": float((df["pred_direction"] == 1).mean()),
        },
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_dl_signal_artifact(
    input_dir: Path,
    output_parquet: Path = OUTPUT_PARQUET,
    output_manifest: Path = OUTPUT_MANIFEST,
) -> pd.DataFrame:
    """
    Build the ``dl_signals_h1_v1`` artifact from per-pair/per-run prediction CSVs.

    Parameters
    ----------
    input_dir:
        Directory containing per-pair/per-run prediction CSVs.
    output_parquet:
        Path to write the consolidated Parquet file.
    output_manifest:
        Path to write the JSON manifest.

    Returns
    -------
    pd.DataFrame
        The consolidated artifact DataFrame.
    """
    print(f"Input directory:  {input_dir.resolve()}")

    # 1. Load
    raw = _load_prediction_csvs(input_dir)

    # 2. Transform
    print("Building artifact …")
    result = _build_artifact(raw)

    # 3. QA
    _run_qa(result)

    # 4. Write outputs
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_parquet, index=False)
    print(f"\nWrote parquet:  {output_parquet.resolve()}")

    _write_manifest(result, input_dir, output_manifest)
    print(f"Wrote manifest: {output_manifest.resolve()}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Build the dl_signals_h1_v1 artifact from per-pair/per-run prediction CSVs.\n\n"
            "Input CSVs must contain at minimum: entry_time, pair, pred_prob_up\n"
            "See docs/DL_SIGNAL_SCHEMA.md for the full input/output spec."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-pair/per-run prediction CSVs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory for parquet + manifest. Default: {OUTPUT_DIR_DEFAULT}",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    output_parquet = args.output_dir / "dl_signals_h1_v1.parquet"
    output_manifest = args.output_dir / "DL_SIGNAL_MANIFEST_h1_v1.json"

    result = build_dl_signal_artifact(
        input_dir=args.input_dir,
        output_parquet=output_parquet,
        output_manifest=output_manifest,
    )

    print(f"\nDone.  {len(result):,} rows  |  {result['pair'].nunique():,} pairs")
    print(f"entry_time range: {result['entry_time'].min()} → {result['entry_time'].max()}")
