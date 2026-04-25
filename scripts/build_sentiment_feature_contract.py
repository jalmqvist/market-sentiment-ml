"""
build_sentiment_feature_contract.py
====================================
Generates the ``sentiment_features_h1_v1`` feature contract artifact.

Output artifacts:
    data/output/features/sentiment_features_h1_v1.parquet
    data/output/features/SENTIMENT_FEATURE_MANIFEST_h1_v1.json

Schema version:  sentiment_features_h1_v1
See also:        docs/SENTIMENT_FEATURE_SCHEMA.md

Usage::

    python build_sentiment_feature_contract.py

Key semantics
-------------
- The hourly grid is derived from FX price files.  ALL timestamps that are
  present in the price files for a given pair are used as ``entry_time``
  values; no manual filtering or hour-invention takes place.
- ``entry_time`` is the H1 bar open timestamp (UTC).
- ``snapshot_time`` is the final corrected UTC timestamp from the research
  dataset (non-hour-aligned).
- At each ``entry_time`` the features reflect the *latest* snapshot with
  ``snapshot_time <= entry_time`` (as-of / forward-fill semantics).
- No backward filling is applied.  If no prior snapshot exists for a pair
  at a given ``entry_time``, feature columns are null and ``has_snapshot``
  is False.
- ``sentiment_change`` is event-based (change between consecutive snapshots
  per pair) and is forward-filled onto the hourly grid.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sentiment_features_h1_v1"
STALE_HOURS_THRESHOLD = 24

PRICE_DIR = Path("data/input/fx")
SOURCE_DATASET = Path("data/output/master_research_dataset.csv")

OUTPUT_DIR = Path("data/output/features")
OUTPUT_PARQUET = OUTPUT_DIR / "sentiment_features_h1_v1.parquet"
OUTPUT_MANIFEST = OUTPUT_DIR / "SENTIMENT_FEATURE_MANIFEST_h1_v1.json"

# Re-use the same helpers from the main builder rather than duplicating them.
from build_fx_sentiment_dataset import (  # noqa: E402
    load_all_mt4_prices,
    get_git_commit_hash,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_group(pair: str) -> str:
    """Return the pair group label for a normalised pair string."""
    return "JPY_cross" if str(pair).lower().endswith("-jpy") else "non_JPY"


def _build_hourly_grid(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-pair hourly grid directly from the price file timestamps.

    Every timestamp present in the price data is kept as-is; no filtering,
    rounding, or invention of hours is performed.

    Returns a DataFrame with columns [pair, entry_time] sorted by
    (pair, entry_time).
    """
    grid = (
        prices[["pair", "timestamp"]]
        .rename(columns={"timestamp": "entry_time"})
        .drop_duplicates()
        .sort_values(["pair", "entry_time"])
        .reset_index(drop=True)
    )
    return grid


def _load_snapshot_events(source_path: Path) -> pd.DataFrame:
    """
    Load the FULL master research dataset and return the unique snapshot
    events with the sentiment feature columns needed for the contract.

    Duplicate (pair, snapshot_time) rows – which can occur when the same
    snapshot is matched to multiple entry bars in the event dataset – are
    deduplicated by keeping the last row (all feature values are identical
    for the same snapshot).
    """
    required_cols = [
        "pair",
        "snapshot_time",
        "net_sentiment",
        "abs_sentiment",
        "crowd_side",
        "extreme_70",
        "extreme_streak_70",
        "side_streak",
        "sentiment_change",
    ]

    df = pd.read_csv(source_path, usecols=required_cols, low_memory=False)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df = df.dropna(subset=["pair", "snapshot_time"])

    # De-duplicate: one row per (pair, snapshot_time) event
    df = (
        df.sort_values(["pair", "snapshot_time"])
        .drop_duplicates(subset=["pair", "snapshot_time"], keep="last")
        .reset_index(drop=True)
    )

    # sentiment_change is already event-based in the source dataset
    # (net_sentiment - prev net_sentiment per pair, computed in
    # load_all_sentiment_files).  We keep it as-is for forward-filling.

    return df


def _as_of_merge(grid: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    For each (pair, entry_time) in the grid, attach the latest snapshot
    with snapshot_time <= entry_time (as-of / LOCF semantics).

    No backward fill: if no prior snapshot exists, feature columns remain
    null and has_snapshot=False.
    """
    out_frames = []

    common_pairs = sorted(
        set(grid["pair"]).intersection(set(snapshots["pair"]))
    )

    for pair in common_pairs:
        g = grid.loc[grid["pair"] == pair].sort_values("entry_time").copy()
        s = (
            snapshots.loc[snapshots["pair"] == pair]
            .sort_values("snapshot_time")
            .copy()
        )

        # pd.merge_asof with direction="backward" gives us the latest
        # snapshot_time <= entry_time (as-of / LOCF).
        merged = pd.merge_asof(
            g,
            s.drop(columns=["pair"]),
            left_on="entry_time",
            right_on="snapshot_time",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["pair"] = pair
        out_frames.append(merged)

    # Pairs that exist only in grid (no snapshots) → keep with nulls
    grid_only_pairs = sorted(set(grid["pair"]) - set(snapshots["pair"]))
    for pair in grid_only_pairs:
        g = grid.loc[grid["pair"] == pair].copy()
        g["snapshot_time"] = pd.NaT
        out_frames.append(g)

    if not out_frames:
        return grid.assign(snapshot_time=pd.NaT)

    return pd.concat(out_frames, ignore_index=True)


def _add_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add snapshot_age_hours, has_snapshot, is_stale."""
    out = df.copy()

    has_snap = out["snapshot_time"].notna() & (
        out["snapshot_time"] <= out["entry_time"]
    )
    out["has_snapshot"] = has_snap

    age = (out["entry_time"] - out["snapshot_time"]).dt.total_seconds() / 3600.0
    out["snapshot_age_hours"] = np.where(has_snap, age, np.nan)
    out["snapshot_age_hours_int"] = np.where(
        has_snap,
        np.floor(out["snapshot_age_hours"]).astype("Int64"),
        pd.NA,
    )

    out["is_stale"] = has_snap & (out["snapshot_age_hours"] > STALE_HOURS_THRESHOLD)

    return out


def build_feature_contract(
    price_dir: Path = PRICE_DIR,
    source_dataset: Path = SOURCE_DATASET,
    output_parquet: Path = OUTPUT_PARQUET,
    output_manifest: Path = OUTPUT_MANIFEST,
) -> pd.DataFrame:
    """
    Build the ``sentiment_features_h1_v1`` feature contract and write
    output artifacts.

    Returns the feature DataFrame.
    """
    # ------------------------------------------------------------------
    # 1.  Load inputs
    # ------------------------------------------------------------------
    print("Loading price files …")
    prices = load_all_mt4_prices(price_dir)
    print(f"  Price rows: {len(prices):,}  |  Pairs: {prices['pair'].nunique():,}")

    print(f"Loading source dataset: {source_dataset} …")
    snapshots = _load_snapshot_events(source_dataset)
    print(f"  Snapshot events: {len(snapshots):,}  |  Pairs: {snapshots['pair'].nunique():,}")

    # ------------------------------------------------------------------
    # 2.  Build hourly grid from ALL price timestamps (per pair)
    # ------------------------------------------------------------------
    grid = _build_hourly_grid(prices)
    print(f"Hourly grid: {len(grid):,} rows across {grid['pair'].nunique():,} pairs")

    # ------------------------------------------------------------------
    # 3.  As-of merge: forward-fill snapshot state onto hourly grid
    # ------------------------------------------------------------------
    print("Running as-of merge (forward-fill, no backward fill) …")
    merged = _as_of_merge(grid, snapshots)

    # ------------------------------------------------------------------
    # 4.  Provenance columns
    # ------------------------------------------------------------------
    merged = _add_provenance_columns(merged)

    # ------------------------------------------------------------------
    # 5.  Structure columns
    # ------------------------------------------------------------------
    merged["pair_group"] = merged["pair"].map(_pair_group)
    merged["schema_version"] = SCHEMA_VERSION

    # ------------------------------------------------------------------
    # 6.  Select and order contract columns
    # ------------------------------------------------------------------
    # Note: rows with no prior snapshot (has_snapshot=False) already have
    # NaN in all feature columns from merge_asof; no explicit nulling is
    # needed here.
    contract_cols = [
        # Keys
        "schema_version",
        "pair",
        "entry_time",
        # Provenance
        "snapshot_time",
        "snapshot_age_hours",
        "snapshot_age_hours_int",
        "has_snapshot",
        "is_stale",
        # Core sentiment
        "net_sentiment",
        "abs_sentiment",
        "crowd_side",
        # Persistence
        "extreme_70",
        "extreme_streak_70",
        "side_streak",
        # Dynamics
        "sentiment_change",
        # Structure
        "pair_group",
    ]

    # Add any contract cols not yet present (defensive)
    for col in contract_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    result = merged[contract_cols].copy()

    # ------------------------------------------------------------------
    # 7.  Type enforcement
    # ------------------------------------------------------------------
    result["entry_time"] = pd.to_datetime(result["entry_time"])
    result["snapshot_time"] = pd.to_datetime(result["snapshot_time"])
    result["net_sentiment"] = pd.to_numeric(result["net_sentiment"], errors="coerce")
    result["abs_sentiment"] = pd.to_numeric(result["abs_sentiment"], errors="coerce")
    result["sentiment_change"] = pd.to_numeric(result["sentiment_change"], errors="coerce")
    result["crowd_side"] = result["crowd_side"].astype("Int64")
    result["extreme_70"] = result["extreme_70"].astype("boolean")
    result["extreme_streak_70"] = result["extreme_streak_70"].astype("Int64")
    result["side_streak"] = result["side_streak"].astype("Int64")
    result["snapshot_age_hours_int"] = result["snapshot_age_hours_int"].astype("Int64")
    result["has_snapshot"] = result["has_snapshot"].astype(bool)
    result["is_stale"] = result["is_stale"].astype(bool)

    result = result.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8.  QA checks
    # ------------------------------------------------------------------
    _run_qa(result)

    # ------------------------------------------------------------------
    # 9.  Write outputs
    # ------------------------------------------------------------------
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_parquet, index=False)
    print(f"\nWrote parquet:  {output_parquet.resolve()}")

    _write_manifest(result, source_dataset, output_manifest)
    print(f"Wrote manifest: {output_manifest.resolve()}")

    return result


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------


def _run_qa(df: pd.DataFrame) -> None:
    """Run minimal QA checks and print a summary."""
    print("\n=== QA checks ===")

    # 1) Uniqueness of (pair, entry_time)
    dups = df.duplicated(subset=["pair", "entry_time"]).sum()
    status = "✓" if dups == 0 else f"✗ {dups:,} duplicates"
    print(f"  (pair, entry_time) uniqueness: {status}")
    if dups > 0:
        raise AssertionError(
            f"Contract violation: {dups:,} duplicate (pair, entry_time) rows found."
        )

    # 2) Causality: snapshot_time <= entry_time when has_snapshot
    snap_rows = df[df["has_snapshot"]]
    violations = (snap_rows["snapshot_time"] > snap_rows["entry_time"]).sum()
    status = "✓" if violations == 0 else f"✗ {violations:,} violations"
    print(f"  snapshot_time <= entry_time causality: {status}")
    if violations > 0:
        raise AssertionError(
            f"Contract violation: {violations:,} rows where snapshot_time > entry_time."
        )

    # 3) Null-rate summary
    total = len(df)
    null_snap = (~df["has_snapshot"]).sum()
    stale = df["is_stale"].sum()
    print(f"  Total rows:       {total:,}")
    print(f"  No snapshot:      {null_snap:,}  ({100*null_snap/total:.1f}%)")
    print(f"  Stale (>{STALE_HOURS_THRESHOLD}h):   {stale:,}  ({100*stale/total:.1f}%)")

    # 4) Per-pair staleness summary
    print("\n  Per-pair staleness summary:")
    summary = (
        df.groupby("pair")
        .agg(
            rows=("entry_time", "count"),
            no_snapshot=("has_snapshot", lambda x: (~x).sum()),
            stale=("is_stale", "sum"),
        )
        .assign(
            pct_no_snapshot=lambda d: (100 * d["no_snapshot"] / d["rows"]).round(1),
            pct_stale=lambda d: (100 * d["stale"] / d["rows"]).round(1),
        )
    )
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    df: pd.DataFrame,
    source_dataset: Path,
    output_manifest: Path,
) -> None:
    """Write the JSON manifest for the feature contract artifact."""

    # Per-pair stats
    pair_stats: dict = {}
    for pair, grp in df.groupby("pair"):
        et = grp["entry_time"].dropna()
        pair_stats[str(pair)] = {
            "row_count": int(len(grp)),
            "entry_time_min": et.min().isoformat() if len(et) else None,
            "entry_time_max": et.max().isoformat() if len(et) else None,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": {
            "path": str(source_dataset),
            "schema_version": "1.0",
        },
        "grid": {
            "description": (
                "Hourly grid derived from FX price files.  "
                "All timestamps present in the price files are used as "
                "entry_time; no manual filtering or hour-invention."
            ),
            "price_dir": str(PRICE_DIR),
        },
        "semantics": {
            "as_of_rule": "latest snapshot_time <= entry_time per pair",
            "fill_rule": "forward-fill per pair; no backward fill",
            "first_snapshot_sentiment_change": "null (no prior snapshot)",
        },
        "stale_hours_threshold": STALE_HOURS_THRESHOLD,
        "git_commit": get_git_commit_hash(),
        "total_rows": int(len(df)),
        "total_pairs": int(df["pair"].nunique()),
        "overall_entry_time_min": df["entry_time"].min().isoformat(),
        "overall_entry_time_max": df["entry_time"].max().isoformat(),
        "pair_stats": pair_stats,
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = build_feature_contract()
    print(f"\nDone.  {len(result):,} rows  |  {result['pair'].nunique():,} pairs")
    print(f"entry_time range: {result['entry_time'].min()} → {result['entry_time'].max()}")
