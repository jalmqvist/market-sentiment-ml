#!/usr/bin/env python3
"""
attach_regimes_to_h1_dataset.py
===============================

Join D1 regime labels (from market-phase-ml) onto an H1 research dataset.

Inputs
------
1) H1 dataset (CSV): data/output/master_research_dataset.csv
   Must contain: pair, entry_time (UTC)

2) Regime dataset (Parquet): data/input/regimes/phase_labels_d1.parquet
   Must contain: pair, timestamp (UTC midnight), phase, is_trending, is_high_vol

Join contract (NO leakage)
--------------------------
- date_utc = floor(entry_time to UTC day)
- join on: (pair, date_utc) == (pair, timestamp)
- No forward-fill, no shifting, no tolerance joins

Transparency-only policy
------------------------
- Missing regimes remain NaN
- We do not fill or invent regimes for missing vendor days

Outputs
-------
- data/output/master_research_dataset_with_regime.csv

Diagnostics (mandatory)
-----------------------
- total_rows, matched_rows, match_rate
- warn if match_rate < 99%
- % missing by pair
- sample rows with missing phase
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# -----------------------------
# Defaults (sourced from config)
# -----------------------------

REGIME_INPUT_DEFAULT = cfg.REGIME_PARQUET_DEFAULT
OUTPUT_DEFAULT = cfg.DATA_PATH_REGIME

VALID_PHASES = cfg.VALID_PHASES

# Warning thresholds (do NOT fail on violations; warn only)
WARN_MATCH_RATE_BELOW: float = 0.95
WARN_MISSING_RATE_ABOVE: float = 0.10
WARN_PAIR_COVERAGE_BELOW: float = 0.50


# -----------------------------
# Helpers
# -----------------------------

def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns: {missing}")


def _ensure_utc_series(s: pd.Series, ctx: str) -> pd.Series:
    """
    Ensure a datetime series is tz-aware UTC.
    - If tz-naive, interpret as UTC (common for CSV parse_dates).
    - If tz-aware, convert to UTC.
    """
    t = pd.to_datetime(s, errors="raise")
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize("UTC")
    else:
        t = t.dt.tz_convert("UTC")
    return t


def _log_missing_samples(df: pd.DataFrame, n: int = 10) -> None:
    missing = df[df["phase"].isna()]
    if missing.empty:
        logger.info("No missing regime rows.")
        return

    cols = ["pair", "entry_time", "date_utc"]
    for extra in ["snapshot_time", "abs_sentiment"]:
        if extra in missing.columns:
            cols.append(extra)

    logger.info("Sample rows with missing phase (n=%d):\n%s", min(n, len(missing)),
                missing[cols].head(n).to_string(index=False))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Attach D1 regime labels to H1 research dataset.")
    p.add_argument("--data", required=True, type=Path, help="H1 master dataset CSV (canonical input)")
    p.add_argument("--regimes", type=Path, default=REGIME_INPUT_DEFAULT, help="D1 regimes parquet")
    p.add_argument("--out", type=Path, default=OUTPUT_DEFAULT, help="Output enriched CSV")
    p.add_argument("--warn-threshold", type=float, default=WARN_MATCH_RATE_BELOW, help="Warn if match rate below this")
    p.add_argument("--missing-sample-n", type=int, default=10, help="Number of missing-phase rows to log")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not args.data.exists():
        raise FileNotFoundError(f"H1 dataset not found: {args.data.resolve()}")
    if not args.regimes.exists():
        raise FileNotFoundError(f"Regime dataset not found: {args.regimes.resolve()}")

    # --- Load H1 dataset ---
    logger.info("Loading H1 dataset: %s", args.data)
    h1 = pd.read_csv(args.data, parse_dates=["entry_time"])
    _require_cols(h1, ["pair", "entry_time"], "H1 dataset")

    rows_before_collapse = len(h1)
    logger.info("H1 dataset loaded: %d rows before collapse", rows_before_collapse)

    # ------------------------------------------------------------------
    # The H1 dataset is event-level (multiple sentiment snapshots per bar).
    # We collapse to one row per (pair, entry_time) using the last snapshot
    # to produce a bar-level dataset suitable for joining with D1 regimes.
    # ------------------------------------------------------------------

    h1 = h1.copy()

    # Strict UTC handling
    h1["entry_time"] = _ensure_utc_series(h1["entry_time"], "H1.entry_time")

    if "snapshot_time" not in h1.columns:
        raise ValueError(
            "H1 dataset is event-level but 'snapshot_time' is missing; cannot collapse to bar-level deterministically."
        )

    # Ensure snapshot_time is parsed and UTC for deterministic ordering
    h1["snapshot_time"] = _ensure_utc_series(h1["snapshot_time"], "H1.snapshot_time")

    # Detect duplicates at bar grain
    dup = int(h1.duplicated(subset=["pair", "entry_time"]).sum())
    if dup > 0:
        logger.info("H1 dataset is event-level: found %d duplicate (pair, entry_time) rows. Collapsing to bar-level...", dup)

        before = len(h1)
        h1 = (
            h1.sort_values("snapshot_time")
              .groupby(["pair", "entry_time"], as_index=False)
              .last()
        )
        after = len(h1)
        logger.info("Collapsed to last snapshot per bar: %d -> %d rows.", before, after)

    # Assert no duplicates remain after collapse (hard requirement now)
    dup_after = int(h1.duplicated(subset=["pair", "entry_time"]).sum())
    if dup_after > 0:
        raise ValueError(f"Collapse failed: still have duplicate (pair, entry_time) rows: {dup_after:,}")

    # Derive join key (UTC midnight boundary)
    h1["date_utc"] = h1["entry_time"].dt.floor("D")  # tz-aware UTC midnight

    # --- Load regime dataset ---
    logger.info("Loading regime dataset: %s", args.regimes)
    regimes = pd.read_parquet(args.regimes)

    _require_cols(regimes, ["pair", "timestamp", "phase", "is_trending", "is_high_vol"], "Regime dataset")

    regimes = regimes.copy()
    regimes["timestamp"] = _ensure_utc_series(regimes["timestamp"], "regimes.timestamp")

    # Hard check: regimes timestamp aligned to 00:00 UTC (contract)
    ts = regimes["timestamp"]
    aligned = (
        (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)
    )
    if not bool(aligned.all()):
        bad_n = int((~aligned).sum())
        raise ValueError(f"Regime dataset timestamp is not aligned to 00:00 UTC for {bad_n} rows")

    # Hard check: no duplicates in regimes key
    dup_reg = regimes.duplicated(subset=["pair", "timestamp"]).sum()
    if dup_reg > 0:
        raise ValueError(f"Regime dataset has duplicate (pair, timestamp) rows: {dup_reg}")

    # Hard check: phase values valid (NaNs not expected in regimes file)
    bad_phases = sorted(set(regimes["phase"].dropna().astype(str)) - VALID_PHASES)
    if bad_phases:
        raise ValueError(f"Regime dataset has invalid phase values: {bad_phases}")

    # Reduce to join columns only (prevents accidental column collisions)
    regimes_keyed = regimes[["pair", "timestamp", "phase", "is_trending", "is_high_vol"]].copy()

    # --- Pair coverage diagnostics ---
    h1_pairs = set(h1["pair"].dropna().astype(str).unique())
    regime_pairs = set(regimes_keyed["pair"].dropna().astype(str).unique())

    canonical_pair_count = len(h1_pairs)
    regime_pair_count = len(regime_pairs)
    pair_coverage_ratio = regime_pair_count / canonical_pair_count if canonical_pair_count else float("nan")

    pairs_missing_in_regimes = sorted(h1_pairs - regime_pairs)
    pairs_only_in_regimes = sorted(regime_pairs - h1_pairs)

    logger.info("Pair coverage: regime_pairs=%d / canonical_pairs=%d = %.4f",
                regime_pair_count, canonical_pair_count, pair_coverage_ratio)

    if pair_coverage_ratio < WARN_PAIR_COVERAGE_BELOW:
        logger.warning(
            "Pair coverage ratio %.4f < %.2f threshold — many canonical pairs lack regime labels.",
            pair_coverage_ratio, WARN_PAIR_COVERAGE_BELOW,
        )

    if pairs_missing_in_regimes:
        logger.warning(
            "Pairs present in H1 but missing in regimes (%d): %s",
            len(pairs_missing_in_regimes), pairs_missing_in_regimes,
        )
    if pairs_only_in_regimes:
        logger.info(
            "Pairs present in regimes but not in H1 (%d): %s",
            len(pairs_only_in_regimes), pairs_only_in_regimes,
        )

    # --------------------------------------------------------------
    # Restrict H1 dataset to pairs supported by regime dataset
    # --------------------------------------------------------------
    if pairs_missing_in_regimes:
        logger.info("Filtering H1 dataset to regime-supported pairs only...")

        rows_before_filter = len(h1)
        pairs_before_filter = h1["pair"].nunique()

        h1 = h1[h1["pair"].isin(regime_pairs)].copy()

        rows_after_filter = len(h1)
        pairs_after_filter = h1["pair"].nunique()

        logger.info(
            "Row reduction: %d -> %d rows | pairs: %d -> %d",
            rows_before_filter, rows_after_filter,
            pairs_before_filter, pairs_after_filter,
        )

    # --- Join (exact match only; no leakage) ---
    logger.info("Joining regimes onto H1 data (exact pair + UTC day match)...")
    out = h1.merge(
        regimes_keyed,
        how="left",
        left_on=["pair", "date_utc"],
        right_on=["pair", "timestamp"],
        validate="m:1",  # many H1 rows per (pair, day), exactly one regime row
    )

    # Drop the right-side join key to keep dataset clean
    out = out.drop(columns=["timestamp"])

    # --- Join validation ---
    total_rows = len(out)
    matched_rows = int(out["phase"].notna().sum())
    missing_rows = total_rows - matched_rows
    match_rate = matched_rows / total_rows if total_rows else float("nan")
    missing_rate_overall = missing_rows / total_rows if total_rows else float("nan")

    logger.info(
        "Join diagnostics: total_rows=%d, matched_rows=%d, match_rate=%.4f",
        total_rows, matched_rows, match_rate,
    )

    if match_rate < WARN_MATCH_RATE_BELOW:
        logger.warning(
            "match_rate %.4f < %.2f threshold — missing regimes likely due to vendor gaps.",
            match_rate, WARN_MATCH_RATE_BELOW,
        )

    if missing_rate_overall > WARN_MISSING_RATE_ABOVE:
        logger.warning(
            "Overall missing regime rate %.4f > %.2f threshold.",
            missing_rate_overall, WARN_MISSING_RATE_ABOVE,
        )

    # Missing by pair (%)
    missing_by_pair = (
        out.assign(missing_phase=out["phase"].isna())
        .groupby("pair")["missing_phase"]
        .mean()
        .sort_values(ascending=False)
        .rename("missing_phase_rate")
        .to_frame()
    )
    missing_by_pair["missing_phase_rate_pct"] = (missing_by_pair["missing_phase_rate"] * 100.0).round(3)

    logger.info("Missing regime rate by pair (pct of H1 rows):\n%s",
                missing_by_pair[["missing_phase_rate_pct"]].to_string())

    # Warn for individual pairs with high missing rate
    high_missing_pairs = missing_by_pair[missing_by_pair["missing_phase_rate"] > WARN_MISSING_RATE_ABOVE]
    if not high_missing_pairs.empty:
        logger.warning(
            "Pairs with missing regime rate > %.0f%%:\n%s",
            WARN_MISSING_RATE_ABOVE * 100,
            high_missing_pairs[["missing_phase_rate_pct"]].to_string(),
        )

    # Log sample rows with missing phase
    _log_missing_samples(out, n=args.missing_sample_n)

    # --- Save ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info("Wrote enriched dataset: %s  rows=%d", args.out, len(out))

    out_valid = out[out["phase"].notna()]
    logger.info(
        "Rows with valid regime: %d (%.2f%%)",
        len(out_valid), len(out_valid) / len(out) * 100 if len(out) else 0,
    )

if __name__ == "__main__":
    main()
