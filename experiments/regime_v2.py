"""
experiments/regime_v2.py
========================
Walk-forward evaluation of the Regime V2 signal.

Refactored from ``walk_forward_regime_v2.py``.  All core logic is preserved;
only structure and safety checks have been added.

Usage::

    python -m experiments.regime_v2
    python -m experiments.regime_v2 --input data/output/master_research_dataset_with_regime.csv \\
                                    --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config as cfg
from evaluation.holdout import holdout_test
from evaluation.metrics import compute_stats, compute_pair_stats
from evaluation.walk_forward import walk_forward_yearly
from pipeline.filters import enforce_global_spacing
from pipeline.signal import apply_regime_v2_signal
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps, require_columns, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime column validation
# ---------------------------------------------------------------------------

#: Columns that must be present in the regime-enriched dataset.
REGIME_REQUIRED_COLUMNS: list[str] = ["phase", "is_trending", "is_high_vol"]


def _require_regime_columns(df: pd.DataFrame, context: str = "") -> None:
    """Raise ``ValueError`` if regime columns are missing from *df*.

    This provides a clear, actionable error when a non-regime dataset is
    accidentally passed to a regime-specific script.
    """
    require_columns(df, REGIME_REQUIRED_COLUMNS, context=context)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path=None) -> pd.DataFrame:
    """Load and prepare the research dataset for Regime V2 evaluation.

    Defaults to ``config.DATA_PATH_REGIME`` (the regime-enriched dataset).
    Raises ``ValueError`` clearly if required regime columns are absent.
    """
    if path is None:
        path = cfg.DATA_PATH_REGIME

    df = read_csv(
        path,
        required_columns=["pair", "time"],
    )

    # Regime scripts require regime-specific columns. Fail clearly rather
    # than producing silently empty or misleading results.
    _require_regime_columns(df, context="regime_v2.load_data")

    df = parse_timestamps(df, "time", context="regime_v2.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    df["pair_group"] = np.where(
        df["pair"].str.contains(cfg.JPY_PAIR_PATTERN, case=False, na=False),
        "JPY_cross",
        "other",
    )

    logger.info("Dataset loaded: %d rows, %d pairs", len(df), df["pair"].nunique())
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def debug_checks(df: pd.DataFrame) -> None:
    """Print column-level diagnostics for debugging."""
    logger.debug("=== DEBUG CHECKS ===")
    logger.debug("pair_group counts:\n%s", df["pair_group"].value_counts().to_string())

    for col in ["saturation_bucket", "crowd_persistence_bucket_70"]:
        if col in df.columns:
            logger.debug(
                "%s counts:\n%s", col, df[col].value_counts(dropna=False).to_string()
            )
        else:
            logger.warning("Column '%s' not found in dataset", col)


def tradeability_check(signal_df: pd.DataFrame, horizon: int) -> dict | None:
    """Return tradeability stats (mean bps, Sharpe) for *signal_df*."""
    ret_col = f"contrarian_ret_{horizon}b"
    require_columns(signal_df, [ret_col], context="tradeability_check")
    if len(signal_df) == 0:
        return None
    mean = signal_df[ret_col].mean()
    std = signal_df[ret_col].std()
    return {
        "mean_return": mean,
        "std": std,
        "mean_bps": mean * 10000,
        "sharpe": mean / std if std != 0 else np.nan,
    }


def frequency_check(signal_df: pd.DataFrame, time_col: str = "timestamp") -> dict | None:
    """Return frequency stats for *signal_df*."""
    if len(signal_df) == 0:
        return None
    tmp = signal_df.copy()
    tmp["date"] = tmp[time_col].dt.date
    per_day = tmp.groupby("date").size()
    return {
        "signals_total": len(signal_df),
        "avg_per_day": per_day.mean(),
        "median_per_day": per_day.median(),
        "max_per_day": per_day.max(),
    }


def distribution_check(signal_df: pd.DataFrame) -> dict | None:
    """Return distribution stats (top pairs, years) for *signal_df*."""
    if len(signal_df) == 0:
        return None
    return {
        "pairs": signal_df["pair"].value_counts().head(5).to_dict(),
        "years": signal_df["year"].value_counts().to_dict(),
    }


def remove_top_pairs(signal_df: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    """Remove the *top_n* most frequent pairs from *signal_df*."""
    if len(signal_df) == 0:
        return signal_df.copy()
    top_pairs = signal_df["pair"].value_counts().head(top_n).index.tolist()
    logger.info("Removing top %d pairs: %s", top_n, top_pairs)
    return signal_df[~signal_df["pair"].isin(top_pairs)].copy()


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def run_evaluation(signal: pd.DataFrame, label: str) -> None:
    """Run full evaluation for a signal subset and print results."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATION: {label}")
    print(f"{'=' * 80}")
    print(f"Signals: {len(signal):,}")

    for horizon in cfg.EVAL_HORIZONS:
        print(f"\n--- Horizon: {horizon} ---")

        signal_spaced = enforce_global_spacing(signal, horizon)

        if warn_if_empty(signal_spaced, context=f"run_evaluation(h={horizon})"):
            print("No signals")
            continue

        wf = walk_forward_yearly(signal_spaced, horizon, start_year=2021)
        if not wf.empty:
            print(wf[["year", "n", "mean", "sharpe"]].to_string(index=False))
        else:
            print("No valid years")

        ret_col = f"contrarian_ret_{horizon}b"
        if ret_col in signal_spaced.columns:
            pooled = signal_spaced[ret_col].dropna()
            summary_sharpe = pooled.mean() / pooled.std() if pooled.std() > 0 else np.nan
            print(f"Summary Sharpe: {summary_sharpe:.4f}")

        hold = holdout_test(signal_spaced, horizon)
        if hold:
            print("Holdout:", hold)

        trade = tradeability_check(signal_spaced, horizon)
        if trade:
            print(f"Tradeability: {trade['mean_bps']:.2f} bps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Walk-forward evaluation: Regime V2 signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=str(cfg.DATA_PATH_REGIME),
        help="Path to regime-enriched research dataset CSV (must contain phase, is_trending, is_high_vol).",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    df = load_data(args.input)
    debug_checks(df)

    signal_raw = apply_regime_v2_signal(df)
    print(f"\nRaw signal count: {len(signal_raw):,}")

    # Per-pair diagnostics
    print("\n=== PER-PAIR PERFORMANCE (48b) ===")
    ret_col_48 = "contrarian_ret_48b"
    if ret_col_48 in signal_raw.columns:
        pair_stats = compute_pair_stats(signal_raw, ret_col_48)
        print(pair_stats.to_string())

    # Evaluations
    run_evaluation(signal_raw, "BASE")
    run_evaluation(remove_top_pairs(signal_raw, top_n=1), "REMOVE TOP 1")
    run_evaluation(remove_top_pairs(signal_raw, top_n=2), "REMOVE TOP 2")

    print("\n--- FREQUENCY ---")
    freq = frequency_check(signal_raw)
    if freq:
        print(freq)

    print("\n--- DISTRIBUTION ---")
    dist = distribution_check(signal_raw)
    if dist:
        print(dist)


if __name__ == "__main__":
    main()
