"""
portfolio/portfolio_builder.py
==============================
Portfolio construction from the canonical behavioral signal.

Refactored from ``portfolio_behavioral_signal.py``.  All core logic is
preserved; structure, config centralisation, and safety checks have been
improved.

Usage::

    python -m portfolio.portfolio_builder --data data/output/master_research_dataset.csv
    python portfolio/portfolio_builder.py --data data/output/master_research_dataset.csv \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution (python portfolio/portfolio_builder.py)
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from evaluation.holdout import holdout_test
from evaluation.metrics import compute_metrics
from evaluation.walk_forward import walk_forward_yearly
from pipeline.filters import (
    cap_signals_per_day,
    enforce_non_overlap,
    select_survivor_pairs,
)
from pipeline.signal import apply_behavioral_signal
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps, require_columns, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the research dataset for portfolio construction.

    Args:
        path: Path to the canonical research dataset CSV.  Required; no default.

    Raises:
        ValueError: If required columns are missing.
    """
    df = read_csv(
        path,
        required_columns=["pair", "time"],
    )

    df = parse_timestamps(df, "time", context="portfolio.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    # Dataset summary
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    print(f"Dataset summary: {len(df):,} rows | {df['pair'].nunique()} unique pairs | {date_min} to {date_max}")

    logger.info("Dataset loaded: %d rows, %d pairs", len(df), df["pair"].nunique())
    return df


# ---------------------------------------------------------------------------
# Equal-weight weighting
# ---------------------------------------------------------------------------

def apply_equal_weight(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add ``pair_weight`` and ``weighted_ret`` columns (true equal weight)."""
    col = f"contrarian_ret_{horizon}b"
    require_columns(df, [col], context="apply_equal_weight")
    out = df.copy()
    out["pair_weight"] = 1.0
    out["weighted_ret"] = out[col]
    return out


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(
    signal: pd.DataFrame,
    survivors: list[str],
    horizon: int,
    *,
    use_trend_filter: bool | None = None,
    max_signals_per_day: int | None = None,
    use_equal_weight: bool | None = None,
) -> pd.DataFrame:
    """Construct the portfolio from *signal* restricted to *survivors*.

    Steps:
    1. Filter to survivor pairs.
    2. Optionally apply trend alignment filter (``trend_alignment_48b == -1``).
    3. Non-overlap deduplication.
    4. Daily cap.
    5. Equal weighting.

    Args:
        signal: Full signal DataFrame.
        survivors: List of qualifying pair names.
        horizon: Return horizon.
        use_trend_filter: Apply trend filter.  Defaults to ``config.USE_TREND_FILTER``.
        max_signals_per_day: Daily cap.  Defaults to ``config.MAX_SIGNALS_PER_DAY``.
        use_equal_weight: Apply equal weighting.  Defaults to ``config.USE_EQUAL_WEIGHT``.

    Returns:
        Portfolio DataFrame.
    """
    if use_trend_filter is None:
        use_trend_filter = cfg.USE_TREND_FILTER
    if max_signals_per_day is None:
        max_signals_per_day = cfg.MAX_SIGNALS_PER_DAY
    if use_equal_weight is None:
        use_equal_weight = cfg.USE_EQUAL_WEIGHT

    df = signal[signal["pair"].isin(survivors)].copy()
    logger.debug("[portfolio] After survivor filter: %d rows", len(df))

    if use_trend_filter:
        trend_col = "trend_alignment_48b"
        if trend_col not in df.columns:
            logger.warning("[portfolio] trend column '%s' missing; skipping trend filter", trend_col)
        else:
            logger.debug("[portfolio] trend counts:\n%s", df[trend_col].value_counts(dropna=False).to_string())
            df = df[df[trend_col] == -1]
            logger.debug("[portfolio] After trend filter: %d rows", len(df))

    df = enforce_non_overlap(df, horizon)
    logger.debug("[portfolio] After non-overlap: %d rows", len(df))

    df = cap_signals_per_day(df, max_signals_per_day)
    logger.debug("[portfolio] After daily cap: %d rows", len(df))

    if use_equal_weight:
        df = apply_equal_weight(df, horizon)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Portfolio construction from behavioral signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV (canonical dataset).",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    logger.info("Loading dataset...")
    df = load_data(args.data)
    print(f"Dataset size: {len(df):,}")

    signal = apply_behavioral_signal(df)
    print(f"Raw signal count: {len(signal):,}")

    if warn_if_empty(signal, context="portfolio.main"):
        print("No signals found. Exiting.")
        return

    for horizon in cfg.EVAL_HORIZONS:
        print(f"\n{'=' * 80}")
        print(f"PORTFOLIO (horizon={horizon})")
        print("=" * 80)

        survivors = select_survivor_pairs(signal, horizon)
        print(f"\nSurvivor pairs: {survivors}")

        if not survivors:
            print("No survivors.")
            continue

        portfolio = build_portfolio(signal, survivors, horizon)
        print(f"\nPortfolio size: {len(portfolio):,}")

        if warn_if_empty(portfolio, context=f"portfolio(h={horizon})"):
            print("Empty portfolio.")
            continue

        # --- Overall metrics ---
        m = compute_metrics(portfolio, horizon)
        if m is None:
            print("Insufficient signals for metrics.")
            continue
        print("\n--- Overall ---")
        print(m)

        # --- Walk-forward ---
        wf = walk_forward_yearly(portfolio, horizon)
        print("\n--- Walk-forward ---")
        if not wf.empty:
            print(wf.to_string(index=False))
            print(f"\nWF Sharpe mean: {wf['sharpe'].mean():.4f}")
        else:
            print("No walk-forward results.")

        # --- Holdout ---
        hold = holdout_test(portfolio, horizon)
        print("\n--- Holdout ---")
        print(hold)

        # --- Tradeability ---
        print("\n--- Tradeability ---")
        print({"bps": m["mean"] * 10000, "sharpe": m["sharpe"]})


if __name__ == "__main__":
    main()
