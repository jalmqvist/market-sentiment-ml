"""
evaluation/walk_forward.py
==========================
Walk-forward evaluation utilities.

All functions accept DataFrames and return DataFrames or scalar statistics.
No I/O is performed here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as cfg
from evaluation.metrics import compute_stats
from utils.validation import require_columns, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yearly walk-forward (simple)
# ---------------------------------------------------------------------------

def walk_forward_yearly(
    df: pd.DataFrame,
    horizon: int,
    *,
    year_col: str = "year",
    min_signals: int | None = None,
    start_year: int | None = None,
) -> pd.DataFrame:
    """Evaluate signal performance year-by-year.

    Args:
        df: Signal DataFrame with *year_col* and the relevant return column.
        horizon: Evaluation horizon.
        year_col: Column containing the calendar year.
        min_signals: Minimum number of observations required for a year to
                     be included.  Defaults to ``config.MIN_SIGNALS_FOR_STATS``.
        start_year: Exclude years strictly before this value (useful to
                    enforce forward-only logic).  ``None`` means include all.

    Returns:
        DataFrame with columns: year, n, mean, std, sharpe, hit_rate.
        Empty DataFrame if no valid years.
    """
    if min_signals is None:
        min_signals = cfg.MIN_SIGNALS_FOR_STATS

    col = f"contrarian_ret_{horizon}b"
    require_columns(df, [year_col, col], context=f"walk_forward_yearly(h={horizon})")

    results = []
    for year in sorted(df[year_col].unique()):
        if start_year is not None and year < start_year:
            continue
        test = df[df[year_col] == year]
        if len(test) < min_signals:
            continue
        stats = compute_stats(test, col)
        stats["year"] = int(year)
        results.append(stats)

    if not results:
        logger.debug("walk_forward_yearly: no valid years for horizon=%d", horizon)
        return pd.DataFrame()

    result = pd.DataFrame(results)[["year", "n", "mean", "std", "sharpe", "hit_rate"]]
    logger.debug("walk_forward_yearly(h=%d): %d years", horizon, len(result))
    return result


# ---------------------------------------------------------------------------
# Expanding-window walk-forward
# ---------------------------------------------------------------------------

def walk_forward_expanding(
    df: pd.DataFrame,
    ret_col: str,
    *,
    year_col: str = "year",
    apply_signal_fn=None,
    spacing_fn=None,
) -> pd.DataFrame:
    """Expanding-window walk-forward: train on all prior years, test on next.

    Args:
        df: Prepared dataset.
        ret_col: Return column name.
        year_col: Column containing the calendar year.
        apply_signal_fn: Optional callable ``(df) -> df`` for signal
                         selection inside each test fold.  ``None`` means
                         use the full test fold.
        spacing_fn: Optional callable ``(df) -> df`` for non-overlap
                    enforcement within each test fold.

    Returns:
        DataFrame with per-fold stats.
    """
    require_columns(df, [year_col, ret_col], context="walk_forward_expanding")

    years = sorted(df[year_col].unique())
    results = []

    for i in range(2, len(years)):
        test_year = years[i]
        test = df[df[year_col] == test_year].copy()

        if len(test) == 0:
            continue

        if apply_signal_fn is not None:
            test = apply_signal_fn(test)

        if spacing_fn is not None:
            test = spacing_fn(test)

        stats = compute_stats(test, ret_col)
        stats["year"] = int(test_year)
        results.append(stats)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Aggregate walk-forward summary
# ---------------------------------------------------------------------------

def wf_summary(wf_df: pd.DataFrame) -> dict:
    """Return mean and std of walk-forward metrics.

    Args:
        wf_df: DataFrame returned by ``walk_forward_yearly`` or similar.

    Returns:
        Dict with mean_mean, mean_sharpe, mean_hit_rate.
    """
    if wf_df.empty:
        return {"mean_mean": np.nan, "mean_sharpe": np.nan, "mean_hit_rate": np.nan}

    return {
        "mean_mean": float(wf_df["mean"].mean()),
        "mean_sharpe": float(wf_df["sharpe"].mean()),
        "mean_hit_rate": float(wf_df["hit_rate"].mean()) if "hit_rate" in wf_df.columns else np.nan,
    }
