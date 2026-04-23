"""
evaluation/metrics.py
=====================
Shared metric computations for the research pipeline.

All functions are pure (no I/O) and return dicts or DataFrames.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as cfg
from utils.validation import require_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, ret_col: str) -> dict:
    """Compute standard performance statistics for a return series.

    Args:
        df: DataFrame containing *ret_col*.
        ret_col: Column name of the return series.

    Returns:
        Dict with keys: n, mean, std, sharpe, hit_rate.
        If *df* is empty, all numeric values are NaN.

    Raises:
        ValueError: If *ret_col* is not present in *df*.
    """
    if len(df) == 0:
        logger.debug("compute_stats: empty DataFrame for col=%s", ret_col)
        return {"n": 0, "mean": np.nan, "std": np.nan, "sharpe": np.nan, "hit_rate": np.nan}

    require_columns(df, [ret_col], context="compute_stats")

    r = df[ret_col].dropna()
    mean = r.mean()
    std = r.std()

    return {
        "n": len(r),
        "mean": mean,
        "std": std,
        "sharpe": mean / std if std > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
    }


def compute_metrics(df: pd.DataFrame, horizon: int) -> dict | None:
    """Convenience wrapper that selects the correct return column for *horizon*.

    Returns ``None`` (rather than raising) if the column is absent or there
    are fewer than ``config.MIN_SIGNALS_FOR_STATS`` observations.

    Args:
        df: Signal DataFrame.
        horizon: Horizon in bars.

    Returns:
        Stats dict or ``None``.
    """
    col = f"contrarian_ret_{horizon}b"
    if col not in df.columns or df.empty:
        logger.debug("compute_metrics: missing col=%s or empty df", col)
        return None

    r = df[col].dropna()
    if len(r) < cfg.MIN_SIGNALS_FOR_STATS:
        logger.debug(
            "compute_metrics: only %d observations (need %d)", len(r), cfg.MIN_SIGNALS_FOR_STATS
        )
        return None

    return compute_stats(df, col)


# ---------------------------------------------------------------------------
# Per-pair metrics
# ---------------------------------------------------------------------------

def compute_pair_stats(
    df: pd.DataFrame,
    ret_col: str,
    *,
    pair_col: str = "pair",
) -> pd.DataFrame:
    """Return a DataFrame with per-pair performance stats.

    Args:
        df: Signal DataFrame.
        ret_col: Return column name.
        pair_col: Column identifying the pair.

    Returns:
        DataFrame indexed by pair with stats columns, sorted by Sharpe desc.
    """
    require_columns(df, [pair_col, ret_col], context="compute_pair_stats")

    rows = []
    for pair, g in df.groupby(pair_col):
        stats = compute_stats(g, ret_col)
        stats[pair_col] = pair
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index(pair_col).sort_values("sharpe", ascending=False)
    logger.debug("compute_pair_stats: %d pairs evaluated for col=%s", len(result), ret_col)
    return result
