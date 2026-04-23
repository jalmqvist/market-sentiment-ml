"""
pipeline/filters.py
===================
Reusable signal-filtering logic shared across experiments.

All filters accept and return DataFrames; no I/O is performed here.
"""

from __future__ import annotations

import logging

import pandas as pd

import config as cfg
from utils.validation import require_columns, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Non-overlap (spacing) filter
# ---------------------------------------------------------------------------

def enforce_non_overlap(
    df: pd.DataFrame,
    horizon: int,
    *,
    time_col: str = "timestamp",
    pair_col: str = "pair",
) -> pd.DataFrame:
    """Remove overlapping signals per pair.

    For each pair, keep only rows that are at least *horizon* hours apart.
    Earlier rows take priority (greedy forward selection).

    Args:
        df: Signal DataFrame, must contain *time_col* and *pair_col*.
        horizon: Minimum gap between signals in hours.
        time_col: Column with signal timestamps.
        pair_col: Column with pair labels.

    Returns:
        Filtered DataFrame (copy).
    """
    require_columns(df, [time_col, pair_col], context="enforce_non_overlap")
    df = df.sort_values(time_col).copy()

    selected_idx: list = []
    last_time: dict = {}

    for idx, row in df.iterrows():
        pair = row[pair_col]
        t = row[time_col]

        if pair not in last_time:
            selected_idx.append(idx)
            last_time[pair] = t
        else:
            delta = (t - last_time[pair]).total_seconds() / 3600
            if delta >= horizon:
                selected_idx.append(idx)
                last_time[pair] = t

    result = df.loc[selected_idx].copy()
    logger.debug(
        "enforce_non_overlap(horizon=%d): %d -> %d rows", horizon, len(df), len(result)
    )
    return result


def enforce_global_spacing(
    df: pd.DataFrame,
    horizon: int,
    *,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Remove overlapping signals globally (across all pairs).

    Keeps only rows that are at least *horizon* hours apart, regardless of pair.
    Earlier rows take priority.

    Args:
        df: Signal DataFrame.
        horizon: Minimum gap in hours.
        time_col: Timestamp column name.

    Returns:
        Filtered DataFrame (copy).
    """
    require_columns(df, [time_col], context="enforce_global_spacing")
    if len(df) == 0:
        return df.copy()

    df = df.sort_values(time_col).reset_index(drop=True)
    selected: list = []
    last_time = None

    for _, row in df.iterrows():
        t = row[time_col]
        if last_time is None:
            selected.append(row)
            last_time = t
        else:
            if (t - last_time).total_seconds() >= horizon * 3600:
                selected.append(row)
                last_time = t

    if not selected:
        return df.iloc[0:0].copy()

    result = pd.DataFrame(selected)
    logger.debug(
        "enforce_global_spacing(horizon=%d): %d -> %d rows",
        horizon,
        len(df),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Daily cap filter
# ---------------------------------------------------------------------------

def cap_signals_per_day(
    df: pd.DataFrame,
    max_per_day: int | None = None,
    *,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Keep at most *max_per_day* signals per calendar day.

    Args:
        df: Signal DataFrame sorted by *time_col*.
        max_per_day: Cap.  Defaults to ``config.MAX_SIGNALS_PER_DAY``.
        time_col: Timestamp column.

    Returns:
        Filtered DataFrame (copy), without the temporary ``date`` column.
    """
    if max_per_day is None:
        max_per_day = cfg.MAX_SIGNALS_PER_DAY
    require_columns(df, [time_col], context="cap_signals_per_day")
    df = df.copy()
    df["_date"] = df[time_col].dt.date
    df = df.sort_values(time_col)
    original_len = len(df)
    df = df.groupby("_date", group_keys=False).head(max_per_day)
    result = df.drop(columns="_date")
    logger.debug(
        "cap_signals_per_day(max=%d): %d -> %d rows", max_per_day, original_len, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Pair-level survivor filter
# ---------------------------------------------------------------------------

def select_survivor_pairs(
    df: pd.DataFrame,
    horizon: int,
    *,
    min_signals: int | None = None,
    min_after_dedup: int | None = None,
    min_sharpe: float | None = None,
    holdout_split_year: int | None = None,
) -> list[str]:
    """Return pairs that pass in-sample quality filters.

    Args:
        df: Signal DataFrame with ``pair`` and the relevant return column.
        horizon: Evaluation horizon (used to select return column).
        min_signals: Minimum raw signal count per pair before dedup.
        min_after_dedup: Minimum signal count after non-overlap dedup.
        min_sharpe: Minimum in-sample Sharpe ratio.
        holdout_split_year: Split year for holdout; pairs must have positive
                            test-set Sharpe.  Uses ``config.HOLDOUT_SPLIT_YEAR``
                            if ``None``.

    Returns:
        List of pairs that pass all filters.
    """
    if min_signals is None:
        min_signals = cfg.SURVIVOR_MIN_SIGNALS
    if min_after_dedup is None:
        min_after_dedup = cfg.SURVIVOR_MIN_AFTER_DEDUP
    if min_sharpe is None:
        min_sharpe = cfg.SURVIVOR_MIN_SHARPE
    if holdout_split_year is None:
        holdout_split_year = cfg.HOLDOUT_SPLIT_YEAR

    col = f"contrarian_ret_{horizon}b"
    require_columns(df, ["pair", col], context="select_survivor_pairs")

    survivors: list[str] = []

    for pair, g in df.groupby("pair"):
        if len(g) < min_signals:
            continue

        g_dedup = enforce_non_overlap(g, horizon)

        if len(g_dedup) < min_after_dedup:
            continue

        r = g_dedup[col].dropna()
        if r.empty:
            continue

        std = r.std()
        sharpe = r.mean() / std if std > 0 else 0.0

        if sharpe <= min_sharpe:
            continue

        # Holdout check
        train = g_dedup[g_dedup["year"] <= holdout_split_year]
        test = g_dedup[g_dedup["year"] >= holdout_split_year + 1]

        if len(test) == 0:
            continue

        r_test = test[col].dropna()
        if r_test.empty:
            continue
        std_test = r_test.std()
        test_sharpe = r_test.mean() / std_test if std_test > 0 else 0.0

        if test_sharpe <= 0:
            continue

        survivors.append(pair)

    logger.debug(
        "select_survivor_pairs(horizon=%d): %d/%d pairs passed",
        horizon,
        len(survivors),
        df["pair"].nunique(),
    )
    return survivors
