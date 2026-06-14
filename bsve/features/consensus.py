"""Consensus-oriented behavioral feature computations."""

from __future__ import annotations

import pandas as pd


def compute_consensus_maturity(
    df: pd.DataFrame,
    *,
    pair_col: str = "pair",
    extreme_flag_col: str = "sentiment_extreme_flag",
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float | None = None,
) -> pd.Series:
    """Compute per-bar maturity (streak length) for extreme consensus states."""
    if extreme_flag_col in df.columns:
        is_extreme = df[extreme_flag_col].fillna(False).astype(bool)
    elif sentiment_col in df.columns and extreme_threshold is not None:
        is_extreme = df[sentiment_col].abs() >= float(extreme_threshold)
    else:
        raise ValueError(
            "provide either an extreme-flag column or a sentiment column with threshold"
        )

    maturity = is_extreme.groupby(df[pair_col], sort=False).cumsum()
    return maturity.where(is_extreme, 0).astype("int64")


def compute_consensus_velocity(
    df: pd.DataFrame,
    *,
    pair_col: str = "pair",
    sentiment_col: str = "net_sentiment",
) -> pd.Series:
    """Compute per-pair first difference in sentiment as consensus velocity."""
    if sentiment_col not in df.columns:
        raise ValueError(f"missing sentiment column: {sentiment_col}")
    return df.groupby(pair_col, sort=False)[sentiment_col].diff().fillna(0.0)
