"""Persistence-oriented behavioral feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_persistence_duration(
    df: pd.DataFrame,
    *,
    signal_col: str = "crowd_side",
    pair_col: str = "pair",
) -> pd.Series:
    """Compute number of consecutive bars the same signal has persisted."""
    if signal_col not in df.columns:
        raise ValueError(f"missing signal column: {signal_col}")

    values = df[signal_col]
    group_change = values.ne(values.shift()) | df[pair_col].ne(df[pair_col].shift())
    return group_change.groupby(df[pair_col], sort=False).cumsum().groupby(
        [df[pair_col], group_change.groupby(df[pair_col], sort=False).cumsum()],
        sort=False,
    ).cumcount() + 1


def compute_transition_flag(
    df: pd.DataFrame,
    *,
    signal_col: str = "crowd_side",
    pair_col: str = "pair",
) -> pd.Series:
    """Compute transition flags where the persistence signal changes."""
    if signal_col not in df.columns:
        raise ValueError(f"missing signal column: {signal_col}")
    changed = df[signal_col].ne(df[signal_col].shift()) | df[pair_col].ne(df[pair_col].shift())
    return changed.astype(np.int8)
