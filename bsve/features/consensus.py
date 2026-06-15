"""Consensus-oriented behavioral feature computations."""

from __future__ import annotations

import pandas as pd


def _episode_local_maturity(is_extreme: pd.Series) -> pd.Series:
    """
    Compute episode-local maturity for a single pair's boolean extreme series.

    Maturity counts bars since the current extreme episode began.  It resets
    to zero whenever the series exits an extreme state.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([True, True, True, False, True, True])
    >>> _episode_local_maturity(s).tolist()
    [1, 2, 3, 0, 1, 2]

    >>> s2 = pd.Series([False, False, True, True, False, True])
    >>> _episode_local_maturity(s2).tolist()
    [0, 0, 1, 2, 0, 1]
    """
    # Assign a unique run-id to each consecutive block of identical values.
    # A new run starts whenever is_extreme changes value (True→False or False→True).
    run_id = (is_extreme != is_extreme.shift()).cumsum()
    # Within each True run, cumcount() gives 0-based position; add 1 for 1-based maturity.
    streak = is_extreme.groupby(run_id).cumcount() + 1
    return streak.where(is_extreme, 0).astype("int64")


def compute_consensus_maturity(
    df: pd.DataFrame,
    *,
    pair_col: str = "pair",
    extreme_flag_col: str = "sentiment_extreme_flag",
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float | None = None,
) -> pd.Series:
    """
    Compute per-bar maturity (streak length) for extreme consensus states.

    Maturity is the number of consecutive bars since the current extreme
    episode began.  It resets to zero whenever the pair exits the extreme
    state or whenever the pair changes.

    Examples
    --------
    Single episode:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "pair": ["A"] * 6,
    ...     "sentiment_extreme_flag": [True, True, True, False, True, True],
    ... })
    >>> compute_consensus_maturity(df).tolist()
    [1, 2, 3, 0, 1, 2]

    Pair boundary resets the counter:

    >>> df2 = pd.DataFrame({
    ...     "pair": ["A", "A", "A", "B", "B", "B"],
    ...     "sentiment_extreme_flag": [True, True, True, True, True, True],
    ... })
    >>> compute_consensus_maturity(df2).tolist()
    [1, 2, 3, 1, 2, 3]
    """
    if extreme_flag_col in df.columns:
        is_extreme = df[extreme_flag_col].fillna(False).astype(bool)
    elif sentiment_col in df.columns and extreme_threshold is not None:
        is_extreme = df[sentiment_col].abs() >= float(extreme_threshold)
    else:
        raise ValueError(
            "provide either an extreme-flag column or a sentiment column with threshold"
        )

    return (
        is_extreme.groupby(df[pair_col], sort=False)
        .apply(_episode_local_maturity)
        .reset_index(level=0, drop=True)
        .reindex(df.index)
        .fillna(0)
        .astype("int64")
    )


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
