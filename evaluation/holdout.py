"""
evaluation/holdout.py
=====================
Holdout test utilities: split a dataset into train/test by year and evaluate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as cfg
from evaluation.metrics import compute_stats
from utils.validation import require_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def train_test_split(
    df: pd.DataFrame,
    split_year: int | None = None,
    *,
    year_col: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train (<= *split_year*) and test (> *split_year*).

    Args:
        df: DataFrame with *year_col*.
        split_year: Last year included in the training set.  Defaults to
                    ``config.HOLDOUT_SPLIT_YEAR``.
        year_col: Column containing the year label.

    Returns:
        Tuple (train_df, test_df).
    """
    if split_year is None:
        split_year = cfg.HOLDOUT_SPLIT_YEAR
    require_columns(df, [year_col], context="train_test_split")
    train = df[df[year_col] <= split_year].copy()
    test = df[df[year_col] >= split_year + 1].copy()
    logger.debug(
        "train_test_split(split=%d): train=%d, test=%d",
        split_year,
        len(train),
        len(test),
    )
    return train, test


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def holdout_test(
    df: pd.DataFrame,
    horizon: int,
    split_year: int | None = None,
    *,
    year_col: str = "year",
    min_signals: int | None = None,
) -> dict:
    """Evaluate signal performance on train and test splits.

    Args:
        df: Signal DataFrame.
        horizon: Return horizon.
        split_year: Last training year.  Defaults to ``config.HOLDOUT_SPLIT_YEAR``.
        year_col: Year column name.
        min_signals: Minimum observations to report stats (rather than None).
                     Defaults to ``config.MIN_SIGNALS_FOR_STATS``.

    Returns:
        Dict with keys ``'train'`` and ``'test'``, each a stats dict or
        ``None`` if there are fewer than *min_signals* observations.
    """
    if min_signals is None:
        min_signals = cfg.MIN_SIGNALS_FOR_STATS
    if split_year is None:
        split_year = cfg.HOLDOUT_SPLIT_YEAR

    col = f"contrarian_ret_{horizon}b"
    require_columns(df, [year_col, col], context=f"holdout_test(h={horizon})")

    train, test = train_test_split(df, split_year, year_col=year_col)

    def _stats(subset: pd.DataFrame) -> dict | None:
        if len(subset) < min_signals:
            return None
        return compute_stats(subset, col)

    result = {"train": _stats(train), "test": _stats(test)}
    logger.debug(
        "holdout_test(h=%d, split=%d): train_n=%d, test_n=%d",
        horizon,
        split_year,
        len(train),
        len(test),
    )
    return result


# ---------------------------------------------------------------------------
# Regime-aware holdout (pre/post 2022)
# ---------------------------------------------------------------------------

def regime_holdout_test(
    df: pd.DataFrame,
    ret_col: str,
    *,
    regime_col: str = "macro_regime",
    pre_label: str = "pre_2022",
    post_label: str = "post_2022",
) -> dict:
    """Evaluate *ret_col* separately for pre- and post-2022 regimes.

    Args:
        df: Signal DataFrame with *regime_col*.
        ret_col: Return column name.
        regime_col: Column containing macro regime labels.
        pre_label: Label for the pre-2022 regime.
        post_label: Label for the post-2022 regime.

    Returns:
        Dict with keys *pre_label* and *post_label*, each a stats dict.
    """
    require_columns(df, [regime_col, ret_col], context="regime_holdout_test")

    pre = df[df[regime_col] == pre_label]
    post = df[df[regime_col] == post_label]

    return {
        pre_label: compute_stats(pre, ret_col),
        post_label: compute_stats(post, ret_col),
    }
