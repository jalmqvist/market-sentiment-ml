"""
pipeline/signal.py
==================
Canonical signal definitions used across all experiments.

Each function accepts a prepared DataFrame (from ``pipeline/build_dataset.py``)
and returns a filtered copy containing only rows that match the signal
condition.

No I/O or feature engineering is performed here.
"""

from __future__ import annotations

import logging

import pandas as pd

import config as cfg
from utils.validation import require_columns, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical behavioral signal (used in discover_behavioral_signal / portfolio)
# ---------------------------------------------------------------------------

def apply_behavioral_signal(
    df: pd.DataFrame,
    *,
    streak_min: int | None = None,
    persistence_buckets: list[str] | None = None,
) -> pd.DataFrame:
    """Return rows matching the canonical behavioral (contrarian) signal.

    Conditions:
      - ``extreme_streak_70`` >= *streak_min*
      - ``crowd_persistence_bucket_70`` in *persistence_buckets*

    Args:
        df: Prepared research dataset.
        streak_min: Minimum extreme-streak count.  Defaults to
                    ``config.SIGNAL_EXTREME_STREAK_MIN``.
        persistence_buckets: Allowed persistence bucket labels.  Defaults to
                             ``config.SIGNAL_PERSISTENCE_BUCKETS``.

    Returns:
        Filtered copy of *df*.
    """
    if streak_min is None:
        streak_min = cfg.SIGNAL_EXTREME_STREAK_MIN
    if persistence_buckets is None:
        persistence_buckets = cfg.SIGNAL_PERSISTENCE_BUCKETS

    require_columns(
        df,
        ["extreme_streak_70", "crowd_persistence_bucket_70"],
        context="apply_behavioral_signal",
    )

    signal = df[
        (df["extreme_streak_70"] >= streak_min)
        & (df["crowd_persistence_bucket_70"].isin(persistence_buckets))
    ].copy()

    logger.debug(
        "apply_behavioral_signal(streak>=%d, persistence=%s): %d/%d rows",
        streak_min,
        persistence_buckets,
        len(signal),
        len(df),
    )
    warn_if_empty(signal, context="apply_behavioral_signal")
    return signal


# ---------------------------------------------------------------------------
# Regime V2 signal (JPY crosses, high persistence + decreasing acceleration)
# ---------------------------------------------------------------------------

def apply_regime_v2_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching the Regime V2 baseline signal.

    Conditions:
      - ``pair_group`` == ``'JPY_cross'``
      - ``crowd_persistence_bucket_70`` == ``'high'``
      - ``acceleration_bucket`` == ``'decreasing'``

    Args:
        df: Prepared research dataset (must have ``pair_group`` computed).

    Returns:
        Filtered copy of *df*.
    """
    require_columns(
        df,
        ["pair_group", "crowd_persistence_bucket_70", "acceleration_bucket"],
        context="apply_regime_v2_signal",
    )

    signal = df[
        (df["pair_group"] == cfg.REGIME_V2_PAIR_GROUP)
        & (df["crowd_persistence_bucket_70"] == cfg.REGIME_V2_PERSISTENCE_BUCKET)
        & (df["acceleration_bucket"] == cfg.REGIME_V2_ACCELERATION_BUCKET)
    ].copy()

    logger.debug(
        "apply_regime_v2_signal: %d/%d rows (JPY, high persistence, decreasing accel)",
        len(signal),
        len(df),
    )
    warn_if_empty(signal, context="apply_regime_v2_signal")
    return signal


# ---------------------------------------------------------------------------
# JPY hypothesis signal (fight-trend + extreme strength)
# ---------------------------------------------------------------------------

def apply_jpy_hypothesis_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching the JPY hypothesis signal.

    Conditions:
      - ``pair_group`` == ``'JPY_cross'``
      - ``fight_trend`` is True (trend_alignment_12b == -1)
      - ``trend_strength_bucket_12b`` == ``'extreme'``

    Args:
        df: Prepared research dataset.

    Returns:
        Filtered copy of *df*.
    """
    require_columns(
        df,
        ["pair_group", "fight_trend", "trend_strength_bucket_12b"],
        context="apply_jpy_hypothesis_signal",
    )

    signal = df[
        (df["pair_group"] == cfg.REGIME_V2_PAIR_GROUP)
        & (df["fight_trend"])
        & (df["trend_strength_bucket_12b"] == "extreme")
    ].copy()

    logger.debug(
        "apply_jpy_hypothesis_signal: %d/%d rows", len(signal), len(df)
    )
    warn_if_empty(signal, context="apply_jpy_hypothesis_signal")
    return signal


# ---------------------------------------------------------------------------
# Parametric signal builder (used in sweep experiments)
# ---------------------------------------------------------------------------

def build_parametric_signal(
    df: pd.DataFrame,
    *,
    streak_threshold: int,
    use_persistence: bool,
    pair_group: str | None = None,
) -> pd.DataFrame:
    """Build a signal with configurable streak threshold and persistence flag.

    Args:
        df: Research dataset with required feature columns.
        streak_threshold: Minimum value of ``extreme_streak_70``.
        use_persistence: If ``True``, also require
                         ``crowd_persistence_bucket_70 == 'high'``.
        pair_group: If provided, restrict to this pair group.  ``None`` means
                    no group filter.

    Returns:
        Filtered copy of *df*.
    """
    require_columns(
        df,
        ["pair_group", "extreme_streak_70", "acceleration_bucket"],
        context="build_parametric_signal",
    )

    cond = (
        (df["extreme_streak_70"] >= streak_threshold)
        & (df["acceleration_bucket"] == "decreasing")
    )

    if pair_group is not None:
        cond = cond & (df["pair_group"] == pair_group)

    if use_persistence:
        require_columns(df, ["crowd_persistence_bucket_70"], context="build_parametric_signal")
        cond = cond & (df["crowd_persistence_bucket_70"] == "high")

    signal = df[cond].copy()
    logger.debug(
        "build_parametric_signal(streak>=%d, persistence=%s, group=%s): %d rows",
        streak_threshold,
        use_persistence,
        pair_group,
        len(signal),
    )
    warn_if_empty(signal, context="build_parametric_signal")
    return signal
