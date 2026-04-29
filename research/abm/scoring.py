"""
research/abm/scoring.py
=======================
Scoring function for ABM calibration quality.

Computes a weighted scalar score from the comparison DataFrame produced by
:func:`~research.abm.calibration.compare_to_data`.  Lower is better.

Metrics used (all as relative differences):
    - std         (weight 0.3)
    - abs_mean    (weight 0.3)
    - autocorr    (weight 0.3)
    - extreme_freq (weight 0.1)

Metrics excluded:
    - mean: too unstable
    - correlations: used as constraints, not objectives
"""

from __future__ import annotations

import math

import pandas as pd


def extract_metric(comparison_df: pd.DataFrame, name: str) -> float:
    """Extract the ``rel_diff`` value for a named statistic.

    Args:
        comparison_df: DataFrame as returned by
            :func:`~research.abm.calibration.compare_to_data`.
        name: Statistic name (value in the ``statistic`` column).

    Returns:
        The ``rel_diff`` float for the requested statistic, or
        ``float("inf")`` if the statistic is absent or NaN.
    """
    mask = comparison_df["statistic"] == name
    if not mask.any():
        return float("inf")
    val = comparison_df.loc[mask, "rel_diff"].values[0]
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("inf")
    if math.isnan(f):
        return float("inf")
    return f


def compute_score(comparison_df: pd.DataFrame) -> float:
    """Compute a scalar calibration score from the comparison table.

    Lower is better.  Returns ``float("inf")`` if any required metric is
    missing or non-finite.

    Args:
        comparison_df: DataFrame as returned by
            :func:`~research.abm.calibration.compare_to_data`.

    Returns:
        Weighted scalar score.
    """
    rel_std = extract_metric(comparison_df, "std")
    rel_abs_mean = extract_metric(comparison_df, "abs_mean")
    rel_autocorr = extract_metric(comparison_df, "autocorr")
    rel_extreme = extract_metric(comparison_df, "extreme_freq")

    metrics = (rel_std, rel_abs_mean, rel_autocorr, rel_extreme)
    if any(not math.isfinite(v) for v in metrics):
        return float("inf")

    score = (
        0.3 * rel_std
        + 0.3 * rel_abs_mean
        + 0.3 * rel_autocorr
        + 0.1 * rel_extreme
    )
    return float(score)
