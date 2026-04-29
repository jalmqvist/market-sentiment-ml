"""
research/abm/calibration.py
============================
Calibration helpers for the retail FX sentiment ABM.

Two complementary functions are provided:

calibrate_from_dataset
    Extract summary statistics from a real research dataset that can serve as
    calibration targets.  These statistics describe the observed distribution
    of ``net_sentiment`` (mean, std, autocorrelation, extreme frequency).

compare_to_data
    Compare a simulation output DataFrame (from :class:`FXSentimentSimulation`)
    to the calibration targets derived from real data.  Returns a
    per-statistic comparison table so the user can assess how well the ABM
    reproduces empirical sentiment dynamics.

The functions impose no optimisation loop.  They are intentionally minimal:
the researcher selects agent mix and parameters manually (or via grid search)
by inspecting the comparison table.

Usage::

    import pandas as pd
    from research.abm.calibration import calibrate_from_dataset, compare_to_data

    real_df = pd.read_csv("data/output/master_research_dataset_core.csv",
                          parse_dates=["snapshot_time", "entry_time"])
    targets = calibrate_from_dataset(real_df)

    # … run simulation …
    comparison = compare_to_data(sim_df, targets)
    print(comparison)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in the real dataset for calibration.
_REQUIRED_REAL_COLS = ["net_sentiment", "pair"]

# Columns that must be present in the simulation output for comparison.
_REQUIRED_SIM_COLS = ["net_sentiment"]


def calibrate_from_dataset(
    df: pd.DataFrame,
    pair: str | None = None,
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float = 70.0,
    autocorr_lag: int = 1,
) -> dict[str, Any]:
    """Compute summary statistics from the real research dataset.

    The returned dictionary serves as calibration targets for the ABM.

    Args:
        df: Real research dataset (e.g. ``master_research_dataset_core.csv``).
            Must contain columns ``net_sentiment`` and ``pair``.
        pair: If provided, restrict statistics to this FX pair only.  When
            ``None`` statistics are computed over all pairs combined.
        sentiment_col: Name of the sentiment column to analyse.
        extreme_threshold: Absolute sentiment level above which a reading is
            considered "extreme" (default 70 matches the dataset's
            ``extreme_70`` flag).
        autocorr_lag: Lag in rows for the autocorrelation computation.

    Returns:
        Dictionary with the following keys:

        - ``mean``: mean net sentiment
        - ``std``: standard deviation of net sentiment
        - ``abs_mean``: mean absolute sentiment (crowd conviction proxy)
        - ``autocorr``: lag-1 autocorrelation (persistence)
        - ``extreme_freq``: fraction of readings where |sentiment| >= threshold
        - ``long_frac``: fraction of readings where net_sentiment > 0
        - ``n_rows``: number of rows used
        - ``pair``: pair filter applied (or ``"all"``)

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [c for c in _REQUIRED_REAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    subset = df.copy()
    if pair is not None:
        subset = subset[subset["pair"] == pair]
        if subset.empty:
            raise ValueError(f"No rows found for pair={pair!r}")

    series = subset[sentiment_col].dropna()
    n = len(series)

    if n == 0:
        raise ValueError("No valid sentiment values after filtering")

    autocorr = float(series.autocorr(lag=autocorr_lag)) if n > autocorr_lag else float("nan")

    stats: dict[str, Any] = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "abs_mean": float(series.abs().mean()),
        "autocorr": autocorr,
        "extreme_freq": float((series.abs() >= extreme_threshold).mean()),
        "long_frac": float((series > 0).mean()),
        "n_rows": n,
        "pair": pair if pair is not None else "all",
    }

    logger.info(
        "Calibration targets (pair=%s, n=%d): mean=%.2f std=%.2f abs_mean=%.2f "
        "autocorr=%.3f extreme_freq=%.3f long_frac=%.3f",
        stats["pair"],
        n,
        stats["mean"],
        stats["std"],
        stats["abs_mean"],
        stats["autocorr"],
        stats["extreme_freq"],
        stats["long_frac"],
    )
    return stats


def compare_to_data(
    sim_df: pd.DataFrame,
    targets: dict[str, Any],
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float = 70.0,
    autocorr_lag: int = 1,
) -> pd.DataFrame:
    """Compare simulation output to real-data calibration targets.

    Args:
        sim_df: Simulation output DataFrame (from
            :meth:`~research.abm.simulation.FXSentimentSimulation.run`).
            Must contain a ``net_sentiment`` column.
        targets: Calibration target dictionary returned by
            :func:`calibrate_from_dataset`.
        sentiment_col: Name of the sentiment column in ``sim_df``.
        extreme_threshold: Threshold for counting extreme readings.
        autocorr_lag: Lag for autocorrelation computation.

    Returns:
        DataFrame with columns:

        - ``statistic``: name of the summary statistic
        - ``simulated``: value from the simulation
        - ``real``: value from the real dataset (calibration target)
        - ``abs_diff``: absolute difference
        - ``rel_diff``: relative difference (|sim - real| / (|real| + ε))

    Raises:
        ValueError: If required columns are missing from ``sim_df``.
    """
    missing = [c for c in _REQUIRED_SIM_COLS if c not in sim_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sim_df: {missing}")

    series = sim_df[sentiment_col].dropna()
    n = len(series)

    if n == 0:
        raise ValueError("sim_df has no valid sentiment values")

    autocorr = float(series.autocorr(lag=autocorr_lag)) if n > autocorr_lag else float("nan")

    sim_stats = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "abs_mean": float(series.abs().mean()),
        "autocorr": autocorr,
        "extreme_freq": float((series.abs() >= extreme_threshold).mean()),
        "long_frac": float((series > 0).mean()),
    }

    rows = []
    for stat, sim_val in sim_stats.items():
        real_val = targets.get(stat, float("nan"))
        try:
            sim_f = float(sim_val)
            real_f = float(real_val)
            abs_diff = abs(sim_f - real_f)
            rel_diff = abs_diff / (abs(real_f) + 1e-12)
        except (TypeError, ValueError):
            abs_diff = float("nan")
            rel_diff = float("nan")
        rows.append(
            {
                "statistic": stat,
                "simulated": sim_val,
                "real": real_val,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Comparison table:\n%s", result.to_string(index=False))
    return result
