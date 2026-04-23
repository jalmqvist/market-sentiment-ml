"""
experiments/sweep.py
====================
Parametric signal sweep over streak thresholds and persistence flags.

Refactored from ``experiment_regime_v2_sweep.py``.  All core logic is
preserved.

Usage::

    python -m experiments.sweep
    python -m experiments.sweep --input data/output/master_research_dataset.csv \\
                                --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config as cfg
from evaluation.holdout import holdout_test as _holdout
from evaluation.metrics import compute_stats
from pipeline.filters import enforce_global_spacing
from pipeline.signal import build_parametric_signal
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path=None) -> pd.DataFrame:
    """Load and prepare the research dataset for the sweep."""
    if path is None:
        path = cfg.MASTER_DATASET_PATH

    df = read_csv(path, required_columns=["pair", "time"])

    df = parse_timestamps(df, "time", context="sweep.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    df["pair_group"] = np.where(
        df["pair"].str.contains(cfg.JPY_PAIR_PATTERN, case=False, na=False),
        "JPY_cross",
        "other",
    )

    logger.info("Dataset loaded: %d rows, %d pairs", len(df), df["pair"].nunique())
    return df


# ---------------------------------------------------------------------------
# Walk-forward summary for a signal
# ---------------------------------------------------------------------------

def _walk_forward(signal_df: pd.DataFrame, horizon: int) -> tuple[float, float, float]:
    """Return (mean_mean, mean_sharpe, mean_hit_rate) across OOS years."""
    if len(signal_df) == 0:
        return np.nan, np.nan, np.nan

    ret_col = f"contrarian_ret_{horizon}b"
    years = sorted(signal_df["year"].unique())
    means, sharpes, hits = [], [], []

    for year in years:
        if year < 2021:
            continue
        test = signal_df[signal_df["year"] == year]
        stats = compute_stats(test, ret_col)
        means.append(stats["mean"])
        sharpes.append(stats["sharpe"])
        hits.append(stats["hit_rate"])

    return float(np.nanmean(means)), float(np.nanmean(sharpes)), float(np.nanmean(hits))


def _holdout_sharpes(
    signal_df: pd.DataFrame, horizon: int
) -> tuple[float, float]:
    """Return (train_sharpe, test_sharpe) for the holdout split."""
    if len(signal_df) == 0:
        return np.nan, np.nan

    result = _holdout(signal_df, horizon)
    train_sharpe = result["train"]["sharpe"] if result["train"] else np.nan
    test_sharpe = result["test"]["sharpe"] if result["test"] else np.nan
    return train_sharpe, test_sharpe


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    df: pd.DataFrame,
    streak_values: list[int] | None = None,
    persistence_values: list[bool] | None = None,
) -> pd.DataFrame:
    """Iterate over parameter combinations and collect walk-forward results.

    Args:
        df: Prepared research dataset.
        streak_values: Streak thresholds to try.  Defaults to [2, 3, 4].
        persistence_values: Persistence flag values.  Defaults to [False, True].

    Returns:
        DataFrame of results sorted by ``wf48_sharpe`` descending.
    """
    if streak_values is None:
        streak_values = [2, 3, 4]
    if persistence_values is None:
        persistence_values = [False, True]

    results = []

    for streak in streak_values:
        for use_persistence in persistence_values:
            signal = build_parametric_signal(
                df,
                streak_threshold=streak,
                use_persistence=use_persistence,
                pair_group=cfg.REGIME_V2_PAIR_GROUP,
            )

            raw_n = len(signal)
            signal_spaced = enforce_global_spacing(signal, horizon=12)
            spaced_n = len(signal_spaced)

            wf12 = _walk_forward(signal_spaced, 12)
            wf48 = _walk_forward(signal_spaced, 48)
            hold12 = _holdout_sharpes(signal_spaced, 12)
            hold48 = _holdout_sharpes(signal_spaced, 48)

            results.append(
                {
                    "streak": streak,
                    "persistence": use_persistence,
                    "raw_n": raw_n,
                    "spaced_n": spaced_n,
                    "wf12_mean": wf12[0],
                    "wf12_sharpe": wf12[1],
                    "wf12_hit": wf12[2],
                    "wf48_mean": wf48[0],
                    "wf48_sharpe": wf48[1],
                    "wf48_hit": wf48[2],
                    "hold12_pre_sharpe": hold12[0],
                    "hold12_post_sharpe": hold12[1],
                    "hold48_pre_sharpe": hold48[0],
                    "hold48_post_sharpe": hold48[1],
                }
            )

    results_df = pd.DataFrame(results).sort_values("wf48_sharpe", ascending=False)
    logger.debug("Sweep complete: %d combinations evaluated", len(results_df))
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Parametric signal sweep (streak × persistence).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=str(cfg.MASTER_DATASET_PATH),
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    df = load_data(args.input)
    results = run_sweep(df)

    print("\n=== EXPERIMENT RESULTS ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
