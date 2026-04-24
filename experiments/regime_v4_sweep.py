"""
experiments/regime_v4_sweep.py
===============================
Robustness sweep framework for the Regime V4 Signal Filter pipeline.

Runs a grid sweep over key parameters of the Signal V2 × Regime Filter
hybrid pipeline and evaluates stability across the parameter space.  The
goal is to identify **flat, stable regions** rather than optimal peaks.

Parameter grid
--------------
``min_n``               : [50, 100, 150, 200]
``min_sharpe``          : [0.0, 0.05, 0.1]
``direction_threshold`` : [0.0002, 0.0005, 0.001]
``threshold``           : [None, 0.5, 1.0]  (Signal V2 position threshold)
``with_direction``      : [True, False]

Metrics per configuration
--------------------------
* ``mean_sharpe``            – mean Sharpe across walk-forward folds
* ``std_sharpe``             – Sharpe std across folds (stability)
* ``mean_coverage``          – mean fraction of test rows with active position
* ``mean_hit_rate``          – mean fraction of active positions with positive P&L
* ``capacity_adj_sharpe``    – mean_sharpe × sqrt(mean_coverage)
* ``mean_n_selected_regimes``– mean number of selected regimes per fold
* ``n_folds``                – number of valid walk-forward folds

Design constraints
------------------
* No lookahead bias — delegates entirely to :func:`walk_forward`.
* Does NOT modify regime_v4_signal_filter logic.
* Pure wrapper experiment.

Usage::

    python experiments/regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv

    # Limit to 10 configs for a quick test
    python experiments/regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv --max-runs 10
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import logging
import math
import sys
from pathlib import Path
from typing import Any

# Safe repo-root sys.path shim for direct execution
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from experiments.regime_v4_signal_filter import (
    _build_regime_base_features,
    _prepare_signal_v2,
    load_data,
    walk_forward,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "min_n": [50, 100, 150, 200],
    "min_sharpe": [0.0, 0.05, 0.1],
    "direction_threshold": [0.0002, 0.0005, 0.001],
    "threshold": [None, 0.5, 1.0],
    "with_direction": [True, False],
}

# Output columns (fixed order as specified)
_OUTPUT_COLS = [
    "min_n",
    "min_sharpe",
    "direction_threshold",
    "threshold",
    "with_direction",
    "mean_sharpe",
    "std_sharpe",
    "mean_coverage",
    "mean_hit_rate",
    "capacity_adj_sharpe",
    "mean_n_selected_regimes",
    "n_folds",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with file-only output (no stdout).

    If *log_file* is ``None``, a timestamped file is created automatically
    in ``logs/``.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v4_sweep_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logger.info("File logging enabled: %s", log_path)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _aggregate_fold_metrics(fold_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate per-fold results into sweep-level metrics.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with ``mean_sharpe``, ``std_sharpe``, ``mean_coverage``,
        ``mean_hit_rate``, ``capacity_adj_sharpe``,
        ``mean_n_selected_regimes``, ``n_folds``.
    """
    if fold_df.empty:
        return {
            "mean_sharpe": float("nan"),
            "std_sharpe": float("nan"),
            "mean_coverage": float("nan"),
            "mean_hit_rate": float("nan"),
            "capacity_adj_sharpe": float("nan"),
            "mean_n_selected_regimes": float("nan"),
            "n_folds": 0,
        }

    sharpes = fold_df["sharpe"].dropna()
    mean_sharpe = float(sharpes.mean()) if len(sharpes) > 0 else float("nan")
    std_sharpe = float(sharpes.std()) if len(sharpes) > 1 else float("nan")
    mean_coverage = float(fold_df["coverage"].mean())
    hit_rates = fold_df["hit_rate"].dropna()
    mean_hit_rate = float(hit_rates.mean()) if len(hit_rates) > 0 else float("nan")
    mean_n_selected = float(fold_df["n_selected_regimes"].mean())
    n_folds = int(len(fold_df))

    # capacity_adj_sharpe = mean_sharpe * sqrt(mean_coverage)
    if not math.isnan(mean_sharpe) and not math.isnan(mean_coverage) and mean_coverage >= 0:
        capacity_adj_sharpe = mean_sharpe * math.sqrt(mean_coverage)
    else:
        capacity_adj_sharpe = float("nan")

    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "mean_coverage": mean_coverage,
        "mean_hit_rate": mean_hit_rate,
        "capacity_adj_sharpe": capacity_adj_sharpe,
        "mean_n_selected_regimes": mean_n_selected,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# Sweep engine
# ---------------------------------------------------------------------------


def run_sweep(
    df: pd.DataFrame,
    param_grid: dict[str, list[Any]] | None = None,
    *,
    max_runs: int | None = None,
    window: int = 96,
) -> pd.DataFrame:
    """Run a grid sweep over the parameter space and collect results.

    Signal V2 features are rebuilt once per unique ``threshold`` value to
    avoid redundant computation.  The remaining parameters (``min_n``,
    ``min_sharpe``, ``direction_threshold``, ``with_direction``) are swept
    without re-computing features.

    Args:
        df: Dataset prepared by :func:`load_data` and
            :func:`_build_regime_base_features` (i.e. ``vol_24b`` present).
        param_grid: Dict mapping parameter names to lists of values.
            Defaults to :data:`DEFAULT_PARAM_GRID`.
        max_runs: Optional cap on the total number of configurations
            evaluated (for debugging).  ``None`` means evaluate all.
        window: Rolling z-score window forwarded to Signal V2.

    Returns:
        DataFrame with columns ``_OUTPUT_COLS`` sorted by
        ``capacity_adj_sharpe`` descending.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    thresholds = param_grid.get("threshold", [None])
    min_ns = param_grid.get("min_n", [100])
    min_sharpes = param_grid.get("min_sharpe", [0.05])
    direction_thresholds = param_grid.get("direction_threshold", [0.0002])
    with_directions = param_grid.get("with_direction", [False])

    # Build the full ordered grid (threshold first so we can group by it)
    all_combos = list(
        itertools.product(
            thresholds,
            min_ns,
            min_sharpes,
            direction_thresholds,
            with_directions,
        )
    )
    total = len(all_combos)
    if max_runs is not None and max_runs < total:
        all_combos = all_combos[:max_runs]
        total = len(all_combos)
        logger.info("max_runs=%d: limiting sweep to %d configurations", max_runs, total)

    logger.info("Starting sweep: %d configurations", total)

    results: list[dict[str, Any]] = []
    _cached_threshold: Any = object()  # sentinel — can never equal a real value
    df_signal: pd.DataFrame | None = None

    for run_idx, (threshold, min_n, min_sharpe, dir_thresh, with_dir) in enumerate(all_combos, start=1):
        # Re-build Signal V2 features only when threshold changes
        if threshold != _cached_threshold:
            logger.info(
                "[%d/%d] Building Signal V2 features (threshold=%s)",
                run_idx,
                total,
                threshold,
            )
            try:
                df_signal = _prepare_signal_v2(df, window=window, threshold=threshold)
            except Exception:
                logger.exception(
                    "Failed to build Signal V2 features for threshold=%s; "
                    "skipping all configurations with this threshold",
                    threshold,
                )
                df_signal = None
            _cached_threshold = threshold

        logger.info(
            "[%d/%d] SWEEP RUN | min_n=%d | min_sharpe=%.4f"
            " | direction=%.4f | threshold=%s | with_direction=%s",
            run_idx,
            total,
            min_n,
            min_sharpe,
            dir_thresh,
            threshold,
            with_dir,
        )

        if df_signal is None:
            metrics = _aggregate_fold_metrics(pd.DataFrame())
            logger.warning(
                "  → SKIPPED (signal build failed for threshold=%s)", threshold
            )
        else:
            try:
                fold_df = walk_forward(
                    df_signal,
                    min_n=min_n,
                    min_sharpe=min_sharpe,
                    with_direction=with_dir,
                    direction_threshold=dir_thresh,
                )
                metrics = _aggregate_fold_metrics(fold_df)
            except Exception:
                logger.exception(
                    "  → ERROR during walk_forward for config [%d/%d]; "
                    "filling with NaN",
                    run_idx,
                    total,
                )
                metrics = _aggregate_fold_metrics(pd.DataFrame())

        cap_adj = metrics["capacity_adj_sharpe"]
        logger.info(
            "  → sharpe=%.4f | coverage=%.4f | cap_adj=%.4f",
            metrics["mean_sharpe"],
            metrics["mean_coverage"],
            cap_adj,
        )

        row: dict[str, Any] = {
            "min_n": min_n,
            "min_sharpe": min_sharpe,
            "direction_threshold": dir_thresh,
            "threshold": threshold,
            "with_direction": with_dir,
        }
        row.update(metrics)
        results.append(row)

    if not results:
        logger.warning("Sweep produced no results")
        return pd.DataFrame(columns=_OUTPUT_COLS)

    results_df = (
        pd.DataFrame(results)[_OUTPUT_COLS]
        .sort_values("capacity_adj_sharpe", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    logger.info("Sweep complete: %d configurations evaluated", len(results_df))
    return results_df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def log_top_configs(results_df: pd.DataFrame, n: int = 10) -> None:
    """Log the top *n* configurations by ``capacity_adj_sharpe``.

    Args:
        results_df: Sorted DataFrame returned by :func:`run_sweep`.
        n: Number of top rows to log.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== TOP %d CONFIGS (by capacity_adj_sharpe) ===", n)
    logger.info(sep)
    if results_df.empty:
        logger.warning("No results to display")
        return
    top = results_df.head(n)
    logger.info("\n%s", top.to_string(index=False))


def log_sweep_summary(results_df: pd.DataFrame) -> None:
    """Log best, median, and worst configurations.

    Args:
        results_df: Sorted DataFrame returned by :func:`run_sweep`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== SWEEP SUMMARY ===")
    logger.info(sep)

    if results_df.empty:
        logger.warning("No results to summarise")
        return

    valid = results_df.dropna(subset=["capacity_adj_sharpe"])
    if valid.empty:
        logger.warning("All configurations produced NaN capacity_adj_sharpe")
        return

    best = valid.iloc[0]
    worst = valid.iloc[-1]
    median_idx = len(valid) // 2
    median = valid.iloc[median_idx]

    def _fmt_row(label: str, row: pd.Series) -> None:
        logger.info(
            "%s config (capacity_adj): min_n=%s | min_sharpe=%s"
            " | dir_thresh=%s | threshold=%s | with_dir=%s"
            " → cap_adj=%.4f | sharpe=%.4f | coverage=%.4f",
            label,
            row["min_n"],
            row["min_sharpe"],
            row["direction_threshold"],
            row["threshold"],
            row["with_direction"],
            row["capacity_adj_sharpe"],
            row["mean_sharpe"],
            row["mean_coverage"],
        )

    _fmt_row("Best   ", best)
    _fmt_row("Median ", median)
    _fmt_row("Worst  ", worst)
    logger.info(sep)


# ---------------------------------------------------------------------------
# CLI (direct execution)
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V4 Robustness Sweep: grid search over key parameters of "
            "the Signal V2 × Regime Filter pipeline.  Evaluates mean Sharpe, "
            "coverage, and capacity-adjusted Sharpe across all walk-forward "
            "folds.  Identifies flat, stable regions in parameter space."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Optional explicit log file path.  When omitted, a timestamped "
            "file is created automatically in logs/."
        ),
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        metavar="N",
        help="Limit the sweep to the first N parameter combinations (for debugging).",
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    logger.info("=== REGIME V4 ROBUSTNESS SWEEP ===")

    # Load data and build regime base features (vol_24b) — done once
    df = load_data(args.data)
    df = _build_regime_base_features(df)
    logger.info("Dataset prepared: %d rows after regime base feature build", len(df))

    # Run the sweep
    results_df = run_sweep(df, max_runs=args.max_runs)

    # Log top configurations
    log_top_configs(results_df, n=10)

    # Log summary (best / median / worst)
    log_sweep_summary(results_df)

    # Save results to CSV
    output_path = Path("data/output/regime_v4_sweep_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to: %s  (rows=%d)", output_path, len(results_df))


if __name__ == "__main__":
    main()
