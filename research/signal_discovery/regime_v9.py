# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v9.py
========================
EVENT-BASED signal pipeline for FX sentiment research — **Version 9**.

V9 replaces the regression model (V8/V8.3) with **event detection + event
scoring**.  Empirical results from V8 showed weak regression IC and sparse
predictability, suggesting that signal appears only in specific high-conviction
conditions.  V9 addresses this directly by:

1. **Defining events** – only high-conviction rows are traded.
2. **Scoring events** – a hand-crafted formula quantifies conviction strength.
3. **Contrarian positioning** – fade extreme sentiment.
4. **Walk-forward with train-only normalization** – no leakage.

Pipeline overview
-----------------
1. **Event flag** – a row is an event when ALL of:

   * ``abs_sentiment >= 70``
   * ``extreme_streak_70 >= 2``

   Only event rows are ever traded.

2. **Raw score** (computed for all rows; applied only at events)::

       score_raw = (
           0.5 * abs_sentiment
         − 0.3 * (net_sentiment × trend_strength_48b)
         + 0.2 * extreme_streak_70
       )

3. **Score normalization** – per fold, z-score parameters (mean, std) are
   estimated on the **train split only** and applied to both train and test
   splits.  No test information enters normalization.

4. **Position** (contrarian)::

       base_direction = −sign(net_sentiment)   # fade extreme sentiment
       position       = base_direction × score_normalized

   Position is set to 0 for non-event rows.

5. **Walk-forward** – same expanding-window logic as V8 (minimum 2 prior
   years before first test year to ensure sufficient training data).

6. **Metrics** (per fold):

   * ``n_events``    – number of event rows in the test split
   * ``coverage``    – ``n_events / len(test_split)``
   * ``mean_score``  – mean normalized score across event rows
   * ``sharpe``      – ``mean(pnl) / std(pnl)``  (event rows only)
   * ``hit_rate``    – fraction of event rows with ``pnl > 0``

Required columns
----------------
``ret_48b``, ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``

Fold output schema
------------------
``["year", "n_events", "coverage", "mean_score", "sharpe", "hit_rate"]``

Logging (per fold)
------------------
* Number of events in test split
* Event coverage %
* Mean normalized score
* Sharpe

Usage (direct)::

    python experiments/regime_v9.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v9.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import datetime
import logging
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
from utils.io import read_csv
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Columns that must be present in the dataset.
_REQUIRED_COLS: list[str] = [
    TARGET_COL,
    "net_sentiment",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
]

#: Event definition thresholds.
_EVENT_ABS_SENTIMENT_MIN: int = 70
_EVENT_STREAK_MIN: int = 2

#: Score formula weights.
_SCORE_W_ABS: float = 0.5
_SCORE_W_TREND: float = 0.3
_SCORE_W_STREAK: float = 0.2

#: Output fold columns.
_FOLD_COLS: list[str] = [
    "year",
    "n_events",
    "coverage",
    "mean_score",
    "sharpe",
    "hit_rate",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is ``None``, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

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
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        log_path = logs_dir / f"regime_v9_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logging.getLogger(__name__).info("File logging enabled: %s", log_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the dataset for Regime V9.

    Reads the canonical research dataset CSV, parses the ``time`` column to
    derive a ``year`` column, and validates that all required columns are
    present.  NaN rows are **not** dropped here; that is deferred to the
    walk-forward step.

    Args:
        path: Path to the master research dataset CSV.

    Returns:
        DataFrame with a ``year`` column added.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing after loading.
    """
    df = read_csv(path, required_columns=["time"] + _REQUIRED_COLS)
    df = parse_timestamps(df, "time", context="regime_v9.load_data")
    df["year"] = df["time"].dt.year

    logger.info(
        "load_data: %d rows, %d pairs, date_range=%s .. %s",
        len(df),
        df["pair"].nunique() if "pair" in df.columns else "N/A",
        df["time"].min(),
        df["time"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Event flag
# ---------------------------------------------------------------------------

def compute_event_flag(df: pd.DataFrame) -> pd.Series:
    """Compute the event flag column.

    A row is flagged as an event when:

    * ``abs_sentiment >= 70``   (high-conviction extreme reading)
    * ``extreme_streak_70 >= 2`` (persistence of at least 2 consecutive bars)

    Args:
        df: DataFrame with ``abs_sentiment`` and ``extreme_streak_70`` columns.

    Returns:
        Integer Series (0/1) aligned with *df*.
    """
    return (
        (df["abs_sentiment"] >= _EVENT_ABS_SENTIMENT_MIN)
        & (df["extreme_streak_70"] >= _EVENT_STREAK_MIN)
    ).astype(int)


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------

def compute_raw_score(df: pd.DataFrame) -> pd.Series:
    """Compute the raw (un-normalized) event score for every row.

    Formula::

        score_raw = (
            0.5 * abs_sentiment
          − 0.3 * (net_sentiment × trend_strength_48b)
          + 0.2 * extreme_streak_70
        )

    The contrarian rationale:

    * ``abs_sentiment`` rewards higher-conviction readings.
    * ``net_sentiment × trend_strength_48b`` penalizes trend-aligned sentiment
      (trend alignment reduces mean-reversion potential).
    * ``extreme_streak_70`` rewards persistence.

    Args:
        df: DataFrame with the required feature columns.

    Returns:
        Float Series aligned with *df*.
    """
    raw = (
        _SCORE_W_ABS * df["abs_sentiment"]
        - _SCORE_W_TREND * (df["net_sentiment"] * df["trend_strength_48b"])
        + _SCORE_W_STREAK * df["extreme_streak_70"]
    )
    return raw.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
) -> pd.DataFrame:
    """Regime-V9 walk-forward: event detection + event scoring.

    For each test year (from the third unique year onward, i.e. at least 2
    training years):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any required column or in ``score_raw``.
    3. Fit z-score normalization parameters (mean, std) on the **train** split.
    4. Normalize scores on both train and test using train parameters.
    5. Compute the event flag; restrict to event rows in the test split.
    6. Build contrarian positions for event rows.
    7. Compute fold-level metrics.

    No test-period information enters score normalization.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            in ``_REQUIRED_COLS``.
        target_col: Forward-return column to evaluate positions against.
        year_col: Column containing calendar year.

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
    if year_col not in df.columns:
        logger.warning("walk_forward: year column '%s' not found", year_col)
        return pd.DataFrame(columns=_FOLD_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "walk_forward: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    # Pre-compute raw score for all rows; NaN rows are dropped per fold
    df = df.copy()
    df["score_raw"] = compute_raw_score(df)
    df["is_event"] = compute_event_flag(df)

    score_cols = _REQUIRED_COLS + ["score_raw"]
    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year].dropna(subset=score_cols)
        test_df = df[df[year_col] == test_year].dropna(subset=score_cols)

        if train_df.empty:
            logger.warning(
                "REGIME V9 [year=%d]: empty train set; skipping fold", test_year
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V9 [year=%d]: empty test set; skipping fold", test_year
            )
            continue

        logger.info(
            "REGIME V9 [year=%d] | train_size=%d | test_size=%d",
            test_year,
            len(train_df),
            len(test_df),
        )

        # ------------------------------------------------------------------
        # Fit z-score parameters on TRAIN only (no leakage)
        # ------------------------------------------------------------------
        score_mean = float(train_df["score_raw"].mean())
        score_std = float(train_df["score_raw"].std())

        if score_std < 1e-10:
            logger.warning(
                "REGIME V9 [year=%d]: train score_std=%.2e near zero; "
                "using unscaled scores",
                test_year,
                score_std,
            )
            score_std = 1.0

        # ------------------------------------------------------------------
        # Normalize test scores using TRAIN parameters
        # ------------------------------------------------------------------
        test_score_norm = (test_df["score_raw"] - score_mean) / score_std

        # ------------------------------------------------------------------
        # Apply event filter on test set
        # ------------------------------------------------------------------
        test_events = test_df["is_event"].values.astype(bool)
        n_total = len(test_df)
        n_events = int(test_events.sum())

        coverage = n_events / n_total if n_total > 0 else 0.0

        logger.info(
            "REGIME V9 [year=%d] | n_events=%d | coverage=%.2f%%",
            test_year,
            n_events,
            100 * coverage,
        )

        if n_events == 0:
            logger.warning(
                "REGIME V9 [year=%d]: no events in test set; skipping fold",
                test_year,
            )
            continue

        # Restrict to event rows
        event_score = test_score_norm.values[test_events]
        net_sent_event = test_df["net_sentiment"].values[test_events]
        ret_event = test_df[target_col].values.astype(float)[test_events]

        mean_score = float(np.mean(event_score))
        logger.info(
            "REGIME V9 [year=%d] | mean_score (normalized)=%.4f",
            test_year,
            mean_score,
        )

        # ------------------------------------------------------------------
        # Contrarian position
        # ------------------------------------------------------------------
        # Fade extreme sentiment: go against the direction of net_sentiment
        base_direction = -np.sign(net_sent_event)
        position = base_direction * event_score

        pnl = position * ret_event

        metrics = _fold_metrics(pnl, n_events, coverage, mean_score)

        logger.info(
            "REGIME V9 FOLD | year=%d | n_events=%5d | coverage=%.2f%%"
            " | mean_score=%+.4f | sharpe=%+.4f | hit_rate=%.4f",
            test_year,
            metrics["n_events"],
            100 * metrics["coverage"],
            metrics["mean_score"],
            metrics["sharpe"],
            metrics["hit_rate"],
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V9: no valid folds produced")
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    pnl: np.ndarray,
    n_events: int,
    coverage: float,
    mean_score: float,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V9.

    Args:
        pnl: PnL series (``position × ret_48b``) for event rows.
        n_events: Number of event rows in the test split.
        coverage: ``n_events / total_test_rows``.
        mean_score: Mean normalized score across event rows.

    Returns:
        Dict with keys: n_events, coverage, mean_score, sharpe, hit_rate.
    """
    n = len(pnl)

    if n < 2:
        return {
            "n_events": n_events,
            "coverage": coverage,
            "mean_score": mean_score,
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
        }

    mean_ret = float(np.mean(pnl))
    std_ret = float(np.std(pnl))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
    hit_rate = float(np.mean(pnl > 0))

    return {
        "n_events": n_events,
        "coverage": coverage,
        "mean_score": mean_score,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with keys: folds, mean_sharpe, mean_hit_rate, mean_coverage.
    """
    if fold_df.empty:
        return {
            "folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
        }
    return {
        "folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V9 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V9: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n_events=%5d | coverage=%.2f%%"
            " | mean_score=%+.4f | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.n_events,
            100 * row.coverage,
            row.mean_score,
            row.sharpe,
            row.hit_rate,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the Regime V9 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        summary: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V9 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V9 SUMMARY: no results")
        return

    logger.info("Folds evaluated  : %d", summary["folds"])
    logger.info("Mean Sharpe      : %+.4f", summary["mean_sharpe"])
    logger.info("Mean hit rate    : %.4f", summary["mean_hit_rate"])
    logger.info("Mean coverage    : %.4f", summary["mean_coverage"])
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | n_events=%5d | coverage=%.2f%%"
            " | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.n_events,
            100 * row.coverage,
            row.sharpe,
            row.hit_rate,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v9.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V9 event-based signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V9: event-based signal pipeline. "
            "Detects high-conviction sentiment events "
            "(abs_sentiment >= 70 AND extreme_streak_70 >= 2) and takes "
            "contrarian positions sized by a normalized event score. "
            "Walk-forward expanding window with train-only score normalization."
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
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    log = logging.getLogger(__name__)
    log.info("=== REGIME V9 (EVENT-BASED) ===")

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v9.main")
    log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(df)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
