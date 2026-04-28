# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v10.py
=========================
EVENT-BASED signal pipeline for FX sentiment research — **Version 10**.

V10 is a structural upgrade of V9. Where V9 treated all detected events
equally, V10 introduces **event ranking and selection**: within each test fold,
events are ranked by their conviction score and only the top fraction (default
20 %) are traded.

Key changes vs V9
-----------------
* **Non-linear score formula** — squares the abs_sentiment term (compounding
  the reward for extreme readings) and applies a stronger penalty for
  trend-aligned sentiment via ``abs(...)``.
* **Event ranking** — events are ranked by normalized score (descending) inside
  each test fold.
* **Top-fraction selection** — only events in the top ``top_frac`` fraction are
  assigned non-zero positions; the rest receive position = 0.
* **Extended logging** — per-fold logs now show ``total_events``,
  ``selected_events``, and ``selection_ratio``.

Pipeline overview
-----------------
1. **Event flag** — a row is an event when ALL of:

   * ``abs_sentiment >= 70``
   * ``extreme_streak_70 >= 2``

   Only event rows are ever eligible for trading.

2. **Raw score** (computed for all rows; applied only at events)::

       score_raw = (
           (abs_sentiment / 100) ** 2
         - 0.5 * abs(net_sentiment * trend_strength_48b)
         + 0.3 * log1p(extreme_streak_70)
       )

3. **Score normalization** — per fold, z-score parameters (mean, std) are
   estimated on the **train split only** and applied to both train and test
   splits.  No test information enters normalization.

4. **Event ranking + selection** (per fold, test set only):

   a. Extract event rows from the test split.
   b. Rank events by normalized score (descending).
   c. Keep only the top ``top_frac`` fraction.
   d. All remaining events receive position = 0.

5. **Position** (contrarian, selected events only)::

       base_direction = -sign(net_sentiment)   # fade extreme sentiment
       position       = base_direction × score_normalized

6. **Walk-forward** — same expanding-window logic as V9 (minimum 2 prior
   years before first test year).

7. **Metrics** (per fold):

   * ``n_total_events``   — events detected in the test split
   * ``n_selected_events`` — events traded (after top-frac selection)
   * ``selection_ratio``  — ``n_selected_events / n_total_events``
   * ``coverage``         — ``n_selected_events / len(test_split)``
   * ``mean_score``       — mean normalized score across *selected* events
   * ``sharpe``           — ``mean(pnl) / std(pnl)``  (selected events only)
   * ``hit_rate``         — fraction of selected events with ``pnl > 0``

Required columns
----------------
``ret_48b``, ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``

Fold output schema
------------------
``["year", "n_total_events", "n_selected_events", "selection_ratio",
   "coverage", "mean_score", "sharpe", "hit_rate"]``

Logging (per fold)
------------------
* Total events in test split
* Selected events after top-frac filter
* Selection ratio
* Event coverage %
* Mean normalized score
* Sharpe

Usage (direct)::

    python experiments/regime_v10.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v10.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3 \\
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

#: Event definition thresholds (unchanged from V9).
_EVENT_ABS_SENTIMENT_MIN: int = 70
_EVENT_STREAK_MIN: int = 2

#: Default fraction of top-ranked events to trade.
_DEFAULT_TOP_FRAC: float = 0.2

#: Output fold columns.
_FOLD_COLS: list[str] = [
    "year",
    "n_total_events",
    "n_selected_events",
    "selection_ratio",
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
        log_path = logs_dir / f"regime_v10_{timestamp}.log"
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
    """Load and prepare the dataset for Regime V10.

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
    df = parse_timestamps(df, "time", context="regime_v10.load_data")
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
    """Compute the event flag column (identical to V9).

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
# Score formula (non-linear — V10 upgrade)
# ---------------------------------------------------------------------------

def compute_raw_score(df: pd.DataFrame) -> pd.Series:
    """Compute the raw (un-normalized) event score for every row.

    V10 non-linear formula::

        score_raw = (
            (abs_sentiment / 100) ** 2
          - 0.5 * abs(net_sentiment * trend_strength_48b)
          + 0.3 * log1p(extreme_streak_70)
        )

    Design rationale:

    * ``(abs_sentiment / 100) ** 2`` — squaring rewards extreme sentiment
      much more than moderate sentiment; the /100 keeps the term in [0, 1].
    * ``-0.5 * abs(net_sentiment * trend_strength_48b)`` — absolute value
      penalizes both bullish and bearish trend-aligned sentiment, reducing
      mean-reversion potential regardless of direction.
    * ``0.3 * log1p(extreme_streak_70)`` — adds diminishing returns for
      streak length.

    Args:
        df: DataFrame with the required feature columns.

    Returns:
        Float Series aligned with *df*.
    """
    raw = (
        (df["abs_sentiment"] / 100.0) ** 2
        - 0.5 * (df["net_sentiment"] * df["trend_strength_48b"]).abs()
        + 0.3 * np.log1p(df["extreme_streak_70"])
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
    top_frac: float = _DEFAULT_TOP_FRAC,
) -> pd.DataFrame:
    """Regime-V10 walk-forward: event detection + ranking + top-frac selection.

    For each test year (from the third unique year onward, i.e. at least 2
    training years):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any required column or in ``score_raw``.
    3. Fit z-score normalization parameters (mean, std) on the **train** split.
    4. Normalize scores on both train and test using train parameters.
    5. Compute the event flag; restrict to event rows in the test split.
    6. **Rank** event rows by normalized score (descending).
    7. **Select** only the top ``top_frac`` fraction of ranked events.
    8. Build contrarian positions for selected events (all others: position=0).
    9. Compute fold-level metrics.

    No test-period information enters score normalization.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            in ``_REQUIRED_COLS``.
        target_col: Forward-return column to evaluate positions against.
        year_col: Column containing calendar year.
        top_frac: Fraction (0, 1] of top-ranked events to trade per fold.

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(
            f"top_frac must be in (0, 1]; got {top_frac}"
        )

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
                "REGIME V10 [year=%d]: empty train set; skipping fold",
                test_year,
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V10 [year=%d]: empty test set; skipping fold",
                test_year,
            )
            continue

        logger.info(
            "REGIME V10 [year=%d] | train_size=%d | test_size=%d",
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
                "REGIME V10 [year=%d]: train score_std=%.2e near zero; "
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
        test_events_mask = test_df["is_event"].values.astype(bool)
        n_total = len(test_df)
        n_total_events = int(test_events_mask.sum())

        logger.info(
            "REGIME V10 [year=%d] | total_events=%d",
            test_year,
            n_total_events,
        )

        if n_total_events == 0:
            logger.warning(
                "REGIME V10 [year=%d]: no events in test set; skipping fold",
                test_year,
            )
            continue

        # Restrict to event rows
        event_score_norm = test_score_norm.values[test_events_mask]
        net_sent_event = test_df["net_sentiment"].values[test_events_mask]
        ret_event = test_df[target_col].values.astype(float)[test_events_mask]

        # ------------------------------------------------------------------
        # Rank events by normalized score (descending) and select top_frac
        # ------------------------------------------------------------------
        n_select = max(1, int(np.ceil(top_frac * n_total_events)))
        # argsort descending → indices of the top-n_select events
        ranked_idx = np.argsort(event_score_norm)[::-1]
        selected_mask = np.zeros(n_total_events, dtype=bool)
        selected_mask[ranked_idx[:n_select]] = True

        n_selected_events = int(selected_mask.sum())
        selection_ratio = n_selected_events / n_total_events
        coverage = n_selected_events / n_total if n_total > 0 else 0.0

        logger.info(
            "REGIME V10 [year=%d] | selected_events=%d | selection_ratio=%.4f"
            " | coverage=%.2f%%",
            test_year,
            n_selected_events,
            selection_ratio,
            100 * coverage,
        )

        # Work only with selected events
        sel_score = event_score_norm[selected_mask]
        sel_net_sent = net_sent_event[selected_mask]
        sel_ret = ret_event[selected_mask]

        mean_score = float(np.mean(sel_score))
        logger.info(
            "REGIME V10 [year=%d] | mean_score (normalized, selected)=%.4f",
            test_year,
            mean_score,
        )

        # ------------------------------------------------------------------
        # Contrarian position (selected events only)
        # ------------------------------------------------------------------
        base_direction = -np.sign(sel_net_sent)
        position = base_direction * sel_score

        pnl = position * sel_ret

        metrics = _fold_metrics(
            pnl,
            n_total_events=n_total_events,
            n_selected_events=n_selected_events,
            selection_ratio=selection_ratio,
            coverage=coverage,
            mean_score=mean_score,
        )

        logger.info(
            "REGIME V10 FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%% | mean_score=%+.4f"
            " | sharpe=%+.4f | hit_rate=%.4f",
            test_year,
            metrics["n_total_events"],
            metrics["n_selected_events"],
            metrics["selection_ratio"],
            100 * metrics["coverage"],
            metrics["mean_score"],
            metrics["sharpe"],
            metrics["hit_rate"],
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V10: no valid folds produced")
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    pnl: np.ndarray,
    *,
    n_total_events: int,
    n_selected_events: int,
    selection_ratio: float,
    coverage: float,
    mean_score: float,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V10.

    Args:
        pnl: PnL series (``position × ret_48b``) for *selected* event rows.
        n_total_events: Total events detected in the test split.
        n_selected_events: Events traded (after top-frac selection).
        selection_ratio: ``n_selected_events / n_total_events``.
        coverage: ``n_selected_events / total_test_rows``.
        mean_score: Mean normalized score across *selected* event rows.

    Returns:
        Dict with keys matching ``_FOLD_COLS`` (minus ``year``).
    """
    n = len(pnl)

    if n < 2:
        return {
            "n_total_events": n_total_events,
            "n_selected_events": n_selected_events,
            "selection_ratio": selection_ratio,
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
        "n_total_events": n_total_events,
        "n_selected_events": n_selected_events,
        "selection_ratio": selection_ratio,
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
        Dict with keys: folds, mean_sharpe, mean_hit_rate, mean_coverage,
        mean_selection_ratio.
    """
    if fold_df.empty:
        return {
            "folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
            "mean_selection_ratio": float("nan"),
        }
    return {
        "folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
        "mean_selection_ratio": float(fold_df["selection_ratio"].mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V10 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V10: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%%"
            " | mean_score=%+.4f | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100 * row.coverage,
            row.mean_score,
            row.sharpe,
            row.hit_rate,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the Regime V10 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        summary: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V10 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V10 SUMMARY: no results")
        return

    logger.info("Folds evaluated      : %d", summary["folds"])
    logger.info("Mean Sharpe          : %+.4f", summary["mean_sharpe"])
    logger.info("Mean hit rate        : %.4f", summary["mean_hit_rate"])
    logger.info("Mean coverage        : %.4f", summary["mean_coverage"])
    logger.info("Mean selection ratio : %.4f", summary["mean_selection_ratio"])
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | total=%5d | selected=%5d | sel_ratio=%.4f"
            " | coverage=%.2f%% | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100 * row.coverage,
            row.sharpe,
            row.hit_rate,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v10.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V10 event-ranking signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V10: event ranking + selection pipeline. "
            "Detects high-conviction sentiment events "
            "(abs_sentiment >= 70 AND extreme_streak_70 >= 2), ranks them by "
            "a non-linear conviction score, and trades only the top fraction. "
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
        "--top-frac",
        type=float,
        default=_DEFAULT_TOP_FRAC,
        metavar="FRAC",
        help=(
            "Fraction of top-ranked events to trade per fold (0 < FRAC <= 1). "
            f"Default: {_DEFAULT_TOP_FRAC}."
        ),
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
    log.info("=== REGIME V10 (EVENT RANKING + SELECTION) ===")
    log.info("top_frac=%.4f", args.top_frac)

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v10.main")
    log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(df, top_frac=args.top_frac)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
