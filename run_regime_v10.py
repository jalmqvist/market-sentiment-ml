"""
run_regime_v10.py
=================
Entry-point script for the Regime V10 event-ranking signal pipeline.

All feature loading, event detection, scoring, walk-forward, and metrics
logic lives in ``experiments/regime_v10.py``; this script is a thin launcher
that configures logging (file-only by default, no stdout) and delegates to
``experiments.regime_v10.main``.

Pipeline summary
----------------
Regime V10 upgrades V9 with **event ranking and selection**:

1. **Event detection** — a row is an event when ``abs_sentiment >= 70`` AND
   ``extreme_streak_70 >= 2``.  Only event rows are ever eligible for trading.

2. **Non-linear event score** (V10 upgrade)::

       score_raw = (
           (abs_sentiment / 100) ** 2
         - 0.5 * abs(net_sentiment * trend_strength_48b)
         + 0.3 * log1p(extreme_streak_70)
       )

   Normalized per fold using **train-only** z-score (no leakage).

3. **Event ranking + selection** — within each test fold, events are ranked
   by their normalized score (descending); only the top ``top_frac`` fraction
   (default 20 %) receive non-zero positions.

4. **Position** (contrarian, selected events only)::

       base_direction = −sign(net_sentiment)
       position       = base_direction × score_normalized

5. **Walk-forward** — expanding window; minimum 2 prior years before the first
   test year.

6. **Metrics per fold**: ``n_total_events``, ``n_selected_events``,
   ``selection_ratio``, ``coverage``, ``mean_score``, ``Sharpe``, ``hit_rate``.

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v10.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v10.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v10.log

    # Custom top-fraction
    python run_regime_v10.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3

    # Verbose mode
    python run_regime_v10.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import config as cfg

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

_DEFAULT_TOP_FRAC: float = 0.2


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.  When omitted, a
            file named ``regime_v10_YYYYMMDD_HHMMSS.log`` is created inside
            ``logs/``.
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


def main(argv=None) -> None:
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

    # Import after logging is configured so module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v10 import (  # noqa: PLC0415
        _REQUIRED_COLS,
        compute_pooled_summary,
        load_data,
        log_final_summary,
        log_fold_results,
        walk_forward,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    _log.info("=== REGIME V10 (EVENT RANKING + SELECTION) ===")
    _log.info("top_frac=%.4f", args.top_frac)

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v10")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(df, top_frac=args.top_frac)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
