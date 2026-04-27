# Legacy experiment — not part of current validated approach\n"""
run_regime_v12.py
=================
Entry-point script for the Regime V12 context-selection + event-ranking pipeline.

All feature loading, event detection, scoring, walk-forward, and metrics
logic lives in ``experiments/regime_v12.py``; this script is a thin launcher
that configures logging (file-only by default, no stdout) and delegates to
``experiments.regime_v12.main``.

Pipeline summary
----------------
Regime V12 upgrades V11 with **context selection** before ranking:

1. **Event detection** — a row is an event when ``abs_sentiment >= 70`` AND
   ``extreme_streak_70 >= 2``.  Only event rows are ever eligible for trading.

2. **Non-linear event score** (inherited from V10/V11)::

       score_raw = (
           (abs_sentiment / 100) ** 2
         - 0.5 * abs(net_sentiment * trend_strength_48b)
         + 0.3 * log1p(extreme_streak_70)
       )

   Normalized per fold using **train-only** z-score (no leakage).

3. **Context key** — each row is assigned a 3-component regime context key::

       context_key = vol_bucket + "_" + trend_bucket + "_" + sentiment_bucket

   All bucket thresholds are derived from training data only (no leakage):

   * ``vol_bucket``       — low / mid / high (rolling vol tertiles)
   * ``trend_bucket``     — down / flat / up (trend_strength_48b tertiles)
   * ``sentiment_bucket`` — low / mid / high (abs_sentiment tertiles)

4. **Context selection (V12 new)** — before ranking, each context is evaluated
   on training events.  A context is *selected* only when:

   * ``n_train_events >= min_context_events`` (default 30)
   * ``train_sharpe >= min_context_sharpe``   (default 0.02, new in V12)

   All other contexts — regardless of how many test events they contain — are
   discarded entirely.

5. **Context-aware ranking + selection** — within each test fold, events in
   *selected* contexts only are grouped by context key, ranked independently
   (descending score), and only the top ``top_frac`` fraction per group is traded.

6. **Position** (contrarian, selected events only)::

       base_direction = −sign(net_sentiment)
       position       = base_direction × score_normalized

7. **Walk-forward** — expanding window; minimum 2 prior years before the first
   test year.

8. **Metrics per fold**: ``n_total_events``, ``n_selected_events``,
   ``selection_ratio``, ``coverage``, ``mean_score``, ``Sharpe``, ``hit_rate``,
   ``n_contexts``, ``n_contexts_selected``, ``n_contexts_rejected``.

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v12.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v12.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v12.log

    # Custom parameters
    python run_regime_v12.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3 \\
        --min-context-events 20 \\
        --min-context-sharpe 0.05

    # Verbose mode
    python run_regime_v12.py \\
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
_DEFAULT_MIN_CONTEXT_EVENTS: int = 30
_DEFAULT_MIN_CONTEXT_SHARPE: float = 0.02


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.  When omitted, a
            file named ``regime_v12_YYYYMMDD_HHMMSS.log`` is created inside
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
        log_path = logs_dir / f"regime_v12_{timestamp}.log"
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
            "Regime V12: context selection + context-aware event ranking pipeline. "
            "Detects high-conviction sentiment events "
            "(abs_sentiment >= 70 AND extreme_streak_70 >= 2), evaluates each "
            "context on training data (n + Sharpe filter), keeps only "
            "signal-bearing contexts, then ranks events WITHIN each surviving "
            "context independently. "
            "Walk-forward expanding window with train-only normalization."
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
            "Fraction of top-ranked events to trade per context group per fold "
            f"(0 < FRAC <= 1). Default: {_DEFAULT_TOP_FRAC}."
        ),
    )
    p.add_argument(
        "--min-context-events",
        type=int,
        default=_DEFAULT_MIN_CONTEXT_EVENTS,
        metavar="N",
        help=(
            "Minimum number of training events required in a context for it to "
            f"be selected. Default: {_DEFAULT_MIN_CONTEXT_EVENTS}."
        ),
    )
    p.add_argument(
        "--min-context-sharpe",
        type=float,
        default=_DEFAULT_MIN_CONTEXT_SHARPE,
        metavar="S",
        help=(
            "Minimum training Sharpe required for a context to be selected. "
            f"Default: {_DEFAULT_MIN_CONTEXT_SHARPE}."
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
    from experiments.regime_v12 import (  # noqa: PLC0415
        _REQUIRED_COLS,
        compute_pooled_summary,
        load_data,
        log_final_summary,
        log_fold_results,
        walk_forward,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    _log.info("=== REGIME V12 (CONTEXT SELECTION + EVENT RANKING) ===")
    _log.info(
        "top_frac=%.4f | min_context_events=%d | min_context_sharpe=%.4f",
        args.top_frac,
        args.min_context_events,
        args.min_context_sharpe,
    )

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v12")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        top_frac=args.top_frac,
        min_context_events=args.min_context_events,
        min_context_sharpe=args.min_context_sharpe,
    )

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
