"""
run_regime_v7.py
================
Entry-point script for the Regime V7 event-based signal pipeline.

All event-detection, walk-forward, and metrics logic lives in
``experiments/regime_v7.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v7.main``.

Pipeline summary
----------------
Regime V7 replaces continuous regime-weighting with discrete *event detection*.
A position is taken only when a statistically validated behavioural event fires
on the test row:

1. **SATURATION_EVENT** – crowd at extreme sentiment (> 70) AND persistent
   (``extreme_streak_70 >= streak_threshold``) AND trending
   (``abs(trend_strength_48b) > trend_threshold``).

2. **DIVERGENCE_EVENT** – sentiment strongly diverges from price momentum
   (``abs(divergence) > divergence_threshold``).

3. **EXHAUSTION_EVENT** – crowd is extreme and persistent BUT the trend is
   weakening (``abs(trend_strength_48b) <= trend_threshold``).

Events are validated on the **training split only** (Sharpe + minimum-n
thresholds).  On the test split:

* If a validated event fires → ``position = tanh(signal_v2_raw) * direction``
* If multiple fire → the event with the highest train-set Sharpe wins.
* If none fire → ``position = 0``.

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v7.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v7.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v7.log

    # Verbose mode
    python run_regime_v7.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Custom thresholds
    python run_regime_v7.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 50 --min-sharpe 0.02 \\
        --streak-threshold 3 --trend-threshold 0.5 \\
        --divergence-threshold 1.0
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import config as cfg

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.  When omitted, a
            file named ``regime_v7_YYYYMMDD_HHMMSS.log`` is created inside
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
        log_path = logs_dir / f"regime_v7_{timestamp}.log"
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
            "Regime V7: event-based signal pipeline. "
            "Detects validated sentiment events (saturation, divergence, "
            "exhaustion) on training data and trades only when those events "
            "fire on the test set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=50,
        metavar="N",
        help="Minimum training event observations for validation. Default: 50.",
    )
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=0.02,
        metavar="S",
        help=(
            "Minimum train-set Sharpe for an event to be considered valid. "
            "Default: 0.02."
        ),
    )
    p.add_argument(
        "--streak-threshold",
        type=int,
        default=3,
        metavar="K",
        help=(
            "Minimum extreme_streak_70 value for saturation / exhaustion events. "
            "Default: 3."
        ),
    )
    p.add_argument(
        "--trend-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help=(
            "Boundary for abs(trend_strength_48b): above → saturation, "
            "at-or-below → exhaustion. Default: 0.5."
        ),
    )
    p.add_argument(
        "--divergence-threshold",
        type=float,
        default=1.0,
        metavar="D",
        help=(
            "Minimum abs(divergence) for the DIVERGENCE_EVENT. Default: 1.0."
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
    from experiments.regime_v7 import (  # noqa: PLC0415
        DEFAULT_MIN_N,
        DEFAULT_MIN_SHARPE,
        DEFAULT_STREAK_THRESHOLD,
        DEFAULT_TREND_THRESHOLD,
        DEFAULT_DIVERGENCE_THRESHOLD,
        TARGET_COL,
        _REQUIRED_COLS,
        compute_pooled_summary,
        load_data,
        log_final_summary,
        log_fold_results,
        walk_forward,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    _log.info(
        "=== REGIME V7 ==="
        " | min_n=%d | min_sharpe=%.4f"
        " | streak_threshold=%d | trend_threshold=%.2f"
        " | divergence_threshold=%.2f",
        args.min_n,
        args.min_sharpe,
        args.streak_threshold,
        args.trend_threshold,
        args.divergence_threshold,
    )

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v7")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        min_sharpe=args.min_sharpe,
        streak_threshold=args.streak_threshold,
        trend_threshold=args.trend_threshold,
        divergence_threshold=args.divergence_threshold,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
