# Legacy experiment — not part of current validated approach\n"""
run_regime_v7_2.py
==================
Entry-point script for the Regime V7.2 interaction-based event scoring pipeline.

All score-building, walk-forward, and metrics logic lives in
``experiments/regime_v7_2.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v7_2.main``.

Pipeline summary
----------------
Regime V7.2 upgrades V7.1 by replacing additive score definitions with
*multiplicative interactions* to correctly model nonlinear behavioural effects:

1. **SATURATION_SCORE** – crowd saturation: z(abs_sentiment) ×
   z(extreme_streak_70) × z(trend_strength_48b), bounded with tanh.

2. **DIVERGENCE_SCORE** – sentiment-price divergence: z(divergence) ×
   z(abs_sentiment), bounded with tanh.

3. **EXHAUSTION_SCORE** – trend exhaustion: z(extreme_streak_70) ×
   (−z(trend_strength_48b)), bounded with tanh.

After computing the raw scores each row is L1-normalised across the three
score types.  The score with the greatest absolute magnitude is then selected.
If it exceeds ``--score-threshold`` a position is taken; otherwise the
position is 0.  All z-score normalisation uses training-split statistics only
(no forward leakage).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v7_2.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v7_2.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v7_2.log

    # Verbose mode
    python run_regime_v7_2.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Custom threshold with score-weighting
    python run_regime_v7_2.py \\
        --data data/output/master_research_dataset.csv \\
        --score-threshold 0.3 --use-score-weighting

    # Minimum-n and score threshold
    python run_regime_v7_2.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 50 --score-threshold 0.5
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
            file named ``regime_v7_2_YYYYMMDD_HHMMSS.log`` is created inside
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
        log_path = logs_dir / f"regime_v7_2_{timestamp}.log"
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
            "Regime V7.2: interaction-based event scoring pipeline. "
            "Computes multiplicative interaction scores (saturation, "
            "divergence, exhaustion), row-normalises them, and trades only "
            "when the strongest signal exceeds a configurable threshold."
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
        help="Minimum training observations for diagnostics warning. Default: 50.",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        metavar="T",
        help=(
            "Minimum best-score magnitude to take a position. "
            "Default: 0.3."
        ),
    )
    p.add_argument(
        "--use-score-weighting",
        action="store_true",
        default=False,
        help=(
            "Weight position by score magnitude instead of applying only "
            "direction.  Default: off (direction-only)."
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
    from experiments.regime_v7_2 import (  # noqa: PLC0415
        DEFAULT_MIN_N,
        DEFAULT_SCORE_THRESHOLD,
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
        "=== REGIME V7.2 ==="
        " | min_n=%d | score_threshold=%.4f | use_score_weighting=%s",
        args.min_n,
        args.score_threshold,
        args.use_score_weighting,
    )

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v7_2")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        score_threshold=args.score_threshold,
        use_score_weighting=args.use_score_weighting,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
