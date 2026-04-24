"""
run_regime_filter_pipeline.py
=============================
Entry-point script for the Regime Filter Pipeline.

Uses regimes to **FILTER** trades (decide *when* to trade), not to predict
return magnitudes.  All walk-forward and regime-statistics logic lives in
``experiments/regime_filter_pipeline.py``; this script is a thin launcher
that configures logging and delegates to
``experiments.regime_filter_pipeline.main``.

Pipeline steps
--------------
1. Load and prepare the research dataset.
2. Compute causal volatility feature (``vol_24b``) via ``build_features``.
3. Build three discrete regime features per fold (training-derived cuts):

   * ``vol_regime``  – tertile of ``vol_24b``
   * ``trend_dir``   – sign of ``trend_strength_48b``
   * ``sent_regime`` – fixed-threshold bin of ``abs_sentiment``

4. Walk-forward regime selection (train only): filter regimes with
   ``n >= min_n``, rank by Sharpe.  Iterate through top ``candidate_k``
   regimes; apply test-side stability filters inside the loop; keep up
   to ``final_k`` that pass (default 5).
5. Test-side stability filters applied **inside** the selection loop:
   drop regimes with ``n_test < min_test_n`` or
   ``coverage_test < min_test_coverage``.
6. Apply regime filter to the test set.
7. Optional direction logic: follow/fade only if ``|train_mean| >= direction_threshold``.
8. Compute per-fold metrics (mean return, Sharpe, hit rate, coverage).
9. Log pooled summary.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_filter_pipeline.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_filter_pipeline.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_filter.log

    # Verbose mode
    python run_regime_filter_pipeline.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Custom thresholds
    python run_regime_filter_pipeline.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 150 --candidate-k 20 --final-k 8 --min-test-coverage 0.03 --min-test-n 30

    # Disable direction logic
    python run_regime_filter_pipeline.py \\
        --data data/output/master_research_dataset.csv \\
        --no-direction

    # Disable persistence filtering
    python run_regime_filter_pipeline.py \\
        --data data/output/master_research_dataset.csv \\
        --min-persistence-folds 0
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
            file named ``regime_filter_pipeline_YYYYMMDD_HHMMSS.log`` is
            created inside ``logs/``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_filter_pipeline_{timestamp}.log"
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
            "Regime Filter Pipeline: use discrete regime features to FILTER "
            "trades (when to trade), not to predict returns.  No LightGBM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(cfg.DATA_PATH),
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
        "--min-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum training observations per regime for selection.  "
            "Defaults to the module-level MIN_REGIME_N constant (100)."
        ),
    )
    p.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Size of the initial candidate pool: maximum number of top regimes "
            "(by training Sharpe) to iterate through in the selection loop.  "
            "Defaults to the module-level CANDIDATE_K constant (15)."
        ),
    )
    p.add_argument(
        "--final-k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Maximum number of regimes to keep after applying test-side "
            "stability filters inside the selection loop.  "
            "Defaults to the module-level FINAL_K constant (5)."
        ),
    )
    p.add_argument(
        "--min-persistence-folds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum number of prior walk-forward folds in which a regime must "
            "have appeared as a candidate (n >= min_n) before it is eligible "
            "for selection.  Set to 0 to disable persistence filtering.  "
            "Defaults to the module-level MIN_PERSISTENCE_FOLDS constant (2)."
        ),
    )
    p.add_argument(
        "--min-test-coverage",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Minimum fraction of test signals a regime must cover to survive "
            "the stability filter.  "
            "Defaults to the module-level MIN_TEST_COVERAGE constant (0.05)."
        ),
    )
    p.add_argument(
        "--min-test-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum raw count of test signals per regime to survive the "
            "stability filter.  "
            "Defaults to the module-level MIN_TEST_N constant (50)."
        ),
    )
    p.add_argument(
        "--with-direction",
        action="store_true",
        default=True,
        help=(
            "Apply direction logic: invert returns for regimes with "
            "negative training mean (fade the crowd).  Enabled by default."
        ),
    )
    p.add_argument(
        "--no-direction",
        dest="with_direction",
        action="store_false",
        help="Disable direction logic; use raw returns for all filtered regimes.",
    )
    p.add_argument(
        "--direction-threshold",
        type=float,
        default=None,
        metavar="THRESH",
        help=(
            "Minimum |train mean| to activate follow/fade direction logic.  "
            "Below this threshold signals are kept unmodified.  "
            "Defaults to the module-level DIRECTION_THRESHOLD constant (0.0002)."
        ),
    )
    p.add_argument(
        "--top-n-log",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of top regimes to log per fold (by training Sharpe).  "
            "Defaults to the module-level TOP_N_LOG constant (5)."
        ),
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    # Import after logging is configured so module-level loggers pick up handlers.
    from experiments.regime_filter_pipeline import (  # noqa: PLC0415
        CANDIDATE_K,
        DIRECTION_THRESHOLD,
        FINAL_K,
        MIN_PERSISTENCE_FOLDS,
        MIN_REGIME_N,
        MIN_TEST_COVERAGE,
        MIN_TEST_N,
        TARGET_COL,
        TOP_N_LOG,
        build_features,
        compute_pooled_summary,
        load_data,
        log_regime_filter_fold_results,
        log_regime_filter_summary,
        regime_filter_walk_forward,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    # Resolve CLI overrides against module-level defaults.
    min_n = args.min_n if args.min_n is not None else MIN_REGIME_N
    candidate_k = args.candidate_k if args.candidate_k is not None else CANDIDATE_K
    final_k = args.final_k if args.final_k is not None else FINAL_K
    min_test_coverage = (
        args.min_test_coverage
        if args.min_test_coverage is not None
        else MIN_TEST_COVERAGE
    )
    min_test_n = args.min_test_n if args.min_test_n is not None else MIN_TEST_N
    direction_threshold = (
        args.direction_threshold
        if args.direction_threshold is not None
        else DIRECTION_THRESHOLD
    )
    min_persistence_folds = (
        args.min_persistence_folds
        if args.min_persistence_folds is not None
        else MIN_PERSISTENCE_FOLDS
    )
    # args.with_direction is always a bool (store_true/store_false with default=True),
    # so we use it directly without a fallback check.
    with_direction = args.with_direction
    top_n_log = args.top_n_log if args.top_n_log is not None else TOP_N_LOG

    df = load_data(args.data)
    df = build_features(df)

    require_columns(
        df,
        ["trend_strength_48b", "abs_sentiment", TARGET_COL],
        context="run_regime_filter_pipeline",
    )

    import logging as _logging  # noqa: PLC0415

    _log = _logging.getLogger(__name__)
    _log.info(
        "=== REGIME FILTER PIPELINE ==="
        " | min_n=%d | candidate_k=%d | final_k=%d | min_test_coverage=%.2f"
        " | min_test_n=%d | direction_threshold=%.4f | with_direction=%s"
        " | min_persistence_folds=%d",
        min_n,
        candidate_k,
        final_k,
        min_test_coverage,
        min_test_n,
        direction_threshold,
        with_direction,
        min_persistence_folds,
    )

    fold_df = regime_filter_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=min_n,
        candidate_k=candidate_k,
        final_k=final_k,
        min_test_coverage=min_test_coverage,
        min_test_n=min_test_n,
        with_direction=with_direction,
        direction_threshold=direction_threshold,
        top_n_log=top_n_log,
        min_persistence_folds=min_persistence_folds,
    )

    log_regime_filter_fold_results(fold_df)

    pooled = compute_pooled_summary(fold_df)
    log_regime_filter_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
