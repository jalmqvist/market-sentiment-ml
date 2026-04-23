"""
run_regime_v3.py
================
Entry-point script for the Regime V3 conditional-performance experiment.

Extends ``experiments.regime_v3`` with logging to *both* stdout and an
optional log file.  All walk-forward and regime-metrics logic lives in
``experiments/regime_v3.py``; this script is a thin launcher that
configures logging and delegates to ``experiments.regime_v3.main``.

Usage::

    # Log to stdout only (default)
    python run_regime_v3.py --data data/output/master_research_dataset.csv

    # Log to stdout + file
    python run_regime_v3.py --data data/output/master_research_dataset.csv \\
                            --log-file logs/regime_v3.log

    # Verbose mode
    python run_regime_v3.py --data data/output/master_research_dataset.csv \\
                            --log-level DEBUG --log-file logs/regime_v3_debug.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config as cfg

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a stdout handler and an optional file handler.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.  The parent directory is
            created automatically if it does not exist.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Stdout handler (always added).
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # File handler (optional).
    if log_file:
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
            "Regime V3: Ridge regression walk-forward with conditional "
            "performance evaluation by vol × trend regime."
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
            "Optional path to a log file.  When provided, log messages are "
            "written to both stdout and this file."
        ),
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    # Import after logging is configured so that module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v3 import (  # noqa: PLC0415
        TARGET_COL,
        build_behavioural_regimes,
        build_features,
        build_regimes,
        behavioural_regime_baseline,
        behavioural_regime_walk_forward,
        load_data,
        log_behavioural_regime_summary,
        log_behavioural_regime_wf,
        log_regime_baseline,
        log_regime_wf,
        print_regime_summary,
        print_wf_summary,
        regime_baseline,
        regime_walk_forward,
        select_features,
        walk_forward_ridge,
    )

    df = load_data(args.data)

    # Step 1: Compute causal volatility feature (vol_24b) and interaction features.
    df = build_features(df)

    # Step 2: Discretise features into vol/trend regimes BEFORE any modeling.
    df = build_regimes(df)

    # Step 2b: Discretise sentiment features into behavioural regimes.
    df = build_behavioural_regimes(df)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: REGIME BASELINE (NO MODEL) — full-dataset regime discovery
    # ------------------------------------------------------------------
    regime_summary = regime_baseline(df)
    log_regime_baseline(regime_summary)

    # ------------------------------------------------------------------
    # Step 3b: BEHAVIOURAL REGIME SUMMARY — full-dataset behavioural
    #          discovery (no model)
    # ------------------------------------------------------------------
    behavioural_summary = behavioural_regime_baseline(df)
    log_behavioural_regime_summary(behavioural_summary)

    # ------------------------------------------------------------------
    # Step 4: REGIME WALK-FORWARD — out-of-sample regime validation
    # ------------------------------------------------------------------
    regime_wf = regime_walk_forward(df)
    log_regime_wf(regime_wf)

    # ------------------------------------------------------------------
    # Step 4b: WALK-FORWARD REGIME PERFORMANCE — out-of-sample
    #          behavioural regime validation
    # ------------------------------------------------------------------
    behavioural_wf = behavioural_regime_walk_forward(df)
    log_behavioural_regime_wf(behavioural_wf)

    # ------------------------------------------------------------------
    # Step 5: MODEL WITHIN REGIME (secondary) — LightGBM walk-forward
    #         trained globally, evaluated per regime
    # ------------------------------------------------------------------
    feature_cols = select_features(df)

    if not feature_cols:
        print("ERROR: No valid feature columns found in dataset. Exiting.")
        sys.exit(1)

    import logging as _logging  # noqa: PLC0415
    _logging.getLogger(__name__).info("=== MODEL WITHIN REGIME ===")

    # Expanding-window LightGBM walk-forward with per-regime evaluation.
    wf_results, regime_model_results = walk_forward_ridge(
        df, feature_cols, regime_col="regime"
    )

    print_wf_summary(wf_results)
    print_regime_summary(regime_model_results)


if __name__ == "__main__":
    main()
