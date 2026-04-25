"""
run_regime_v8.py
================
Entry-point script for the Regime V8 model-based signal pipeline.

All feature loading, model training, walk-forward, and metrics logic lives in
``experiments/regime_v8.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v8.main``.

Pipeline summary
----------------
Regime V8 replaces handcrafted scoring with a **learned alpha function**.
A LightGBM regressor is trained on six sentiment and market features to
predict ``ret_48b`` directly.  Positions are taken as ``sign(prediction)``,
and performance is evaluated using a strict expanding-window walk-forward:

* **Features**: ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
  ``trend_strength_48b``, ``divergence``, ``signal_v2_raw``

* **Walk-forward**: for each test year, the model is trained exclusively on
  all prior years (no forward leakage).

* **Signal**: ``position = sign(model.predict(X_test))``

* **Metrics per fold**: n, mean return, Sharpe, hit_rate, IC (Spearman
  correlation between predictions and realized returns).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v8.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v8.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v8.log

    # Verbose mode
    python run_regime_v8.py \\
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


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.  When omitted, a
            file named ``regime_v8_YYYYMMDD_HHMMSS.log`` is created inside
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
        log_path = logs_dir / f"regime_v8_{timestamp}.log"
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
            "Regime V8: model-based signal pipeline. "
            "Trains LightGBM to predict ret_48b from sentiment and market "
            "features using walk-forward validation, and evaluates trading "
            "performance using sign(prediction)."
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

    # Import after logging is configured so module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v8 import (  # noqa: PLC0415
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
    _log.info("=== REGIME V8 ===")

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v8")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(df, target_col=TARGET_COL)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
