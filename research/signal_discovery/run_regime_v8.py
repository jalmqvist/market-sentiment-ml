# Legacy experiment — not part of current validated approach\n"""
run_regime_v8.py
================
Entry-point script for the Regime V8.3 model-based signal pipeline.

All feature loading, model training, walk-forward, and metrics logic lives in
``experiments/regime_v8.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v8.main``.

Pipeline summary
----------------
Regime V8.3 extends V8.2 with **interaction features** that expose conditional
signal structure (sentiment extremes × trend context × persistence).
A LightGBM regressor is trained on sentiment, market, and interaction features to
predict ``ret_48b`` directly.  Predictions are normalised by their standard
deviation and clipped to [-3, 3] to produce continuous positions.  Only the
top ``top_frac`` of predictions (by absolute magnitude) are traded each fold;
the remainder are zeroed out.
Performance is evaluated using a strict expanding-window walk-forward:

* **Features**: ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
  ``trend_strength_48b``, ``divergence``, ``signal_v2_raw``,
  ``sent_x_trend``, ``extreme_x_trend``, ``streak_x_sent``, ``streak_x_trend``

* **Walk-forward**: for each test year, the model is trained exclusively on
  all prior years (no forward leakage).

* **Signal**::

      pred_std    = std(pred)
      scaled_pred = pred / pred_std  if pred_std > 1e-10 else pred
      scaled_pred = clip(scaled_pred, -3, 3)
      score       = abs(pred)
      position    = where(score >= quantile(score, 1-top_frac), scaled_pred, 0)

* **Metrics per fold**: n (traded rows), mean return, Sharpe, hit_rate, IC
  (Spearman correlation between predictions and realized returns, all rows).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v8.py --data data/output/master_research_dataset.csv

    # Use custom top-frac (trade top 30% of signals)
    python run_regime_v8.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3

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


def _top_frac_arg(value: str) -> float:
    """Argparse type for ``--top-frac``: validates the value is in (0, 1]."""
    try:
        frac = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--top-frac must be a float, got: {value!r}")
    if not (0 < frac <= 1):
        raise argparse.ArgumentTypeError(
            f"--top-frac must be in (0, 1], got: {frac}"
        )
    return frac


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V8.3: model-based signal pipeline with interaction features, "
            "continuous position sizing, and top-k filtering. "
            "Trains LightGBM to predict ret_48b from sentiment and market "
            "features using walk-forward validation, normalises prediction "
            "magnitude into continuous positions, and trades only the top "
            "top_frac of predictions ranked by absolute magnitude."
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
        "--top-frac",
        type=_top_frac_arg,
        default=0.2,
        metavar="FRAC",
        help=(
            "Fraction of predictions to trade per fold, ranked by absolute "
            "prediction magnitude.  Must be in (0, 1].  Default is 0.2 "
            "(top 20%%)."
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
    _log.info("=== REGIME V8.3 ===")

    df = load_data(args.data)

    require_columns(df, _REQUIRED_COLS, context="run_regime_v8")
    _log.info("Dataset ready: %d rows", len(df))

    CORE_FEATURES = [
        "net_sentiment",
        "abs_sentiment",
        "extreme_streak_70",
        "trend_strength_48b",
    ]

    OPTIONAL_FEATURES = [
        "divergence",
        "signal_v2_raw",
        "sent_x_trend",
        "extreme_x_trend",
        "streak_x_sent",
        "streak_x_trend",
    ]

    available_features = [
        col for col in CORE_FEATURES + OPTIONAL_FEATURES
        if col in df.columns
    ]

    missing_optional = [
        col for col in OPTIONAL_FEATURES
        if col not in df.columns
    ]

    if missing_optional:
        _log.warning("Missing optional features: %s", missing_optional)

    if len(available_features) < 2:
        raise ValueError("Not enough features available")

    _log.info("Using features: %s", available_features)

    fold_df = walk_forward(
        df,
        feature_cols=available_features,
        target_col=TARGET_COL,
        top_frac=args.top_frac,
    )

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
