"""
run_regime_v4.py
================
Entry-point script for the Regime V4 continuous regime-conditioned signal
pipeline.

All walk-forward and regime-weighting logic lives in
``experiments/regime_v4.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v4.main``.

Pipeline steps
--------------
1. Load and prepare the research dataset.
2. Compute causal volatility feature (``vol_24b``) via ``build_features``.
3. Build a 4-component discrete regime key per fold (training-derived cuts):

   * ``vol_regime``         – tertile of ``vol_24b``
   * ``trend_dir``          – sign of ``trend_strength_48b``
   * ``trend_strength_bin`` – tertile of ``abs(trend_strength_48b)``
   * ``sent_regime``        – fixed-threshold bin of ``abs_sentiment``

4. Walk-forward regime weighting (train only): compute per-regime Sharpe;
   convert to smooth weight via ``tanh(sharpe / std_sharpe)`` (default) or
   ``sharpe / max_abs_sharpe`` (``--normalize-weights``).
5. Apply to test set: ``position = sign(net_sentiment) * weight``.
6. Compute per-fold metrics (n, mean, Sharpe, hit rate, coverage, avg weight).
7. Log pooled summary (mean Sharpe, hit rate, coverage, avg weight).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v4.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v4.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v4.log

    # Verbose mode
    python run_regime_v4.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Alternative weight normalization
    python run_regime_v4.py \\
        --data data/output/master_research_dataset.csv \\
        --normalize-weights

    # Custom min-n and log file
    python run_regime_v4.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 150 --log-file logs/regime_v4_n150.log
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
            file named ``regime_v4_YYYYMMDD_HHMMSS.log`` is created inside
            ``logs/``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v4_{timestamp}.log"
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
            "Regime V4: continuous regime-conditioned signal pipeline. "
            "Converts regime Sharpe → smooth weight; applies multiplicatively "
            "to sign(net_sentiment).  No filtering — always produces a position."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(cfg.DATA_PATH),
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum training observations per regime for non-zero weight.  "
            "Defaults to the module-level MIN_REGIME_N constant (100)."
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
    p.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help=(
            "Use sharpe / max_abs_sharpe instead of tanh(sharpe / std_sharpe) "
            "for regime weight computation."
        ),
    )
    p.add_argument(
        "--top-n-log",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of top regimes (by |weight|) to log per fold.  "
            "Defaults to the module-level TOP_N_LOG constant (5)."
        ),
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    # Import after logging is configured so that module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v4 import (  # noqa: PLC0415
        MIN_REGIME_N,
        TARGET_COL,
        TOP_N_LOG,
        build_features,
        compute_pooled_summary,
        load_data,
        log_final_summary,
        log_fold_results,
        regime_v4_walk_forward,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    # Resolve CLI overrides against module-level defaults.
    min_n = args.min_n if args.min_n is not None else MIN_REGIME_N
    top_n_log = args.top_n_log if args.top_n_log is not None else TOP_N_LOG

    _log = logging.getLogger(__name__)
    _log.info(
        "=== REGIME V4 ==="
        " | min_n=%d | normalize_weights=%s",
        min_n,
        args.normalize_weights,
    )

    df = load_data(args.data)
    df = build_features(df)

    require_columns(
        df,
        ["trend_strength_48b", "abs_sentiment", "net_sentiment", TARGET_COL],
        context="run_regime_v4",
    )

    fold_df = regime_v4_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=min_n,
        normalize_weights=args.normalize_weights,
        top_n_log=top_n_log,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
