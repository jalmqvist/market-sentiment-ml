"""
run_regime_v5.py
================
Entry-point script for the Regime V5 continuous signal blending pipeline.

All walk-forward, regime-weighting, and behavioral-scoring logic lives in
``experiments/regime_v5.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.regime_v5.main``.

Pipeline summary
----------------
Regime V5 blends three multiplicative signal layers continuously:

1. **Base signal** — Signal V2 raw composite (divergence + shock – exhaustion),
   passed through ``tanh`` to bound to ``(-1, +1)``.

2. **Regime score** — 4-component regime key (vol × trend_dir × trend_strength
   × sentiment intensity); train-only Sharpe converted via
   ``tanh(sharpe / std_sharpe)``.  Unknown regimes receive weight 0.

3. **Behavioral score** — ``tanh(0.5 * persistence_z + 0.5 * saturation_z)``
   where z-scores are standardised using training-set mean / std only::

       persistence_z  = zscore(extreme_streak_70)  [train stats]
       saturation_z   = zscore(abs_sentiment)       [train stats]

Final position::

    position = base_signal × regime_score × behavior_score

No filtering, no thresholding, no discrete regime selection.  All statistical
parameters are derived from training data only (strict walk-forward).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v5.py --data data/output/master_research_dataset.csv

    # Explicit log file
    python run_regime_v5.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/regime_v5.log

    # Verbose mode
    python run_regime_v5.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Custom min-n and window
    python run_regime_v5.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 150 --window 96
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
            file named ``regime_v5_YYYYMMDD_HHMMSS.log`` is created inside
            ``logs/``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v5_{timestamp}.log"
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
            "Regime V5: continuous signal blending — Signal V2 base signal, "
            "regime weights (V4 logic), and behavioral scoring "
            "(persistence × saturation).  "
            "No filtering — always produces a position."
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
        default=100,
        metavar="N",
        help=(
            "Minimum training observations per regime for non-zero weight. "
            "Default: 100."
        ),
    )
    p.add_argument(
        "--window",
        type=int,
        default=96,
        metavar="N",
        help=(
            "Rolling z-score window size in bars for Signal V2 features. "
            "Default: 96."
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

    # Import after logging is configured so that module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v5 import (  # noqa: PLC0415
        compute_pooled_summary,
        load_data,
        log_final_summary,
        log_fold_results,
        regime_v5_walk_forward,
        TARGET_COL,
    )
    from utils.validation import require_columns  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    _log.info(
        "=== REGIME V5 === | min_n=%d | window=%d",
        args.min_n,
        args.window,
    )

    df = load_data(args.data, window=args.window)

    require_columns(
        df,
        [TARGET_COL, "signal_v2_raw", "year"],
        context="run_regime_v5",
    )
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = regime_v5_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
