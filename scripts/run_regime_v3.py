"""
scripts/run_regime_v3.py
========================
Launcher for experiments.regime_v3 with strict logging (stdout + file),
modeled after run_pipeline_strict.py.

Usage::

    python -m scripts.run_regime_v3 --data data/output/master_research_dataset.csv
    python -m scripts.run_regime_v3 --data data/output/master_research_dataset.csv --log-level DEBUG
    python -m scripts.run_regime_v3 --data data/output/master_research_dataset.csv --log-file logs/regime_v3.log
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import config as cfg

log = logging.getLogger("scripts.run_regime_v3")


def _configure_logging(level: str, log_file: str | None) -> Path:
    """Configure logging to stdout AND a log file.

    Only adds handlers if the root logger has none, avoiding double
    initialization when main() is called multiple times (e.g. in tests
    or notebooks).

    If log_file is None, create logs/regime_v3_YYYYMMDD_HHMMSS.log.
    Returns the resolved log file path.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(log_level)

    if not root.handlers:
        # stdout handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    # file handler
    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v3_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Only add the file handler if it is not already attached.
    existing_files = {
        h.baseFilename
        for h in root.handlers
        if isinstance(h, logging.FileHandler)
    }
    resolved_log_path = log_path.resolve()
    if str(resolved_log_path) not in existing_files:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.info("Regime V3 log file: %s", log_path)
    return resolved_log_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run regime_v3 with logging to file + stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=getattr(cfg, "LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Optional log file path. If omitted, defaults to "
            "logs/regime_v3_YYYYMMDD_HHMMSS.log"
        ),
    )

    # Backwards-compat: accept but ignore (for now).
    # Remove later if you prefer hard-fail.
    p.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="(Deprecated) Accepted for backwards compatibility; currently ignored.",
    )

    args = p.parse_args(argv)

    _configure_logging(args.log_level, args.log_file)

    if args.target_scale is not None:
        log.warning(
            "--target-scale is deprecated and currently ignored (value=%s).",
            args.target_scale,
        )

    from experiments import regime_v3

    # IMPORTANT: do NOT call regime_v3.main(); it will reconfigure logging.
    df = regime_v3.load_data(args.data)
    df = regime_v3.build_features(df)
    df = regime_v3.build_regimes(df)

    feature_cols = regime_v3.select_features(df)
    if not feature_cols:
        log.error("No valid feature columns found in dataset. Exiting.")
        return 1

    if regime_v3.TARGET_COL not in df.columns:
        log.error("Target column '%s' not found. Exiting.", regime_v3.TARGET_COL)
        return 1

    wf_results, regime_results = regime_v3.walk_forward_ridge(
        df, feature_cols, regime_col="regime"
    )

    regime_v3.print_wf_summary(wf_results)
    regime_v3.print_regime_summary(regime_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())