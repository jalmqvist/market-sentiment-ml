"""
pipeline/build_dataset.py
=========================
CLI entry-point for building the master research dataset.

This module wraps the logic in ``build_fx_sentiment_dataset.py`` and
exposes it via a clean argparse interface.  All hardcoded paths and
thresholds are sourced from ``config.py``.

Usage::

    python -m pipeline.build_dataset
    python -m pipeline.build_dataset --sentiment-dir data/input/sentiment \\
                                     --price-dir data/input/fx \\
                                     --output data/output/master_research_dataset.csv

The underlying build logic lives in the root-level script
``build_fx_sentiment_dataset.py`` which is preserved as the canonical
implementation.  This module imports and delegates to it so that behaviour
is identical.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config as cfg
from utils.io import setup_logging

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the master FX sentiment research dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sentiment-dir",
        type=Path,
        default=cfg.SENTIMENT_DIR,
        help="Directory containing sentiment snapshot CSVs.",
    )
    p.add_argument(
        "--price-dir",
        type=Path,
        default=cfg.PRICE_DIR,
        help="Directory containing hourly FX price CSVs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=cfg.MASTER_DATASET_PATH,
        help="Output path for the master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    logger.info("Building master research dataset...")
    logger.info("  Sentiment dir : %s", args.sentiment_dir)
    logger.info("  Price dir     : %s", args.price_dir)
    logger.info("  Output        : %s", args.output)

    # Import and delegate to the canonical build script
    try:
        import build_fx_sentiment_dataset as builder
    except ImportError:
        logger.error(
            "Could not import build_fx_sentiment_dataset.  "
            "Ensure the repository root is on the Python path."
        )
        sys.exit(1)

    master = builder.build_master_dataset(
        sentiment_dir=args.sentiment_dir,
        price_dir=args.price_dir,
        output_file=args.output,
        horizons=cfg.HORIZONS,
    )

    builder.quick_summary(master, horizons=cfg.HORIZONS)
    logger.info("Done.  Rows: %d", len(master))


if __name__ == "__main__":
    main()
