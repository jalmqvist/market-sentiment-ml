"""
scripts/build_dataset.py
========================
CLI entry-point for building a versioned master research dataset.

Usage::

    python scripts/build_dataset.py --version 1.1.0
    python scripts/build_dataset.py --version 1.1.0 --tag post_validation
    python scripts/build_dataset.py \\
        --version 1.1.0 \\
        --tag post_validation \\
        --sentiment-dir data/input/sentiment \\
        --price-dir data/input/fx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.io import setup_logging

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a versioned master FX sentiment research dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        required=True,
        help="Dataset version string, e.g. '1.1.0'.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional descriptive tag, e.g. 'post_validation'.",
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
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    version = args.version

    logger.info("Building master research dataset")
    logger.info("  Version       : %s", version)
    if args.tag:
        logger.info("  Tag           : %s", args.tag)
    logger.info("  Sentiment dir : %s", args.sentiment_dir)
    logger.info("  Price dir     : %s", args.price_dir)

    output_dir = cfg.OUTPUT_DIR / version
    logger.info("  Output dir    : %s", output_dir)

    try:
        import scripts.build_fx_sentiment_dataset as builder
    except ImportError:
        logger.error(
            "Could not import build_fx_sentiment_dataset. "
            "Ensure the repository root is on the Python path."
        )
        sys.exit(1)

    master = builder.build_master_dataset(
        sentiment_dir=args.sentiment_dir,
        price_dir=args.price_dir,
        horizons=cfg.HORIZONS,
        version=version,
        tag=args.tag,
    )

    full_path = output_dir / "master_research_dataset.csv"
    core_path = output_dir / "master_research_dataset_core.csv"
    extended_path = output_dir / "master_research_dataset_extended.csv"
    manifest_path = output_dir / "DATASET_MANIFEST.json"

    # ------------------------------------------------------------------
    # Enrich all saved variants with volatility + regime features
    # ------------------------------------------------------------------
    try:
        import scripts.build_dataset_vol as vol_module
    except ImportError:
        logger.error(
            "Could not import build_dataset_vol. "
            "Volatility and regime features will NOT be added."
        )
        vol_module = None

    if vol_module is not None:
        for variant_name, variant_path in [
            ("full", full_path),
            ("core", core_path),
            ("extended", extended_path),
        ]:
            if not variant_path.exists():
                logger.warning("Variant file not found, skipping: %s", variant_path)
                continue

            logger.info("Enriching %s variant with vol + regime features ...", variant_name)
            df_v = pd.read_csv(variant_path)

            df_v = vol_module.add_volatility_features(df_v)
            df_v = builder.add_regime_features(df_v)

            df_v.to_csv(variant_path, index=False)
            logger.info(
                "  %s variant: %d rows, vol+regime features added", variant_name, len(df_v)
            )

        # Update manifest with features_added metadata
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            manifest["features_added"] = {
                "version": "1.3.0",
                "volatility": ["vol_12b", "vol_48b"],
                "trend": ["trend_strength"],
                "regime_flags": ["is_trending", "is_high_vol"],
                "regime_label": ["regime"],
                "regime_values": ["HVTF", "LVTF", "HVR", "LVR"],
                "trend_threshold": builder.TREND_THRESHOLD,
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            logger.info("Updated dataset manifest with features_added: %s", manifest_path)

    logger.info("Dataset build complete")
    logger.info("  Version       : %s", version)
    logger.info("  Rows (full)   : %d", len(master))
    logger.info("  Output paths:")
    logger.info("    Full     : %s", full_path)
    logger.info("    Core     : %s", core_path)
    logger.info("    Extended : %s", extended_path)
    logger.info("    Manifest : %s", manifest_path)

    # Update data/output/latest symlink to point at this version
    latest_link = cfg.OUTPUT_DIR / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version)
        logger.info("  Latest link  : %s -> %s", latest_link, version)
    except OSError as exc:
        logger.warning("Could not update 'latest' symlink: %s", exc)


if __name__ == "__main__":
    main()
