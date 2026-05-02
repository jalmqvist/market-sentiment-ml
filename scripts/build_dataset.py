#!/usr/bin/env python3
from __future__ import annotations

"""
scripts/build_dataset.py
========================
CLI entry-point for building a versioned master research dataset.

Usage::

    python scripts/build_dataset.py --version 1.1.0
    python scripts/build_dataset.py --version 1.1.0 --tag post_validation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg
from utils.logging import setup_experiment_logging

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a versioned master FX sentiment research dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--sentiment-dir", type=Path, default=cfg.SENTIMENT_DIR)
    p.add_argument("--price-dir", type=Path, default=cfg.PRICE_DIR)
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument("--no-log-file", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    log_file = setup_experiment_logging(
        experiment_type="dataset",
        tag=args.version,
        log_level=args.log_level,
        no_log_file=getattr(args, "no_log_file", False),
        log_dir=REPO_ROOT / "logs",
    )

    if log_file:
        logger.info("Logging to %s", log_file)

    version = args.version
    output_dir = cfg.OUTPUT_DIR / version

    logger.info("=== Dataset Build ===")
    logger.info("dataset_version: %s", version)
    logger.info("tag: %s", args.tag)

    # -----------------------------
    # Import builder
    # -----------------------------
    import scripts.build_fx_sentiment_dataset as builder

    master = builder.build_master_dataset(
        sentiment_dir=args.sentiment_dir,
        price_dir=args.price_dir,
        horizons=cfg.HORIZONS,
        version=version,
        tag=args.tag,
    )

    logger.info("raw_rows: %d", len(master))
    logger.info("pairs: %d", master["pair"].nunique() if "pair" in master.columns else 0)

    full_path = output_dir / "master_research_dataset.csv"
    core_path = output_dir / "master_research_dataset_core.csv"
    extended_path = output_dir / "master_research_dataset_extended.csv"
    manifest_path = output_dir / "DATASET_MANIFEST.json"

    logger.info("output_full: %s", full_path)
    logger.info("output_core: %s", core_path)
    logger.info("output_extended: %s", extended_path)

    # -----------------------------
    # Add vol + regime features
    # -----------------------------
    try:
        import scripts.build_dataset_vol as vol_module
    except ImportError:
        logger.error("build_dataset_vol not found — skipping vol/regime features")
        vol_module = None

    if vol_module:
        logger.info("Adding volatility, trend, and regime features")
        for name, path in [
            ("full", full_path),
            ("core", core_path),
            ("extended", extended_path),
        ]:
            if not path.exists():
                continue

            df = pd.read_csv(path)
            rows_before = len(df)

            df = vol_module.add_volatility_features(df)
            logger.info("vol_features_added: vol_12b, vol_48b (%s)", name)

            df = builder.add_regime_features(df)
            logger.info("regime_features_added: trend_vol_adj_strength, is_trending, is_high_vol, regime (%s)", name)

            # Add classification target using vol-adjusted threshold (preferred formula)
            if "ret_48b" in df.columns and "vol_48b" in df.columns:
                df["target_cls"] = (df["ret_48b"] > 0.1 * df["vol_48b"]).astype(int)
                df = df.reset_index(drop=True)
                logger.info(
                    "target_cls_added (%s): class_balance=%.3f  n=%d",
                    name,
                    df["target_cls"].mean(),
                    len(df),
                )
            elif "ret_48b" in df.columns:
                df["target_cls"] = (df["ret_48b"] > 0).astype(int)
                df = df.reset_index(drop=True)
                logger.info(
                    "target_cls_added_simple (%s): class_balance=%.3f  n=%d",
                    name,
                    df["target_cls"].mean(),
                    len(df),
                )

            df.to_csv(path, index=False)
            logger.info("Enriched %s dataset (%d rows)", name, len(df))

        # update manifest
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            manifest["features_added"] = {
                "version": version,
                "volatility": ["vol_12b", "vol_48b"],
                "trend": ["trend_vol_adj_strength"],
                "regime_flags": ["is_trending", "is_high_vol"],
                "regime_label": ["regime"],
                "trend_threshold": builder.TREND_THRESHOLD,
                "classification_target": ["target_cls"],
                "sentiment_v2": ["sentiment", "sentiment_delta_12b", "sentiment_z", "sentiment_extreme"],
            }

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    logger.info("Dataset build complete")

    # -----------------------------
    # Update latest symlink
    # -----------------------------
    latest = cfg.OUTPUT_DIR / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(version)
    except OSError:
        logger.warning("Could not update latest symlink")


if __name__ == "__main__":
    main()