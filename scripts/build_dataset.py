#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg
from utils.logging import setup_experiment_logging

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--sentiment-dir", type=Path, default=cfg.SENTIMENT_DIR)
    p.add_argument("--price-dir", type=Path, default=cfg.PRICE_DIR)
    p.add_argument("--log-level", default=cfg.LOG_LEVEL)
    p.add_argument("--no-log-file", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    log_file = setup_experiment_logging(
        experiment_type="dataset",
        tag=args.version,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=REPO_ROOT / "logs",
    )

    if log_file:
        logger.info("Logging to %s", log_file)

    import scripts.build_fx_sentiment_dataset as builder

    master = builder.build_master_dataset(
        sentiment_dir=args.sentiment_dir,
        price_dir=args.price_dir,
        horizons=cfg.HORIZONS,
        version=args.version,
        tag=args.tag,
    )

    output_dir = cfg.OUTPUT_DIR / args.version

    full_path = output_dir / "master_research_dataset.csv"
    core_path = output_dir / "master_research_dataset_core.csv"
    extended_path = output_dir / "master_research_dataset_extended.csv"
    manifest_path = output_dir / "DATASET_MANIFEST.json"

    import scripts.build_dataset_vol as vol_module

    for name, path in [("full", full_path), ("core", core_path), ("extended", extended_path)]:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        df = vol_module.add_volatility_features(df)
        df = builder.add_regime_features(df)

        if "ret_48b" in df.columns and "vol_48b" in df.columns:
            threshold = 0.1 * df["vol_48b"]
            df["target_cls"] = (df["ret_48b"] > threshold).astype(int)

            logger.info(
                "target_cls (%s): pos_rate=%.3f n=%d",
                name,
                df["target_cls"].mean(),
                len(df),
            )

        df.to_csv(path, index=False)
        logger.info("Enriched %s dataset (%d rows)", name, len(df))

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        manifest["features_added"] = {
            "version": args.version,
            "volatility": ["vol_12b", "vol_48b"],
            "regime": ["trend_vol_adj_strength", "is_trending", "is_high_vol"],
            "target": ["target_cls"],
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info("Dataset build complete")


if __name__ == "__main__":
    main()