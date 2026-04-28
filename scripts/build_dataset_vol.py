from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.io import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a versioned master FX sentiment research dataset (with volatility).",
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
    return p.parse_args(argv)


# ---------------------------------------------------------------------
# Volatility computation (SAFE)
# ---------------------------------------------------------------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding volatility features (vol_12b, vol_48b)")

    df = df.copy()

    # -----------------------------------------------------------------
    # Ensure proper ordering (CRITICAL)
    # -----------------------------------------------------------------
    df = df.sort_values(["pair", "snapshot_time"])

    # -----------------------------------------------------------------
    # Ensure return column exists
    # -----------------------------------------------------------------
    if "ret_1b" not in df.columns:
        raise ValueError("Expected 'ret_1b' column not found in dataset")

    # -----------------------------------------------------------------
    # Rolling volatility (STRICTLY BACKWARD LOOKING)
    # -----------------------------------------------------------------
    df["vol_12b"] = (
        df.groupby("pair")["ret_1b"]
        .rolling(window=12, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["vol_48b"] = (
        df.groupby("pair")["ret_1b"]
        .rolling(window=48, min_periods=48)
        .std()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------------------------------------------
    # Safety checks
    # -----------------------------------------------------------------
    logger.info(
        "Volatility stats (12b): mean=%.6f std=%.6f",
        df["vol_12b"].mean(),
        df["vol_12b"].std(),
    )

    logger.info(
        "Volatility stats (48b): mean=%.6f std=%.6f",
        df["vol_48b"].mean(),
        df["vol_48b"].std(),
    )

    return df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(argv=None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    version = args.version

    logger.info("Building master research dataset (with volatility)")
    logger.info("  Version       : %s", version)
    if args.tag:
        logger.info("  Tag           : %s", args.tag)

    output_dir = cfg.OUTPUT_DIR / version
    logger.info("  Output dir    : %s", output_dir)

    try:
        import scripts.build_fx_sentiment_dataset as builder
    except ImportError:
        logger.error("Could not import dataset builder")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Build base dataset
    # -----------------------------------------------------------------
    master = builder.build_master_dataset(
        sentiment_dir=args.sentiment_dir,
        price_dir=args.price_dir,
        horizons=cfg.HORIZONS,
        version=version,
        tag=args.tag,
    )

    logger.info("Base dataset built: %d rows", len(master))

    # -----------------------------------------------------------------
    # ADD VOLATILITY FEATURES
    # -----------------------------------------------------------------
    master = add_volatility_features(master)

    # ------------------------------------------------------------
    # OVERWRITE DATASET WITH VOLATILITY FEATURES
    # ------------------------------------------------------------
    logger.info("Overwriting dataset files with volatility features")

    full_path = output_dir / "master_research_dataset.csv"
    core_path = output_dir / "master_research_dataset_core.csv"

    # Save full
    master.to_csv(full_path, index=False)

    # Rebuild core (drop NaNs)
    core = master.dropna().copy()
    core.to_csv(core_path, index=False)

    logger.info("Saved dataset with volatility features")

    # -----------------------------------------------------------------
    # Save (overwrite existing structure)
    # -----------------------------------------------------------------
    full_path = output_dir / "master_research_dataset.csv"
    core_path = output_dir / "master_research_dataset_core.csv"
    extended_path = output_dir / "master_research_dataset_extended.csv"
    manifest_path = output_dir / "DATASET_MANIFEST.json"

    logger.info("Dataset build complete (with volatility)")
    logger.info("  Rows (full)   : %d", len(master))
    logger.info("  Output paths:")
    logger.info("    Full     : %s", full_path)
    logger.info("    Core     : %s", core_path)
    logger.info("    Extended : %s", extended_path)
    logger.info("    Manifest : %s", manifest_path)

    # -----------------------------------------------------------------
    # Update latest symlink
    # -----------------------------------------------------------------
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
