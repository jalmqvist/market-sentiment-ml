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


def _canonical_dataset_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "full": output_dir / "master_research_dataset.csv",
        "core": output_dir / "master_research_dataset_core.csv",
        "extended": output_dir / "master_research_dataset_extended.csv",
    }


def _fail_if_canonical_exists(canonical_paths: dict[str, Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in canonical_paths.values() if path.exists()]
    if existing:
        joined = ", ".join(str(p) for p in existing)
        raise FileExistsError(
            "Refusing to overwrite canonical dataset(s) without --force: "
            f"{joined}"
        )


def _validate_augment_only_inputs(
    canonical_paths: dict[str, Path],
    behavioral_surface: Path | None,
) -> None:
    if behavioral_surface is None:
        raise ValueError("--augment-only requires --behavioral-surface PATH.")

    missing = [path for path in canonical_paths.values() if not path.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Augment-only mode requires existing canonical datasets. "
            f"Missing: {joined}"
        )


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--sentiment-dir", type=Path, default=cfg.SENTIMENT_DIR)
    p.add_argument("--price-dir", type=Path, default=cfg.PRICE_DIR)
    p.add_argument("--log-level", default=cfg.LOG_LEVEL)
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument(
        "--behavioral-surface",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a frozen Behavioral Surface Parquet produced by BSVE. "
            "When supplied, behaviorally-augmented dataset variants are written alongside "
            "the original datasets. The original datasets are never modified."
        ),
    )
    p.add_argument(
        "--augment-only",
        action="store_true",
        help=(
            "Skip canonical dataset construction and only run Behavioral Surface "
            "augmentation against existing canonical dataset files."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow overwriting existing canonical dataset files. Without this flag, "
            "existing canonical outputs cause a fail-fast error."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    output_dir = cfg.OUTPUT_DIR / args.version
    canonical_paths = _canonical_dataset_paths(output_dir)
    manifest_path = output_dir / "DATASET_MANIFEST.json"

    log_file = setup_experiment_logging(
        experiment_type="dataset",
        tag=args.version,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=REPO_ROOT / "logs",
    )

    if log_file:
        logger.info("Logging to %s", log_file)

    if args.augment_only:
        _validate_augment_only_inputs(canonical_paths, args.behavioral_surface)
    else:
        _fail_if_canonical_exists(canonical_paths, args.force)

    import scripts.build_fx_sentiment_dataset as builder

    if not args.augment_only:
        builder.build_master_dataset(
            sentiment_dir=args.sentiment_dir,
            price_dir=args.price_dir,
            horizons=cfg.HORIZONS,
            version=args.version,
            tag=args.tag,
        )

        import scripts.build_dataset_vol as vol_module

        for name, path in canonical_paths.items():
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

    # ------------------------------------------------------------------
    # Optional Behavioral Surface augmentation
    # ------------------------------------------------------------------
    if args.behavioral_surface is not None:
        from bsve.dataset_augmentation import run_behavioral_augmentation

        logger.info(
            "Behavioral Surface augmentation requested: %s", args.behavioral_surface
        )

        base_dataset_paths = {
            variant_label: path
            for variant_label, path in canonical_paths.items()
            if path.exists()
        }

        run_behavioral_augmentation(
            surface_path=args.behavioral_surface,
            dataset_version=args.version,
            output_dir=output_dir,
            base_dataset_paths=base_dataset_paths,
            base_manifest_path=manifest_path if manifest_path.exists() else None,
        )


if __name__ == "__main__":
    main()