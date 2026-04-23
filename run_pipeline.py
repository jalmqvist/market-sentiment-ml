from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config as cfg

log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def _ensure_file_nonempty(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{what} is empty (0 bytes): {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the FX sentiment pipeline step-by-step (canonical dataset + optional regimes layer)."
    )

    # Step 1 inputs
    ap.add_argument("--sentiment-dir", type=Path, default=getattr(cfg, "SENTIMENT_DIR", Path("data/input/sentiment")))
    ap.add_argument("--price-dir", type=Path, default=getattr(cfg, "PRICE_DIR", Path("data/input/fx")))

    # Outputs (canonical + regime-enriched)
    ap.add_argument("--out-dir", type=Path, default=Path("data/output"))
    ap.add_argument(
        "--canonical-dataset",
        type=Path,
        default=getattr(cfg, "DATA_PATH", getattr(cfg, "MASTER_DATASET_PATH", None)) or Path("data/output/master_research_dataset.csv"),
        help="Canonical dataset output (base dataset).",
    )
    ap.add_argument(
        "--regime-dataset",
        type=Path,
        default=getattr(cfg, "DATA_PATH_REGIME", None) or Path("data/output/master_research_dataset_with_regime.csv"),
        help="Regime-enriched dataset output (optional layer).",
    )

    # Regime attachment inputs
    ap.add_argument(
        "--regimes-parquet",
        type=Path,
        default=None,
        help=(
            "Path to phase_labels_d1.parquet. If omitted, attach_regimes_to_h1_dataset.py uses its own default."
        ),
    )

    ap.add_argument("--log-level", default=getattr(cfg, "LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ap.add_argument(
        "--skip",
        default="",
        help="Comma-separated steps to skip: build,regimes (example: --skip build)",
    )

    args = ap.parse_args()
    _configure_logging(args.log_level)

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    # Ensure output directory exists (for default relative paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize outputs: if user gave just filenames, resolve under out-dir
    canonical_path = args.canonical_dataset
    if canonical_path.parent == Path("."):
        canonical_path = args.out_dir / canonical_path

    regime_path = args.regime_dataset
    if regime_path.parent == Path("."):
        regime_path = args.out_dir / regime_path

    # -----------------------
    # 1) Build canonical dataset
    # -----------------------
    if "build" not in skip:
        _run(
            [
                sys.executable,
                "-m",
                "pipeline.build_dataset",
                "--sentiment-dir",
                str(args.sentiment_dir),
                "--price-dir",
                str(args.price_dir),
                "--output",
                str(canonical_path),
                "--log-level",
                args.log_level,
            ]
        )

    _ensure_file_nonempty(canonical_path, "Canonical dataset CSV")
    log.info("Canonical dataset ready: %s", canonical_path)

    # -----------------------
    # 2) Optional regimes layer
    # -----------------------
    if "regimes" not in skip:
        cmd = [
            sys.executable,
            "attach_regimes_to_h1_dataset.py",
            "--data",
            str(canonical_path),
            "--out",
            str(regime_path),
        ]
        if args.regimes_parquet:
            cmd += ["--regimes", str(args.regimes_parquet)]

        _run(cmd)
        _ensure_file_nonempty(regime_path, "Regime-enriched dataset CSV")
        log.info("Regime-enriched dataset ready: %s", regime_path)
    else:
        log.info("Skipped regimes attachment step.")

    # -----------------------
    # Guidance: which dataset to use where
    # -----------------------
    log.info("Downstream defaults:")
    log.info("  - Use CANONICAL dataset for discovery / WF / portfolio / sanity checks:")
    log.info("      %s", canonical_path)
    log.info("  - Use REGIME dataset ONLY for regime experiments / conditioning / feature engineering:")
    log.info("      %s", regime_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())