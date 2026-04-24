"""
run_pipeline_strict.py
======================
Strict deterministic pipeline orchestrator.

Enforces exact execution order and dataset separation:

    build_dataset → [attach_regimes] → discovery → portfolio → [regime_v2]

Rules:
- discovery and portfolio are REQUIRED stages.
- attach_regimes and regime_v2 are OPTIONAL (skipped by default unless --regime-dataset is provided).
- walk_forward is a library; it is NEVER invoked via subprocess.
- All stages receive their dataset path via --data explicitly.
- After all pipeline stages, validate_pipeline_extended.py is executed.

Logging:
- All runs write logs to a timestamped file only (no stdout):
    logs/pipeline_strict_YYYYMMDD_HHMMSS.log
- All subprocess output (stdout + stderr) is captured and logged.
- Deterministic fingerprints (dataset hash, discovery artifact hash) are
  logged at the end of each run.

Usage::

    # Canonical pipeline (discovery + portfolio only):
    python run_pipeline_strict.py \\
        --canonical-dataset data/output/master_research_dataset.csv

    # Full pipeline (with regime stages):
    python run_pipeline_strict.py \\
        --canonical-dataset data/output/master_research_dataset.csv \\
        --regime-dataset    data/output/master_research_dataset_with_regime.csv \\
        --regimes-parquet   /path/to/phase_labels_d1.parquet

    # Skip dataset build (data files already exist):
    python run_pipeline_strict.py \\
        --canonical-dataset data/output/master_research_dataset.csv \\
        --skip build

    # Skip validation at the end:
    python run_pipeline_strict.py \\
        --canonical-dataset data/output/master_research_dataset.csv \\
        --skip validate
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config as cfg

log = logging.getLogger("run_pipeline_strict")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> logging.FileHandler:
    """Configure logging to a timestamped file only (no stdout).

    Writes only to the log file so that automated runs stay quiet.
    Returns the FileHandler so callers can retrieve the log file path.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(log_level)

    # file handler — logs/pipeline_strict_YYYYMMDD_HHMMSS.log
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_strict_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("Pipeline log file: %s", log_file)
    return file_handler


def _run(cmd: list[str]) -> None:
    """Run *cmd*, capture stdout+stderr, log both, and raise RuntimeError on non-zero exit."""
    log.info("Running: %s", " ".join(map(str, cmd)))
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout or ""
    if output.strip():
        for line in output.splitlines():
            log.info("[subprocess] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(str(c) for c in cmd)}")


def _ensure_file_nonempty(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{what} is empty (0 bytes): {path}")


def _hash_file(path: Path) -> str:
    """Return MD5 hex digest of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_csv_summary(path: Path, label: str) -> None:
    """Log a compact summary of a CSV dataset (rows, pairs, date range)."""
    try:
        df = pd.read_csv(path, low_memory=False)
        rows = len(df)
        pairs = df["pair"].nunique() if "pair" in df.columns else "n/a"
        date_range = "n/a"
        for col in ("entry_time", "time", "timestamp"):
            if col in df.columns:
                try:
                    ts = pd.to_datetime(df[col], errors="coerce")
                    date_range = f"{ts.min()} .. {ts.max()}"
                except Exception:
                    pass
                break
        log.info(
            "[summary] %s: rows=%s, pairs=%s, date_range=%s",
            label, f"{rows:,}", pairs, date_range,
        )
    except Exception as exc:
        log.warning("[summary] Could not summarise %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Strict deterministic pipeline orchestrator. "
            "Enforces: build → [attach_regimes] → discovery → portfolio → [regime_v2] → validate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset paths (canonical = required; regime = optional) ---
    ap.add_argument(
        "--canonical-dataset",
        type=Path,
        required=True,
        help=(
            "Path to canonical research dataset CSV "
            "(data/output/master_research_dataset.csv). "
            "Used by: discovery, portfolio."
        ),
    )
    ap.add_argument(
        "--regime-dataset",
        type=Path,
        default=None,
        help=(
            "Path to regime-enriched dataset CSV "
            "(data/output/master_research_dataset_with_regime.csv). "
            "When provided, attach_regimes and regime_v2 stages are executed."
        ),
    )

    # --- Build inputs ---
    ap.add_argument(
        "--sentiment-dir",
        type=Path,
        default=getattr(cfg, "SENTIMENT_DIR", Path("data/input/sentiment")),
        help="Input directory for sentiment snapshot CSVs (used by build stage).",
    )
    ap.add_argument(
        "--price-dir",
        type=Path,
        default=getattr(cfg, "PRICE_DIR", Path("data/input/fx")),
        help="Input directory for hourly FX price CSVs (used by build stage).",
    )

    # --- Regime attachment input ---
    ap.add_argument(
        "--regimes-parquet",
        type=Path,
        default=None,
        help=(
            "Path to phase_labels_d1.parquet for regime attachment. "
            "If omitted, attach_regimes_to_h1_dataset.py uses its own default."
        ),
    )

    # --- Validation ---
    ap.add_argument(
        "--reference-dataset",
        type=Path,
        default=None,
        help="Path to reference dataset for hash/parity validation (optional).",
    )

    # --- Control ---
    ap.add_argument(
        "--log-level",
        default=getattr(cfg, "LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    ap.add_argument(
        "--skip",
        default="",
        help=(
            "Comma-separated stages to skip. "
            "Valid values: build, attach_regimes, discovery, portfolio, regime_v2, validate. "
            "Example: --skip build,validate"
        ),
    )

    args = ap.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.log_level)

    skip: set[str] = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    canonical_path: Path = args.canonical_dataset
    regime_path: Path | None = args.regime_dataset

    # Discovery artifact path (written by discovery stage, consumed by portfolio stage)
    discovery_artifact_path: Path = Path("data/output/discovery_results.json")

    run_regime_stages = regime_path is not None

    log.info("=" * 70)
    log.info("STRICT PIPELINE ORCHESTRATOR")
    log.info("  canonical dataset : %s", canonical_path)
    log.info("  regime dataset    : %s", regime_path or "<not provided>")
    log.info("  skip              : %s", skip or "<none>")
    log.info("=" * 70)

    # -----------------------
    # 1) Build canonical dataset
    # -----------------------
    if "build" not in skip:
        log.info("[1/6] Building canonical dataset...")
        _run(
            [
                sys.executable,
                "-m",
                "pipeline.build_dataset",
                "--sentiment-dir", str(args.sentiment_dir),
                "--price-dir",     str(args.price_dir),
                "--output",        str(canonical_path),
                "--log-level",     args.log_level,
            ]
        )
    else:
        log.info("[1/6] Skipped: build")

    _ensure_file_nonempty(canonical_path, "Canonical dataset")
    log.info("Canonical dataset ready: %s", canonical_path)
    _log_csv_summary(canonical_path, "canonical dataset")

    # -----------------------
    # 2) Attach regime labels (optional)
    # -----------------------
    if run_regime_stages and "attach_regimes" not in skip:
        log.info("[2/6] Attaching regime labels...")
        cmd = [
            sys.executable,
            "attach_regimes_to_h1_dataset.py",
            "--data", str(canonical_path),
            "--out",  str(regime_path),
        ]
        if args.regimes_parquet:
            cmd += ["--regimes", str(args.regimes_parquet)]
        _run(cmd)
        _ensure_file_nonempty(regime_path, "Regime-enriched dataset")
        log.info("Regime-enriched dataset ready: %s", regime_path)
        _log_csv_summary(regime_path, "regime dataset")
    else:
        if run_regime_stages:
            log.info("[2/6] Skipped: attach_regimes")
        else:
            log.info("[2/6] Skipped: attach_regimes (no --regime-dataset provided)")

    # -----------------------
    # 3) Signal discovery (REQUIRED) — canonical dataset only
    # -----------------------
    if "discovery" not in skip:
        log.info("[3/6] Running signal discovery (canonical dataset)...")
        _run(
            [
                sys.executable,
                "-m",
                "experiments.discovery",
                "--data",             str(canonical_path),
                "--out-artifact",     str(discovery_artifact_path),
                "--log-level",        args.log_level,
            ]
        )
        if discovery_artifact_path.exists():
            log.info("[3/6] Discovery artifact written: %s", discovery_artifact_path)
            try:
                artifact_data = json.loads(discovery_artifact_path.read_text())
                for horizon, info in artifact_data.get("horizons", {}).items():
                    selected = info.get("selected_pairs", [])
                    log.info(
                        "[summary] discovery(h=%s): %d selected pairs: %s",
                        horizon, len(selected), selected,
                    )
            except Exception as exc:
                log.warning("[3/6] Could not parse discovery artifact: %s", exc)
    else:
        log.info("[3/6] Skipped: discovery")

    # -----------------------
    # 4) Portfolio construction (REQUIRED) — canonical dataset only
    # -----------------------
    if "portfolio" not in skip:
        log.info("[4/6] Running portfolio construction (canonical dataset)...")
        portfolio_cmd = [
            sys.executable,
            "-m",
            "portfolio.portfolio_builder",
            "--data",      str(canonical_path),
            "--log-level", args.log_level,
        ]
        if discovery_artifact_path.exists():
            portfolio_cmd += ["--discovery-artifact", str(discovery_artifact_path)]
            log.info("[4/6] Using discovery artifact for pair selection: %s", discovery_artifact_path)
        _run(portfolio_cmd)
    else:
        log.info("[4/6] Skipped: portfolio")

    # -----------------------
    # 5) Regime experiments (OPTIONAL) — regime dataset only
    # -----------------------
    if run_regime_stages and "regime_v2" not in skip:
        log.info("[5/6] Running regime experiments (regime dataset)...")
        _ensure_file_nonempty(regime_path, "Regime-enriched dataset (regime_v2 input)")
        _run(
            [
                sys.executable,
                "-m",
                "experiments.regime_v2",
                "--data",      str(regime_path),
                "--log-level", args.log_level,
            ]
        )
    else:
        if run_regime_stages:
            log.info("[5/6] Skipped: regime_v2")
        else:
            log.info("[5/6] Skipped: regime_v2 (no --regime-dataset provided)")

    # -----------------------
    # 6) Extended validation (fail pipeline if validation fails)
    # -----------------------
    if "validate" not in skip:
        log.info("[6/6] Running extended pipeline validation...")

        if regime_path is None or not regime_path.exists():
            log.warning(
                "Regime dataset not available; skipping validation (requires both datasets)."
            )
        else:
            validate_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "validation" / "validate_pipeline_extended.py"),
                "--data",        str(canonical_path),
                "--data-regime", str(regime_path),
            ]
            if args.reference_dataset:
                validate_cmd += ["--reference", str(args.reference_dataset)]
            _run(validate_cmd)
    else:
        log.info("[6/6] Skipped: validate")

    # -----------------------
    # Final summary with fingerprints
    # -----------------------
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("  canonical dataset : %s", canonical_path)
    if run_regime_stages:
        log.info("  regime dataset    : %s", regime_path)

    # Deterministic fingerprints
    try:
        canonical_hash = _hash_file(canonical_path)
        log.info("  dataset hash      : %s", canonical_hash)
    except Exception as exc:
        log.warning("  dataset hash      : <error: %s>", exc)

    if discovery_artifact_path.exists():
        try:
            artifact_hash = _hash_file(discovery_artifact_path)
            log.info("  discovery artifact: %s  hash=%s", discovery_artifact_path, artifact_hash)
        except Exception as exc:
            log.warning("  discovery artifact hash: <error: %s>", exc)
    else:
        log.info("  discovery artifact: <not present>")

    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
