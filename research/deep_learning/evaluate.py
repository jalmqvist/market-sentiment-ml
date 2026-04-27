#!/usr/bin/env python3
"""
research/deep_learning/evaluate.py
=====================================
Evaluate DL predictions using the same methodology as validate_signal_raw.py.

Converts raw model predictions to a directional signal::

    signal = sign(prediction)

Then measures:
  - Sharpe ratio
  - Hit rate

Runs four scenarios (mirroring validate_signal_raw.py):
  - baseline   — raw signal
  - shift(1)   — signal shifted 1 bar forward
  - shift(5)   — signal shifted 5 bars forward
  - shuffled   — signal randomly permuted (null distribution)

Usage::

    python research/deep_learning/evaluate.py \\
        --predictions data/output/1.1.0/dl/predictions_price_sentiment.csv

Or evaluate all feature sets for a given version::

    python research/deep_learning/evaluate.py \\
        --dataset-version 1.1.0 \\
        --feature-set price_sentiment
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]  # market-sentiment-ml/


def _setup_logging() -> None:
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dl_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
    print(f"Logging to: {log_file}", flush=True)


_setup_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics — mirroring validate_signal_raw.py
# ---------------------------------------------------------------------------


def _compute_positions(signal: pd.Series) -> pd.Series:
    return np.sign(signal).astype(float)


def _compute_metrics(df: pd.DataFrame) -> dict:
    """Compute Sharpe and hit-rate from a DataFrame with 'position' and target."""
    pnl = df["position"] * df["ret_48b"]

    n = len(pnl)
    if n == 0:
        return {"n": 0, "sharpe": 0.0, "hit_rate": 0.0}

    mean = pnl.mean()
    std = pnl.std()
    sharpe = float(mean / std) if std > 1e-12 else 0.0
    hit_rate = float((pnl > 0).mean())

    return {"n": int(n), "sharpe": sharpe, "hit_rate": hit_rate}


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def _run_scenario(df: pd.DataFrame, label: str) -> dict:
    """Evaluate a DataFrame that already has 'signal' and 'position' set."""
    metrics = _compute_metrics(df)
    density = float((df["signal"] != 0).mean())

    logger.info(
        "[%s] n=%d | sharpe=%.4f | hit=%.4f | density=%.4f",
        label,
        metrics["n"],
        metrics["sharpe"],
        metrics["hit_rate"],
        density,
    )
    return {**metrics, "density": density, "label": label}


def _evaluate(df: pd.DataFrame, shifts: list[int]) -> list[dict]:
    """Run all evaluation scenarios and return a list of result dicts."""
    results = []

    # --- baseline ---
    base = df.copy()
    base["position"] = _compute_positions(base["signal"])
    results.append(_run_scenario(base, "baseline"))

    # --- shift(N) ---
    for n in shifts:
        shifted = df.copy()
        shifted["signal"] = shifted["signal"].shift(n)
        shifted = shifted.dropna(subset=["signal"])
        shifted["position"] = _compute_positions(shifted["signal"])
        results.append(_run_scenario(shifted, f"shift({n})"))

    # --- shuffled ---
    shuffled = df.copy()
    shuffled["signal"] = np.random.permutation(shuffled["signal"].values)
    shuffled["position"] = _compute_positions(shuffled["signal"])
    results.append(_run_scenario(shuffled, "shuffled"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions:
        return Path(args.predictions)

    if not args.dataset_version:
        raise ValueError("Either --predictions or --dataset-version must be provided")

    import config as cfg

    return (
        cfg.OUTPUT_DIR
        / args.dataset_version
        / "dl"
        / f"predictions_{args.feature_set}.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DL predictions (mirrors validate_signal_raw.py)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--predictions", help="Path to predictions CSV")
    group.add_argument("--dataset-version", help="Dataset version (e.g. 1.1.0)")

    parser.add_argument(
        "--feature-set",
        default="price_sentiment",
        choices=["price_only", "price_sentiment"],
        help="Used to locate predictions file when --dataset-version is given",
    )
    parser.add_argument(
        "--shifts",
        default="1,5",
        help="Comma-separated shift values for the shift test (default: 1,5)",
    )
    args = parser.parse_args()

    shifts = [int(s) for s in args.shifts.split(",")]

    pred_path = _resolve_predictions_path(args)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        raise FileNotFoundError(pred_path)

    logger.info("=== DL Evaluation ===")
    logger.info("predictions: %s", pred_path)

    df = pd.read_csv(pred_path)

    required = {"prediction", "ret_48b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions file: {missing}")

    # Convert raw model output to directional signal
    df["signal"] = np.sign(df["prediction"])

    logger.info("loaded %d rows from %s", len(df), pred_path)

    results = _evaluate(df, shifts=shifts)

    print("\n=== Evaluation Results ===")
    for r in results:
        print(
            f"  [{r['label']:<14}]  "
            f"n={r['n']:>6}  "
            f"sharpe={r['sharpe']:+.4f}  "
            f"hit={r['hit_rate']:.4f}  "
            f"density={r['density']:.4f}"
        )


if __name__ == "__main__":
    main()
