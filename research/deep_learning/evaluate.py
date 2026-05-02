#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.logging import setup_experiment_logging

log_file = setup_experiment_logging(
    experiment_type="evaluate",
    tag="dl",
    log_dir=_REPO_ROOT / "logs",
)

logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------


def _compute_regression_metrics(df):
    pnl = df["position"] * df["ret_48b"]

    if len(pnl) == 0:
        return 0.0, 0.0

    sharpe = pnl.mean() / pnl.std() if pnl.std() > 1e-12 else 0.0
    hit = (pnl > 0).mean()
    return sharpe, hit


def _compute_classification_metrics(df):
    """Compute accuracy, precision, recall, F1 from prediction logits/probs."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    if "target_cls" not in df.columns:
        logger.warning("No 'target_cls' column in predictions file; skipping classification metrics")
        return None

    # Support both logit and probability columns
    if "logit" in df.columns:
        pred_labels = (df["logit"] > 0).astype(int)
    elif "prediction" in df.columns:
        pred_labels = (df["prediction"] > 0.5).astype(int)
    else:
        logger.warning("No 'logit' or 'prediction' column; skipping classification metrics")
        return None

    y = df["target_cls"].astype(int)

    accuracy = float((pred_labels == y).mean())
    precision = float(precision_score(y, pred_labels, zero_division=0))
    recall = float(recall_score(y, pred_labels, zero_division=0))
    f1 = float(f1_score(y, pred_labels, zero_division=0))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _run(df, label):
    results = {}

    # Regression / Sharpe metrics (requires ret_48b and signal columns)
    if "ret_48b" in df.columns and "signal" in df.columns:
        sharpe, hit = _compute_regression_metrics(df)
        density = (df["signal"] != 0).mean()
        results["sharpe"] = sharpe
        results["hit_rate"] = hit
        results["density"] = density
        logger.info(
            "[%s] n=%d sharpe=%.4f hit=%.4f density=%.4f",
            label,
            len(df),
            sharpe,
            hit,
            density,
        )

    # Classification metrics
    cls_metrics = _compute_classification_metrics(df)
    if cls_metrics is not None:
        results.update(cls_metrics)
        logger.info(
            "[%s] accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            label,
            cls_metrics["accuracy"],
            cls_metrics["precision"],
            cls_metrics["recall"],
            cls_metrics["f1"],
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--shifts", default="1,5")
    args = parser.parse_args()

    shifts = [int(x) for x in args.shifts.split(",")]

    df = pd.read_csv(args.predictions)

    # Build signal: logits → sign; probabilities → {-1, +1} via >0.5 threshold
    if "logit" in df.columns:
        df["signal"] = np.sign(df["logit"])
    elif "prediction" in df.columns:
        # prediction may be a probability [0,1] or raw score
        # Use >0.5 threshold so signal is always ±1 (avoids np.sign returning 1 for all probs)
        df["signal"] = np.where(df["prediction"] > 0.5, 1.0, -1.0)
    else:
        logger.error("Predictions file must contain 'logit' or 'prediction' column")
        sys.exit(1)

    # baseline
    base = df.copy()
    base["position"] = base["signal"]
    _run(base, "baseline")

    # shifts
    for s in shifts:
        shifted = df.copy()
        shifted["signal"] = shifted["signal"].shift(s)
        shifted = shifted.dropna(subset=["signal"])
        shifted["position"] = shifted["signal"]
        _run(shifted, f"shift({s})")

    # shuffled
    shuffled = df.copy()
    shuffled["signal"] = np.random.permutation(shuffled["signal"])
    shuffled["position"] = shuffled["signal"]
    _run(shuffled, "shuffled")


if __name__ == "__main__":
    main()
