#!/usr/bin/env python3
"""
research/deep_learning/train.py
================================
Train the MLP on a versioned dataset and write predictions + metrics.

Usage::

    python research/deep_learning/train.py \\
        --dataset-version 1.1.0 \\
        --feature-set price_sentiment \\
        --epochs 50

Outputs (written to ``data/output/<version>/dl/``):
    - ``predictions_<feature_set>.csv``   — test-set predictions
    - ``metrics_<feature_set>.json``      — basic evaluation metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging — must be configured before any project imports so the logger
# hierarchy picks up the file handler.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # market-sentiment-ml/


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
# Project imports (after logging setup)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

from research.deep_learning.dataset_loader import (
    get_features,
    load_dataset,
    to_tensors,
    train_test_split,
)
from research.deep_learning.model import MLP


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float = 1e-3,
) -> None:
    """Train *model* in-place using MSE loss and Adam optimiser."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info("epoch %d/%d  loss=%.6f", epoch, epochs, loss.item())


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrors validate_signal_raw logic)
# ---------------------------------------------------------------------------


def _compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute Sharpe and hit-rate from raw predictions."""
    position = np.sign(predictions).astype(float)
    pnl = position * targets

    n = len(pnl)
    if n == 0:
        return {"n": 0, "sharpe": 0.0, "hit_rate": 0.0, "mse": 0.0}

    mean = pnl.mean()
    std = pnl.std()
    sharpe = float(mean / std) if std > 1e-12 else 0.0
    hit_rate = float((pnl > 0).mean())
    mse = float(np.mean((predictions - targets) ** 2))

    return {"n": int(n), "sharpe": sharpe, "hit_rate": hit_rate, "mse": mse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on versioned dataset")
    parser.add_argument("--dataset-version", required=True, help="e.g. 1.1.0")
    parser.add_argument(
        "--feature-set",
        default="price_sentiment",
        choices=["price_only", "price_sentiment"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--variant",
        default="core",
        choices=["full", "core", "extended"],
        help="Dataset variant to load",
    )
    args = parser.parse_args()

    logger.info("=== DL Training ===")
    logger.info(
        "config: version=%s feature_set=%s epochs=%d hidden_dim=%d lr=%s",
        args.dataset_version,
        args.feature_set,
        args.epochs,
        args.hidden_dim,
        args.lr,
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = load_dataset(args.dataset_version, variant=args.variant)
    logger.info("dataset version=%s  rows=%d", args.dataset_version, len(df))

    X, y = get_features(df, args.feature_set)
    logger.info("feature_set=%s  shape=%s", args.feature_set, X.shape)

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, df)

    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=args.hidden_dim)
    logger.info("model: MLP(input_dim=%d, hidden_dim=%d)", input_dim, args.hidden_dim)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    _train(model, X_train_t, y_train_t, epochs=args.epochs, lr=args.lr)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        test_preds = model(X_test_t).numpy()

    train_metrics = _compute_metrics(train_preds, y_train)
    test_metrics = _compute_metrics(test_preds, y_test)

    logger.info("train metrics: %s", train_metrics)
    logger.info("test  metrics: %s", test_metrics)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    import config as cfg

    out_dir = cfg.OUTPUT_DIR / args.dataset_version / "dl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    test_idx_start = len(X_train)
    pred_df = df.iloc[test_idx_start:].copy().reset_index(drop=True)
    pred_df["prediction"] = test_preds
    pred_df["signal"] = np.sign(test_preds)

    pred_path = out_dir / f"predictions_{args.feature_set}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("predictions saved to %s", pred_path)

    # Metrics JSON
    metrics_payload = {
        "dataset_version": args.dataset_version,
        "feature_set": args.feature_set,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "train": train_metrics,
        "test": test_metrics,
    }
    metrics_path = out_dir / f"metrics_{args.feature_set}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("metrics saved to %s", metrics_path)

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
