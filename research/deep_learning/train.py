#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
# Reproducibility
# ---------------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

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


def _train(model, X_train, y_train, epochs, lr):
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


def _compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    position = np.sign(predictions)
    pnl = position * targets

    mean = pnl.mean()
    std = pnl.std()
    sharpe = float(mean / std) if std > 1e-12 else 0.0
    hit_rate = float((pnl > 0).mean())
    mse = float(np.mean((predictions - targets) ** 2))

    return {
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "mse": mse,
        "n": int(len(pnl)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--feature-set", default="price_sentiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variant", default="core")
    args = parser.parse_args()

    logger.info("=== DL Training ===")
    logger.info(vars(args))

    # Load
    df = load_dataset(args.dataset_version, variant=args.variant)
    X, y, df_clean = get_features(df, args.feature_set)

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, df_clean)

    # ------------------------------------------------------------------
    # NORMALIZATION (CRITICAL FIX)
    # ------------------------------------------------------------------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    logger.info("feature normalization applied (train stats)")
    logger.info("X_train stats: mean=%.4f std=%.4f", X_train.mean(), X_train.std())

    # Convert to tensors
    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    # Model
    model = MLP(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim)

    # Train
    _train(model, X_train_t, y_train_t, args.epochs, args.lr)

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        test_preds = model(X_test_t).numpy()

    train_metrics = _compute_metrics(train_preds, y_train)
    test_metrics = _compute_metrics(test_preds, y_test)

    logger.info("train metrics: %s", train_metrics)
    logger.info("test metrics: %s", test_metrics)

    # Save
    import config as cfg

    out_dir = cfg.OUTPUT_DIR / args.dataset_version / "dl"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_idx_start = len(X_train)
    pred_df = df_clean.iloc[test_idx_start:].copy().reset_index(drop=True)
    pred_df["prediction"] = test_preds  # NO signal column here

    pred_path = out_dir / f"predictions_{args.feature_set}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = out_dir / f"metrics_{args.feature_set}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "train": train_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
