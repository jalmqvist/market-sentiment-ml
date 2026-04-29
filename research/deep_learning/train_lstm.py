from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.io import setup_logging
import config as cfg

from research.deep_learning.feature_sets import FEATURE_SETS
from research.deep_learning.lstm_dataset import (
    build_sequences,
    train_test_split_sequences,
)
from research.deep_learning.lstm_model import LSTMModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--feature-set", required=True)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_dataset(version: str) -> pd.DataFrame:
    path = Path(cfg.OUTPUT_DIR) / version / "master_research_dataset_core.csv"
    logger.info(f"Loading dataset: {path}")
    df = pd.read_csv(path)

    if "snapshot_time" in df.columns:
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])

    return df


def normalize_sequences(X_train: np.ndarray, X_test: np.ndarray):
    """
    Normalize using train statistics only (no leakage).
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    logger.info("Sequence normalization applied (train stats)")

    return X_train, X_test


def compute_metrics(y_true, y_pred):
    """
    Same philosophy as MLP:
    - Sharpe from sign(prediction)
    - hit rate
    - mse
    """
    pred_sign = np.sign(y_pred)
    pnl = pred_sign * y_true

    sharpe = pnl.mean() / (pnl.std() + 1e-8)
    hit = (np.sign(y_true) == pred_sign).mean()
    mse = ((y_true - y_pred) ** 2).mean()

    return {
        "sharpe": float(sharpe),
        "hit_rate": float(hit),
        "mse": float(mse),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging("INFO")

    logger.info("=== LSTM Training ===")
    logger.info(vars(args))

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    df = load_dataset(args.dataset_version)

    features = FEATURE_SETS[args.feature_set]
    target = "ret_48b"

    # -----------------------------------------------------------------
    # Build sequences
    # -----------------------------------------------------------------
    X, y = build_sequences(df, features, target, args.seq_len)

    logger.info(f"Sequences built: X={X.shape}, y={y.shape}")

    # -----------------------------------------------------------------
    # Train/test split (chronological)
    # -----------------------------------------------------------------
    (X_train, y_train), (X_test, y_test) = train_test_split_sequences(X, y)

    # -----------------------------------------------------------------
    # Normalize (train stats only)
    # -----------------------------------------------------------------
    X_train, X_test = normalize_sequences(X_train, X_test)

    # -----------------------------------------------------------------
    # Convert to torch
    # -----------------------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    input_dim = X_train.shape[2]
    model = LSTMModel(input_dim=input_dim, hidden_dim=args.hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()

        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            logger.info(f"epoch {epoch}/{args.epochs} loss={loss.item():.6f}")

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        test_preds = model(X_test_t).numpy()

    train_metrics = compute_metrics(y_train, train_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    logger.info(f"train metrics: {train_metrics}")
    logger.info(f"test metrics: {test_metrics}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # -----------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------
    out_dir = Path(cfg.OUTPUT_DIR) / args.dataset_version / "dl"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"predictions_lstm_{args.feature_set}.csv"

    df_out = pd.DataFrame({
        "prediction": test_preds,
        "ret_48b": y_test,
    })

    df_out.to_csv(out_path, index=False)

    logger.info(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
