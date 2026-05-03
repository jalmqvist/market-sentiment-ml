import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.deep_learning.dataset_loader import load_dataset


# =========================
# Logging
# =========================
def setup_logging(name: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{name}_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging to {log_path}")


# =========================
# Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0).astype(int)

    accuracy = (y_true == y_pred_bin).mean()

    tp = ((y_true == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true == 1) & (y_pred_bin == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    returns = y_true * y_pred_bin
    sharpe = returns.mean() / (returns.std() + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sharpe": float(sharpe),
        "n": int(len(y_true)),
    }


# =========================
# Sequence builder (per pair)
# =========================
def build_sequences_per_pair(df, features, target_col, seq_len):
    X_list, y_list = [], []

    for pair in df["pair"].unique():
        df_p = df[df["pair"] == pair].sort_values("timestamp")

        X_vals = df_p[features].values.astype(np.float32)
        y_vals = df_p[target_col].values.astype(np.float32)

        if len(df_p) < seq_len:
            continue

        for i in range(len(df_p) - seq_len):
            X_list.append(X_vals[i:i + seq_len])
            y_list.append(y_vals[i + seq_len - 1])

    if len(X_list) == 0:
        raise ValueError("No sequences created — check filters")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pairs")
    parser.add_argument("--regime")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=24)

    args = parser.parse_args()

    setup_logging(f"lstm_{args.feature_set}")

    logging.info("=== LSTM Training ===")
    logging.info(vars(args))

    # -------------------------
    # Load
    # -------------------------
    df = load_dataset(args.dataset_version, variant="core")

    # -------------------------
    # Timestamp handling
    # -------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError("No timestamp column found")

    # -------------------------
    # Filtering
    # -------------------------
    if args.pairs:
        pairs = [p.lower().replace("usd", "-usd") for p in args.pairs.split(",")]
        df = df[df["pair"].isin(pairs)]

    if args.regime:
        df = df[df["regime"] == args.regime]

    logging.info(f"rows_after_filter: {len(df)}")

    # -------------------------
    # Target (SIGN-BASED ✅)
    # -------------------------
    ret_col = f"ret_{args.target_horizon}b"

    if ret_col not in df.columns:
        available = [c for c in df.columns if c.startswith("ret_")]
        raise ValueError(f"{ret_col} not found. Available: {available}")

    df["target_direction"] = (df[ret_col] > 0).astype(int)

    # -------------------------
    # Features
    # -------------------------
    base_features = [
        "trend_12b",
        "trend_vol_adj_strength",
        "is_trending",
        "is_high_vol",
    ]

    sentiment_features = [c for c in df.columns if "sentiment" in c]

    if args.feature_set == "price_trend_sentiment":
        features = base_features + sentiment_features
    else:
        features = base_features

    features = [f for f in features if f in df.columns]

    df = df.dropna(subset=features + ["target_direction", "timestamp"]).copy()

    # -------------------------
    # Split (time-based)
    # -------------------------
    df = df.sort_values("timestamp")

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    # -------------------------
    # Build sequences
    # -------------------------
    X_train, y_train = build_sequences_per_pair(
        df_train, features, "target_direction", args.seq_len
    )

    X_test, y_test = build_sequences_per_pair(
        df_test, features, "target_direction", args.seq_len
    )

    logging.info(f"train sequences: {len(X_train)}")
    logging.info(f"test sequences: {len(X_test)}")

    logging.info(f"class_balance_train: {y_train.mean():.3f}")
    logging.info(f"class_balance_test: {y_test.mean():.3f}")

    # -------------------------
    # Normalize
    # -------------------------
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # -------------------------
    # Torch
    # -------------------------
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)

    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    model = LSTMModel(X_train.shape[2], args.hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(args.epochs):
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logging.info(f"epoch {epoch+1} loss={loss.item():.6f}")

    # -------------------------
    # Eval
    # -------------------------
    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        test_preds = model(X_test_t).numpy()

    logging.info(f"pred_positive_rate_test: {(test_preds > 0).mean():.3f}")

    train_metrics = compute_metrics(y_train, train_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    logging.info(f"train metrics: {train_metrics}")
    logging.info(f"test metrics: {test_metrics}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()