import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.deep_learning.dataset_loader import load_dataset
from research.deep_learning.lstm_dataset import (
    build_sequences,
    train_test_split_sequences,
)


# =========================
# Base feature sets
# =========================
BASE_FEATURES = {
    "price_trend": [
        "trend_12b",
        "trend_vol_adj_strength",
        "is_trending",
        "is_high_vol",
    ],
}


# =========================
# Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()  # (N,)


# =========================
# Utils
# =========================
def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/lstm_{pd.Timestamp.now('UTC'):%Y%m%d_%H%M%S}.log")
        ]
    )


def normalize_pair(pair: str) -> str:
    pair = pair.lower().replace("/", "").replace("_", "")
    return pair[:3] + "-" + pair[3:]


def normalize_sequences(X_train, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def compute_metrics(preds, y_true):
    preds = np.array(preds)
    y_true = np.array(y_true)

    accuracy = (preds == y_true).mean()

    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    pnl = preds * y_true
    sharpe = pnl.mean() / (pnl.std() + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sharpe": float(sharpe),
        "n": len(preds),
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--regime", type=str, default=None)

    parser.add_argument("--mode", choices=["classification", "regression"], default="classification")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--label-quantile", type=float, default=0.6)
    parser.add_argument("--label-mode", type=str, default="sign", choices=["sign", "threshold"])

    args = parser.parse_args()

    # Default seq_len to target_horizon when not explicitly provided
    if args.seq_len is None:
        args.seq_len = args.target_horizon

    setup_logging()

    logging.info("=== LSTM Training ===")
    logging.info(vars(args))
    logging.info({
        "target_horizon": args.target_horizon,
        "label_mode": args.label_mode,
        "label_quantile": args.label_quantile,
    })

    # =========================
    # Load dataset
    # =========================
    df = load_dataset(args.dataset_version, variant="core")

    # =========================
    # Filter
    # =========================
    rows_before = len(df)

    if args.pairs:
        pairs = [normalize_pair(p) for p in args.pairs.split(",")]
        df = df[df["pair"].isin(pairs)]

    if args.regime:
        df = df[df["regime"] == args.regime]

    logging.info(f"rows_before_filter: {rows_before}")
    logging.info(f"rows_after_filter: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "Filtering resulted in 0 rows. "
            "Check pair format and regime availability."
        )

    # =========================
    # Feature selection
    # =========================
    if args.feature_set == "price_trend":
        features = BASE_FEATURES["price_trend"]

    elif args.feature_set == "price_trend_sentiment":
        base = BASE_FEATURES["price_trend"]

        sentiment_cols = [
            c for c in df.columns if "sentiment" in c.lower()
        ]

        if not sentiment_cols:
            raise ValueError("No sentiment columns found in dataset")

        logging.info(f"Detected sentiment features: {sentiment_cols}")

        features = base + sentiment_cols

    else:
        raise ValueError(f"Unknown feature_set: {args.feature_set}")

    # =========================
    # Target
    # =========================
    target = f"ret_{args.target_horizon}b"
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    if args.mode == "classification":
        if args.label_mode == "sign":
            df["target_direction"] = (df[target] > 0).astype(int)
        elif args.label_mode == "threshold":
            q = df[target].abs().quantile(args.label_quantile)
            df["target_direction"] = np.nan
            df.loc[df[target] > q, "target_direction"] = 1
            df.loc[df[target] < -q, "target_direction"] = 0
            before = len(df)
            df = df.dropna(subset=["target_direction"])
            after = len(df)
            logging.info(f"label_threshold: {q:.6f}")
            logging.info(f"rows_after_label_filter: {after} (dropped {before - after})")
        logging.info(f"class_balance: {df['target_direction'].mean():.3f}")
        target = "target_direction"

    df = df.dropna(subset=features + [target]).copy()

    # =========================
    # Build sequences
    # =========================
    X, y = build_sequences(df, features, target, args.seq_len)

    logging.info(f"Sequences built: X={X.shape}, y={y.shape}")

    # =========================
    # Split
    # =========================
    (X_train, y_train), (X_test, y_test) = train_test_split_sequences(X, y)

    # =========================
    # Normalize
    # =========================
    X_train, X_test = normalize_sequences(X_train, X_test)

    # =========================
    # Tensors
    # =========================
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1)

    # =========================
    # Model
    # =========================
    model = LSTMModel(input_dim=X.shape[2], hidden_dim=args.hidden_dim)

    loss_fn = nn.BCEWithLogitsLoss() if args.mode == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # =========================
    # Training
    # =========================
    for epoch in range(1, args.epochs + 1):
        model.train()

        optimizer.zero_grad()

        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            logging.info(f"epoch {epoch}/{args.epochs} loss={loss.item():.6f}")

    # =========================
    # Evaluation
    # =========================
    model.eval()

    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        test_preds = model(X_test_t).numpy()

    if args.mode == "classification":
        train_probs = torch.sigmoid(torch.tensor(train_preds)).numpy()
        test_probs = torch.sigmoid(torch.tensor(test_preds)).numpy()

        train_preds_bin = (train_probs > 0.5).astype(int)
        test_preds_bin = (test_probs > 0.5).astype(int)

        train_metrics = compute_metrics(train_preds_bin, y_train)
        test_metrics = compute_metrics(test_preds_bin, y_test)
    else:
        train_metrics = compute_metrics(train_preds, y_train)
        test_metrics = compute_metrics(test_preds, y_test)

    logging.info(f"train metrics: {train_metrics}")
    logging.info(f"test metrics: {test_metrics}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()