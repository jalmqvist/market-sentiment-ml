import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.deep_learning.dataset_loader import load_dataset


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def normalize_pairs(pair_str):
    def norm(p):
        p = p.strip().lower()
        return f"{p[:3]}-{p[3:]}" if "-" not in p else p
    return [norm(p) for p in pair_str.split(",")]


def build_sequences(df, features, target, seq_len):
    df = df.sort_values("snapshot_time")

    X, y = [], []
    data = df[features].values
    target_vals = df[target].values

    for i in range(len(df) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target_vals[i + seq_len])

    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"accuracy": np.nan, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (y_true == y_pred).mean()

    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pairs", type=str)
    parser.add_argument("--regime", type=str)
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--label-quantile", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=24)

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    pair_str = args.pairs.strip().lower().replace(",", "-") if args.pairs else "all"
    regime_str = args.regime.strip().lower() if args.regime else "all"
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    log_file = f"logs/lstm_{pair_str}_{regime_str}_h{args.target_horizon}_q{args.label_quantile}_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info("=== LSTM Training ===")
    logging.info(f"CONFIG | model=lstm pair={args.pairs} regime={args.regime} horizon={args.target_horizon} quantile={args.label_quantile}")

    df = load_dataset(args.dataset_version, variant="core")

    if args.pairs:
        pairs = normalize_pairs(args.pairs)
        logging.info(f"normalized_pairs: {pairs}")
        df = df[df["pair"].isin(pairs)]

    if args.regime:
        df = df[df["regime"] == args.regime]

    logging.info(f"rows_after_filter: {len(df)}")

    if len(df) < 50:
        logging.warning(f"SKIP | reason=too_few_rows | rows={len(df)}")
        return

    ret_col = f"ret_{args.target_horizon}b"
    if ret_col not in df.columns:
        logging.warning(f"{ret_col} missing → fallback to ret_24b")
        ret_col = "ret_24b"

    threshold = float(df[ret_col].abs().quantile(args.label_quantile))
    logging.info(f"label_threshold: {threshold:.6f}")

    df["target_direction"] = (df[ret_col] > threshold).astype(int)

    BASE_FEATURES = [
        "trend_12b", "trend_24b", "trend_48b",
        "vol_12b", "vol_48b",
        "net_sentiment", "abs_sentiment",
        "sentiment_change", "sentiment_z"
    ]

    features = [c for c in BASE_FEATURES if c in df.columns]
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    features = [c for c in features if c in numeric_cols]

    logging.info(f"using_features: {features}")

    df = df.copy()
    df[features] = df[features].fillna(0.0)
    df = df.dropna(subset=["target_direction"])

    X, y = build_sequences(df, features, "target_direction", args.seq_len)

    if len(X) < 50:
        logging.warning(f"SKIP | reason=too_few_sequences | seqs={len(X)}")
        return

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) == 0:
        logging.warning("SKIP | reason=empty_test_set | rows=0")
        return

    if y_train.sum() == 0:
        logging.warning("SKIP | reason=no_positive_class")
        return

    logging.info(f"class_balance_train: {y_train.mean():.3f}")
    logging.info(f"class_balance_test: {y_test.mean():.3f}")

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32)

    model = LSTMModel(X.shape[2], args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train).float()

    for epoch in range(args.epochs):
        preds = model(Xt)
        loss = loss_fn(preds, yt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
            logging.info(f"epoch {epoch+1} loss={loss.item():.6f}")

    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)

    logging.info(f"pred_positive_rate_test: {preds.mean():.3f}")

    metrics = compute_metrics(y_test, preds)
    metrics["n"] = len(y_test)

    logging.info("=== Test metrics ===")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")


if __name__ == "__main__":
    main()