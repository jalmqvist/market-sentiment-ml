from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.logging import setup_experiment_logging
import config as cfg

from research.deep_learning.feature_sets import FEATURE_SETS, TARGET, TARGET_CLS
from research.deep_learning.lstm_dataset import (
    build_sequences,
    train_test_split_sequences,
)
from research.deep_learning.lstm_model import LSTMModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _normalize_pair(pair: str) -> str:
    """Normalize a pair string to the dataset format (e.g. 'EURUSD' → 'eur-usd')."""
    pair = pair.strip()
    letters = "".join(c for c in pair if c.isalpha())
    if len(letters) == 6:
        return f"{letters[:3].lower()}-{letters[3:].lower()}"
    return pair.lower().replace("_", "-")


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
    p.add_argument("--tag", default=None, help="Log/config file tag (default: feature-set)")
    p.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated list of pairs to filter, e.g. 'EURUSD,GBPUSD,NZDUSD'",
    )
    p.add_argument(
        "--regime",
        default=None,
        choices=["HVTF", "LVTF", "HVR", "LVR"],
        help="Filter dataset to a single market regime (requires dataset version >= 1.3.0)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging; write to stdout only",
    )
    p.add_argument(
        "--mode",
        default="classification",
        choices=["regression", "classification"],
        help="Training mode: 'classification' (default) or 'regression'",
    )
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


def compute_regression_metrics(y_true, y_pred):
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


def compute_classification_metrics(logits, y_true):
    from sklearn.metrics import precision_score, recall_score, f1_score

    pred_labels = (logits > 0).astype(int)
    y = y_true.astype(int)

    accuracy = float((pred_labels == y).mean())
    precision = float(precision_score(y, pred_labels, zero_division=0))
    recall = float(recall_score(y, pred_labels, zero_division=0))
    f1 = float(f1_score(y, pred_labels, zero_division=0))

    probs = 1.0 / (1.0 + np.exp(-logits))
    # Convert binary labels (0/1) to position signs (-1/+1) for Sharpe computation
    position = np.where(probs > 0.5, 1.0, -1.0)
    ret_proxy = position * (2.0 * y_true - 1.0)  # (0/1) → (-1/+1)
    std = ret_proxy.std()
    sharpe = float(ret_proxy.mean() / std) if std > 1e-12 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sharpe": sharpe,
        "n": len(y_true),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    tag = args.tag if args.tag is not None else args.feature_set

    log_file = setup_experiment_logging(
        experiment_type="lstm",
        tag=tag,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=cfg.REPO_ROOT / "logs",
    )

    if log_file is not None:
        logger.info("Logging to %s", log_file)

    logger.info("=== LSTM Training ===")
    logger.info("experiment_type=lstm")
    logger.info("cli_command: %s", " ".join(sys.argv))
    logger.info(vars(args))
    logger.info("feature_set: %s", args.feature_set)
    logger.info("dataset_version: %s", args.dataset_version)
    logger.info("mode: %s", args.mode)

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    df = load_dataset(args.dataset_version)

    dataset_path = str(Path(cfg.OUTPUT_DIR) / args.dataset_version / "master_research_dataset_core.csv")
    logger.info("dataset_path: %s", dataset_path)

    # -----------------------------------------------------------------
    # Regime-aware filtering (BEFORE sequence building)
    # -----------------------------------------------------------------
    rows_before_filter = len(df)

    if args.pairs:
        parsed_pairs = [_normalize_pair(p) for p in args.pairs.split(",")]
        df = df[df["pair"].isin(parsed_pairs)].reset_index(drop=True)

    if args.regime:
        df = df[df["regime"] == args.regime].reset_index(drop=True)

    rows_after_filter = len(df)
    logger.info("pairs: %s", args.pairs or "all")
    logger.info("regime: %s", args.regime or "all")
    logger.info("rows_before_filter: %d", rows_before_filter)
    logger.info("rows_after_filter: %d", rows_after_filter)

    # -----------------------------------------------------------------
    # Config snapshot
    # -----------------------------------------------------------------
    if log_file is not None:
        config_payload = {
            "experiment_type": "lstm",
            "cli_command": " ".join(sys.argv),
            "dataset_path": dataset_path,
            "dataset_version": args.dataset_version,
            "feature_set": args.feature_set,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "pairs": args.pairs,
            "regime": args.regime,
            "rows_before_filter": rows_before_filter,
            "rows_after_filter": rows_after_filter,
        }
        log_file.with_suffix(".json").write_text(json.dumps(config_payload, indent=2))
        logger.info("Config snapshot: %s", log_file.with_suffix(".json"))

    features = FEATURE_SETS[args.feature_set]
    is_classification = args.mode == "classification"
    target = TARGET_CLS if is_classification else TARGET

    # -----------------------------------------------------------------
    # Build sequences
    # -----------------------------------------------------------------
    X, y = build_sequences(df, features, target, args.seq_len)

    logger.info(f"Sequences built: X={X.shape}, y={y.shape}")

    if is_classification:
        logger.info("class_balance: %.3f", y.mean())

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
    if is_classification:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
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
        train_preds = model(X_train_t).numpy().squeeze()
        test_preds = model(X_test_t).numpy().squeeze()

    if is_classification:
        train_metrics = compute_classification_metrics(train_preds, y_train)
        test_metrics = compute_classification_metrics(test_preds, y_test)
    else:
        train_metrics = compute_regression_metrics(y_train, train_preds)
        test_metrics = compute_regression_metrics(y_test, test_preds)

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

    mode_suffix = f"_{args.mode}"
    out_path = out_dir / f"predictions_lstm_{args.feature_set}{mode_suffix}.csv"

    if is_classification:
        df_out = pd.DataFrame({
            "logit": test_preds,
            "prediction": 1.0 / (1.0 + np.exp(-test_preds)),
            "target_cls": y_test,
        })
    else:
        df_out = pd.DataFrame({
            "prediction": test_preds,
            "ret_48b": y_test,
        })

    df_out.to_csv(out_path, index=False)

    logger.info(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
