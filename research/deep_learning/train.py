#!/usr/bin/env python3
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

# ---------------------------------------------------------------------------
# Repo root (NO config dependency)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
OUTPUT_DIR = REPO_ROOT / "data" / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from utils.logging import setup_experiment_logging
from research.deep_learning.dataset_loader import (
    get_features,
    load_dataset,
    to_tensors,
    train_test_split,
)
from research.deep_learning.feature_sets import TARGET, TARGET_CLS
from research.deep_learning.model import MLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pair(pair: str) -> str:
    pair = pair.strip()
    letters = "".join(c for c in pair if c.isalpha())
    if len(letters) == 6:
        return f"{letters[:3].lower()}-{letters[3:].lower()}"
    return pair.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(model, X_train, y_train, epochs, lr, criterion):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info("epoch %d/%d  loss=%.6f", epoch, epochs, loss.item())


def _compute_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
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


def _compute_classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score

    pred_labels = (logits > 0).astype(int)
    y = targets.astype(int)

    accuracy = float((pred_labels == y).mean())
    precision = float(precision_score(y, pred_labels, zero_division=0))
    recall = float(recall_score(y, pred_labels, zero_division=0))
    f1 = float(f1_score(y, pred_labels, zero_division=0))

    # Sharpe using sigmoid probability as position proxy
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Convert binary labels (0/1) to position signs (-1/+1) for Sharpe computation
    position = np.where(probs > 0.5, 1.0, -1.0)
    ret_proxy = position * (2.0 * targets - 1.0)  # (0/1) → (-1/+1)
    std = ret_proxy.std()
    sharpe = float(ret_proxy.mean() / std) if std > 1e-12 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sharpe": sharpe,
        "n": int(len(targets)),
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
    parser.add_argument("--tag", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--regime", default=None, choices=["HVTF", "LVTF", "HVR", "LVR"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument(
        "--mode",
        default="classification",
        choices=["regression", "classification"],
        help="Training mode: 'classification' (default) or 'regression'",
    )
    args = parser.parse_args()

    tag = args.tag if args.tag else args.feature_set

    log_file = setup_experiment_logging(
        experiment_type="mlp",
        tag=tag,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=LOG_DIR,
    )

    if log_file:
        logger.info("Logging to %s", log_file)

    logger.info("=== MLP Training ===")
    logger.info(vars(args))
    logger.info("feature_set: %s", args.feature_set)
    logger.info("dataset_version: %s", args.dataset_version)
    logger.info("mode: %s", args.mode)

    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = load_dataset(args.dataset_version, variant=args.variant)

    suffix = "" if args.variant == "full" else f"_{args.variant}"
    dataset_path = OUTPUT_DIR / args.dataset_version / f"master_research_dataset{suffix}.csv"
    logger.info("dataset_path: %s", dataset_path)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    rows_before = len(df)

    if args.pairs:
        pairs = [_normalize_pair(p) for p in args.pairs.split(",")]
        df = df[df["pair"].isin(pairs)]

    if args.regime:
        df = df[df["regime"] == args.regime]

    df = df.reset_index(drop=True)

    rows_after = len(df)

    logger.info("pairs: %s", args.pairs or "all")
    logger.info("regime: %s", args.regime or "all")
    logger.info("rows_before_filter: %d", rows_before)
    logger.info("rows_after_filter: %d", rows_after)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    is_classification = args.mode == "classification"
    target = TARGET_CLS if is_classification else TARGET

    X, y, df_clean = get_features(df, args.feature_set, target=target)

    if is_classification:
        logger.info("class_balance: %.3f", y.mean())

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, df_clean)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    logger.info("feature normalization applied")

    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    if is_classification:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MLP(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim)

    _train(model, X_train_t, y_train_t, args.epochs, args.lr, criterion)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).numpy().squeeze()
        test_preds = model(X_test_t).numpy().squeeze()

    if is_classification:
        train_metrics = _compute_classification_metrics(train_preds, y_train)
        test_metrics = _compute_classification_metrics(test_preds, y_test)
    else:
        train_metrics = _compute_regression_metrics(train_preds, y_train)
        test_metrics = _compute_regression_metrics(test_preds, y_test)

    logger.info("train metrics: %s", train_metrics)
    logger.info("test metrics: %s", test_metrics)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = OUTPUT_DIR / args.dataset_version / "dl"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = df_clean.iloc[len(X_train):].copy().reset_index(drop=True)
    if is_classification:
        # Save raw logits; sigmoid gives probability
        pred_df["logit"] = test_preds
        pred_df["prediction"] = 1.0 / (1.0 + np.exp(-test_preds))
    else:
        pred_df["prediction"] = test_preds

    mode_suffix = f"_{args.mode}"
    pred_path = out_dir / f"predictions_{args.feature_set}{mode_suffix}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = out_dir / f"metrics_{args.feature_set}{mode_suffix}.json"
    metrics_path.write_text(json.dumps({
        "config": vars(args),
        "train": train_metrics,
        "test": test_metrics,
    }, indent=2))

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()