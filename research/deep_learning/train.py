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
# Imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config as cfg
from utils.logging import setup_experiment_logging
from research.deep_learning.dataset_loader import (
    get_features,
    load_dataset,
    to_tensors,
    train_test_split,
)
from research.deep_learning.model import MLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pair(pair: str) -> str:
    """Normalize a pair string to the dataset format (e.g. 'EURUSD' → 'eur-usd')."""
    pair = pair.strip()
    letters = "".join(c for c in pair if c.isalpha())
    if len(letters) == 6:
        return f"{letters[:3].lower()}-{letters[3:].lower()}"
    return pair.lower().replace("_", "-")


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
    parser.add_argument("--tag", default=None, help="Log/config file tag (default: feature-set)")
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated list of pairs to filter, e.g. 'EURUSD,GBPUSD,NZDUSD'",
    )
    parser.add_argument(
        "--regime",
        default=None,
        choices=["HVTF", "LVTF", "HVR", "LVR"],
        help="Filter dataset to a single market regime (requires dataset version >= 1.3.0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging; write to stdout only",
    )
    args = parser.parse_args()

    tag = args.tag if args.tag is not None else args.feature_set

    log_file = setup_experiment_logging(
        experiment_type="mlp",
        tag=tag,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=cfg.REPO_ROOT / "logs",
    )

    if log_file is not None:
        logger.info("Logging to %s", log_file)

    logger.info("=== MLP Training ===")
    logger.info("experiment_type=mlp")
    logger.info("cli_command: %s", " ".join(sys.argv))
    logger.info(vars(args))

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load
    df = load_dataset(args.dataset_version, variant=args.variant)

    suffix = "" if args.variant == "full" else f"_{args.variant}"
    dataset_path = str(cfg.OUTPUT_DIR / args.dataset_version / f"master_research_dataset{suffix}.csv")
    logger.info("dataset_path: %s", dataset_path)

    # ------------------------------------------------------------------
    # Regime-aware filtering (BEFORE feature extraction)
    # ------------------------------------------------------------------
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

    # Config snapshot
    if log_file is not None:
        config_payload = {
            "experiment_type": "mlp",
            "cli_command": " ".join(sys.argv),
            "dataset_path": dataset_path,
            "dataset_version": args.dataset_version,
            "feature_set": args.feature_set,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "variant": args.variant,
            "pairs": args.pairs,
            "regime": args.regime,
            "rows_before_filter": rows_before_filter,
            "rows_after_filter": rows_after_filter,
        }
        log_file.with_suffix(".json").write_text(json.dumps(config_payload, indent=2))
        logger.info("Config snapshot: %s", log_file.with_suffix(".json"))

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
