import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.deep_learning.dataset_loader import load_dataset


# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------
# Metrics
# ---------------------------
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------
# Pair normalization
# ---------------------------
def normalize_pairs(pair_str):
    def norm(p):
        p = p.strip().lower()
        return f"{p[:3]}-{p[3:]}" if "-" not in p else p

    return [norm(p) for p in pair_str.split(",")]


# ---------------------------
# Main
# ---------------------------
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

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    pair_str = args.pairs.strip().lower().replace(",", "-") if args.pairs else "all"
    regime_str = args.regime.strip().lower() if args.regime else "all"
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    log_file = (
        f"logs/mlp_{pair_str}_{regime_str}_"
        f"h{args.target_horizon}_q{args.label_quantile}_{ts}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info("=== MLP Training ===")
    logging.info(
        f"CONFIG | model=mlp pair={args.pairs} regime={args.regime} "
        f"horizon={args.target_horizon} quantile={args.label_quantile}"
    )

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

    X = df[features].values.astype("float32")
    y = df["target_direction"].values.astype("float32")

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Explicit metadata split to preserve alignment robustness
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy().reset_index(drop=True)

    logging.info(f"train_size: {len(X_train)} | test_size: {len(X_test)}")

    if len(X_test) == 0:
        logging.warning("SKIP | reason=empty_test_set | rows=0")
        return

    if y_train.sum() == 0:
        logging.warning("SKIP | reason=no_positive_class")
        return

    # Normalize
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Class imbalance handling
    pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32)

    logging.info(f"pos_weight: {pos_weight.item():.3f}")

    # Train
    model = MLP(X.shape[1], args.hidden_dim)

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
            logging.info(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.6f}")

    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        probs = torch.sigmoid(logits).numpy()

        # Flatten explicitly to avoid parquet/object dtype ambiguity
        pred_prob_up = np.asarray(probs).reshape(-1).astype("float64")

        preds = (pred_prob_up > 0.5).astype(int)

    logging.info(f"pred_positive_rate_test: {preds.mean():.3f}")

    metrics = compute_metrics(y_test, preds)
    metrics["n"] = len(y_test)

    logging.info("=== Test metrics ===")

    for k, v in metrics.items():
        logging.info(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Export per-run prediction artifact (parquet + manifest)
    # ------------------------------------------------------------------
    _scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"

    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

    from write_dl_prediction_artifact import (  # noqa: E402
        write_dl_prediction_artifact,
        PREDICTIONS_DIR_DEFAULT,
    )

    # Explicit normalization for canonical downstream joins
    export_pairs = (
        df_test["pair"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    export_entry_time = pd.to_datetime(df_test["entry_time"])

    # Single inference-time timestamp for the exported prediction batch
    prediction_timestamp = pd.Timestamp.utcnow().tz_localize(None)

    # Derived canonical signal representation
    signal_strength = (2.0 * pred_prob_up) - 1.0

    # Integrity checks
    n_dupes = (
        pd.DataFrame({
            "pair": export_pairs,
            "entry_time": export_entry_time,
        })
        .duplicated()
        .sum()
    )

    assert n_dupes == 0, (
        f"Duplicate (pair, entry_time) rows in test set: "
        f"{n_dupes} duplicates found"
    )

    n_nans = int(np.isnan(pred_prob_up).sum())

    assert n_nans == 0, (
        f"NaN values in pred_prob_up: "
        f"{n_nans} of {len(pred_prob_up)} rows are NaN"
    )

    assert np.isfinite(pred_prob_up).all(), (
        "Non-finite values detected in pred_prob_up"
    )

    assert np.isfinite(signal_strength).all(), (
        "Non-finite values detected in signal_strength"
    )

    assert (
        (pred_prob_up >= 0.0).all() and
        (pred_prob_up <= 1.0).all()
    ), "pred_prob_up outside [0, 1]"

    assert (
        (signal_strength >= -1.0).all() and
        (signal_strength <= 1.0).all()
    ), "signal_strength outside [-1, +1]"

    for _pair, _grp in (
        pd.DataFrame({
            "pair": export_pairs,
            "entry_time": export_entry_time,
        })
        .groupby("pair")
    ):
        _et = pd.to_datetime(_grp["entry_time"])

        assert _et.is_monotonic_increasing, (
            f"Non-monotonic entry_time for pair {_pair!r}"
        )

    pred_df = pd.DataFrame({
        "pair": export_pairs.values,
        "entry_time": export_entry_time.values,
        "pred_prob_up": pred_prob_up,
        "signal_strength": signal_strength,
        "prediction_timestamp": prediction_timestamp,
    })

    # Canonical ordering + stable schema
    pred_df = pred_df[
        [
            "pair",
            "entry_time",
            "pred_prob_up",
            "signal_strength",
            "prediction_timestamp",
        ]
    ]

    pred_df["pred_prob_up"] = (
        pred_df["pred_prob_up"].astype("float64")
    )

    pred_df["signal_strength"] = (
        pred_df["signal_strength"].astype("float64")
    )

    pred_df["prediction_timestamp"] = pd.to_datetime(
        pred_df["prediction_timestamp"]
    )

    pred_df = pred_df.sort_values(
        ["pair", "entry_time"]
    ).reset_index(drop=True)

    dl_regime = args.regime if args.regime else "MIXED"

    identity = {
        "model": "mlp",
        "dl_regime": dl_regime,
        "target_horizon": args.target_horizon,
        "feature_set": args.feature_set,
    }

    provenance = {
        "dataset_version": args.dataset_version,
    }

    pq_path, mf_path = write_dl_prediction_artifact(
        df=pred_df,
        identity=identity,
        provenance=provenance,
        output_dir=PREDICTIONS_DIR_DEFAULT,
    )

    logging.info("artifact_parquet: %s", pq_path)
    logging.info("artifact_manifest: %s", mf_path)


if __name__ == "__main__":
    main()
