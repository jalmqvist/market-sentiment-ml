import argparse
import logging
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
    parser.add_argument(
        "--train-pairs",
        type=str,
        default=None,
        help=(
            "Optional training pair universe. If omitted, falls back to --pairs "
            "(backward-compatible behavior)."
        ),
    )
    parser.add_argument(
        "--predict-pairs",
        type=str,
        default=None,
        help=(
            "Optional inference/export pair universe. If omitted, falls back to --pairs "
            "(backward-compatible behavior)."
        ),
    )
    parser.add_argument("--regime", type=str)
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--label-quantile", type=float, default=0.5)
    parser.add_argument(
        "--export-after-year",
        type=int,
        default=None,
        help=(
            "Optional export-only filter: keep only prediction rows with entry_time.year "
            ">= this year when writing parquet artifacts. Does NOT affect training, "
            "validation, metrics, or test split."
        ),
    )
    parser.add_argument(
        "--export-before-year",
        type=int,
        default=None,
        help=(
            "Optional export-only filter: keep only prediction rows with entry_time.year "
            "<= this year when writing parquet artifacts. Does NOT affect training, "
            "validation, metrics, or test split."
        ),
    )
    parser.add_argument(
        "--export-split",
        choices=["test", "all"],
        default="test",
        help=(
            "Which split to export predictions for. "
            "'test' exports only the held-out test split (current behavior). "
            "'all' exports predictions for the full filtered dataset (train+test) "
            "for proof-of-integration overlap. Training and evaluation behavior "
            "remain unchanged."
        ),
    )

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    train_pairs_arg = args.train_pairs if args.train_pairs is not None else args.pairs
    predict_pairs_arg = args.predict_pairs if args.predict_pairs is not None else args.pairs

    train_pairs = normalize_pairs(train_pairs_arg) if train_pairs_arg else None
    predict_pairs = normalize_pairs(predict_pairs_arg) if predict_pairs_arg else None

    pair_str = train_pairs_arg.strip().lower().replace(",", "-") if train_pairs_arg else "all"
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
        f"CONFIG | model=mlp pair={args.pairs} train_pairs={train_pairs_arg} "
        f"predict_pairs={predict_pairs_arg} regime={args.regime} "
        f"horizon={args.target_horizon} quantile={args.label_quantile}"
    )

    df_base = load_dataset(args.dataset_version, variant="core")
    df = df_base.copy()
    infer_df = df_base.copy()

    if args.regime:
        df = df[df["regime"] == args.regime]
        infer_df = infer_df[infer_df["regime"] == args.regime]

    if train_pairs:
        logging.info(f"normalized_train_pairs: {train_pairs}")
        df = df[df["pair"].isin(train_pairs)]
    if predict_pairs:
        logging.info(f"normalized_predict_pairs: {predict_pairs}")
        infer_df = infer_df[infer_df["pair"].isin(predict_pairs)]

    logging.info(f"rows_after_filter: {len(df)}")
    logging.info(f"inference_rows_after_filter: {len(infer_df)}")

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
    model = MLP(X_train.shape[1], args.hidden_dim)

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
        # --- Test probabilities (used for metrics; unchanged behavior) ---
        logits_test = model(torch.tensor(X_test))
        probs_test = torch.sigmoid(logits_test).numpy()
        pred_prob_up_test = np.asarray(probs_test).reshape(-1).astype("float64")

        # Binary predictions for test metrics
        preds = (pred_prob_up_test > 0.5).astype(int)

        # --- Full-dataset probabilities (export-only option; does NOT affect metrics) ---
        # Build normalized full matrix using the SAME train-derived mean/std.
        X_all_norm = (X - mean) / std
        logits_all = model(torch.tensor(X_all_norm.astype("float32")))
        probs_all = torch.sigmoid(logits_all).numpy()
        pred_prob_up_all = np.asarray(probs_all).reshape(-1).astype("float64")

        # --- Inference-only probabilities on predict universe ---
        infer_work_df = infer_df.copy()
        infer_work_df[features] = infer_work_df[features].fillna(0.0)
        X_infer = infer_work_df[features].values.astype("float32")
        X_infer_norm = (X_infer - mean) / std
        logits_infer = model(torch.tensor(X_infer_norm.astype("float32")))
        probs_infer = torch.sigmoid(logits_infer).numpy()
        pred_prob_up_infer = np.asarray(probs_infer).reshape(-1).astype("float64")

    logging.info(f"pred_positive_rate_test: {preds.mean():.3f}")

    metrics = compute_metrics(y_test, preds)
    metrics["n"] = len(y_test)

    logging.info("=== Test metrics ===")

    for k, v in metrics.items():
        logging.info(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Export per-run prediction artifact (parquet + manifest)
    # ------------------------------------------------------------------

    # Use package import instead of sys.path mutation
    from scripts.write_dl_prediction_artifact import (
        write_dl_prediction_artifact,
        PREDICTIONS_DIR_DEFAULT,
    )

    # ------------------------------------------------------------------
    # Choose export frame (export-only; training/eval unchanged)
    # ------------------------------------------------------------------
    use_predict_universe = (
        args.train_pairs is not None or
        args.predict_pairs is not None
    )
    if use_predict_universe:
        export_meta_df = infer_df.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_infer
        logging.info("[export] predict universe (inference-only)")
    elif args.export_split == "all":
        export_meta_df = df.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_all
        logging.info("[export] export_split=all (train+test)")
    else:
        export_meta_df = df_test.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_test
        logging.info("[export] export_split=test (held-out test only)")

    if len(export_meta_df) == 0:
        raise ValueError(
            "Export produced 0 rows for selected predict universe. "
            "Adjust --predict-pairs/--pairs/--regime filters."
        )

    # Explicit normalization for canonical downstream joins
    export_pairs = (
        export_meta_df["pair"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    export_entry_time = pd.to_datetime(
        export_meta_df["entry_time"]
    ).dt.tz_localize(None)

    # Single inference-time timestamp for the exported prediction batch
    # (avoid deprecated Timestamp.utcnow)
    prediction_timestamp = pd.Timestamp.now("UTC").tz_localize(None)

    # Derived canonical signal representation
    signal_strength = (2.0 * export_pred_prob_up) - 1.0

    # Identity metadata
    dl_regime = args.regime if args.regime else "MIXED"

    model_name = "mlp"
    target_horizon = int(args.target_horizon)
    feature_set = str(args.feature_set)

    # ------------------------------------------------------------------
    # Collapse multiple intra-hour sentiment snapshots mapping to the
    # same (pair, entry_time) H1 bar.
    #
    # This preserves the richer internal snapshot-level dataset while
    # enforcing the DL artifact contract:
    #
    #   unique(pair, entry_time)
    #
    # We average probabilities/signal strength across snapshots that
    # fall into the same H1 bar.
    # ------------------------------------------------------------------

    pred_df = pd.DataFrame({
        "pair": export_pairs,
        "entry_time": export_entry_time,
        "pred_prob_up": export_pred_prob_up.astype("float64"),
        "signal_strength": signal_strength.astype("float64"),
        "model": model_name,
        "feature_set": feature_set,
        "dl_regime": dl_regime,
        "target_horizon": target_horizon,
        "prediction_timestamp": prediction_timestamp,
    })

    pre_collapse_rows = len(pred_df)

    pred_df = (
        pred_df
        .groupby(["pair", "entry_time"], as_index=False)
        .agg({
            "pred_prob_up": "mean",
            "signal_strength": "mean",
            "model": "first",
            "feature_set": "first",
            "dl_regime": "first",
            "target_horizon": "first",
            "prediction_timestamp": "first",
        })
    )

    post_collapse_rows = len(pred_df)

    collapsed_rows = pre_collapse_rows - post_collapse_rows

    if collapsed_rows > 0:
        logging.warning(
            "[export] collapsed %d intra-hour duplicate rows "
            "to enforce unique(pair, entry_time)",
            collapsed_rows,
        )

    # Re-extract canonical export arrays after collapse
    export_pairs = pred_df["pair"]
    export_entry_time = pred_df["entry_time"]
    export_pred_prob_up = pred_df["pred_prob_up"].values
    signal_strength = pred_df["signal_strength"].values

    # Final integrity check
    n_dupes = (
        pred_df[["pair", "entry_time"]]
        .duplicated()
        .sum()
    )

    assert n_dupes == 0, (
        f"Duplicate (pair, entry_time) rows remain after collapse: "
        f"{n_dupes}"
    )

    assert np.isfinite(export_pred_prob_up).all(), (
        "Non-finite values detected in pred_prob_up"
    )

    assert np.isfinite(signal_strength).all(), (
        "Non-finite values detected in signal_strength"
    )

    assert (
            (export_pred_prob_up >= 0.0).all() and
            (export_pred_prob_up <= 1.0).all()
    ), "pred_prob_up outside [0, 1]"

    assert (
            (signal_strength >= -1.0).all() and
            (signal_strength <= 1.0).all()
    ), "signal_strength outside [-1, +1]"

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Surface identity columns MUST exist in parquet rows
    # because MPML surface loader filters directly on parquet schema.
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "pair": export_pairs.values,
        "entry_time": export_entry_time.values,

        "pred_prob_up": export_pred_prob_up.astype("float64"),
        "signal_strength": signal_strength.astype("float64"),

        "prediction_timestamp": prediction_timestamp,

        # Surface identity columns
        "model": model_name,
        "dl_regime": dl_regime,
        "target_horizon": pd.Series(
            [target_horizon] * len(export_pred_prob_up),
            dtype="Int64",
        ),
        "feature_set": feature_set,
    })

    # Stable ordering
    pred_df = pred_df.sort_values(
        ["pair", "entry_time"]
    ).reset_index(drop=True)

    # Per-pair monotonicity check
    for _pair, _grp in pred_df.groupby("pair"):
        assert _grp["entry_time"].is_monotonic_increasing, (
            f"Non-monotonic entry_time for pair {_pair!r}"
        )

    identity = {
        "model": model_name,
        "dl_regime": dl_regime,
        "target_horizon": target_horizon,
        "feature_set": feature_set,
    }

    provenance = {
        "dataset_version": args.dataset_version,
        "training_pairs": train_pairs if train_pairs is not None else "all",
        "inference_pairs": predict_pairs if predict_pairs is not None else "all",
    }

    # --- Export-only window filter (does not affect training/eval/metrics) ---
    if (args.export_after_year is not None) or (args.export_before_year is not None):
        if "entry_time" not in pred_df.columns:
            raise ValueError("pred_df missing required column 'entry_time' for export-year filtering")

        pred_df["entry_time"] = pd.to_datetime(pred_df["entry_time"])
        before = len(pred_df)

        export_mask = pd.Series(True, index=pred_df.index)

        if args.export_after_year is not None:
            export_mask &= (pred_df["entry_time"].dt.year >= int(args.export_after_year))

        if args.export_before_year is not None:
            export_mask &= (pred_df["entry_time"].dt.year <= int(args.export_before_year))

        pred_df = pred_df.loc[export_mask].copy()
        after = len(pred_df)

        print(
            f"[export] export_after_year={args.export_after_year} "
            f"export_before_year={args.export_before_year} rows: {before:,} -> {after:,}"
        )

        if len(pred_df) == 0:
            raise ValueError(
                "Export produced 0 rows after export filters. "
                "Adjust export window or export split."
            )

    logging.info(
        "[export] pred_df rows=%s pairs=%s entry_time_range=%s -> %s",
        len(pred_df),
        pred_df["pair"].nunique(),
        pred_df["entry_time"].min(),
        pred_df["entry_time"].max(),
    )
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
