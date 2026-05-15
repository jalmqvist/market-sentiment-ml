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
    X, y, meta_rows = [], [], []

    sort_col = "snapshot_time" if "snapshot_time" in df.columns else "entry_time"
    meta_cols = ["pair", "entry_time", "snapshot_time"]
    if "regime" in df.columns:
        meta_cols.append("regime")
    meta_cols = [c for c in meta_cols if c in df.columns]

    for pair, pair_df in df.groupby("pair", sort=True):
        pair_df = pair_df.sort_values(sort_col).reset_index(drop=True)
        if len(pair_df) <= seq_len:
            continue

        data = pair_df[features].values
        target_vals = pair_df[target].values

        for i in range(len(pair_df) - seq_len):
            target_idx = i + seq_len
            X.append(data[i:target_idx])
            y.append(target_vals[target_idx])
            meta_row = pair_df.loc[target_idx, meta_cols].to_dict()
            meta_row["__seq_idx"] = len(X) - 1
            meta_rows.append(meta_row)

    if len(X) == 0:
        empty_meta = pd.DataFrame(columns=meta_cols)
        return (
            np.empty((0, seq_len, len(features)), dtype="float32"),
            np.empty((0,), dtype="float32"),
            empty_meta,
        )

    meta_df = pd.DataFrame(meta_rows)
    if "snapshot_time" in meta_df.columns:
        meta_df["snapshot_time"] = pd.to_datetime(meta_df["snapshot_time"])
        meta_df = meta_df.sort_values(["snapshot_time", "pair"]).reset_index(drop=True)
    order = meta_df["__seq_idx"].to_numpy()

    X_arr = np.asarray(X, dtype="float32")[order]
    y_arr = np.asarray(y, dtype="float32")[order]
    meta_df = meta_df.drop(columns="__seq_idx")
    meta_df = meta_df.reset_index(drop=True)
    return X_arr, y_arr, meta_df


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
    parser.add_argument("--seq-len", type=int, default=24)
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
    use_predict_universe = (
        args.predict_pairs is not None or
        train_pairs != predict_pairs
    )

    pair_str = train_pairs_arg.strip().lower().replace(",", "-") if train_pairs_arg else "all"
    regime_str = args.regime.strip().lower() if args.regime else "all"
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    log_file = f"logs/lstm_{pair_str}_{regime_str}_h{args.target_horizon}_q{args.label_quantile}_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info("=== LSTM Training ===")
    logging.info(
        f"CONFIG | model=lstm pair={args.pairs} train_pairs={train_pairs_arg} "
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

    training_pairs_provenance = sorted(
        df["pair"].astype(str).str.strip().str.lower().unique().tolist()
    )
    inference_pairs_provenance = sorted(
        infer_df["pair"].astype(str).str.strip().str.lower().unique().tolist()
    )

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

    X, y, meta_df = build_sequences(df, features, "target_direction", args.seq_len)

    if len(X) < 50:
        logging.warning(f"SKIP | reason=too_few_sequences | seqs={len(X)}")
        return

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    meta_test = meta_df.iloc[split:].copy().reset_index(drop=True)

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
        logits_test = model(torch.tensor(X_test))
        probs_test = torch.sigmoid(logits_test).numpy()
        pred_prob_up_test = np.asarray(probs_test).reshape(-1).astype("float64")

        preds = (pred_prob_up_test > 0.5).astype(int)

        X_all_norm = (X.astype("float32") - mean) / std
        logits_all = model(torch.tensor(X_all_norm))
        probs_all = torch.sigmoid(logits_all).numpy()
        pred_prob_up_all = np.asarray(probs_all).reshape(-1).astype("float64")

        if use_predict_universe:
            infer_work_df = infer_df.copy()
            infer_work_df[features] = infer_work_df[features].fillna(0.0)
            infer_work_df["target_direction"] = 0
            X_infer, _, infer_meta_df = build_sequences(
                infer_work_df, features, "target_direction", args.seq_len
            )
            if len(X_infer) == 0:
                raise ValueError(
                    "Predict universe produced 0 valid LSTM sequences. "
                    f"predict_pairs={predict_pairs_arg!r}, regime={args.regime!r}, "
                    f"seq_len={args.seq_len}."
                )
            X_infer_norm = (X_infer.astype("float32") - mean) / std
            logits_infer = model(torch.tensor(X_infer_norm))
            probs_infer = torch.sigmoid(logits_infer).numpy()
            pred_prob_up_infer = np.asarray(probs_infer).reshape(-1).astype("float64")

    logging.info(f"pred_positive_rate_test: {preds.mean():.3f}")

    metrics = compute_metrics(y_test, preds)
    metrics["n"] = len(y_test)

    logging.info("=== Test metrics ===")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")

    from scripts.write_dl_prediction_artifact import (
        write_dl_prediction_artifact,
        PREDICTIONS_DIR_DEFAULT,
    )

    if use_predict_universe:
        export_meta_df = infer_meta_df.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_infer
        logging.info("[export] predict universe (inference-only)")
    elif args.export_split == "all":
        export_meta_df = meta_df.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_all
        logging.info("[export] export_split=all (train+test)")
    else:
        export_meta_df = meta_test.copy().reset_index(drop=True)
        export_pred_prob_up = pred_prob_up_test
        logging.info("[export] export_split=test (held-out test only)")

    if len(export_meta_df) == 0:
        raise ValueError(
            "Export produced 0 rows for selected predict universe. "
            f"train_pairs={train_pairs_arg!r}, "
            f"predict_pairs={predict_pairs_arg!r}, regime={args.regime!r}."
        )
    if len(export_meta_df) != len(export_pred_prob_up):
        raise ValueError(
            "Metadata/prediction length mismatch in export path: "
            f"meta_rows={len(export_meta_df)} preds={len(export_pred_prob_up)}"
        )

    export_pairs = (
        export_meta_df["pair"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    export_entry_time = pd.to_datetime(
        export_meta_df["entry_time"]
    ).dt.tz_localize(None)

    # Keep tz-naive UTC to match downstream artifact contract.
    prediction_timestamp = pd.Timestamp.now("UTC").tz_localize(None)
    signal_strength = (2.0 * export_pred_prob_up) - 1.0

    dl_regime = args.regime if args.regime else "MIXED"
    model_name = "lstm"
    target_horizon = int(args.target_horizon)
    feature_set = str(args.feature_set)

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

    export_pairs = pred_df["pair"]
    export_entry_time = pred_df["entry_time"]
    export_pred_prob_up = pred_df["pred_prob_up"].values
    signal_strength = pred_df["signal_strength"].values

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

    pred_df = pd.DataFrame({
        "pair": export_pairs.values,
        "entry_time": export_entry_time.values,
        "pred_prob_up": export_pred_prob_up.astype("float64"),
        "signal_strength": signal_strength.astype("float64"),
        "prediction_timestamp": prediction_timestamp,
        "model": model_name,
        "dl_regime": dl_regime,
        "target_horizon": target_horizon,
        "feature_set": feature_set,
    })
    pred_df["target_horizon"] = pred_df["target_horizon"].astype("Int64")
    pred_df = pred_df.sort_values(
        ["pair", "entry_time"]
    ).reset_index(drop=True)

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
        "training_pairs": training_pairs_provenance,
        "inference_pairs": inference_pairs_provenance,
    }

    if (args.export_after_year is not None) or (args.export_before_year is not None):
        pred_df["entry_time"] = pd.to_datetime(pred_df["entry_time"])
        before = len(pred_df)
        export_mask = pd.Series(True, index=pred_df.index)
        if args.export_after_year is not None:
            export_mask &= (pred_df["entry_time"].dt.year >= int(args.export_after_year))
        if args.export_before_year is not None:
            export_mask &= (pred_df["entry_time"].dt.year <= int(args.export_before_year))
        pred_df = pred_df.loc[export_mask].copy()
        after = len(pred_df)
        logging.info(
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
