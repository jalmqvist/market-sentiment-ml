from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def resolve_time_column(df: pd.DataFrame) -> str:
    for col in ["snapshot_time", "entry_time", "timestamp", "time"]:
        if col in df.columns:
            return col
    raise ValueError("Dataset is missing a supported time column")


def resolve_target_column(df: pd.DataFrame, target_horizon: int) -> str:
    preferred = f"ret_{target_horizon}b"
    if preferred in df.columns:
        return preferred
    if "ret_24b" in df.columns:
        return "ret_24b"
    raise ValueError("Dataset is missing target return columns")


def deterministic_seed(*parts: object, base_seed: int = 42) -> int:
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2 ** 32 - 1)


def filter_window(
    df: pd.DataFrame,
    *,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col], errors="coerce")
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    return df.loc[mask].copy()


def build_binary_labels(
    df: pd.DataFrame,
    *,
    target_col: str,
    threshold: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["pair", "entry_time", "y_true"])
    work = df.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], errors="coerce").dt.tz_localize(None)
    work = work.dropna(subset=["pair", "entry_time", target_col])
    work["y_true"] = (work[target_col] > threshold).astype(int)
    out = (
        work[["pair", "entry_time", "y_true"]]
        .groupby(["pair", "entry_time"], as_index=False)
        .agg({"y_true": "mean"})
    )
    out["y_true"] = (out["y_true"] >= 0.5).astype(int)
    return out


def train_threshold(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    label_quantile: float,
) -> float | None:
    if train_df.empty:
        return None
    vals = pd.to_numeric(train_df[target_col], errors="coerce").dropna().abs()
    if vals.empty:
        return None
    return float(vals.quantile(label_quantile))


def match_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if predictions.empty or labels.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    pred = predictions.copy()
    pred["pair"] = pred["pair"].astype(str).str.strip().str.lower()
    pred["entry_time"] = pd.to_datetime(pred["entry_time"], errors="coerce").dt.tz_localize(None)
    pred["pred_prob_up"] = pd.to_numeric(pred["pred_prob_up"], errors="coerce")

    lab = labels.copy()
    lab["pair"] = lab["pair"].astype(str).str.strip().str.lower()
    lab["entry_time"] = pd.to_datetime(lab["entry_time"], errors="coerce").dt.tz_localize(None)

    merged = pred.merge(lab, on=["pair", "entry_time"], how="inner")
    merged = merged.dropna(subset=["pred_prob_up", "y_true"])

    y_true = merged["y_true"].astype(int).to_numpy()
    y_prob = merged["pred_prob_up"].astype(float).to_numpy()
    return y_true, y_prob
