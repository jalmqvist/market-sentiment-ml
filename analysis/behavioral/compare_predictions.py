from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prediction_artifact(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    return df


def summarize_prediction_artifact(path: Path) -> dict[str, object]:
    df = load_prediction_artifact(path)
    row: dict[str, object] = {
        "artifact_file": path.name,
        "prediction_row_count": int(len(df)),
        "pair_count": int(df["pair"].nunique()) if "pair" in df.columns else 0,
    }
    if "entry_time" in df.columns and len(df) > 0:
        row["entry_time_min"] = df["entry_time"].min()
        row["entry_time_max"] = df["entry_time"].max()
        row["entry_time_unique"] = int(df["entry_time"].nunique())
    else:
        row["entry_time_min"] = pd.NaT
        row["entry_time_max"] = pd.NaT
        row["entry_time_unique"] = 0

    for col in ["pred_prob_up", "signal_strength"]:
        if col in df.columns and len(df) > 0:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(series.mean()) if len(series) else np.nan
            row[f"{col}_std"] = float(series.std()) if len(series) else np.nan
            row[f"{col}_p05"] = float(series.quantile(0.05)) if len(series) else np.nan
            row[f"{col}_p50"] = float(series.quantile(0.50)) if len(series) else np.nan
            row[f"{col}_p95"] = float(series.quantile(0.95)) if len(series) else np.nan
        else:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"] = np.nan
            row[f"{col}_p05"] = np.nan
            row[f"{col}_p50"] = np.nan
            row[f"{col}_p95"] = np.nan
    return row


def compare_mlp_lstm_predictions(
    *,
    mlp_path: Path | None,
    lstm_path: Path | None,
    surface_id: str,
    state_id: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "surface_id": surface_id,
        "state_id": state_id,
        "agreement_common_rows": 0,
        "agreement_rate": np.nan,
        "pred_prob_correlation": np.nan,
        "signal_strength_correlation": np.nan,
    }

    if mlp_path is None or lstm_path is None:
        return row

    mlp = load_prediction_artifact(mlp_path)
    lstm = load_prediction_artifact(lstm_path)

    keys = [c for c in ["pair", "entry_time"] if c in mlp.columns and c in lstm.columns]
    if len(keys) != 2:
        return row

    merged = mlp.merge(
        lstm,
        on=keys,
        how="inner",
        suffixes=("_mlp", "_lstm"),
    )
    row["agreement_common_rows"] = int(len(merged))
    if len(merged) == 0:
        return row

    pred_prob_cols = ["pred_prob_up_mlp", "pred_prob_up_lstm"]
    signal_cols = ["signal_strength_mlp", "signal_strength_lstm"]

    if set(pred_prob_cols).issubset(merged.columns):
        a = pd.to_numeric(merged[pred_prob_cols[0]], errors="coerce")
        b = pd.to_numeric(merged[pred_prob_cols[1]], errors="coerce")
        valid = a.notna() & b.notna()
        if valid.any():
            a_valid = a[valid]
            b_valid = b[valid]
            row["agreement_rate"] = float(((a_valid >= 0.5) == (b_valid >= 0.5)).mean())
            row["pred_prob_correlation"] = float(a_valid.corr(b_valid))

    if set(signal_cols).issubset(merged.columns):
        sa = pd.to_numeric(merged[signal_cols[0]], errors="coerce")
        sb = pd.to_numeric(merged[signal_cols[1]], errors="coerce")
        valid_s = sa.notna() & sb.notna()
        if valid_s.any():
            row["signal_strength_correlation"] = float(sa[valid_s].corr(sb[valid_s]))

    return row
