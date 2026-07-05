"""Scientific metrics for behavioral prediction artifacts.

These metrics describe model behavior independently of trading performance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Element-wise binary entropy H(p) = -p·log₂(p) - (1-p)·log₂(1-p)."""
    eps = 1e-10
    p = np.clip(p, eps, 1.0 - eps)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _normalised_entropy(counts: np.ndarray) -> float:
    """Normalised Shannon entropy in [0, 1] for a count array."""
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    raw = float(-(probs * np.log2(probs)).sum())
    return raw / np.log2(len(counts))


def compute_prediction_metrics(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.1,
    artifact_file: str | None = None,
    surface_id: str | None = None,
    state_id: str | None = None,
) -> dict[str, object]:
    """Compute scientific metrics for a single prediction artifact DataFrame.

    Parameters
    ----------
    df:
        Prediction artifact with at least ``pred_prob_up`` and optionally
        ``signal_strength``, ``pair``, ``entry_time``.
    confidence_threshold:
        Minimum |pred_prob_up - 0.5| to count a prediction as materially
        informative for the *effective prediction coverage* metric.
    artifact_file:
        Optional file label included in the returned dict for traceability.
    surface_id / state_id:
        Optional provenance labels included verbatim in the returned dict.
    """
    row: dict[str, object] = {
        "artifact_file": artifact_file or "",
        "surface_id": surface_id,
        "state_id": state_id,
        "n_predictions": int(len(df)),
    }

    # ------------------------------------------------------------------
    # Pair balance
    # ------------------------------------------------------------------
    if "pair" in df.columns and len(df) > 0:
        pair_counts = df["pair"].value_counts()
        row["n_pairs"] = int(pair_counts.shape[0])
        row["pair_balance"] = round(_normalised_entropy(pair_counts.values), 4)
    else:
        row["n_pairs"] = 0
        row["pair_balance"] = np.nan

    # ------------------------------------------------------------------
    # Timestamp coverage
    # ------------------------------------------------------------------
    time_col = next(
        (c for c in ["entry_time", "timestamp", "snapshot_time"] if c in df.columns),
        None,
    )
    if time_col and len(df) > 0:
        ts = pd.to_datetime(df[time_col], errors="coerce").dropna()
        row["timestamp_unique"] = int(ts.nunique())
        row["timestamp_min"] = ts.min()
        row["timestamp_max"] = ts.max()
        if len(ts) >= 2:
            row["coverage_days"] = round(
                (ts.max() - ts.min()).total_seconds() / 86400.0, 2
            )
        else:
            row["coverage_days"] = 0.0
    else:
        row["timestamp_unique"] = 0
        row["timestamp_min"] = pd.NaT
        row["timestamp_max"] = pd.NaT
        row["coverage_days"] = np.nan

    # ------------------------------------------------------------------
    # Prediction probability distribution
    # ------------------------------------------------------------------
    if "pred_prob_up" in df.columns and len(df) > 0:
        probs = pd.to_numeric(df["pred_prob_up"], errors="coerce").dropna().values
        n = len(probs)
        if n > 0:
            entropy_vals = _binary_entropy(probs)
            row["prediction_entropy_mean"] = round(float(entropy_vals.mean()), 6)
            row["prediction_entropy_std"] = round(float(entropy_vals.std()), 6)

            confidence = np.abs(probs - 0.5)
            row["prediction_confidence_mean"] = round(float(confidence.mean()), 6)
            row["prediction_confidence_std"] = round(float(confidence.std()), 6)
            # Sharpness: mean confidence normalised to [0, 1]
            row["sharpness"] = round(float(confidence.mean() * 2.0), 6)

            # Effective prediction coverage: fraction with |p - 0.5| > threshold
            row["effective_prediction_coverage"] = round(
                float((confidence > confidence_threshold).mean()), 6
            )

            # Distribution quantiles
            for pct, label in [(5, "p05"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")]:
                row[f"pred_prob_{label}"] = round(float(np.percentile(probs, pct)), 6)
            row["pred_prob_mean"] = round(float(probs.mean()), 6)
            row["pred_prob_std"] = round(float(probs.std()), 6)

            # Fraction predicting UP vs DOWN
            row["pred_fraction_up"] = round(float((probs >= 0.5).mean()), 6)
        else:
            for k in [
                "prediction_entropy_mean", "prediction_entropy_std",
                "prediction_confidence_mean", "prediction_confidence_std",
                "sharpness", "effective_prediction_coverage",
                "pred_prob_p05", "pred_prob_p25", "pred_prob_p50",
                "pred_prob_p75", "pred_prob_p95",
                "pred_prob_mean", "pred_prob_std", "pred_fraction_up",
            ]:
                row[k] = np.nan
    else:
        for k in [
            "prediction_entropy_mean", "prediction_entropy_std",
            "prediction_confidence_mean", "prediction_confidence_std",
            "sharpness", "effective_prediction_coverage",
            "pred_prob_p05", "pred_prob_p25", "pred_prob_p50",
            "pred_prob_p75", "pred_prob_p95",
            "pred_prob_mean", "pred_prob_std", "pred_fraction_up",
        ]:
            row[k] = np.nan

    # ------------------------------------------------------------------
    # Signal-strength distribution
    # ------------------------------------------------------------------
    if "signal_strength" in df.columns and len(df) > 0:
        signals = pd.to_numeric(df["signal_strength"], errors="coerce").dropna().values
        if len(signals) > 0:
            for pct, label in [(5, "p05"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")]:
                row[f"signal_strength_{label}"] = round(float(np.percentile(signals, pct)), 6)
            row["signal_strength_mean"] = round(float(signals.mean()), 6)
            row["signal_strength_std"] = round(float(signals.std()), 6)
            row["signal_strength_positive_fraction"] = round(float((signals > 0).mean()), 6)
        else:
            for k in [
                "signal_strength_p05", "signal_strength_p25", "signal_strength_p50",
                "signal_strength_p75", "signal_strength_p95",
                "signal_strength_mean", "signal_strength_std",
                "signal_strength_positive_fraction",
            ]:
                row[k] = np.nan
    else:
        for k in [
            "signal_strength_p05", "signal_strength_p25", "signal_strength_p50",
            "signal_strength_p75", "signal_strength_p95",
            "signal_strength_mean", "signal_strength_std",
            "signal_strength_positive_fraction",
        ]:
            row[k] = np.nan

    return row


def compute_prediction_metrics_from_path(
    path: Path,
    *,
    confidence_threshold: float = 0.1,
    surface_id: str | None = None,
    state_id: str | None = None,
) -> dict[str, object]:
    """Load a prediction artifact from *path* and compute metrics."""
    df = pd.read_parquet(path)
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    return compute_prediction_metrics(
        df,
        confidence_threshold=confidence_threshold,
        artifact_file=path.name,
        surface_id=surface_id,
        state_id=state_id,
    )
