"""
research/abm/calibration.py
============================
Calibration helpers for the retail FX sentiment ABM.

This version extends the original implementation with:

1. Standard moment matching (mean, std, etc.)
2. Sentiment persistence (autocorr)
3. Extreme clustering
4. CRITICAL: relationship to returns
   - contemporaneous correlation
   - forward correlation (predictive test)

These additional metrics are required to validate that the ABM reproduces:
→ "sentiment reacts to price but does not predict returns"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_REAL_COLS = ["net_sentiment", "pair"]
_REQUIRED_SIM_COLS = ["net_sentiment"]


# ============================================================
# Helpers
# ============================================================

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# REAL DATA CALIBRATION
# ============================================================

def calibrate_from_dataset(
    df: pd.DataFrame,
    pair: str | None = None,
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float = 70.0,
    autocorr_lag: int = 1,
) -> dict[str, Any]:
    missing = [c for c in _REQUIRED_REAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    subset = df.copy()

    if pair is not None:
        subset = subset[subset["pair"] == pair]
        if subset.empty:
            raise ValueError(f"No rows found for pair={pair!r}")

    sort_col = next(
        (c for c in ("entry_time", "snapshot_time") if c in subset.columns), None
    )
    if sort_col is not None:
        subset = subset.sort_values(sort_col)

    series = subset[sentiment_col].dropna()
    n = len(series)

    if n == 0:
        raise ValueError("No valid sentiment values after filtering")

    autocorr = float(series.autocorr(lag=autocorr_lag)) if n > autocorr_lag else float("nan")

    # --------------------------------------------------------
    # RETURN RELATIONSHIPS (CRITICAL)
    # --------------------------------------------------------

    corr_contemporaneous = float("nan")
    corr_forward = float("nan")

    if "ret_1b" in subset.columns:
        tmp = subset[[sentiment_col, "ret_1b"]].dropna()
        if len(tmp) > 10:
            corr_contemporaneous = _safe_corr(
                tmp[sentiment_col].values,
                tmp["ret_1b"].values,
            )

    if "ret_48b" in subset.columns:
        tmp = subset[[sentiment_col, "ret_48b"]].dropna()
        if len(tmp) > 10:
            corr_forward = _safe_corr(
                tmp[sentiment_col].values,
                tmp["ret_48b"].values,
            )

    stats: dict[str, Any] = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "abs_mean": float(series.abs().mean()),
        "autocorr": autocorr,
        "extreme_freq": float((series.abs() >= extreme_threshold).mean()),
        "long_frac": float((series > 0).mean()),
        "corr_contemporaneous": corr_contemporaneous,
        "corr_forward": corr_forward,
        "n_rows": n,
        "pair": pair if pair is not None else "all",
    }

    logger.info(
        "Calibration targets (pair=%s): mean=%.2f std=%.2f abs_mean=%.2f "
        "autocorr=%.3f extreme=%.3f long=%.3f "
        "corr_t=%.3f corr_fwd=%.3f",
        stats["pair"],
        stats["mean"],
        stats["std"],
        stats["abs_mean"],
        stats["autocorr"],
        stats["extreme_freq"],
        stats["long_frac"],
        stats["corr_contemporaneous"],
        stats["corr_forward"],
    )

    return stats


# ============================================================
# SIMULATION COMPARISON
# ============================================================

def compare_to_data(
    sim_df: pd.DataFrame,
    targets: dict[str, Any],
    sentiment_col: str = "net_sentiment",
    extreme_threshold: float = 70.0,
    autocorr_lag: int = 1,
) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_SIM_COLS if c not in sim_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sim_df: {missing}")

    series = sim_df[sentiment_col].dropna()
    n = len(series)

    if n == 0:
        raise ValueError("sim_df has no valid sentiment values")

    autocorr = float(series.autocorr(lag=autocorr_lag)) if n > autocorr_lag else float("nan")

    # --------------------------------------------------------
    # RETURNS FROM SIM PRICE
    # --------------------------------------------------------

    corr_contemporaneous = float("nan")
    corr_forward = float("nan")

    if "price" in sim_df.columns:
        price = sim_df["price"].values

        returns = np.diff(price)
        sent_aligned = series.values[:len(returns)]

        if len(returns) > 10:
            corr_contemporaneous = _safe_corr(sent_aligned, returns)

        fwd_returns = returns[1:]
        sent_fwd = series.values[:len(fwd_returns)]

        if len(fwd_returns) > 10:
            corr_forward = _safe_corr(sent_fwd, fwd_returns)

    sim_stats = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "abs_mean": float(series.abs().mean()),
        "autocorr": autocorr,
        "extreme_freq": float((series.abs() >= extreme_threshold).mean()),
        "long_frac": float((series > 0).mean()),
        "corr_contemporaneous": corr_contemporaneous,
        "corr_forward": corr_forward,
    }

    rows = []

    for stat, sim_val in sim_stats.items():
        real_val = targets.get(stat, float("nan"))

        try:
            sim_f = float(sim_val)
            real_f = float(real_val)
            abs_diff = abs(sim_f - real_f)
            rel_diff = abs_diff / (abs(real_f) + 1e-12)
        except Exception:
            abs_diff = float("nan")
            rel_diff = float("nan")

        rows.append(
            {
                "statistic": stat,
                "simulated": sim_val,
                "real": real_val,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    result = pd.DataFrame(rows)

    logger.info("Comparison table:\n%s", result.to_string(index=False))

    return result