from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.walkforward.evaluate import compute_predictive_metrics
from analysis.walkforward.utils import deterministic_seed


def _constant_baseline(prob: float, n: int) -> np.ndarray:
    return np.full(shape=n, fill_value=float(prob), dtype=float)


def build_control_rows(
    *,
    y_true: np.ndarray,
    y_prob_behavioral: np.ndarray,
    fold_id: int,
    model: str,
    surface_id: str,
    state_id: str,
    train_positive_rate: float | None,
    regime_train_df: pd.DataFrame | None,
    regime_test_df: pd.DataFrame | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    if y_true.size == 0:
        return rows

    base_meta = {
        "fold": int(fold_id),
        "model": model,
        "surface_id": surface_id,
        "state_id": state_id,
    }

    # Permutation baseline: destroy label association within this fold.
    rng = np.random.default_rng(deterministic_seed("perm", fold_id, model, surface_id, state_id))
    y_perm = y_true.copy()
    rng.shuffle(y_perm)
    rows.append(
        {
            **base_meta,
            "baseline": "permutation",
            **compute_predictive_metrics(y_perm, y_prob_behavioral),
        }
    )

    # Base-rate baseline: constant predictor from train fold positive frequency.
    p_base = float(train_positive_rate) if train_positive_rate is not None else float(y_true.mean())
    y_prob_base = _constant_baseline(p_base, y_true.size)
    rows.append(
        {
            **base_meta,
            "baseline": "base_rate",
            **compute_predictive_metrics(y_true, y_prob_base),
        }
    )

    # Random matched-partition baseline: random probabilities matched to fold sample count.
    rng_rand = np.random.default_rng(deterministic_seed("matched", fold_id, model, surface_id, state_id))
    y_prob_rand = rng_rand.uniform(0.0, 1.0, size=y_true.size)
    rows.append(
        {
            **base_meta,
            "baseline": "random_matched_partition",
            **compute_predictive_metrics(y_true, y_prob_rand),
        }
    )

    # Trend/volatility baseline from regime-conditioned train frequencies.
    if regime_train_df is not None and regime_test_df is not None and "regime" in regime_train_df.columns and "regime" in regime_test_df.columns:
        work_train = regime_train_df.copy()
        work_test = regime_test_df.copy()
        work_train = work_train.dropna(subset=["regime", "y_true"])
        work_test = work_test.dropna(subset=["regime", "y_true"])

        if not work_train.empty and not work_test.empty:
            regime_map = work_train.groupby("regime")["y_true"].mean().to_dict()
            fallback = float(work_train["y_true"].mean())
            reg_probs = work_test["regime"].map(regime_map).fillna(fallback).astype(float).to_numpy()
            reg_true = work_test["y_true"].astype(int).to_numpy()
            rows.append(
                {
                    **base_meta,
                    "baseline": "trend_volatility",
                    **compute_predictive_metrics(reg_true, reg_probs),
                }
            )

    return rows
