# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v4.py
========================
Continuous regime-conditioned signal pipeline.

Unlike ``regime_filter_pipeline``, which selects a discrete set of "good"
regimes and produces trades only for those, this pipeline assigns a **smooth
weight** to every regime based on its historical Sharpe ratio and applies that
weight multiplicatively to a base sentiment signal.  No trades are suppressed
entirely (unless the weight happens to be zero), and no top-k selection is
performed.

Pipeline design
---------------
1. **Regime definition** (4-component key, training-derived cuts)

   * ``vol_regime``         – tertile of ``vol_24b``               (low/mid/high)
   * ``trend_dir``          – sign of ``trend_strength_48b``       (down/flat/up)
   * ``trend_strength_bin`` – tertile of ``abs(trend_strength_48b)`` (low/mid/high)
   * ``sent_regime``        – fixed-threshold bin of ``abs_sentiment``
                              (sent_low/sent_mid/sent_high)

   Regime key::

       regime_key = f"{vol_regime}__{trend_dir}__{trend_strength_bin}__{sent_regime}"

2. **Walk-forward loop** – strict expanding window:

   * ``train_df = df[df.year < test_year]``
   * ``test_df  = df[df.year == test_year]``

3. **Regime stats on TRAIN only** – per regime:

   * ``n``, ``mean``, ``std``, ``sharpe = mean / std``

4. **Regime → weight conversion**

   Default (tanh scaling)::

       weight = tanh(sharpe / std_sharpe)

   Alternative (normalize, via ``--normalize-weights``)::

       weight = sharpe / max_abs_sharpe

   Rules:
   * If ``n < min_n`` → ``weight = 0``
   * Regimes absent from the train map → ``weight = 0``
   * ``std_sharpe`` is computed across all eligible regimes (n >= min_n) for
     the current fold.

5. **Apply to test set**

   * ``base_signal = sign(net_sentiment)``
   * ``position   = base_signal * weight``

6. **Metrics per fold** (on ``position * ret_48b``)

   * ``n``            – number of non-zero positions
   * ``mean``         – mean weighted return
   * ``sharpe``       – mean / std of weighted returns
   * ``hit_rate``     – fraction of non-zero positions with positive return
   * ``coverage``     – fraction of test rows with non-zero weight
   * ``avg_weight``   – mean absolute weight across all test rows

7. **Fold output schema**

   ``["year", "n", "mean", "sharpe", "hit_rate", "coverage", "avg_weight"]``

8. **Final summary**

   Mean Sharpe, mean hit rate, mean coverage, mean abs weight across folds.

9. **Logging** (per fold)

   * Number of eligible regimes in train (n >= min_n)
   * ``std_sharpe`` (or ``max_abs_sharpe`` in normalize mode)
   * Top-5 regimes by absolute weight (sharpe + weight)
   * Weight distribution: min / max / mean across eligible regimes

No leakage: all regime weights are derived exclusively from training data.
No filtering: every test row receives a position (weight may be zero).

Usage::

    python experiments/regime_v4.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v4.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 100 --normalize-weights --log-level DEBUG
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any

# Safe repo-root sys.path shim for direct execution
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from experiments.regime_v3 import build_features, load_data
from utils.validation import require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (override via CLI)
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Minimum training observations per regime for it to receive a non-zero weight.
MIN_REGIME_N: int = 100

#: Sentiment intensity bins for ``sent_regime``.
SENT_BINS: list[float] = [0.0, 50.0, 70.0, float("inf")]
SENT_LABELS: list[str] = ["sent_low", "sent_mid", "sent_high"]

#: Number of top regimes (by |weight|) to log per fold.
TOP_N_LOG: int = 5


# ---------------------------------------------------------------------------
# Regime feature building (4-component key)
# ---------------------------------------------------------------------------

def _compute_train_cuts(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute quantile cut points for vol_regime and trend_strength_bin.

    Both are derived **from training data only** to avoid lookahead bias.

    Args:
        train_df: Training-period DataFrame.  Must contain ``vol_24b`` and
            ``trend_strength_48b``.

    Returns:
        Dict with keys ``"vol"`` and/or ``"trend_strength"`` mapping to 1-D
        NumPy arrays of bin edges.  Keys are absent when the corresponding
        column is missing or too small to produce 3 distinct bins.
    """
    cuts: dict[str, np.ndarray] = {}

    if "vol_24b" in train_df.columns:
        valid_vol = train_df["vol_24b"].dropna()
        if len(valid_vol) >= 3:
            try:
                _, vol_bins = pd.qcut(
                    valid_vol, q=3, retbins=True, duplicates="drop"
                )
                cuts["vol"] = vol_bins
            except ValueError:
                logger.debug(
                    "_compute_train_cuts: vol qcut failed; skipping vol_regime"
                )

    if "trend_strength_48b" in train_df.columns:
        valid_ts = train_df["trend_strength_48b"].dropna().abs()
        if len(valid_ts) >= 3:
            try:
                _, ts_bins = pd.qcut(
                    valid_ts, q=3, retbins=True, duplicates="drop"
                )
                cuts["trend_strength"] = ts_bins
            except ValueError:
                logger.debug(
                    "_compute_train_cuts: trend_strength qcut failed; "
                    "skipping trend_strength_bin"
                )

    return cuts


def _build_regime_key(
    df: pd.DataFrame, cuts: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Add regime feature columns and a 4-component ``regime_key`` to *df*.

    Adds four causal discrete features:

    * ``vol_regime``         – tertile of ``vol_24b``.
    * ``trend_dir``          – sign of ``trend_strength_48b``.
    * ``trend_strength_bin`` – tertile of ``abs(trend_strength_48b)``.
    * ``sent_regime``        – fixed-threshold bin of ``abs_sentiment``.

    All four are combined into ``regime_key``.  Rows where any constituent
    feature is NaN receive NaN in ``regime_key`` and contribute a zero weight
    in the signal step.

    Args:
        df: Slice to label (train or test).
        cuts: Output of :func:`_compute_train_cuts` for this fold.

    Returns:
        Copy of *df* with the four feature columns and ``regime_key`` appended.
    """
    out = df.copy()

    # --- vol_regime: tertile of vol_24b ---
    if "vol_24b" in out.columns and "vol" in cuts:
        bins = np.copy(cuts["vol"])
        bins[0] = -np.inf
        bins[-1] = np.inf
        n_bins = len(bins) - 1
        labels_vol = ["vol_low", "vol_mid", "vol_high"][:n_bins]
        result = pd.cut(
            out["vol_24b"],
            bins=bins,
            labels=labels_vol,
            include_lowest=True,
        )
        out["vol_regime"] = result.astype(str)
        out.loc[out["vol_24b"].isna(), "vol_regime"] = np.nan
    else:
        out["vol_regime"] = np.nan
        logger.debug(
            "_build_regime_key: vol_24b or vol cuts missing; vol_regime=NaN"
        )

    # --- trend_dir: sign(trend_strength_48b) ---
    if "trend_strength_48b" in out.columns:
        sign = np.sign(out["trend_strength_48b"].fillna(0.0))
        out["trend_dir"] = np.select(
            [sign < 0, sign > 0],
            ["trend_down", "trend_up"],
            default="trend_flat",
        )
        out.loc[out["trend_strength_48b"].isna(), "trend_dir"] = np.nan
    else:
        out["trend_dir"] = np.nan
        logger.debug(
            "_build_regime_key: trend_strength_48b missing; trend_dir=NaN"
        )

    # --- trend_strength_bin: tertile of abs(trend_strength_48b) ---
    if "trend_strength_48b" in out.columns and "trend_strength" in cuts:
        bins_ts = np.copy(cuts["trend_strength"])
        bins_ts[0] = -np.inf
        bins_ts[-1] = np.inf
        n_bins_ts = len(bins_ts) - 1
        labels_ts = ["ts_low", "ts_mid", "ts_high"][:n_bins_ts]
        abs_ts = out["trend_strength_48b"].abs()
        result_ts = pd.cut(
            abs_ts,
            bins=bins_ts,
            labels=labels_ts,
            include_lowest=True,
        )
        out["trend_strength_bin"] = result_ts.astype(str)
        out.loc[out["trend_strength_48b"].isna(), "trend_strength_bin"] = np.nan
    else:
        out["trend_strength_bin"] = np.nan
        logger.debug(
            "_build_regime_key: trend_strength_48b or trend_strength cuts "
            "missing; trend_strength_bin=NaN"
        )

    # --- sent_regime: fixed-threshold bins of abs_sentiment ---
    if "abs_sentiment" in out.columns:
        result_sent = pd.cut(
            out["abs_sentiment"],
            bins=SENT_BINS,
            labels=SENT_LABELS,
            right=False,
            include_lowest=True,
        )
        out["sent_regime"] = result_sent.astype(str)
        out.loc[out["abs_sentiment"].isna(), "sent_regime"] = np.nan
    else:
        out["sent_regime"] = np.nan
        logger.debug(
            "_build_regime_key: abs_sentiment missing; sent_regime=NaN"
        )

    # --- regime_key: 4-component concatenation ---
    valid_mask = (
        out["vol_regime"].notna()
        & out["trend_dir"].notna()
        & out["trend_strength_bin"].notna()
        & out["sent_regime"].notna()
    )
    combined = (
        out["vol_regime"].astype(str)
        + "__"
        + out["trend_dir"].astype(str)
        + "__"
        + out["trend_strength_bin"].astype(str)
        + "__"
        + out["sent_regime"].astype(str)
    )
    out["regime_key"] = combined.where(valid_mask)

    n_valid = int(valid_mask.sum())
    n_regimes = out["regime_key"].nunique()
    logger.debug(
        "_build_regime_key: %d rows assigned to %d unique regime keys",
        n_valid,
        n_regimes,
    )
    return out


# ---------------------------------------------------------------------------
# Regime statistics from training data
# ---------------------------------------------------------------------------

def _compute_regime_stats(
    train_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    min_n: int = MIN_REGIME_N,
) -> dict[str, dict[str, float]]:
    """Compute per-regime return statistics on training data only.

    Groups *train_df* by ``regime_key`` and computes ``n``, ``mean``,
    ``std``, and ``sharpe = mean / std`` for each group.  Only regimes with
    ``n >= min_n`` and a non-NaN Sharpe are included.

    Args:
        train_df: Training slice with ``regime_key`` and *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        min_n: Minimum observations for a regime to be eligible for weighting.

    Returns:
        Dict mapping ``regime_key`` → stat dict
        ``{"n", "mean", "std", "sharpe"}``.
    """
    if "regime_key" not in train_df.columns:
        logger.warning(
            "_compute_regime_stats: regime_key not found in train data"
        )
        return {}

    valid = train_df.dropna(subset=["regime_key", target_col])
    if valid.empty:
        logger.warning(
            "_compute_regime_stats: no valid rows in training data"
        )
        return {}

    stats: dict[str, dict[str, float]] = {}

    for regime_label, grp in valid.groupby("regime_key"):
        returns = grp[target_col].values
        n = len(returns)
        if n < min_n:
            logger.debug(
                "_compute_regime_stats: regime=%s skipped (n=%d < min_n=%d)",
                regime_label,
                n,
                min_n,
            )
            continue
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        sharpe = mean / std if std > 1e-10 else np.nan
        if np.isnan(sharpe):
            logger.debug(
                "_compute_regime_stats: regime=%s skipped (sharpe=NaN, std~0)",
                regime_label,
            )
            continue
        stats[str(regime_label)] = {
            "n": float(n),
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
        }

    logger.debug(
        "_compute_regime_stats: %d eligible regimes (n>=%d)",
        len(stats),
        min_n,
    )
    return stats


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_regime_weight(sharpe: float, std_sharpe: float) -> float:
    """Convert a regime Sharpe to a smooth weight via tanh scaling.

    Args:
        sharpe: Per-regime Sharpe ratio (from training data).
        std_sharpe: Standard deviation of Sharpe ratios across all eligible
            regimes in the current fold (from training data only).

    Returns:
        Weight in ``(-1, +1)``.  Returns 0.0 when *std_sharpe* is near zero.
    """
    if std_sharpe < 1e-10:
        return 0.0
    return float(np.tanh(sharpe / std_sharpe))


def _build_weight_map(
    regime_stats: dict[str, dict[str, float]],
    *,
    normalize_weights: bool = False,
) -> tuple[dict[str, float], float]:
    """Build a regime → weight mapping from training-derived Sharpe ratios.

    Args:
        regime_stats: Output of :func:`_compute_regime_stats`.
        normalize_weights: If ``True`` use ``weight = sharpe / max_abs_sharpe``
            instead of ``tanh(sharpe / std_sharpe)``.

    Returns:
        ``(weight_map, scale_value)`` where *weight_map* maps each eligible
        regime key to its computed weight and *scale_value* is the normalization
        denominator (``std_sharpe`` or ``max_abs_sharpe``).
    """
    if not regime_stats:
        return {}, 0.0

    sharpes = np.array([v["sharpe"] for v in regime_stats.values()])

    if normalize_weights:
        max_abs_sharpe = float(np.max(np.abs(sharpes)))
        scale_value = max_abs_sharpe
        if max_abs_sharpe < 1e-10:
            weight_map = {k: 0.0 for k in regime_stats}
            return weight_map, scale_value
        weight_map = {
            k: float(v["sharpe"] / max_abs_sharpe)
            for k, v in regime_stats.items()
        }
    else:
        std_sharpe = float(np.std(sharpes))
        scale_value = std_sharpe
        weight_map = {
            k: compute_regime_weight(v["sharpe"], std_sharpe)
            for k, v in regime_stats.items()
        }

    return weight_map, scale_value


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    weighted_returns: np.ndarray,
    weights: np.ndarray,
    n_total_test: int,
) -> dict[str, float]:
    """Compute fold-level performance metrics.

    Args:
        weighted_returns: ``position * ret_48b`` for all non-zero positions.
        weights: Absolute weight for every test row (including zero-weight rows),
            used to compute ``avg_weight`` and ``coverage``.
        n_total_test: Total number of test rows (denominator for coverage).

    Returns:
        Dict with keys ``n``, ``mean``, ``sharpe``, ``hit_rate``,
        ``coverage``, ``avg_weight``.
    """
    n = len(weighted_returns)
    coverage = float(n / n_total_test) if n_total_test > 0 else 0.0
    avg_weight = float(np.mean(weights)) if len(weights) > 0 else 0.0

    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
            "coverage": coverage,
            "avg_weight": avg_weight,
        }

    mean = float(np.mean(weighted_returns))
    std = float(np.std(weighted_returns))
    sharpe = mean / std if std > 1e-10 else np.nan
    hit_rate = float(np.mean(weighted_returns > 0))
    return {
        "n": n,
        "mean": mean,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "coverage": coverage,
        "avg_weight": avg_weight,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def regime_v4_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_n: int = MIN_REGIME_N,
    normalize_weights: bool = False,
    top_n_log: int = TOP_N_LOG,
) -> pd.DataFrame:
    """Regime-V4 walk-forward: continuous regime-conditioned signal.

    Performs a strictly causal expanding-window walk-forward:

    1. For each test year (from the third unique year onward):
    2. Compute quantile cut points for ``vol_regime`` and
       ``trend_strength_bin`` from ``train_df`` only.
    3. Build 4-component regime keys on both ``train_df`` and ``test_df``
       using those training-derived cuts.
    4. Compute per-regime stats (n, mean, std, sharpe) on ``train_df``; only
       regimes with ``n >= min_n`` and a valid Sharpe are eligible.
    5. Compute ``std_sharpe`` (or ``max_abs_sharpe``) across eligible regimes.
    6. Build ``weight_map``: for each eligible regime:
       * Default: ``weight = tanh(sharpe / std_sharpe)``
       * Normalize: ``weight = sharpe / max_abs_sharpe``
       * Regimes absent from map → ``weight = 0``
    7. Apply to test set:
       * ``base_signal = sign(net_sentiment)``
       * ``position   = base_signal * weight``
    8. Compute fold metrics on rows where ``position != 0``.

    No test-period return information enters the weight computation.
    Every test row receives a position (weight may be zero).

    Args:
        df: Full dataset (after ``build_features``).  Must contain
            ``vol_24b``, ``trend_strength_48b``, ``abs_sentiment``,
            ``net_sentiment``, and *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum training observations per regime to be eligible for
            weighting (default 100).
        normalize_weights: If ``True``, use ``sharpe / max_abs_sharpe``
            instead of ``tanh(sharpe / std_sharpe)``.
        top_n_log: Number of top regimes (by |weight|) to log per fold.

    Returns:
        DataFrame ``fold_df`` with schema
        ``["year", "n", "mean", "sharpe", "hit_rate", "coverage", "avg_weight"]``.
        One row per test fold where at least 2 non-zero positions exist.
    """
    _COLS = ["year", "n", "mean", "sharpe", "hit_rate", "coverage", "avg_weight"]

    if year_col not in df.columns:
        logger.warning(
            "regime_v4_walk_forward: year column '%s' not found", year_col
        )
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "regime_v4_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year]
        test_df = df[df[year_col] == test_year]

        # --- Step 1: quantile cuts from training data only ---
        cuts = _compute_train_cuts(train_df)

        # --- Step 2: build 4-component regime keys ---
        train_labeled = _build_regime_key(train_df, cuts)
        test_labeled = _build_regime_key(test_df, cuts)

        n_unique_train_regimes = int(
            train_labeled["regime_key"].dropna().nunique()
        )

        # --- Step 3: regime stats (train only) ---
        regime_stats = _compute_regime_stats(
            train_labeled,
            target_col=target_col,
            min_n=min_n,
        )
        n_eligible = len(regime_stats)

        # --- Step 4: build weight map ---
        weight_map, scale_value = _build_weight_map(
            regime_stats,
            normalize_weights=normalize_weights,
        )

        scale_label = "max_abs_sharpe" if normalize_weights else "std_sharpe"
        logger.info(
            "REGIME V4 [year=%d] | n_unique_train_regimes=%d | n_eligible=%d"
            " | %s=%.4f",
            test_year,
            n_unique_train_regimes,
            n_eligible,
            scale_label,
            scale_value,
        )

        # --- Per-fold weight distribution logging ---
        if weight_map:
            w_vals = list(weight_map.values())
            logger.info(
                "  WEIGHT DISTRIBUTION | min=%+.4f | max=%+.4f | mean=%+.4f",
                min(w_vals),
                max(w_vals),
                float(np.mean(w_vals)),
            )

            # Top-N regimes by absolute weight
            top_regimes = sorted(
                weight_map.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:top_n_log]
            for regime_label, w in top_regimes:
                sharpe_val = regime_stats[regime_label]["sharpe"]
                logger.info(
                    "  TOP REGIME | regime=%-60s | sharpe=%+.4f | weight=%+.4f",
                    regime_label,
                    sharpe_val,
                    w,
                )
        else:
            logger.warning(
                "REGIME V4 [year=%d]: no eligible regimes — all weights zero",
                test_year,
            )

        # --- Step 5: apply to test set ---
        test_valid = test_labeled.dropna(subset=[target_col])
        if test_valid.empty:
            logger.warning(
                "REGIME V4 [year=%d]: no valid test rows; skipping fold",
                test_year,
            )
            continue

        n_total_test = len(test_valid)

        # base_signal = sign(net_sentiment); missing → 0
        if "net_sentiment" not in test_valid.columns:
            logger.warning(
                "REGIME V4 [year=%d]: net_sentiment missing; "
                "base_signal=0 for all rows",
                test_year,
            )
            base_signal = np.zeros(n_total_test)
        else:
            base_signal = np.sign(test_valid["net_sentiment"].fillna(0.0).values)

        # weight per row (0 for unknown / ineligible regimes)
        row_weight = (
            test_valid["regime_key"]
            .map(weight_map)
            .fillna(0.0)
            .values
        )

        position = base_signal * row_weight

        # Non-zero-position rows only for metrics
        active_mask = position != 0.0
        active_positions = position[active_mask]
        active_returns = test_valid.loc[
            test_valid.index[active_mask], target_col
        ].values
        weighted_returns = active_positions * active_returns

        # avg_weight uses absolute weight across ALL test rows
        abs_weights_all = np.abs(row_weight)

        m = _fold_metrics(weighted_returns, abs_weights_all, n_total_test)

        fold_rows.append(
            {
                "year": int(test_year),
                "n": m["n"],
                "mean": m["mean"],
                "sharpe": m["sharpe"],
                "hit_rate": m["hit_rate"],
                "coverage": m["coverage"],
                "avg_weight": m["avg_weight"],
            }
        )

        logger.info(
            "REGIME V4 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | coverage=%.1f%% | avg_weight=%.4f",
            test_year,
            m["n"],
            m["mean"] if not np.isnan(m["mean"]) else float("nan"),
            m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
            m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
            m["coverage"] * 100,
            m["avg_weight"],
        )

    if not fold_rows:
        logger.warning(
            "REGIME V4: no valid folds produced (min_n=%d)", min_n
        )
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(fold_rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pooled summary
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`regime_v4_walk_forward`.

    Returns:
        Dict with keys ``n_folds``, ``mean_sharpe``, ``mean_hit_rate``,
        ``mean_coverage``, ``mean_avg_weight``.  Float values are NaN when
        *fold_df* is empty.
    """
    if fold_df.empty:
        return {
            "n_folds": 0,
            "mean_sharpe": np.nan,
            "mean_hit_rate": np.nan,
            "mean_coverage": np.nan,
            "mean_avg_weight": np.nan,
        }
    return {
        "n_folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
        "mean_avg_weight": float(fold_df["avg_weight"].mean()),
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`regime_v4_walk_forward`.
    """
    logger.info("=== REGIME V4 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V4: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%% | avg_weight=%.4f",
            row.year,
            row.n,
            row.mean if not np.isnan(row.mean) else float("nan"),
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
            row.avg_weight,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    pooled: dict[str, float | int],
) -> None:
    """Log the consolidated final summary of the Regime V4 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`regime_v4_walk_forward`.
        pooled: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V4 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V4 SUMMARY: no results")
        return

    logger.info("Folds evaluated   : %d", pooled["n_folds"])
    logger.info(
        "Mean Sharpe       : %+.4f",
        pooled["mean_sharpe"]
        if not np.isnan(pooled["mean_sharpe"])
        else float("nan"),
    )
    logger.info(
        "Mean hit rate     : %.4f",
        pooled["mean_hit_rate"]
        if not np.isnan(pooled["mean_hit_rate"])
        else float("nan"),
    )
    logger.info(
        "Mean coverage     : %.1f%%",
        pooled["mean_coverage"] * 100,
    )
    logger.info(
        "Mean abs weight   : %.4f",
        pooled["mean_avg_weight"],
    )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Main (module-level CLI)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V4 walk-forward pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V4: continuous regime-conditioned signal pipeline. "
            "Converts regime Sharpe → smooth weight; applies multiplicatively "
            "to sign(net_sentiment).  No filtering — always produces a position."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(cfg.DATA_PATH),
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_REGIME_N,
        metavar="N",
        help="Minimum training observations per regime for non-zero weight.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Optional explicit log file path.  When omitted, a timestamped "
            "file is created automatically in logs/."
        ),
    )
    p.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help=(
            "Use sharpe / max_abs_sharpe instead of tanh(sharpe / std_sharpe) "
            "for regime weight computation."
        ),
    )
    p.add_argument(
        "--top-n-log",
        type=int,
        default=TOP_N_LOG,
        metavar="N",
        help="Number of top regimes (by |weight|) to log per fold.",
    )
    args = p.parse_args(argv)

    # Configure basic logging so this module is usable standalone
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%dT%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if args.log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v4_{timestamp}.log"
    else:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(file_handler)
    logger.info("File logging enabled: %s", log_path)

    df = load_data(args.data)
    df = build_features(df)

    require_columns(
        df,
        ["trend_strength_48b", "abs_sentiment", "net_sentiment", TARGET_COL],
        context="regime_v4.main",
    )

    logger.info(
        "=== REGIME V4 ==="
        " | min_n=%d | normalize_weights=%s",
        args.min_n,
        args.normalize_weights,
    )

    fold_df = regime_v4_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        normalize_weights=args.normalize_weights,
        top_n_log=args.top_n_log,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
