"""
experiments/regime_filter_pipeline.py
======================================
Regime-filter pipeline: use discrete regime features to FILTER trades
(i.e. decide *when* to trade), not to predict return magnitudes.

Unlike ``regime_v3``, which conditions a LightGBM model on regimes, this
pipeline is purely rule-based:

1. **Regime feature building** – three causal, discrete features are created:

   * ``vol_regime``  – tertile of ``vol_24b``         (low / mid / high).
   * ``trend_dir``   – sign of ``trend_strength_48b`` (down / flat / up).
   * ``sent_regime`` – fixed-threshold bin of ``abs_sentiment``
                        (low [0–50) / mid [50–70) / high [70+]).

   These three features are combined into a single string ``regime_key``
   (e.g. ``"vol_low__trend_up__sent_high"``).

2. **Walk-forward evaluation (strict, no leakage)** – expanding window:

   For each ``test_year`` (starting from the third unique year):

   * ``train_df = df[df.year < test_year]``
   * ``test_df  = df[df.year == test_year]``

   Quantile cut points for ``vol_regime`` are derived **from training data
   only** per fold.

3. **Regime statistics (train only)** – group ``train_df`` by ``regime_key``
   and compute: ``n``, ``mean`` return, ``std``, ``sharpe = mean / std``.

4. **Regime selection** – among regimes with ``n >= min_n``, rank by
   training Sharpe descending and keep the top ``top_k`` (default 5).

5. **Test-set stability filters** – after selecting top-k regimes from
   training data, drop any regime where (in the test set):

   * ``n_test_regime < min_test_n``                    (default 50)
   * ``n_test_regime / total_test_signals < min_test_coverage`` (default 0.05)

   These filters use only counts (no return information) so they introduce
   no lookahead bias.

6. **Apply filter to test set** – keep only rows where
   ``regime_key ∈ surviving_regimes``.

7. **Optional direction logic** – for each surviving regime:

   * If ``abs(train_mean) >= direction_threshold`` (default 0.0002):

     * ``train mean > 0`` → follow (``effective_return = +return``)
     * ``train mean < 0`` → fade  (``effective_return = −return``)

   * Otherwise, keep signals unmodified (multiplier = +1).

8. **Metrics per fold** – mean return, Sharpe, hit rate, coverage
   (fraction of test signals kept).

9. **Logging** – uses the existing ``logging`` system; no ``print`` calls.
   Per-fold log: n_candidate_regimes, n_selected (top-k), n_surviving
   (after test filters), coverage before and after filtering.

10. **Output schema** – consistent with the rest of the pipeline:

    * ``fold_df``  – ``["year", "n", "mean", "sharpe", "hit_rate", "coverage"]``
    * ``pooled``   – ``{"n_folds", "mean_return", "mean_sharpe",
                        "mean_hit_rate", "mean_coverage"}``

No model predictions are used.  No forward-looking return information is
introduced — regime selection is re-computed per fold from training data.

Usage::

    python -m experiments.regime_filter_pipeline \
        --data data/output/master_research_dataset.csv

    python experiments/regime_filter_pipeline.py \
        --data data/output/master_research_dataset.csv \
        --min-n 100 --top-k 5 --with-direction \
        --log-level INFO
"""

from __future__ import annotations

import argparse
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

#: Minimum training observations per regime for it to be eligible for selection.
MIN_REGIME_N: int = 100

#: Minimum training Sharpe ratio for a regime to be selected.
#: Kept for backward compatibility; no longer used by the default selection
#: logic which uses candidate_k/final_k ranking instead of a hard Sharpe threshold.
MIN_REGIME_SHARPE: float = 0.05

#: Size of the initial candidate pool: top regimes by Sharpe to consider
#: before applying test-side stability filters.
CANDIDATE_K: int = 15

#: Number of regimes to keep after iterating through the candidate pool and
#: applying test-side stability filters.
FINAL_K: int = 5

#: Kept for backward compatibility; maps to FINAL_K.
TOP_K: int = FINAL_K

#: Minimum fraction of total test signals a regime must cover to survive.
MIN_TEST_COVERAGE: float = 0.05

#: Minimum raw count of test signals a regime must have to survive.
MIN_TEST_N: int = 50

#: Minimum |train mean| required to apply follow/fade direction logic.
#: Below this threshold, signals are kept unmodified (multiplier = +1).
DIRECTION_THRESHOLD: float = 0.0005

#: When True, invert returns for regimes whose training mean return is negative
#: (fade the crowd); when False, always use the raw return sign.
WITH_DIRECTION: bool = True

#: Number of top regimes to log per fold.
TOP_N_LOG: int = 5

#: Minimum number of prior walk-forward folds in which a regime must have
#: appeared as a candidate (n >= min_n) before it is eligible for selection.
#: Set to 0 to disable persistence filtering.
MIN_PERSISTENCE_FOLDS: int = 2

#: Sentiment intensity bins for ``sent_regime``.
SENT_BINS: list[float] = [0.0, 50.0, 70.0, float("inf")]
SENT_LABELS: list[str] = ["sent_low", "sent_mid", "sent_high"]


# ---------------------------------------------------------------------------
# Regime feature building
# ---------------------------------------------------------------------------

def _compute_train_cuts(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute quantile cut points for *vol_regime*.

    Computed from **training data only** to avoid lookahead bias.
    The returned cut-point arrays are passed to :func:`_build_regime_key`
    to discretize both the training slice (for regime statistics) and the
    test slice (for filtering).

    Args:
        train_df: Training-period DataFrame.  Must contain ``vol_24b``.

    Returns:
        Dict with key ``"vol"`` mapping to a 1-D NumPy array of bin edges
        (including ``−∞`` and ``+∞`` sentinels set by the caller).
        An empty dict is returned when the column is missing or too small.
    """
    cuts: dict[str, np.ndarray] = {}

    if "vol_24b" in train_df.columns:
        valid_vol = train_df["vol_24b"].dropna()
        if len(valid_vol) >= 3:
            try:
                _, vol_bins = pd.qcut(valid_vol, q=3, retbins=True, duplicates="drop")
                cuts["vol"] = vol_bins
            except ValueError:
                logger.debug("_compute_train_cuts: vol qcut failed; skipping vol_regime")

    return cuts


def _build_regime_key(df: pd.DataFrame, cuts: dict[str, np.ndarray]) -> pd.DataFrame:
    """Add regime feature columns and a combined ``regime_key`` to *df*.

    Adds three causal discrete features (all computed from past data):

    * ``vol_regime``  – tertile of ``vol_24b`` (low / mid / high).
    * ``trend_dir``   – sign of ``trend_strength_48b``
                        (trend_down / trend_flat / trend_up).
    * ``sent_regime`` – fixed-threshold bin of ``abs_sentiment``
                        (sent_low / sent_mid / sent_high).

    All three are combined into ``regime_key``.  Rows where *any*
    constituent feature is ``NaN`` receive ``NaN`` in ``regime_key``
    and are excluded from downstream analysis.

    Args:
        df: Slice to label (train or test).
        cuts: Output of :func:`_compute_train_cuts` for this fold.

    Returns:
        Copy of *df* with columns ``vol_regime``, ``trend_dir``,
        ``sent_regime``, and ``regime_key`` appended.
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
        if "vol_24b" not in out.columns:
            logger.debug("_build_regime_key: vol_24b missing; vol_regime=NaN")
        else:
            logger.debug("_build_regime_key: vol cut points missing; vol_regime=NaN")

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
        logger.debug("_build_regime_key: trend_strength_48b missing; trend_dir=NaN")

    # --- sent_regime: fixed-threshold bins of abs_sentiment ---
    if "abs_sentiment" in out.columns:
        result = pd.cut(
            out["abs_sentiment"],
            bins=SENT_BINS,
            labels=SENT_LABELS,
            right=False,
            include_lowest=True,
        )
        out["sent_regime"] = result.astype(str)
        out.loc[out["abs_sentiment"].isna(), "sent_regime"] = np.nan
    else:
        out["sent_regime"] = np.nan
        logger.debug("_build_regime_key: abs_sentiment missing; sent_regime=NaN")

    # --- regime_key: concatenation of all three features ---
    valid_mask = (
        out["vol_regime"].notna()
        & out["trend_dir"].notna()
        & out["sent_regime"].notna()
    )
    combined = (
        out["vol_regime"].astype(str)
        + "__"
        + out["trend_dir"].astype(str)
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
    ``std``, and ``sharpe = mean / std`` for each group.  Only regimes
    with ``n >= min_n`` are returned; no Sharpe threshold is applied here
    (top-k selection happens in :func:`regime_filter_walk_forward`).

    Args:
        train_df: Training slice with ``regime_key`` and *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        min_n: Minimum observations required for a regime to be included.

    Returns:
        Dict mapping ``regime_key`` → stat dict
        (``{"n", "mean", "std", "sharpe"}``).  All regimes meeting the
        ``min_n`` threshold are included, regardless of Sharpe.
    """
    if "regime_key" not in train_df.columns:
        logger.warning("_compute_regime_stats: regime_key not found in train data")
        return {}

    valid = train_df.dropna(subset=["regime_key", target_col])
    if valid.empty:
        logger.warning("_compute_regime_stats: no valid rows in training data")
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
                "_compute_regime_stats: regime=%s skipped (sharpe=NaN, std=0)",
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
        "_compute_regime_stats: %d candidate regimes (n>=%d)",
        len(stats),
        min_n,
    )
    return stats


# ---------------------------------------------------------------------------
# Fold metric computation
# ---------------------------------------------------------------------------

def _fold_metrics(returns_arr: np.ndarray) -> dict[str, float]:
    """Compute fold-level performance metrics from a return array.

    Args:
        returns_arr: Array of (possibly direction-adjusted) realised returns.

    Returns:
        Dict with keys ``n``, ``mean``, ``std``, ``sharpe``, ``hit_rate``.
    """
    n = len(returns_arr)
    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
        }
    mean = float(np.mean(returns_arr))
    std = float(np.std(returns_arr))
    sharpe = mean / std if std > 1e-10 else np.nan
    hit_rate = float(np.mean(returns_arr > 0))
    return {"n": n, "mean": mean, "std": std, "sharpe": sharpe, "hit_rate": hit_rate}


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def regime_filter_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_n: int = MIN_REGIME_N,
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
    min_test_coverage: float = MIN_TEST_COVERAGE,
    min_test_n: int = MIN_TEST_N,
    with_direction: bool = WITH_DIRECTION,
    direction_threshold: float = DIRECTION_THRESHOLD,
    top_n_log: int = TOP_N_LOG,
    min_persistence_folds: int = MIN_PERSISTENCE_FOLDS,
) -> pd.DataFrame:
    """Regime-filter walk-forward: select regimes from train, apply to test.

    Performs a strictly causal expanding-window walk-forward:

    1. For each test year (starting from the third unique year):
    2. Compute quantile cut points for ``vol_regime`` from ``train_df``
       (all years prior to test year).
    3. Apply regime feature building to both ``train_df`` and ``test_df``
       using those training-derived cut points.
    4. Compute per-regime return statistics on ``train_df`` only.
    5. Apply persistence filter: require each regime to have appeared as a
       candidate (n >= min_n) in at least ``min_persistence_folds`` prior
       walk-forward folds.
    6. Sort surviving candidates by training Sharpe (descending).  Iterate
       through the top ``candidate_k`` entries; keep the first ``final_k``
       that pass the test-side stability filters (n_test >= min_test_n and
       coverage_test >= min_test_coverage).  Filters are applied inside the
       selection loop — no separate post-hoc filtering step.
    7. Filter ``test_df`` to selected regimes.
    8. Optionally apply direction logic per regime: follow/fade only when
       ``abs(train_mean) >= direction_threshold``; otherwise keep raw sign.
    9. Compute and log fold-level metrics + coverage.

    No test-period return information is ever used for regime selection or
    direction classification.  Test-side stability filters use only signal
    counts, not return values.

    Args:
        df: Full dataset (after :func:`build_features`).  Must contain
            ``vol_24b``, ``trend_strength_48b``, ``abs_sentiment``, and
            *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum training observations per regime (default 100).
        candidate_k: Maximum number of top regimes (by Sharpe) to consider
            when iterating through the selection loop (default 15).
        final_k: Maximum number of regimes to keep after test-side filters
            inside the selection loop (default 5).
        min_test_coverage: Minimum fraction of test signals a regime must
            cover to pass the in-loop stability filter (default 0.05).
        min_test_n: Minimum raw count of test signals per regime to pass
            the in-loop stability filter (default 50).
        with_direction: If ``True``, apply follow/fade direction logic for
            regimes where ``abs(train_mean) >= direction_threshold``.
        direction_threshold: Minimum |train mean| to activate follow/fade
            logic (default 0.0005).  Below this threshold the signal is
            kept unmodified.
        top_n_log: Number of top selected regimes to log per fold, by
            training Sharpe.
        min_persistence_folds: Minimum number of prior walk-forward folds
            in which a regime must have been a candidate (n >= min_n) before
            it is eligible for selection in the current fold (default 2).
            Set to 0 to disable persistence filtering.

    Returns:
        DataFrame ``fold_df`` with schema
        ``["year", "n", "mean", "sharpe", "hit_rate", "coverage"]``.
        One row per test fold where at least 2 filtered signals exist.
    """
    _COLS = ["year", "n", "mean", "sharpe", "hit_rate", "coverage"]

    if year_col not in df.columns:
        logger.warning(
            "regime_filter_walk_forward: year column '%s' not found", year_col
        )
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "regime_filter_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    fold_rows: list[dict[str, Any]] = []

    # Track how many prior walk-forward folds each regime was a candidate in.
    regime_seen_count: dict[str, int] = {}

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year]
        test_df = df[df[year_col] == test_year]

        # --- Step 1: quantile cuts from training data ---
        cuts = _compute_train_cuts(train_df)

        # --- Step 2: build regime keys on train and test ---
        train_labeled = _build_regime_key(train_df, cuts)
        test_labeled = _build_regime_key(test_df, cuts)

        # Number of all unique regime keys (any n) in training data.
        n_unique_train_regimes = int(
            train_labeled["regime_key"].dropna().nunique()
        )

        # --- Step 3: regime stats (train only, all n>=min_n candidates) ---
        regime_stats = _compute_regime_stats(
            train_labeled,
            target_col=target_col,
            min_n=min_n,
        )
        n_after_min_n = len(regime_stats)

        # --- Step 4: persistence filter ---
        if min_persistence_folds > 0:
            persistent_stats = {
                k: v
                for k, v in regime_stats.items()
                if regime_seen_count.get(k, 0) >= min_persistence_folds
            }
        else:
            persistent_stats = dict(regime_stats)

        # Sort all persistent candidates by training Sharpe (descending).
        sorted_candidates = sorted(
            persistent_stats.items(),
            key=lambda kv: kv[1]["sharpe"],
            reverse=True,
        )

        # --- Step 5: test-valid rows (used for stability checks inside loop) ---
        test_valid = test_labeled.dropna(subset=[target_col])
        n_total_test = len(test_valid)

        # Coverage if all n>=min_n regimes were selected (pre-filter baseline).
        min_n_regimes = set(regime_stats.keys())
        coverage_before = (
            len(test_valid[test_valid["regime_key"].isin(min_n_regimes)])
            / n_total_test
            if n_total_test > 0
            else 0.0
        )

        # --- Step 6: selection loop — test filter applied inside ---
        selected_stats: dict[str, dict[str, float]] = {}
        n_checked = 0
        for regime_label, rstats in sorted_candidates:
            if len(selected_stats) >= final_k:
                break
            if n_checked >= candidate_k:
                break
            n_checked += 1

            n_test_regime = int(
                (test_valid["regime_key"] == regime_label).sum()
            )
            cov_test = (
                n_test_regime / n_total_test if n_total_test > 0 else 0.0
            )

            if n_test_regime < min_test_n:
                logger.debug(
                    "  SELECTION LOOP [year=%d] | regime=%s skipped "
                    "(n_test=%d < min_test_n=%d)",
                    test_year,
                    regime_label,
                    n_test_regime,
                    min_test_n,
                )
                continue
            if cov_test < min_test_coverage:
                logger.debug(
                    "  SELECTION LOOP [year=%d] | regime=%s skipped "
                    "(coverage_test=%.3f < min_test_coverage=%.3f)",
                    test_year,
                    regime_label,
                    cov_test,
                    min_test_coverage,
                )
                continue

            selected_stats[regime_label] = rstats

        surviving_regimes = set(selected_stats.keys())
        n_final_selected = len(surviving_regimes)

        # --- Step 7: apply filter to test set ---
        test_filtered = test_valid[test_valid["regime_key"].isin(surviving_regimes)]
        n_filtered = len(test_filtered)
        coverage_after = n_filtered / n_total_test if n_total_test > 0 else 0.0

        # --- Per-fold logging ---
        logger.info(
            "REGIME FILTER PIPELINE [year=%d]"
            " | n_candidate_regimes=%d | n_after_min_n=%d"
            " | n_checked_in_selection=%d | n_final_selected=%d"
            " | coverage_before=%.1f%% | coverage_after=%.1f%% (%d/%d signals)",
            test_year,
            n_unique_train_regimes,
            n_after_min_n,
            n_checked,
            n_final_selected,
            coverage_before * 100,
            coverage_after * 100,
            n_filtered,
            n_total_test,
        )

        top_regimes_sorted = sorted(
            selected_stats.items(),
            key=lambda kv: kv[1]["sharpe"],
            reverse=True,
        )[:top_n_log]
        for regime_label, rstats in top_regimes_sorted:
            logger.info(
                "  TOP REGIME | regime=%-50s | n=%5d"
                " | mean=%+.6f | sharpe=%+.4f",
                regime_label,
                int(rstats["n"]),
                rstats["mean"],
                rstats["sharpe"],
            )

        # --- Update persistence tracker for future folds ---
        for regime_label in regime_stats:
            regime_seen_count[regime_label] = (
                regime_seen_count.get(regime_label, 0) + 1
            )

        if n_filtered < 2:
            logger.warning(
                "REGIME FILTER PIPELINE: year=%d skipped "
                "(n_filtered=%d < 2 after stability filters)",
                test_year,
                n_filtered,
            )
            continue

        # --- Step 8: optional direction logic ---
        if with_direction and selected_stats:
            direction_map: dict[str, float] = {}
            for r, s in selected_stats.items():
                if abs(s["mean"]) >= direction_threshold:
                    direction_map[r] = 1.0 if s["mean"] > 0 else -1.0
                else:
                    # Train mean too weak to determine direction; keep raw sign.
                    direction_map[r] = 1.0
            multiplier = test_filtered["regime_key"].map(direction_map).fillna(0.0)
            active_mask = multiplier != 0.0
            returns_arr = (
                multiplier[active_mask]
                * test_filtered.loc[active_mask, target_col]
            ).values
        else:
            returns_arr = test_filtered[target_col].values

        # --- Step 9: fold metrics ---
        m = _fold_metrics(returns_arr)

        fold_rows.append(
            {
                "year": int(test_year),
                "n": m["n"],
                "mean": m["mean"],
                "sharpe": m["sharpe"],
                "hit_rate": m["hit_rate"],
                "coverage": coverage_after,
            }
        )

        logger.info(
            "REGIME FILTER PIPELINE | year=%d | n=%5d"
            " | mean=%+.6f | sharpe=%+.4f | hit_rate=%.4f | coverage=%.1f%%",
            test_year,
            m["n"],
            m["mean"],
            m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
            m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
            coverage_after * 100,
        )

    if not fold_rows:
        logger.warning(
            "REGIME FILTER PIPELINE: no valid folds produced "
            "(min_n=%d, candidate_k=%d, final_k=%d)",
            min_n,
            candidate_k,
            final_k,
        )
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(fold_rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pooled summary
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`regime_filter_walk_forward`.

    Returns:
        Dict with keys ``n_folds``, ``mean_return``, ``mean_sharpe``,
        ``mean_hit_rate``, ``mean_coverage``.  All float values are
        ``NaN`` when *fold_df* is empty.
    """
    if fold_df.empty:
        return {
            "n_folds": 0,
            "mean_return": np.nan,
            "mean_sharpe": np.nan,
            "mean_hit_rate": np.nan,
            "mean_coverage": np.nan,
        }
    return {
        "n_folds": len(fold_df),
        "mean_return": float(fold_df["mean"].mean()),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_regime_filter_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results under the REGIME FILTER PIPELINE header.

    Args:
        fold_df: DataFrame returned by :func:`regime_filter_walk_forward`.
    """
    logger.info("=== REGIME FILTER PIPELINE — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME FILTER PIPELINE: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%%",
            row.year,
            row.n,
            row.mean,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate,
            row.coverage * 100,
        )


def log_regime_filter_summary(
    fold_df: pd.DataFrame,
    pooled: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the regime-filter pipeline.

    Args:
        fold_df: DataFrame returned by :func:`regime_filter_walk_forward`.
        pooled: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME FILTER PIPELINE — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME FILTER PIPELINE SUMMARY: no results")
        return

    logger.info(
        "Folds evaluated : %d",
        pooled["n_folds"],
    )
    logger.info(
        "Mean return      : %+.6f",
        pooled["mean_return"],
    )
    logger.info(
        "Mean Sharpe      : %+.4f",
        pooled["mean_sharpe"] if not np.isnan(pooled["mean_sharpe"]) else float("nan"),
    )
    logger.info(
        "Mean hit rate    : %.4f",
        pooled["mean_hit_rate"],
    )
    logger.info(
        "Mean coverage    : %.1f%%",
        pooled["mean_coverage"] * 100,
    )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime Filter Pipeline: use discrete regime features to FILTER "
            "trades (when to trade), not to predict returns.  No LightGBM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(cfg.DATA_PATH),
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_REGIME_N,
        metavar="N",
        help="Minimum training observations per regime for selection.",
    )
    p.add_argument(
        "--candidate-k",
        type=int,
        default=CANDIDATE_K,
        metavar="K",
        help=(
            "Size of the initial candidate pool: maximum number of top regimes "
            "(by training Sharpe) to iterate through in the selection loop."
        ),
    )
    p.add_argument(
        "--final-k",
        type=int,
        default=FINAL_K,
        metavar="K",
        help=(
            "Maximum number of regimes to keep after applying test-side "
            "stability filters inside the selection loop."
        ),
    )
    p.add_argument(
        "--min-persistence-folds",
        type=int,
        default=MIN_PERSISTENCE_FOLDS,
        metavar="N",
        help=(
            "Minimum number of prior walk-forward folds in which a regime must "
            "have appeared as a candidate (n >= min_n) before it is eligible "
            "for selection.  Set to 0 to disable persistence filtering."
        ),
    )
    p.add_argument(
        "--min-test-coverage",
        type=float,
        default=MIN_TEST_COVERAGE,
        metavar="FRAC",
        help=(
            "Minimum fraction of test signals a regime must cover to survive "
            "the test-set stability filter."
        ),
    )
    p.add_argument(
        "--min-test-n",
        type=int,
        default=MIN_TEST_N,
        metavar="N",
        help="Minimum raw count of test signals per regime to survive stability filter.",
    )
    p.add_argument(
        "--with-direction",
        action="store_true",
        default=WITH_DIRECTION,
        help=(
            "Apply direction logic: invert returns for regimes with "
            "negative training mean (fade the crowd)."
        ),
    )
    p.add_argument(
        "--no-direction",
        dest="with_direction",
        action="store_false",
        help="Disable direction logic; use raw returns for all filtered regimes.",
    )
    p.add_argument(
        "--direction-threshold",
        type=float,
        default=DIRECTION_THRESHOLD,
        metavar="THRESH",
        help=(
            "Minimum |train mean| to activate follow/fade direction logic.  "
            "Below this threshold signals are kept unmodified."
        ),
    )
    p.add_argument(
        "--top-n-log",
        type=int,
        default=TOP_N_LOG,
        metavar="N",
        help="Number of top regimes to log per fold (by training Sharpe).",
    )
    args = p.parse_args(argv)

    from utils.io import setup_logging  # noqa: PLC0415

    setup_logging(args.log_level)

    df = load_data(args.data)
    df = build_features(df)

    require_columns(
        df,
        ["trend_strength_48b", "abs_sentiment", TARGET_COL],
        context="regime_filter_pipeline.main",
    )

    logger.info(
        "=== REGIME FILTER PIPELINE ==="
        " | min_n=%d | candidate_k=%d | final_k=%d | min_test_coverage=%.2f"
        " | min_test_n=%d | direction_threshold=%.4f | with_direction=%s"
        " | min_persistence_folds=%d",
        args.min_n,
        args.candidate_k,
        args.final_k,
        args.min_test_coverage,
        args.min_test_n,
        args.direction_threshold,
        args.with_direction,
        args.min_persistence_folds,
    )

    fold_df = regime_filter_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        min_test_coverage=args.min_test_coverage,
        min_test_n=args.min_test_n,
        with_direction=args.with_direction,
        direction_threshold=args.direction_threshold,
        top_n_log=args.top_n_log,
        min_persistence_folds=args.min_persistence_folds,
    )

    log_regime_filter_fold_results(fold_df)

    pooled = compute_pooled_summary(fold_df)
    log_regime_filter_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
