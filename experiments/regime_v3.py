"""
experiments/regime_v3.py
========================
Regime-discovery-first experiment: identify whether sentiment alpha is
concentrated in specific discrete regimes BEFORE any model-based prediction.

Pipeline order
--------------
1. **Feature discretization** – ``build_regimes()`` adds:

   * ``vol_bucket`` (low / mid / high tertile of ``vol_24b``)
   * ``trend_bucket`` (weak / mid / strong tertile of ``trend_strength_48b``)
   * ``trend_sign`` (sign of ``trend_strength_48b``)
   * ``regime`` = ``vol_bucket + "_" + trend_sign``

2. **REGIME BASELINE (NO MODEL)** – ``regime_baseline()`` groups the full
   dataset by ``regime`` and computes mean return, std, Sharpe, and hit-rate
   for each regime.  No model is trained or used.

3. **REGIME WALK-FORWARD** – ``regime_walk_forward()`` repeats the same
   computation on each test-year slice (expanding window, same discipline as
   the main walk-forward).  Validates whether regime structure is stable
   out-of-sample.

4. **MODEL WITHIN REGIME (secondary)** – ``walk_forward_ridge()`` trains a
   LightGBM model globally and evaluates predictions broken down by regime.

Output DataFrames
-----------------
* ``regime_summary`` – schema ``["regime", "n", "mean", "std", "sharpe",
  "hit_rate"]``; sorted by Sharpe descending.
* ``regime_wf`` – schema ``["year", "regime", "n", "mean", "sharpe",
  "hit_rate"]``; per test-year regime performance.

All features are strictly causal at ``entry_time``:

* **Sentiment** – ``net_sentiment``, ``abs_sentiment``, ``sentiment_change``,
  ``side_streak``, ``extreme_streak_70``, ``extreme_streak_80``
* **Trend** – ``trend_strength_12b``, ``trend_strength_48b``,
  ``trend_dir_12b``, ``trend_dir_48b``
* **Volatility** – ``vol_24b``
* **Interaction** – ``abs_sent_x_trend12b``, ``abs_sent_x_trend48b``,
  ``abs_sent_x_vol24b``, ``extreme70_x_trend48b``

Only columns listed in ``SAFE_FEATURES`` are ever used as model inputs.
Any column matching ``ret_*`` or ``contrarian_ret_*`` is explicitly
prohibited.

Usage::

    python -m experiments.regime_v3 --data data/output/master_research_dataset.csv
    python experiments/regime_v3.py --data data/output/master_research_dataset.csv \\
                                    --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution (python experiments/regime_v3.py)
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from evaluation.walk_forward import wf_summary
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: LightGBM parameters.  Using canonical names to avoid duplicates:
#: - ``min_data_in_leaf`` (not ``min_child_samples``)
#: - ``min_gain_to_split`` (not ``min_split_gain``)
LGBM_PARAMS: dict = {
    "objective": "regression",
    "verbosity": -1,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "min_gain_to_split": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

#: Minimum training observations before a LightGBM model is fit (stability guard).
#: 10 000 rows is a conservative lower bound for tree-based models with up to
#: 31 leaves and 200 estimators, ensuring that each leaf has sufficient coverage
#: and that leave-one-out CV statistics are meaningful.
MIN_TRAIN_OBS: int = 10_000

#: Minimum test-set observations per regime before per-regime metrics are
#: computed.  Regimes with fewer samples are skipped to avoid unreliable IC /
#: Sharpe estimates driven by noise in very small subsets.
MIN_REGIME_OBS: int = 50

#: Whitelisted causal feature columns.  Only these may be used as model inputs.
#: Any column matching ``ret_*`` or ``contrarian_ret_*`` is explicitly prohibited.
SAFE_FEATURES: list[str] = [
    # Sentiment features – available at snapshot_time (causal)
    "net_sentiment",
    "abs_sentiment",
    "sentiment_change",
    "side_streak",
    "extreme_streak_70",
    "extreme_streak_80",
    # Trend features – backward-looking past-price columns (causal)
    "trend_strength_12b",
    "trend_strength_48b",
    "trend_dir_12b",
    "trend_dir_48b",
    # Volatility feature added by build_features
    "vol_24b",
    # Interaction features added by build_features
    "abs_sent_x_trend12b",
    "abs_sent_x_trend48b",
    "abs_sent_x_vol24b",
    "extreme70_x_trend48b",
]

#: Sentinel sub-lists kept for backward compatibility with callers that
#: reference the individual category lists.
SENTIMENT_FEATURES: list[str] = [
    "net_sentiment",
    "abs_sentiment",
    "sentiment_change",
    "side_streak",
    "extreme_streak_70",
    "extreme_streak_80",
]

TREND_FEATURES: list[str] = [
    "trend_strength_12b",
    "trend_strength_48b",
    "trend_dir_12b",
    "trend_dir_48b",
]

VOLATILITY_FEATURES: list[str] = ["vol_24b"]

INTERACTION_FEATURES: list[str] = [
    "abs_sent_x_trend12b",
    "abs_sent_x_trend48b",
    "abs_sent_x_vol24b",
    "extreme70_x_trend48b",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal volatility and interaction features, plus a regime diagnostic.

    ``vol_24b`` is the rolling 24-bar standard deviation of bar-to-bar
    ``entry_close`` returns within each pair.  Because the rolling window
    references only past observations (``min_periods=5``), there is no
    lookahead.

    ``vol_bucket`` is a three-way volatility quantile bucket computed via
    ``pd.qcut(vol_24b, 3)`` for regime diagnostics.  It is not used as a
    model input; it is stored for distribution logging only.

    Interaction features are products of base features computed after
    ``vol_24b`` is available.  Each interaction is only created when both
    constituent columns are present; missing base columns are silently skipped
    (a debug message is emitted).

    Args:
        df: Dataset with ``pair``, ``entry_time``, and ``entry_close``.

    Returns:
        Copy of *df* with ``vol_24b``, ``vol_bucket``, and available
        interaction columns added.
    """
    require_columns(df, ["pair", "entry_time", "entry_close"], context="build_features")

    out = df.copy()
    out = out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    # Bar-to-bar return per pair (pct_change is causal: uses only past bars).
    out["_bar_ret"] = out.groupby("pair")["entry_close"].pct_change()

    # Rolling 24-bar volatility per pair – no lookahead.
    out["vol_24b"] = out.groupby("pair")["_bar_ret"].transform(
        lambda s: s.rolling(24, min_periods=5).std()
    )
    out = out.drop(columns=["_bar_ret"])

    logger.debug("vol_24b: %d non-null values", out["vol_24b"].notna().sum())

    # Regime diagnostic: volatility buckets (low / mid / high).
    # Only computed for rows with valid vol_24b; not used for modeling.
    # Requires at least 3 non-null values to produce 3 distinct quantile buckets.
    valid_vol = out["vol_24b"].notna()
    if valid_vol.sum() >= 3:  # minimum for a 3-way qcut
        out.loc[valid_vol, "vol_bucket"] = pd.qcut(
            out.loc[valid_vol, "vol_24b"],
            q=3,
            labels=["low", "mid", "high"],
        )
        bucket_dist = out["vol_bucket"].value_counts().sort_index()
        logger.info("vol_bucket distribution: %s", bucket_dist.to_dict())
    else:
        out["vol_bucket"] = np.nan
        logger.warning("build_features: not enough vol_24b values for vol_bucket qcut")

    # Interaction features – only created when both operands are present.
    _interactions: list[tuple[str, str, str]] = [
        ("abs_sent_x_trend12b", "abs_sentiment", "trend_strength_12b"),
        ("abs_sent_x_trend48b", "abs_sentiment", "trend_strength_48b"),
        ("abs_sent_x_vol24b", "abs_sentiment", "vol_24b"),
        ("extreme70_x_trend48b", "extreme_streak_70", "trend_strength_48b"),
    ]
    for new_col, col_a, col_b in _interactions:
        if col_a in out.columns and col_b in out.columns:
            out[new_col] = out[col_a] * out[col_b]
            logger.debug(
                "Interaction %s: %d non-null values",
                new_col,
                out[new_col].notna().sum(),
            )
        else:
            missing_bases = [c for c in (col_a, col_b) if c not in out.columns]
            logger.debug(
                "Skipping interaction %s: base column(s) missing: %s",
                new_col,
                missing_bases,
            )

    return out


def select_features(df: pd.DataFrame) -> list[str]:
    """Return feature column names that are present in *df*.

    Filters ``SAFE_FEATURES`` to columns that actually exist in *df*.
    Missing candidates emit a warning.

    Also performs leakage protection: raises ``ValueError`` if any selected
    feature column starts with ``ret_`` or matches ``contrarian_ret_*``, as
    these are forward-looking target columns that must never be used as inputs.

    Args:
        df: Dataset (after ``build_features``).

    Returns:
        List of available feature column names (subset of ``SAFE_FEATURES``).

    Raises:
        ValueError: If a leaking column is detected in the candidate list.
    """
    # Leakage guard: assert SAFE_FEATURES itself contains no forbidden columns.
    leaking = [
        c for c in SAFE_FEATURES
        if c.startswith("ret_") or c.startswith("contrarian_ret_")
    ]
    if leaking:
        raise ValueError(
            f"Leakage detected: SAFE_FEATURES contains forward-looking columns: {leaking}"
        )

    feature_cols = [c for c in SAFE_FEATURES if c in df.columns]
    missing = [c for c in SAFE_FEATURES if c not in df.columns]
    if missing:
        logger.warning("Feature columns not in dataset (skipped): %s", missing)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)
    return feature_cols


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def build_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime classification columns to *df*.

    Performs feature discretization before any modeling:

    * ``vol_bucket`` – tertile bucket of ``vol_24b`` (``"low"`` / ``"mid"``
      / ``"high"``).  When ``vol_bucket`` is already present (computed by
      ``build_features``), that column is reused directly.  Otherwise it is
      recomputed via ``pd.qcut`` from ``vol_24b``.
    * ``trend_bucket`` – tertile bucket of ``trend_strength_48b``
      (``"weak"`` / ``"mid"`` / ``"strong"``).  Rows where
      ``trend_strength_48b`` is missing are assigned ``"unknown"``.
    * ``trend_sign`` – ``np.sign(trend_strength_48b)``: ``-1.0``, ``0.0``,
      or ``1.0``.
    * ``regime`` – combined label ``vol_bucket + "_" + trend_sign``, e.g.
      ``"low_-1.0"`` or ``"high_1.0"``.  Rows without a valid ``vol_bucket``
      are set to ``NaN`` so they are excluded from regime analysis.

    Args:
        df: Dataset (after ``build_features``).  Must contain ``vol_24b``
            and ``trend_strength_48b`` for full regime assignment; missing
            columns are handled gracefully with a warning.

    Returns:
        Copy of *df* with the four regime columns added.
    """
    out = df.copy()

    # --- vol_bucket ---
    if "vol_bucket" in out.columns:
        # Reuse the vol_bucket already computed by build_features (avoids a
        # second qcut pass over the same data).
        n_assigned = int(out["vol_bucket"].notna().sum())
        logger.debug(
            "build_regimes: reusing vol_bucket from build_features (%d rows)",
            n_assigned,
        )
    elif "vol_24b" in out.columns:
        logger.warning(
            "build_regimes: vol_bucket not found; recomputing from vol_24b"
        )
        valid_mask = out["vol_24b"].notna()
        out.loc[valid_mask, "vol_bucket"] = pd.qcut(
            out.loc[valid_mask, "vol_24b"],
            q=3,
            labels=["low", "mid", "high"],
        )
        n_assigned = int(valid_mask.sum())
        logger.debug("build_regimes: vol_bucket assigned for %d rows", n_assigned)
    else:
        logger.warning("build_regimes: vol_24b not found; vol_bucket will be NaN")
        out["vol_bucket"] = np.nan

    # --- trend_bucket (discretization of trend_strength_48b) ---
    if "trend_strength_48b" in out.columns:
        valid_trend = out["trend_strength_48b"].notna()
        if valid_trend.sum() >= 3:
            out["trend_bucket"] = "unknown"
            out.loc[valid_trend, "trend_bucket"] = pd.qcut(
                out.loc[valid_trend, "trend_strength_48b"],
                q=3,
                labels=["weak", "mid", "strong"],
            ).astype(str)
        else:
            out["trend_bucket"] = "unknown"
            logger.warning(
                "build_regimes: not enough trend_strength_48b values for trend_bucket qcut"
            )
        bucket_dist = out["trend_bucket"].value_counts().sort_index()
        logger.info("trend_bucket distribution: %s", bucket_dist.to_dict())
    else:
        out["trend_bucket"] = "unknown"
        logger.warning(
            "build_regimes: trend_strength_48b not found; trend_bucket set to 'unknown'"
        )

    # --- trend_sign ---
    if "trend_strength_48b" not in out.columns:
        logger.warning(
            "build_regimes: trend_strength_48b not found; trend_sign will be 0"
        )
        out["trend_sign"] = 0.0
    else:
        out["trend_sign"] = np.sign(out["trend_strength_48b"])

    # --- regime label = vol_bucket + "_" + trend_sign ---
    out["regime"] = out["vol_bucket"].astype(str) + "_" + out["trend_sign"].astype(str)
    # Rows where vol_bucket was NaN produce 'nan_...' labels; mark these as NaN.
    out.loc[out["vol_bucket"].isna(), "regime"] = np.nan
    n_regimes = out["regime"].nunique()
    logger.info("build_regimes: %d unique regime labels", n_regimes)
    return out



# ---------------------------------------------------------------------------
# Return-based (model-free) regime metrics helper
# ---------------------------------------------------------------------------

def _direct_regime_metrics(returns: np.ndarray) -> dict:
    """Compute return-based metrics for a single regime subset (no model).

    Args:
        returns: Array of realised returns (e.g. ``ret_48b``) for the subset.

    Returns:
        Dict with keys ``n``, ``mean``, ``std``, ``sharpe``, ``hit_rate``.
    """
    n = len(returns)
    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
        }

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else np.nan
    hit_rate = float(np.mean(returns > 0))

    return {
        "n": n,
        "mean": mean_ret,
        "std": std_ret,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# REGIME BASELINE (NO MODEL) — full-dataset regime discovery
# ---------------------------------------------------------------------------

def regime_baseline(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "regime",
    min_n: int = MIN_REGIME_OBS,
) -> pd.DataFrame:
    """Compute regime statistics on the full dataset without any model.

    Groups observations by ``regime_col`` and computes mean return, std,
    Sharpe, and hit-rate for each regime.  Regimes with fewer than ``min_n``
    observations are skipped.

    This is the primary discovery step: no model is trained or used.

    Args:
        df: Full dataset (after ``build_regimes``).
        target_col: Forward-return column to summarise (default ``ret_48b``).
        regime_col: Column containing regime labels (default ``"regime"``).
        min_n: Minimum observations required per regime.

    Returns:
        DataFrame with schema ``["regime", "n", "mean", "std", "sharpe",
        "hit_rate"]``, sorted by Sharpe descending.
    """
    _COLS = ["regime", "n", "mean", "std", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning("regime_baseline: '%s' column not found", regime_col)
        return pd.DataFrame(columns=_COLS)

    valid = df.dropna(subset=[regime_col, target_col])
    rows: list[dict] = []

    for regime_label, grp in valid.groupby(regime_col):
        returns = grp[target_col].values
        if len(returns) < min_n:
            logger.warning(
                "REGIME BASELINE: regime=%s skipped (n=%d < %d)",
                regime_label,
                len(returns),
                min_n,
            )
            continue
        m = _direct_regime_metrics(returns)
        rows.append({"regime": str(regime_label), **m})

    if not rows:
        logger.warning("REGIME BASELINE (NO MODEL): no valid regimes found")
        return pd.DataFrame(columns=_COLS)

    result = (
        pd.DataFrame(rows)[_COLS]
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# REGIME WALK-FORWARD — per-year test-set regime discovery (no model)
# ---------------------------------------------------------------------------

def regime_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "regime",
    year_col: str = "year",
    min_n: int = MIN_REGIME_OBS,
) -> pd.DataFrame:
    """Validate regime structure out-of-sample using an expanding window.

    For each test year (starting from the third unique year), computes
    regime metrics on the **test set only** using raw returns — no model
    predictions are involved.  This validates whether the regime structure
    identified in ``regime_baseline`` is stable out-of-sample.

    Guardrails:
    * Regimes with fewer than ``min_n`` test-set observations are skipped.
    * No training data is used; metrics depend only on the test-year slice.

    Args:
        df: Full dataset (after ``build_regimes``).
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing regime labels (default ``"regime"``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum test-set observations required per regime.

    Returns:
        DataFrame with schema ``["year", "regime", "n", "mean", "sharpe",
        "hit_rate"]``.
    """
    _COLS = ["year", "regime", "n", "mean", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning("regime_walk_forward: '%s' column not found", regime_col)
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "regime_walk_forward: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame(columns=_COLS)

    rows: list[dict] = []

    for i in range(2, len(years)):
        test_year = years[i]
        test = df[df[year_col] == test_year].dropna(subset=[regime_col, target_col])

        if test.empty:
            logger.debug(
                "REGIME WALK-FORWARD: year=%d — empty test set, skipping", test_year
            )
            continue

        for regime_label, grp in test.groupby(regime_col):
            returns = grp[target_col].values
            if len(returns) < min_n:
                logger.warning(
                    "REGIME WALK-FORWARD: year=%d | regime=%s skipped (n=%d < %d)",
                    test_year,
                    regime_label,
                    len(returns),
                    min_n,
                )
                continue
            m = _direct_regime_metrics(returns)
            rows.append(
                {
                    "year": int(test_year),
                    "regime": str(regime_label),
                    "n": m["n"],
                    "mean": m["mean"],
                    "sharpe": m["sharpe"],
                    "hit_rate": m["hit_rate"],
                }
            )

    if not rows:
        logger.warning("REGIME WALK-FORWARD: no valid regime/year combinations found")
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structured logging display for new regime outputs
# ---------------------------------------------------------------------------

def log_regime_baseline(regime_summary: pd.DataFrame) -> None:
    """Log regime_summary under a clearly labelled section header.

    Args:
        regime_summary: DataFrame returned by ``regime_baseline``.
    """
    logger.info("=== REGIME BASELINE (NO MODEL) ===")
    if regime_summary.empty:
        logger.warning("REGIME BASELINE (NO MODEL): no results to display")
        return
    for row in regime_summary.itertuples(index=False):
        logger.info(
            "REGIME BASELINE | regime=%-12s | n=%5d | mean=%+.6f | std=%.6f"
            " | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.n,
            row.mean,
            row.std,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_regime_wf(regime_wf: pd.DataFrame) -> None:
    """Log regime_wf under a clearly labelled section header.

    Args:
        regime_wf: DataFrame returned by ``regime_walk_forward``.
    """
    logger.info("=== REGIME WALK-FORWARD ===")
    if regime_wf.empty:
        logger.warning("REGIME WALK-FORWARD: no results to display")
        return
    for row in regime_wf.itertuples(index=False):
        logger.info(
            "REGIME WALK-FORWARD | year=%d | regime=%-12s | n=%5d"
            " | mean=%+.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.regime,
            row.n,
            row.mean,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


# ---------------------------------------------------------------------------
# Per-regime metrics helper (model-based: uses predictions vs actuals)
# ---------------------------------------------------------------------------

def _regime_metrics(y_pred: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute IC, Sharpe and hit rate for a single regime subset.

    Args:
        y_pred: Model predictions for the subset.
        y_test: Realised returns for the subset.

    Returns:
        Dict with keys ``n``, ``ic``, ``sharpe``, ``hit_rate``.
    """
    n = len(y_pred)
    if n < 2:
        return {
            "n": n,
            "ic": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
        }

    ic, _ = stats.spearmanr(y_pred, y_test)

    signal_dir = np.sign(y_pred)
    active = signal_dir != 0
    signal_returns = signal_dir[active] * y_test[active]
    n_active = int(active.sum())

    if n_active > 1:
        sr_mean = float(np.mean(signal_returns))
        sr_std = float(np.std(signal_returns))
        sharpe = sr_mean / sr_std if sr_std > 1e-10 else np.nan
        hit_rate = float(np.mean(signal_returns > 0))
    else:
        sharpe = np.nan
        hit_rate = np.nan

    return {
        "n": n,
        "ic": float(ic) if not np.isnan(ic) else np.nan,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# Walk-forward LightGBM regression
# ---------------------------------------------------------------------------

def walk_forward_ridge(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_train_obs: int = MIN_TRAIN_OBS,
    regime_col: str | None = None,
    lgbm_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward with LightGBM regression.

    Follows the same discipline as
    ``evaluation.walk_forward.walk_forward_expanding``: for each test year
    (starting from index 2 in the sorted unique-year list), the model is
    trained on *all prior years* and evaluated on the current year.

    **Stability guard**: folds where the training set has fewer than
    ``min_train_obs`` rows (default 10 000) are skipped with a warning.

    **Target demeaning**: ``y_train`` is demeaned per fold before fitting.
    The fold mean is subtracted from training targets to remove any
    in-sample level bias.  Test targets are *not* modified, preventing
    leakage of test information into training.

    When *regime_col* is provided the test-set predictions are also broken
    down by regime and metrics are computed for each regime group.

    Args:
        df: Dataset with feature columns, *target_col*, and *year_col*.
        feature_cols: Columns to use as predictors (must be a subset of
            ``SAFE_FEATURES``; no ``ret_*`` or ``contrarian_ret_*`` allowed).
        target_col: Regression target column (default ``ret_48b``).
        year_col: Column containing the calendar year.
        min_train_obs: Minimum training rows required per fold.  Folds
            below this threshold are skipped.  Default is 10 000.
        regime_col: Optional column name containing regime labels.  When
            provided, per-regime metrics are computed for each fold.
        lgbm_params: Optional override for LightGBM parameters.  Defaults
            to ``LGBM_PARAMS``.

    Returns:
        A 2-tuple ``(wf_df, regime_df)``:

        * *wf_df* – one row per test fold:
          year, n_train, n_test, ic, signal_sharpe, signal_hit_rate, r2.
        * *regime_df* – one row per (fold, regime) combination:
          year, regime, n, ic, signal_sharpe, signal_hit_rate.
          Empty if *regime_col* is ``None`` or no valid folds.
    """
    require_columns(
        df, [year_col, target_col] + feature_cols, context="walk_forward_ridge"
    )

    params = dict(LGBM_PARAMS) if lgbm_params is None else dict(lgbm_params)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "walk_forward_ridge: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame(), pd.DataFrame()

    results: list[dict] = []
    regime_results: list[dict] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_years = years[:i]

        train = df[df[year_col].isin(train_years)].dropna(
            subset=feature_cols + [target_col]
        )
        test = df[df[year_col] == test_year].dropna(
            subset=feature_cols + [target_col]
        )

        # Walk-forward stability guard: require sufficient training data.
        if len(train) < min_train_obs:
            logger.warning(
                "Skipping test_year=%d: %d train rows (need %d)",
                test_year,
                len(train),
                min_train_obs,
            )
            continue

        if len(test) == 0:
            logger.debug("Skipping test_year=%d: empty test set", test_year)
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].values
        X_test = test[feature_cols]
        y_test = test[target_col].values

        # Demean y_train per fold to remove in-sample level bias.
        # Test targets are NOT modified; no test information leaks into training.
        y_train_mean = np.mean(y_train)
        y_train_demeaned = y_train - y_train_mean

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train_demeaned)

        y_pred = model.predict(X_test)

        # IC: Spearman rank correlation between predictions and actuals.
        ic, _ = stats.spearmanr(y_pred, y_test)

        # Signal: sign of prediction → long (+1) or short (−1) position.
        signal_dir = np.sign(y_pred)
        active = signal_dir != 0
        signal_returns = signal_dir[active] * y_test[active]

        n_active = int(active.sum())
        if n_active > 1:
            sr_mean = float(np.mean(signal_returns))
            sr_std = float(np.std(signal_returns))
            sharpe = sr_mean / sr_std if sr_std > 1e-10 else np.nan
            hit_rate = float(np.mean(signal_returns > 0))
        else:
            sharpe = np.nan
            hit_rate = np.nan

        # R² on the test set.
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

        results.append(
            {
                "year": int(test_year),
                "n_train": len(train),
                "n_test": len(test),
                "ic": float(ic) if not np.isnan(ic) else np.nan,
                "signal_sharpe": sharpe,
                "signal_hit_rate": hit_rate,
                "r2": float(r2),
            }
        )

        logger.debug(
            "year=%d | train=%d | test=%d | IC=%.4f | Sharpe=%.4f | R2=%.4f",
            test_year,
            len(train),
            len(test),
            ic,
            sharpe if not np.isnan(sharpe) else float("nan"),
            r2 if not np.isnan(r2) else float("nan"),
        )

        # Per-regime metrics for this fold (TEST set only – no train data used here).
        if regime_col is not None and regime_col in test.columns:
            regime_labels = test[regime_col].values
            for regime_label in np.unique(regime_labels[pd.notna(regime_labels)]):
                mask = regime_labels == regime_label
                n_regime = int(mask.sum())
                if n_regime < MIN_REGIME_OBS:
                    logger.warning(
                        "year=%d | regime=%s | skipped (n=%d < %d)",
                        test_year,
                        regime_label,
                        n_regime,
                        MIN_REGIME_OBS,
                    )
                    continue
                m = _regime_metrics(y_pred[mask], y_test[mask])
                regime_results.append(
                    {
                        "year": int(test_year),
                        "regime": str(regime_label),
                        **m,
                    }
                )
                logger.info(
                    "year=%d | regime=%s | n=%d | IC=%.4f | Sharpe=%.4f | hit_rate=%.4f",
                    test_year,
                    regime_label,
                    m["n"],
                    m["ic"] if not np.isnan(m["ic"]) else float("nan"),
                    m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
                    m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
                )

    if not results:
        logger.warning("walk_forward_ridge: no valid folds produced")
        return pd.DataFrame(), pd.DataFrame()

    return pd.DataFrame(results), pd.DataFrame(regime_results)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the research dataset for Regime V3 evaluation.

    Args:
        path: Path to the master research dataset CSV.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing.
    """
    df = read_csv(
        path,
        required_columns=["pair", "time", "entry_time", "entry_close"],
    )

    df = parse_timestamps(df, "time", context="regime_v3.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    df = parse_timestamps(df, "entry_time", context="regime_v3.load_data")

    df["pair_group"] = np.where(
        df["pair"].str.contains(cfg.JPY_PAIR_PATTERN, case=False, na=False),
        "JPY_cross",
        "other",
    )

    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    logger.info(
        "Dataset summary: %d rows | %d unique pairs | %s to %s",
        len(df),
        df["pair"].nunique(),
        date_min,
        date_max,
    )
    logger.info("Dataset loaded: %d rows, %d pairs", len(df), df["pair"].nunique())
    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_wf_summary(wf_df: pd.DataFrame) -> None:
    """Print per-fold results and pooled aggregate statistics.

    Also calls ``evaluation.walk_forward.wf_summary`` (renamed columns for
    compatibility) so that the standard library summary is included.

    Args:
        wf_df: DataFrame returned by ``walk_forward_ridge``.
    """
    if wf_df.empty:
        print("No walk-forward results.")
        return

    print("\n=== WALK-FORWARD RESULTS (LightGBM → ret_48b) ===")
    display_cols = [
        "year",
        "n_train",
        "n_test",
        "ic",
        "signal_sharpe",
        "signal_hit_rate",
        "r2",
    ]
    display_cols = [c for c in display_cols if c in wf_df.columns]
    print(wf_df[display_cols].to_string(index=False))

    # Aggregate via the standard walk_forward library helper.
    compat = wf_df.rename(
        columns={
            "signal_sharpe": "sharpe",
            "ic": "mean",
            "signal_hit_rate": "hit_rate",
        }
    )
    summary = wf_summary(compat)

    print("\n--- POOLED SUMMARY ---")
    print(f"  Mean IC:           {wf_df['ic'].mean():.4f}")
    print(f"  Mean Sharpe:       {wf_df['signal_sharpe'].mean():.4f}")
    print(f"  Mean Hit Rate:     {wf_df['signal_hit_rate'].mean():.4f}")
    print(f"  Median R\u00b2:         {wf_df['r2'].median():.4f}")
    print(f"  Folds evaluated:   {len(wf_df)}")
    logger.info("wf_summary (library): %s", summary)


def print_regime_summary(regime_df: pd.DataFrame) -> None:
    """Print per-fold regime metrics and pooled per-regime aggregates.

    Outputs two tables:

    1. Per-fold breakdown: ``year | regime | n | IC | Sharpe | hit rate``
    2. Pooled (across all folds) per-regime summary.

    Args:
        regime_df: DataFrame returned as the second element of
            ``walk_forward_ridge`` when *regime_col* is provided.
    """
    if regime_df.empty:
        print("No regime-level results.")
        return

    print("\n=== PER-FOLD REGIME METRICS ===")
    display_cols = ["year", "regime", "n", "ic", "sharpe", "hit_rate"]
    display_cols = [c for c in display_cols if c in regime_df.columns]
    print(
        regime_df.sort_values(["year", "regime"])[display_cols].to_string(index=False)
    )

    # Pooled metrics per regime across all folds (weighted by sample size).
    def _wmean(grp: pd.DataFrame, col: str) -> float:
        weights = grp["n"].values.astype(float)
        vals = grp[col].values.astype(float)
        valid = ~np.isnan(vals) & (weights > 0)
        if valid.sum() == 0:
            return np.nan
        return float(np.average(vals[valid], weights=weights[valid]))

    pooled_rows = []
    for regime_label, grp in regime_df.groupby("regime"):
        pooled_rows.append(
            {
                "regime": regime_label,
                "total_n": int(grp["n"].sum()),
                "ic": _wmean(grp, "ic"),
                "sharpe": _wmean(grp, "sharpe"),
                "hit_rate": _wmean(grp, "hit_rate"),
                "folds": len(grp),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values("regime").reset_index(drop=True)

    print("\n=== POOLED REGIME METRICS (across all folds) ===")
    print(pooled.to_string(index=False))
    logger.info("Pooled regime metrics computed for %d regimes", len(pooled))
    for row in pooled.itertuples(index=False):
        logger.info(
            "pooled | regime=%s | total_n=%d | IC=%.4f | Sharpe=%.4f | hit_rate=%.4f | folds=%d",
            row.regime,
            row.total_n,
            row.ic,
            row.sharpe,
            row.hit_rate,
            row.folds,
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V3: regime discovery via discretization, then optional "
            "LightGBM walk-forward predicting ret_48b (no leakage)."
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
    )
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    df = load_data(args.data)

    # Step 1: Compute causal volatility feature (vol_24b) and interaction features.
    df = build_features(df)

    # Step 2: Discretise features into regimes BEFORE any modeling.
    df = build_regimes(df)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: REGIME BASELINE (NO MODEL) — full-dataset regime discovery
    # ------------------------------------------------------------------
    regime_summary = regime_baseline(df)
    log_regime_baseline(regime_summary)

    # ------------------------------------------------------------------
    # Step 4: REGIME WALK-FORWARD — out-of-sample regime validation
    # ------------------------------------------------------------------
    regime_wf = regime_walk_forward(df)
    log_regime_wf(regime_wf)

    # ------------------------------------------------------------------
    # Step 5: MODEL WITHIN REGIME (secondary) — LightGBM walk-forward
    #         trained globally, evaluated per regime
    # ------------------------------------------------------------------
    feature_cols = select_features(df)

    if not feature_cols:
        logger.warning("No valid feature columns found; skipping MODEL WITHIN REGIME.")
        return

    logger.info("=== MODEL WITHIN REGIME ===")
    wf_results, regime_model_results = walk_forward_ridge(
        df, feature_cols, regime_col="regime"
    )

    print_wf_summary(wf_results)
    print_regime_summary(regime_model_results)


if __name__ == "__main__":
    main()
