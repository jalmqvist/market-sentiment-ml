"""
experiments/regime_v3.py
========================
Regime-discovery-first experiment: identify whether sentiment alpha is
concentrated in specific discrete regimes BEFORE any model-based prediction.
Extends into a filtered-signal trading pipeline using ``TOP_REGIMES``.

Pipeline order
--------------
1. **Feature discretization** – ``build_regimes()`` adds:

   * ``vol_bucket`` (low / mid / high tertile of ``vol_24b``)
   * ``trend_bucket`` (weak / mid / strong tertile of ``trend_strength_48b``)
   * ``trend_sign`` (sign of ``trend_strength_48b``)
   * ``regime`` = ``vol_bucket + "_" + trend_sign``

2. **BEHAVIOURAL REGIME DISCRETIZATION** – ``build_behavioural_regimes()``
   adds sentiment-persistence bucket columns and constructs:

   * ``abs_sent_bucket`` – intensity of crowd sentiment (low / mid / high /
     extreme based on fixed thresholds).
   * ``side_streak_bucket`` – persistence of crowd positioning (short / medium
     / long based on run-length).
   * ``extreme_streak_bucket`` – exhaustion signal (none / mild / extreme
     based on extreme-sentiment run length).
   * ``behavioural_regime`` = ``abs_sent_bucket + "_" + side_streak_bucket +
     "_" + extreme_streak_bucket``
   * ``crowding_regime`` = ``side_streak_bucket + "_" + abs_sent_bucket``
     (simplified 2-axis regime used by the trading filter).

3. **REGIME BASELINE (NO MODEL)** – ``regime_baseline()`` groups the full
   dataset by ``regime`` and computes mean return, std, Sharpe, and hit-rate
   for each regime.  No model is trained or used.

4. **BEHAVIOURAL REGIME SUMMARY** – ``behavioural_regime_baseline()`` does
   the same using ``behavioural_regime``, revealing whether alpha is
   concentrated in crowding/exhaustion regimes.

5. **REGIME WALK-FORWARD** – ``regime_walk_forward()`` repeats the same
   computation on each test-year slice (expanding window, same discipline as
   the main walk-forward).  Validates whether regime structure is stable
   out-of-sample.

6. **WALK-FORWARD REGIME PERFORMANCE** – ``behavioural_regime_walk_forward()``
   does the same for ``behavioural_regime``.

7. **REGIME FILTER** – ``apply_regime_filter()`` marks rows active when their
   ``crowding_regime`` is in ``TOP_REGIMES``.  Filter is deterministic and
   leakage-free; no forward-looking information is used.

8. **FULL DATASET PERFORMANCE** – ``full_dataset_performance()`` computes
   aggregate metrics on the full (unfiltered) dataset as baseline comparison.

9. **FILTERED PERFORMANCE** – ``filtered_regime_baseline()`` computes
   aggregate metrics on the regime-filtered dataset only.

10. **WALK-FORWARD FILTERED PERFORMANCE** – ``filtered_regime_walk_forward()``
    computes per-year OOS metrics on filtered signals (no refitting of the
    regime list).

11. **COVERAGE SUMMARY** – ``compute_coverage_summary()`` reports the fraction
    of signals retained after the regime filter.

12. **MODEL WITHIN REGIME (secondary)** – ``walk_forward_ridge()`` trains a
    LightGBM model globally and evaluates predictions broken down by regime.

Output DataFrames
-----------------
* ``regime_summary`` – schema ``["regime", "n", "mean", "std", "sharpe",
  "hit_rate"]``; sorted by Sharpe descending.
* ``regime_wf`` – schema ``["year", "regime", "n", "mean", "sharpe",
  "hit_rate"]``; per test-year regime performance.
* ``behavioural_regime_summary`` – same schema as ``regime_summary``; uses
  the behavioural regime label.
* ``behavioural_regime_wf`` – same schema as ``regime_wf``; uses the
  behavioural regime label.
* ``filtered_performance_summary`` – schema ``["n", "mean", "std", "sharpe",
  "hit_rate"]``; aggregate metrics on the regime-filtered dataset.
* ``filtered_wf`` – schema ``["year", "n", "mean", "sharpe", "hit_rate"]``;
  per-year OOS metrics on filtered signals.
* ``coverage_summary`` – schema ``["total_signals", "filtered_signals",
  "coverage_ratio"]``; signal coverage diagnostics.

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

#: Minimum observations per crowding regime (simplified 2-axis definition).
#: Higher threshold than MIN_REGIME_OBS to ensure statistical stability with
#: fewer, broader regime buckets.
MIN_CROWDING_REGIME_OBS: int = 100

#: Hardcoded list of high-quality crowding regimes selected from regime_v3
#: discovery results.  Each entry must match the ``crowding_regime`` label
#: format: ``side_streak_bucket + "_" + abs_sent_bucket``
#: (e.g. ``"long_extreme"``, ``"medium_high"``).
#: Update this list to change which regimes are active in the trading filter.
TOP_REGIMES: list[str] = [
    "long_extreme",
    "medium_high",
]

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
# Behavioural regime discretization
# ---------------------------------------------------------------------------

def build_behavioural_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural regime bucket columns and a combined regime label.

    Discretises sentiment-derived features into fixed-threshold buckets that
    capture crowding intensity, persistence, and exhaustion — the primary
    behavioural signals identified in feature-importance analysis.

    Adds the following columns:

    * ``abs_sent_bucket`` – sentiment intensity bucket based on ``abs_sentiment``:

      - ``"low"``     (<50)
      - ``"mid"``     (50–70)
      - ``"high"``    (70–85)
      - ``"extreme"`` (≥85)

    * ``side_streak_bucket`` – crowd-side persistence bucket based on the
      absolute value of ``side_streak``:

      - ``"short"``   (1–2 bars)
      - ``"medium"``  (3–5 bars)
      - ``"long"``    (6+ bars)

    * ``extreme_streak_bucket`` – exhaustion signal bucket based on
      ``extreme_streak_70``:

      - ``"none"``    (0)
      - ``"mild"``    (1–2)
      - ``"extreme"`` (3+)

    * ``behavioural_regime`` – composite label constructed as::

          abs_sent_bucket + "_" + side_streak_bucket + "_" + extreme_streak_bucket

      Rows where any constituent bucket is ``NaN`` receive ``NaN`` in
      ``behavioural_regime`` and are excluded from downstream regime analysis.

    Args:
        df: Dataset (after ``build_features``).  Should contain
            ``abs_sentiment``, ``side_streak``, and ``extreme_streak_70``.

    Returns:
        Copy of *df* with the four new columns appended.
    """
    out = df.copy()

    # --- abs_sent_bucket ---
    if "abs_sentiment" in out.columns:
        out["abs_sent_bucket"] = pd.cut(
            out["abs_sentiment"],
            bins=[-np.inf, 50, 70, 85, np.inf],
            labels=["low", "mid", "high", "extreme"],
            right=False,  # [low, 50), [50, 70), [70, 85), [85, ∞)
        ).astype(str)
        # pd.cut returns "nan" for NaN inputs; replace with proper NaN.
        out.loc[out["abs_sentiment"].isna(), "abs_sent_bucket"] = np.nan
        dist = out["abs_sent_bucket"].value_counts(dropna=True).sort_index()
        logger.info("abs_sent_bucket distribution: %s", dist.to_dict())
    else:
        out["abs_sent_bucket"] = np.nan
        logger.warning(
            "build_behavioural_regimes: abs_sentiment not found; abs_sent_bucket will be NaN"
        )

    # --- side_streak_bucket ---
    if "side_streak" in out.columns:
        streak_abs = out["side_streak"].abs()
        out["side_streak_bucket"] = pd.cut(
            streak_abs,
            bins=[0, 2, 5, np.inf],
            labels=["short", "medium", "long"],
            right=True,         # [0, 2] → short, (2, 5] → medium, (5, ∞) → long
            include_lowest=True,  # makes first bin [0, 2] (closed on left)
        ).astype(str)
        out.loc[out["side_streak"].isna(), "side_streak_bucket"] = np.nan
        dist = out["side_streak_bucket"].value_counts(dropna=True).sort_index()
        logger.info("side_streak_bucket distribution: %s", dist.to_dict())
    else:
        out["side_streak_bucket"] = np.nan
        logger.warning(
            "build_behavioural_regimes: side_streak not found; side_streak_bucket will be NaN"
        )

    # --- extreme_streak_bucket ---
    if "extreme_streak_70" in out.columns:
        ext = out["extreme_streak_70"]
        out["extreme_streak_bucket"] = pd.cut(
            ext,
            bins=[-np.inf, 0, 2, np.inf],
            labels=["none", "mild", "extreme"],
            right=True,   # (-∞, 0] → none, (0, 2] → mild, (2, ∞) → extreme
        ).astype(str)
        dist = out["extreme_streak_bucket"].value_counts(dropna=True).sort_index()
        logger.info("extreme_streak_bucket distribution: %s", dist.to_dict())
    else:
        out["extreme_streak_bucket"] = np.nan
        logger.warning(
            "build_behavioural_regimes: extreme_streak_70 not found; "
            "extreme_streak_bucket will be NaN"
        )

    # --- behavioural_regime = abs_sent_bucket + "_" + side_streak_bucket
    #                          + "_" + extreme_streak_bucket ---
    valid_mask = (
        out["abs_sent_bucket"].notna()
        & out["side_streak_bucket"].notna()
        & out["extreme_streak_bucket"].notna()
    )
    combined = (
        out["abs_sent_bucket"].astype(str)
        + "_"
        + out["side_streak_bucket"].astype(str)
        + "_"
        + out["extreme_streak_bucket"].astype(str)
    )
    # Mask invalid rows (where any bucket is NaN) with NaN; .where() preserves
    # string dtype and sets non-matching rows to NaN.
    out["behavioural_regime"] = combined.where(valid_mask)
    n_valid = int(valid_mask.sum())
    n_regimes = out["behavioural_regime"].nunique()
    logger.info(
        "build_behavioural_regimes: %d rows assigned to %d unique behavioural regimes",
        n_valid,
        n_regimes,
    )

    # --- crowding_regime = side_streak_bucket + "_" + abs_sent_bucket ---
    # Simplified 2-axis regime: excludes extreme_streak_bucket to reduce
    # dimensionality and improve sample sizes per regime.
    crowding_valid_mask = (
        out["abs_sent_bucket"].notna()
        & out["side_streak_bucket"].notna()
    )
    crowding_combined = (
        out["side_streak_bucket"].astype(str)
        + "_"
        + out["abs_sent_bucket"].astype(str)
    )
    out["crowding_regime"] = crowding_combined.where(crowding_valid_mask)
    n_crowding_regimes = out["crowding_regime"].nunique()
    logger.info(
        "build_behavioural_regimes: %d unique crowding regimes (2-axis simplified)",
        n_crowding_regimes,
    )

    return out



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
# BEHAVIOURAL REGIME SUMMARY (NO MODEL) — full-dataset behavioural discovery
# ---------------------------------------------------------------------------

def behavioural_regime_baseline(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "behavioural_regime",
    min_n: int = MIN_REGIME_OBS,
) -> pd.DataFrame:
    """Compute regime statistics using behavioural regime labels (no model).

    Groups observations by ``behavioural_regime`` (or *regime_col*) and
    computes mean return, std, Sharpe, and hit-rate for each regime.  Regimes
    with fewer than ``min_n`` observations are skipped and logged at DEBUG
    level.

    This is the primary behavioural discovery step: no model is trained or
    used.  Identifies whether sentiment alpha is concentrated in specific
    crowding/exhaustion regimes.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        target_col: Forward-return column to summarise (default ``ret_48b``).
        regime_col: Column containing behavioural regime labels (default
            ``"behavioural_regime"``).
        min_n: Minimum observations required per regime; regimes below this
            threshold are skipped.

    Returns:
        DataFrame with schema ``["regime", "n", "mean", "std", "sharpe",
        "hit_rate"]``, sorted by Sharpe descending.  This is the
        ``behavioural_regime_summary`` output artifact.
    """
    _COLS = ["regime", "n", "mean", "std", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning("behavioural_regime_baseline: '%s' column not found", regime_col)
        return pd.DataFrame(columns=_COLS)

    valid = df.dropna(subset=[regime_col, target_col])
    rows: list[dict] = []

    for regime_label, grp in valid.groupby(regime_col):
        returns = grp[target_col].values
        if len(returns) < min_n:
            logger.debug(
                "BEHAVIOURAL REGIME SUMMARY: regime=%s skipped (n=%d < %d)",
                regime_label,
                len(returns),
                min_n,
            )
            continue
        m = _direct_regime_metrics(returns)
        rows.append({"regime": str(regime_label), **m})

    if not rows:
        logger.warning("BEHAVIOURAL REGIME SUMMARY: no valid regimes found")
        return pd.DataFrame(columns=_COLS)

    result = (
        pd.DataFrame(rows)[_COLS]
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# WALK-FORWARD REGIME PERFORMANCE — per-year behavioural regime discovery
# ---------------------------------------------------------------------------

def behavioural_regime_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "behavioural_regime",
    year_col: str = "year",
    min_n: int = MIN_REGIME_OBS,
) -> pd.DataFrame:
    """Validate behavioural regime structure out-of-sample (expanding window).

    For each test year (starting from the third unique year), computes
    regime metrics on the **test set only** using raw returns — no model
    predictions are involved.  Validates whether the behavioural regime
    structure identified in ``behavioural_regime_baseline`` is stable
    out-of-sample.

    Regimes with fewer than ``min_n`` test-set observations are skipped and
    logged at DEBUG level.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing behavioural regime labels (default
            ``"behavioural_regime"``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum test-set observations required per regime.

    Returns:
        DataFrame with schema ``["year", "regime", "n", "mean", "sharpe",
        "hit_rate"]``.  This is the ``behavioural_regime_wf`` output artifact.
    """
    _COLS = ["year", "regime", "n", "mean", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning(
            "behavioural_regime_walk_forward: '%s' column not found", regime_col
        )
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "behavioural_regime_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    rows: list[dict] = []

    for i in range(2, len(years)):
        test_year = years[i]
        test = df[df[year_col] == test_year].dropna(subset=[regime_col, target_col])

        if test.empty:
            logger.debug(
                "WALK-FORWARD REGIME PERFORMANCE: year=%d — empty test set, skipping",
                test_year,
            )
            continue

        for regime_label, grp in test.groupby(regime_col):
            returns = grp[target_col].values
            if len(returns) < min_n:
                logger.debug(
                    "WALK-FORWARD REGIME PERFORMANCE: year=%d | regime=%s skipped (n=%d < %d)",
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
        logger.warning(
            "WALK-FORWARD REGIME PERFORMANCE: no valid regime/year combinations found"
        )
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CROWDING REGIME SUMMARY (NO MODEL) — simplified 2-axis regime discovery
# ---------------------------------------------------------------------------

def crowding_regime_baseline(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "crowding_regime",
    min_n: int = MIN_CROWDING_REGIME_OBS,
) -> pd.DataFrame:
    """Compute crowding regime statistics on the full dataset without any model.

    Groups observations by the simplified 2-axis crowding regime
    (``side_streak_bucket + "_" + abs_sent_bucket``) and computes mean
    return, std, Sharpe, and hit-rate for each regime.  Regimes with fewer
    than ``min_n`` observations are skipped and logged at WARNING level.

    This is the primary crowding discovery step: no model is trained or used.
    Compared to ``behavioural_regime_baseline``, this uses only two axes
    (removing ``extreme_streak_bucket``) to reduce sparsity and improve
    statistical stability.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        target_col: Forward-return column to summarise (default ``ret_48b``).
        regime_col: Column containing crowding regime labels (default
            ``"crowding_regime"``).
        min_n: Minimum observations required per regime.

    Returns:
        DataFrame ``crowding_regime_summary`` with schema
        ``["regime", "n", "mean", "std", "sharpe", "hit_rate"]``,
        sorted by Sharpe descending.
    """
    _COLS = ["regime", "n", "mean", "std", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning("crowding_regime_baseline: '%s' column not found", regime_col)
        return pd.DataFrame(columns=_COLS)

    valid = df.dropna(subset=[regime_col, target_col])
    rows: list[dict] = []

    for regime_label, grp in valid.groupby(regime_col):
        returns = grp[target_col].values
        if len(returns) < min_n:
            logger.warning(
                "CROWDING REGIME SUMMARY: regime=%s skipped (n=%d < %d)",
                regime_label,
                len(returns),
                min_n,
            )
            continue
        m = _direct_regime_metrics(returns)
        rows.append({"regime": str(regime_label), **m})

    if not rows:
        logger.warning("CROWDING REGIME SUMMARY: no valid regimes found")
        return pd.DataFrame(columns=_COLS)

    result = (
        pd.DataFrame(rows)[_COLS]
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# CROWDING REGIME WALK-FORWARD — per-year crowding regime validation
# ---------------------------------------------------------------------------

def crowding_regime_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "crowding_regime",
    year_col: str = "year",
    min_n: int = MIN_CROWDING_REGIME_OBS,
) -> pd.DataFrame:
    """Validate crowding regime structure out-of-sample using an expanding window.

    For each test year (starting from the third unique year), computes
    regime metrics on the **test set only** using raw returns — no model
    predictions are involved.  Logs ``year | regime | n | sharpe | hit_rate``
    for each valid (year, regime) combination.

    Regimes with fewer than ``min_n`` test-set observations are skipped and
    logged at WARNING level.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing crowding regime labels (default
            ``"crowding_regime"``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum test-set observations required per regime.

    Returns:
        DataFrame ``crowding_regime_wf`` with schema
        ``["year", "regime", "n", "mean", "sharpe", "hit_rate"]``.
    """
    _COLS = ["year", "regime", "n", "mean", "sharpe", "hit_rate"]

    if regime_col not in df.columns:
        logger.warning(
            "crowding_regime_walk_forward: '%s' column not found", regime_col
        )
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "crowding_regime_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    rows: list[dict] = []

    for i in range(2, len(years)):
        test_year = years[i]
        test = df[df[year_col] == test_year].dropna(subset=[regime_col, target_col])

        if test.empty:
            logger.debug(
                "WALK-FORWARD REGIME PERFORMANCE: year=%d — empty test set, skipping",
                test_year,
            )
            continue

        for regime_label, grp in test.groupby(regime_col):
            returns = grp[target_col].values
            if len(returns) < min_n:
                logger.warning(
                    "WALK-FORWARD REGIME PERFORMANCE: year=%d | regime=%s skipped (n=%d < %d)",
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
            logger.info(
                "WALK-FORWARD REGIME PERFORMANCE | year=%d | regime=%-20s | n=%5d"
                " | sharpe=%+.4f | hit_rate=%.4f",
                test_year,
                regime_label,
                m["n"],
                m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
                m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
            )

    if not rows:
        logger.warning(
            "WALK-FORWARD REGIME PERFORMANCE: no valid regime/year combinations found"
        )
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SECONDARY VOL FILTER — secondary conditioning on top crowding regimes
# ---------------------------------------------------------------------------

def secondary_vol_filter(
    df: pd.DataFrame,
    crowding_summary: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "crowding_regime",
    vol_col: str = "vol_24b",
    top_n: int = 5,
    min_n: int = MIN_CROWDING_REGIME_OBS,
) -> pd.DataFrame:
    """Apply secondary vol conditioning to the top crowding regimes.

    For each of the top *top_n* crowding regimes (by Sharpe), splits
    observations into ``"high_vol"`` and ``"low_vol"`` subsets using a
    global median split on ``vol_24b``.  Performance metrics are computed
    for each ``(regime, vol_group)`` combination.

    Vol is used **only** as a secondary split within already-identified top
    regimes; it is **not** included in the regime key itself.  This avoids
    the combinatorial explosion of a vol × regime cross product.

    Subsets with fewer than ``min_n`` observations are skipped and logged
    at WARNING level.

    Args:
        df: Full dataset (after ``build_behavioural_regimes`` and
            ``build_features``).
        crowding_summary: DataFrame returned by ``crowding_regime_baseline``,
            sorted by Sharpe descending.
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing crowding regime labels (default
            ``"crowding_regime"``).
        vol_col: Volatility column used for median split (default ``vol_24b``).
        top_n: Number of top regimes to apply secondary conditioning to.
        min_n: Minimum observations per vol subset to include in results.

    Returns:
        DataFrame with schema
        ``["regime", "vol_group", "n", "mean", "std", "sharpe", "hit_rate"]``.
    """
    _COLS = ["regime", "vol_group", "n", "mean", "std", "sharpe", "hit_rate"]

    if crowding_summary.empty:
        logger.warning("secondary_vol_filter: crowding_summary is empty")
        return pd.DataFrame(columns=_COLS)
    if regime_col not in df.columns:
        logger.warning(
            "secondary_vol_filter: '%s' column not found in dataset", regime_col
        )
        return pd.DataFrame(columns=_COLS)
    if vol_col not in df.columns:
        logger.warning(
            "secondary_vol_filter: '%s' column not found; skipping vol filter",
            vol_col,
        )
        return pd.DataFrame(columns=_COLS)

    top_regimes = crowding_summary.head(top_n)["regime"].tolist()
    valid = df.dropna(subset=[regime_col, target_col, vol_col])
    vol_median = float(valid[vol_col].median())
    logger.info(
        "secondary_vol_filter: vol_24b median=%.6f; applying to top %d regimes",
        vol_median,
        len(top_regimes),
    )

    rows: list[dict] = []

    for regime_label in top_regimes:
        regime_df = valid[valid[regime_col] == regime_label]
        for vol_group, vol_mask in [
            ("low_vol", regime_df[vol_col] <= vol_median),
            ("high_vol", regime_df[vol_col] > vol_median),
        ]:
            subset = regime_df[vol_mask]
            returns = subset[target_col].values
            if len(returns) < min_n:
                logger.warning(
                    "SECONDARY VOL FILTER: regime=%s | vol_group=%s skipped (n=%d < %d)",
                    regime_label,
                    vol_group,
                    len(returns),
                    min_n,
                )
                continue
            m = _direct_regime_metrics(returns)
            rows.append(
                {
                    "regime": str(regime_label),
                    "vol_group": vol_group,
                    **m,
                }
            )

    if not rows:
        logger.warning("SECONDARY VOL FILTER (TOP REGIMES): no valid results")
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(rows)[_COLS].reset_index(drop=True)


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


def log_behavioural_regime_summary(regime_summary: pd.DataFrame, *, top_n: int = 5) -> None:
    """Log behavioural_regime_summary under the BEHAVIOURAL REGIME SUMMARY header.

    Logs all regimes sorted by Sharpe descending, then explicitly highlights
    the top *top_n* and bottom *top_n* regimes.

    Args:
        regime_summary: DataFrame returned by ``behavioural_regime_baseline``.
        top_n: Number of top/bottom regimes to highlight (default 5).
    """
    logger.info("=== BEHAVIOURAL REGIME SUMMARY ===")
    if regime_summary.empty:
        logger.warning("BEHAVIOURAL REGIME SUMMARY: no results to display")
        return

    for row in regime_summary.itertuples(index=False):
        logger.info(
            "BEHAVIOURAL REGIME SUMMARY | regime=%-30s | n=%5d | mean=%+.6f"
            " | std=%.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.n,
            row.mean,
            row.std,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )

    # Highlight top and bottom regimes.
    n_rows = len(regime_summary)
    n_top = min(top_n, n_rows)
    logger.info("--- TOP %d BEHAVIOURAL REGIMES (by Sharpe) ---", n_top)
    for row in regime_summary.head(n_top).itertuples(index=False):
        logger.info(
            "TOP | regime=%-30s | n=%5d | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.n,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )

    n_bottom = min(top_n, n_rows)
    logger.info("--- BOTTOM %d BEHAVIOURAL REGIMES (by Sharpe) ---", n_bottom)
    for row in regime_summary.tail(n_bottom).itertuples(index=False):
        logger.info(
            "BOTTOM | regime=%-30s | n=%5d | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.n,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_behavioural_regime_wf(regime_wf: pd.DataFrame) -> None:
    """Log behavioural_regime_wf under the WALK-FORWARD REGIME PERFORMANCE header.

    Args:
        regime_wf: DataFrame returned by ``behavioural_regime_walk_forward``.
    """
    logger.info("=== WALK-FORWARD REGIME PERFORMANCE ===")
    if regime_wf.empty:
        logger.warning("WALK-FORWARD REGIME PERFORMANCE: no results to display")
        return
    for row in regime_wf.itertuples(index=False):
        logger.info(
            "WALK-FORWARD REGIME PERFORMANCE | year=%d | regime=%-30s | n=%5d"
            " | mean=%+.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.regime,
            row.n,
            row.mean,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_crowding_regime_summary(regime_summary: pd.DataFrame) -> None:
    """Log crowding_regime_summary under the CROWDING REGIME SUMMARY section.

    Logs all regimes sorted by Sharpe descending.

    Args:
        regime_summary: DataFrame returned by ``crowding_regime_baseline``.
    """
    logger.info("=== CROWDING REGIME SUMMARY ===")
    if regime_summary.empty:
        logger.warning("CROWDING REGIME SUMMARY: no results to display")
        return
    for row in regime_summary.itertuples(index=False):
        logger.info(
            "CROWDING REGIME SUMMARY | regime=%-20s | n=%5d | mean=%+.6f"
            " | std=%.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.n,
            row.mean,
            row.std,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_crowding_regime_wf(regime_wf: pd.DataFrame) -> None:
    """Log crowding_regime_wf under the WALK-FORWARD REGIME PERFORMANCE section.

    Args:
        regime_wf: DataFrame returned by ``crowding_regime_walk_forward``.
    """
    logger.info("=== WALK-FORWARD REGIME PERFORMANCE ===")
    if regime_wf.empty:
        logger.warning("WALK-FORWARD REGIME PERFORMANCE: no results to display")
        return
    for row in regime_wf.itertuples(index=False):
        logger.info(
            "WALK-FORWARD REGIME PERFORMANCE | year=%d | regime=%-20s | n=%5d"
            " | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.regime,
            row.n,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_secondary_vol_filter(vol_filter_df: pd.DataFrame) -> None:
    """Log secondary_vol_filter results under the SECONDARY VOL FILTER section.

    Args:
        vol_filter_df: DataFrame returned by ``secondary_vol_filter``.
    """
    logger.info("=== SECONDARY VOL FILTER (TOP REGIMES) ===")
    if vol_filter_df.empty:
        logger.warning("SECONDARY VOL FILTER (TOP REGIMES): no results to display")
        return
    for row in vol_filter_df.itertuples(index=False):
        logger.info(
            "SECONDARY VOL FILTER | regime=%-20s | vol_group=%-8s | n=%5d"
            " | mean=%+.6f | std=%.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.regime,
            row.vol_group,
            row.n,
            row.mean,
            row.std,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


# ---------------------------------------------------------------------------
# REGIME FILTER — apply TOP_REGIMES filter to dataset
# ---------------------------------------------------------------------------

def apply_regime_filter(
    df: pd.DataFrame,
    top_regimes: list[str] | None = None,
    *,
    regime_col: str = "crowding_regime",
) -> pd.DataFrame:
    """Add an ``is_active`` boolean column marking rows in the top regimes.

    Rows whose ``regime_col`` value is in *top_regimes* are marked
    ``is_active = True``; all other rows (including NaN regime rows) are
    marked ``False``.  The filter is deterministic and leakage-free because
    ``crowding_regime`` is computed entirely from past/contemporaneous
    sentiment data before any target column is observed.

    Args:
        df: Dataset (after ``build_behavioural_regimes``).
        top_regimes: List of regime labels to activate.  Defaults to the
            module-level ``TOP_REGIMES`` constant.
        regime_col: Column containing regime labels (default
            ``"crowding_regime"``).

    Returns:
        Copy of *df* with an ``is_active`` boolean column appended.
    """
    if top_regimes is None:
        top_regimes = TOP_REGIMES

    out = df.copy()

    if regime_col not in out.columns:
        logger.warning(
            "apply_regime_filter: '%s' column not found; is_active will be False",
            regime_col,
        )
        out["is_active"] = False
        return out

    out["is_active"] = out[regime_col].isin(top_regimes)
    n_active = int(out["is_active"].sum())
    n_total = len(out)
    logger.info(
        "apply_regime_filter: %d / %d rows active (top_regimes=%s)",
        n_active,
        n_total,
        top_regimes,
    )
    return out


# ---------------------------------------------------------------------------
# FULL DATASET PERFORMANCE — aggregate metrics without regime filter
# ---------------------------------------------------------------------------

def full_dataset_performance(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Compute aggregate performance metrics on the full (unfiltered) dataset.

    Computes n, mean return, std, Sharpe, and hit-rate across all rows where
    *target_col* is non-null.  This acts as the baseline comparison for the
    filtered performance.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        target_col: Forward-return column to summarise (default ``ret_48b``).

    Returns:
        Single-row DataFrame with schema
        ``["n", "mean", "std", "sharpe", "hit_rate"]``.
        This is the ``full_performance_summary`` output artifact.
    """
    _COLS = ["n", "mean", "std", "sharpe", "hit_rate"]
    valid = df.dropna(subset=[target_col])
    if valid.empty:
        logger.warning("full_dataset_performance: no valid rows found")
        return pd.DataFrame(columns=_COLS)

    m = _direct_regime_metrics(valid[target_col].values)
    return pd.DataFrame([m])[_COLS]


# ---------------------------------------------------------------------------
# FILTERED PERFORMANCE — aggregate metrics on regime-filtered dataset
# ---------------------------------------------------------------------------

def filtered_regime_baseline(
    df: pd.DataFrame,
    top_regimes: list[str] | None = None,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "crowding_regime",
) -> pd.DataFrame:
    """Compute aggregate performance metrics on the regime-filtered dataset.

    Filters the dataset to rows whose ``regime_col`` is in *top_regimes* and
    computes n, mean return, std, Sharpe, and hit-rate.

    Guardrails:
    * ``is_active`` (if present) or an inline ``isin`` filter is applied;
      no forward-looking information is introduced.
    * NaN regime rows are always excluded from the filtered set.

    Args:
        df: Dataset (after ``apply_regime_filter`` or at minimum after
            ``build_behavioural_regimes``).
        top_regimes: List of regime labels to include.  Defaults to
            ``TOP_REGIMES``.
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing regime labels (default
            ``"crowding_regime"``).

    Returns:
        Single-row DataFrame with schema
        ``["n", "mean", "std", "sharpe", "hit_rate"]``.
        This is the ``filtered_performance_summary`` output artifact.
    """
    _COLS = ["n", "mean", "std", "sharpe", "hit_rate"]

    if top_regimes is None:
        top_regimes = TOP_REGIMES

    if regime_col not in df.columns:
        logger.warning(
            "filtered_regime_baseline: '%s' column not found; returning empty",
            regime_col,
        )
        return pd.DataFrame(columns=_COLS)

    # Use is_active column if already computed; otherwise apply filter inline.
    if "is_active" in df.columns:
        df_filtered = df[df["is_active"]].dropna(subset=[target_col])
    else:
        df_filtered = df[df[regime_col].isin(top_regimes)].dropna(subset=[target_col])

    if df_filtered.empty:
        logger.warning(
            "filtered_regime_baseline: no rows remain after regime filter "
            "(top_regimes=%s)",
            top_regimes,
        )
        return pd.DataFrame(columns=_COLS)

    m = _direct_regime_metrics(df_filtered[target_col].values)
    return pd.DataFrame([m])[_COLS]


# ---------------------------------------------------------------------------
# WALK-FORWARD FILTERED PERFORMANCE — per-year metrics on filtered signals
# ---------------------------------------------------------------------------

def filtered_regime_walk_forward(
    df: pd.DataFrame,
    top_regimes: list[str] | None = None,
    *,
    target_col: str = TARGET_COL,
    regime_col: str = "crowding_regime",
    year_col: str = "year",
    min_n: int = MIN_CROWDING_REGIME_OBS,
) -> pd.DataFrame:
    """Walk-forward validation of filtered-regime performance (no model).

    For each test year (starting from the third unique year), applies the same
    regime filter and computes aggregate metrics on the filtered test-year
    slice.  No refitting is done; the filter list is fixed (``top_regimes``).

    Guardrails:
    * Regime filter applied identically across all years (no look-ahead).
    * Years where the filtered test set has fewer than ``min_n`` rows are
      skipped and logged at WARNING level.

    Args:
        df: Full dataset (after ``build_behavioural_regimes``).
        top_regimes: Regime labels to include.  Defaults to ``TOP_REGIMES``.
        target_col: Forward-return column (default ``ret_48b``).
        regime_col: Column containing regime labels (default
            ``"crowding_regime"``).
        year_col: Column containing calendar year (default ``"year"``).
        min_n: Minimum filtered rows required per test year.

    Returns:
        DataFrame with schema
        ``["year", "n", "mean", "sharpe", "hit_rate"]``.
        This is the ``filtered_wf`` output artifact.
    """
    _COLS = ["year", "n", "mean", "sharpe", "hit_rate"]

    if top_regimes is None:
        top_regimes = TOP_REGIMES

    if regime_col not in df.columns:
        logger.warning(
            "filtered_regime_walk_forward: '%s' column not found", regime_col
        )
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "filtered_regime_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    rows: list[dict] = []

    for i in range(2, len(years)):
        test_year = years[i]
        test_all = df[df[year_col] == test_year].dropna(subset=[target_col])

        # Apply regime filter (no refitting — same list across all years).
        test_filtered = test_all[test_all[regime_col].isin(top_regimes)]

        if len(test_filtered) < min_n:
            logger.warning(
                "WALK-FORWARD FILTERED: year=%d skipped (n=%d < %d after filter)",
                test_year,
                len(test_filtered),
                min_n,
            )
            continue

        m = _direct_regime_metrics(test_filtered[target_col].values)
        rows.append(
            {
                "year": int(test_year),
                "n": m["n"],
                "mean": m["mean"],
                "sharpe": m["sharpe"],
                "hit_rate": m["hit_rate"],
            }
        )
        logger.info(
            "WALK-FORWARD FILTERED PERFORMANCE | year=%d | n=%5d"
            " | mean=%+.6f | sharpe=%+.4f | hit_rate=%.4f",
            test_year,
            m["n"],
            m["mean"],
            m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
            m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
        )

    if not rows:
        logger.warning(
            "WALK-FORWARD FILTERED PERFORMANCE: no valid years after regime filter"
        )
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# COVERAGE SUMMARY — how many signals survive the regime filter
# ---------------------------------------------------------------------------

def compute_coverage_summary(
    df: pd.DataFrame,
    top_regimes: list[str] | None = None,
    *,
    regime_col: str = "crowding_regime",
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Compute coverage diagnostics for the regime filter.

    Counts total signals (rows with non-null *target_col*), filtered signals
    (rows also passing the regime filter), and the coverage ratio.

    Args:
        df: Dataset (after ``build_behavioural_regimes``).
        top_regimes: Regime labels to include.  Defaults to ``TOP_REGIMES``.
        regime_col: Column containing regime labels (default
            ``"crowding_regime"``).
        target_col: Target column used to determine signal eligibility.

    Returns:
        Single-row DataFrame with schema
        ``["total_signals", "filtered_signals", "coverage_ratio"]``.
        This is the ``coverage_summary`` output artifact.
    """
    _COLS = ["total_signals", "filtered_signals", "coverage_ratio"]

    if top_regimes is None:
        top_regimes = TOP_REGIMES

    valid = df.dropna(subset=[target_col])
    total = len(valid)

    if regime_col not in df.columns:
        logger.warning(
            "compute_coverage_summary: '%s' column not found; filtered=0",
            regime_col,
        )
        return pd.DataFrame(
            [{"total_signals": total, "filtered_signals": 0, "coverage_ratio": 0.0}]
        )[_COLS]

    filtered = int(valid[regime_col].isin(top_regimes).sum())
    coverage_ratio = filtered / total if total > 0 else 0.0

    logger.info(
        "Coverage: %.1f%% of signals retained after regime filter "
        "(%d / %d signals; top_regimes=%s)",
        coverage_ratio * 100,
        filtered,
        total,
        top_regimes,
    )
    return pd.DataFrame(
        [
            {
                "total_signals": total,
                "filtered_signals": filtered,
                "coverage_ratio": coverage_ratio,
            }
        ]
    )[_COLS]


# ---------------------------------------------------------------------------
# Logging helpers for filtered-signal pipeline sections
# ---------------------------------------------------------------------------

def log_full_dataset_performance(full_summary: pd.DataFrame) -> None:
    """Log full-dataset performance under the FULL DATASET PERFORMANCE header.

    Args:
        full_summary: DataFrame returned by ``full_dataset_performance``.
    """
    logger.info("=== FULL DATASET PERFORMANCE ===")
    if full_summary.empty:
        logger.warning("FULL DATASET PERFORMANCE: no results to display")
        return
    row = full_summary.iloc[0]
    logger.info(
        "FULL DATASET | n=%5d | mean=%+.6f | std=%.6f"
        " | sharpe=%+.4f | hit_rate=%.4f",
        int(row["n"]),
        row["mean"],
        row["std"],
        row["sharpe"] if not np.isnan(row["sharpe"]) else float("nan"),
        row["hit_rate"] if not np.isnan(row["hit_rate"]) else float("nan"),
    )


def log_filtered_performance(filtered_summary: pd.DataFrame) -> None:
    """Log filtered-dataset performance under the FILTERED PERFORMANCE header.

    Args:
        filtered_summary: DataFrame returned by ``filtered_regime_baseline``.
    """
    logger.info("=== FILTERED PERFORMANCE ===")
    if filtered_summary.empty:
        logger.warning("FILTERED PERFORMANCE: no results to display")
        return
    row = filtered_summary.iloc[0]
    logger.info(
        "FILTERED | n=%5d | mean=%+.6f | std=%.6f"
        " | sharpe=%+.4f | hit_rate=%.4f",
        int(row["n"]),
        row["mean"],
        row["std"],
        row["sharpe"] if not np.isnan(row["sharpe"]) else float("nan"),
        row["hit_rate"] if not np.isnan(row["hit_rate"]) else float("nan"),
    )


def log_filtered_wf(filtered_wf: pd.DataFrame) -> None:
    """Log filtered walk-forward results under WALK-FORWARD FILTERED PERFORMANCE.

    Args:
        filtered_wf: DataFrame returned by ``filtered_regime_walk_forward``.
    """
    logger.info("=== WALK-FORWARD FILTERED PERFORMANCE ===")
    if filtered_wf.empty:
        logger.warning("WALK-FORWARD FILTERED PERFORMANCE: no results to display")
        return
    for row in filtered_wf.itertuples(index=False):
        logger.info(
            "WALK-FORWARD FILTERED | year=%d | n=%5d"
            " | mean=%+.6f | sharpe=%+.4f | hit_rate=%.4f",
            row.year,
            row.n,
            row.mean,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
        )


def log_coverage_summary(coverage_summary: pd.DataFrame) -> None:
    """Log coverage diagnostics under the COVERAGE SUMMARY header.

    Args:
        coverage_summary: DataFrame returned by ``compute_coverage_summary``.
    """
    logger.info("=== COVERAGE SUMMARY ===")
    if coverage_summary.empty:
        logger.warning("COVERAGE SUMMARY: no results to display")
        return
    row = coverage_summary.iloc[0]
    logger.info(
        "Coverage: %.1f%% of signals retained after regime filter"
        " | total=%d | filtered=%d",
        float(row["coverage_ratio"]) * 100,
        int(row["total_signals"]),
        int(row["filtered_signals"]),
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

    # Step 2: Discretise features into vol/trend regimes BEFORE any modeling.
    df = build_regimes(df)

    # Step 2b: Discretise sentiment features into behavioural regimes.
    df = build_behavioural_regimes(df)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: REGIME BASELINE (NO MODEL) — full-dataset regime discovery
    # ------------------------------------------------------------------
    regime_summary = regime_baseline(df)
    log_regime_baseline(regime_summary)

    # ------------------------------------------------------------------
    # Step 3b: BEHAVIOURAL REGIME SUMMARY — full-dataset behavioural
    #          discovery (no model)
    # ------------------------------------------------------------------
    behavioural_summary = behavioural_regime_baseline(df)
    log_behavioural_regime_summary(behavioural_summary)

    # ------------------------------------------------------------------
    # Step 3c: CROWDING REGIME SUMMARY — simplified 2-axis regime discovery
    # ------------------------------------------------------------------
    crowding_summary = crowding_regime_baseline(df)
    log_crowding_regime_summary(crowding_summary)

    # ------------------------------------------------------------------
    # Step 4: REGIME WALK-FORWARD — out-of-sample regime validation
    # ------------------------------------------------------------------
    regime_wf = regime_walk_forward(df)
    log_regime_wf(regime_wf)

    # ------------------------------------------------------------------
    # Step 4b: WALK-FORWARD REGIME PERFORMANCE — out-of-sample
    #          behavioural regime validation
    # ------------------------------------------------------------------
    behavioural_wf = behavioural_regime_walk_forward(df)
    log_behavioural_regime_wf(behavioural_wf)

    # ------------------------------------------------------------------
    # Step 4c: CROWDING REGIME WALK-FORWARD — out-of-sample crowding
    #          regime validation (simplified 2-axis)
    # ------------------------------------------------------------------
    crowding_wf = crowding_regime_walk_forward(df)
    log_crowding_regime_wf(crowding_wf)

    # ------------------------------------------------------------------
    # Step 4d: SECONDARY VOL FILTER — secondary conditioning on top
    #          crowding regimes (no combinatorial regime expansion)
    # ------------------------------------------------------------------
    vol_filter_results = secondary_vol_filter(df, crowding_summary)
    log_secondary_vol_filter(vol_filter_results)

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
