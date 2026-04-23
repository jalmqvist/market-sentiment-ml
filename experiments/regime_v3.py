"""
experiments/regime_v3.py
========================
LightGBM walk-forward experiment: predict ``ret_48b`` from
sentiment + volatility + trend features with no lookahead leakage.

The walk-forward loop uses the same expanding-window discipline as
``evaluation.walk_forward.walk_forward_expanding``: for each test year
(starting from the third unique year in the dataset), the model is trained
on all prior years and evaluated on the current year.

All features are strictly causal at ``entry_time``:

* **Sentiment** – ``net_sentiment``, ``abs_sentiment``, ``sentiment_change``,
  ``side_streak``, ``extreme_streak_70``, ``extreme_streak_80``
* **Trend** – ``trend_strength_12b``, ``trend_strength_48b``,
  ``trend_dir_12b``, ``trend_dir_48b`` (backward-looking past-price
  features already present in the dataset)
* **Volatility** – ``vol_24b``: rolling 24-bar std of bar-to-bar returns
  derived from ``entry_close``, computed per pair using only past bars
* **Interaction** – ``abs_sent_x_trend12b``, ``abs_sent_x_trend48b``,
  ``abs_sent_x_vol24b``, ``extreme70_x_trend48b``: products of base
  features that capture non-linear regime signals

Only columns listed in ``SAFE_FEATURES`` are ever used as model inputs.
Any column matching ``ret_*`` or ``contrarian_ret_*`` is explicitly
prohibited and triggers a ``ValueError`` if accidentally included.

The experiment also evaluates **conditional performance by regime**.  Each
observation is assigned to one of up to nine regimes formed by crossing a
three-way volatility bucket (``vol_regime``: low / mid / high, based on
tertiles of ``vol_24b``) with the sign of ``trend_strength_48b``
(``trend_sign``: −1 / 0 / +1).  The combined label is stored in the
``regime`` column as, e.g., ``"low_-1.0"`` or ``"high_1.0"``.

Per-regime metrics (IC, Sharpe, hit rate, sample size) are computed for
each walk-forward test fold and pooled across all folds.

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

    Creates three columns:

    * ``vol_regime`` – tertile bucket of ``vol_24b`` (``"low"`` / ``"mid"``
      / ``"high"``).  Quantile thresholds are computed on the full dataset;
      ``NaN`` rows are left as ``NaN``.
    * ``trend_sign`` – ``np.sign(trend_strength_48b)``: ``-1.0``, ``0.0``,
      or ``1.0``.
    * ``regime`` – combined label, e.g. ``"low_-1.0"`` or ``"high_1.0"``.

    Args:
        df: Dataset (after ``build_features``).  Must contain ``vol_24b``
            and ``trend_strength_48b`` for full regime assignment; missing
            columns are handled gracefully with a warning.

    Returns:
        Copy of *df* with the three regime columns added.
    """
    out = df.copy()

    if "vol_24b" not in out.columns:
        logger.warning("build_regimes: vol_24b not found; vol_regime will be NaN")
        out["vol_regime"] = np.nan
    else:
        valid_mask = out["vol_24b"].notna()
        out.loc[valid_mask, "vol_regime"] = pd.qcut(
            out.loc[valid_mask, "vol_24b"],
            q=3,
            labels=["low", "mid", "high"],
        )
        n_assigned = int(valid_mask.sum())
        logger.debug("build_regimes: vol_regime assigned for %d rows", n_assigned)

    if "trend_strength_48b" not in out.columns:
        logger.warning(
            "build_regimes: trend_strength_48b not found; trend_sign will be 0"
        )
        out["trend_sign"] = 0.0
    else:
        out["trend_sign"] = np.sign(out["trend_strength_48b"])

    out["regime"] = out["vol_regime"].astype(str) + "_" + out["trend_sign"].astype(str)
    # Rows where vol_24b was NaN produce 'nan_...' labels; mark these as NaN.
    out.loc[out["vol_24b"].isna(), "regime"] = np.nan
    n_regimes = out["regime"].nunique()
    logger.info("build_regimes: %d unique regime labels", n_regimes)
    return out


# ---------------------------------------------------------------------------
# Per-regime metrics helper
# ---------------------------------------------------------------------------

def _regime_metrics(y_pred: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute IC, Sharpe and hit rate for a single regime subset.

    Args:
        y_pred: Model predictions for the subset.
        y_test: Realised returns for the subset.

    Returns:
        Dict with keys ``n``, ``ic``, ``signal_sharpe``, ``signal_hit_rate``.
    """
    n = len(y_pred)
    if n < 2:
        return {
            "n": n,
            "ic": np.nan,
            "signal_sharpe": np.nan,
            "signal_hit_rate": np.nan,
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
        "signal_sharpe": sharpe,
        "signal_hit_rate": hit_rate,
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

        # Per-regime metrics for this fold.
        if regime_col is not None and regime_col in test.columns:
            regime_labels = test[regime_col].values
            for regime_label in np.unique(regime_labels[pd.notna(regime_labels)]):
                mask = regime_labels == regime_label
                m = _regime_metrics(y_pred[mask], y_test[mask])
                regime_results.append(
                    {
                        "year": int(test_year),
                        "regime": str(regime_label),
                        **m,
                    }
                )
                logger.debug(
                    "year=%d | regime=%s | n=%d | IC=%.4f | Sharpe=%.4f | hit_rate=%.4f",
                    test_year,
                    regime_label,
                    m["n"],
                    m["ic"] if not np.isnan(m["ic"]) else float("nan"),
                    m["signal_sharpe"] if not np.isnan(m["signal_sharpe"]) else float("nan"),
                    m["signal_hit_rate"] if not np.isnan(m["signal_hit_rate"]) else float("nan"),
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
    display_cols = ["year", "regime", "n", "ic", "signal_sharpe", "signal_hit_rate"]
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
                "n": int(grp["n"].sum()),
                "ic": _wmean(grp, "ic"),
                "signal_sharpe": _wmean(grp, "signal_sharpe"),
                "signal_hit_rate": _wmean(grp, "signal_hit_rate"),
                "folds": len(grp),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values("regime").reset_index(drop=True)

    print("\n=== POOLED REGIME METRICS (across all folds) ===")
    print(pooled.to_string(index=False))
    logger.info("Pooled regime metrics computed for %d regimes", len(pooled))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V3: Ridge regression walk-forward predicting ret_48b "
            "from sentiment + volatility + trend features (no leakage)."
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

    # Compute causal volatility feature (vol_24b).
    df = build_features(df)

    # Classify each observation into a vol × trend regime.
    df = build_regimes(df)

    # Determine which feature columns are available in this dataset.
    feature_cols = select_features(df)

    if not feature_cols:
        print("ERROR: No valid feature columns found in dataset. Exiting.")
        sys.exit(1)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # Expanding-window Ridge walk-forward with per-regime evaluation.
    wf_results, regime_results = walk_forward_ridge(
        df, feature_cols, regime_col="regime"
    )

    print_wf_summary(wf_results)
    print_regime_summary(regime_results)


if __name__ == "__main__":
    main()
