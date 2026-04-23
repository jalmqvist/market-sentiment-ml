"""
experiments/regime_v3.py
========================
Ridge regression walk-forward experiment: predict ``ret_48b`` from
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

Features columns present in the dataset are used; missing columns emit a
warning and are silently dropped.

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

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import config as cfg
from evaluation.walk_forward import wf_summary
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Candidate regularisation strengths for RidgeCV (leave-one-out CV).
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

#: Minimum training observations before a Ridge model is fit.
MIN_TRAIN_OBS: int = 50

#: Sentiment features – all available at ``snapshot_time`` (causal).
SENTIMENT_FEATURES: list[str] = [
    "net_sentiment",
    "abs_sentiment",
    "sentiment_change",
    "side_streak",
    "extreme_streak_70",
    "extreme_streak_80",
]

#: Trend features – backward-looking past-price columns (causal).
TREND_FEATURES: list[str] = [
    "trend_strength_12b",
    "trend_strength_48b",
    "trend_dir_12b",
    "trend_dir_48b",
]

#: Volatility feature added by ``build_features``.
VOLATILITY_FEATURES: list[str] = ["vol_24b"]

#: Interaction features added by ``build_features`` (products of base features).
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
    """Add causal volatility and interaction features.

    ``vol_24b`` is the rolling 24-bar standard deviation of bar-to-bar
    ``entry_close`` returns within each pair.  Because the rolling window
    references only past observations (``min_periods=5``), there is no
    lookahead.

    Interaction features are products of base features computed after
    ``vol_24b`` is available.  Each interaction is only created when both
    constituent columns are present; missing base columns are silently skipped
    (a debug message is emitted).

    Args:
        df: Dataset with ``pair``, ``entry_time``, and ``entry_close``.

    Returns:
        Copy of *df* with ``vol_24b`` and available interaction columns added.
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

    Combines SENTIMENT_FEATURES + TREND_FEATURES + VOLATILITY_FEATURES and
    filters to columns that actually exist.  Missing candidates emit a
    warning.

    Args:
        df: Dataset (after ``build_features``).

    Returns:
        List of available feature column names.
    """
    candidates = SENTIMENT_FEATURES + TREND_FEATURES + VOLATILITY_FEATURES + INTERACTION_FEATURES
    feature_cols = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        logger.warning("Feature columns not in dataset (skipped): %s", missing)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)
    return feature_cols


# ---------------------------------------------------------------------------
# Walk-forward Ridge regression
# ---------------------------------------------------------------------------

def walk_forward_ridge(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    min_train_obs: int = MIN_TRAIN_OBS,
) -> pd.DataFrame:
    """Expanding-window walk-forward with Ridge regression.

    Follows the same discipline as
    ``evaluation.walk_forward.walk_forward_expanding``: for each test year
    (starting from index 2 in the sorted unique-year list), the model is
    trained on *all prior years* and evaluated on the current year.

    Feature scaling (``StandardScaler``) is fit on training data only and
    applied to test data – no leakage.

    Args:
        df: Dataset with feature columns, *target_col*, and *year_col*.
        feature_cols: Columns to use as predictors.
        target_col: Regression target column (default ``ret_48b``).
        year_col: Column containing the calendar year.
        alphas: Regularisation strengths for ``RidgeCV``.
        min_train_obs: Minimum training observations required per fold.

    Returns:
        DataFrame with one row per test fold:
        year, n_train, n_test, alpha, ic, signal_sharpe, signal_hit_rate, r2.
        Empty DataFrame if no valid folds.
    """
    require_columns(
        df, [year_col, target_col] + feature_cols, context="walk_forward_ridge"
    )

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "walk_forward_ridge: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame()

    results = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_years = years[:i]

        train = df[df[year_col].isin(train_years)].dropna(
            subset=feature_cols + [target_col]
        )
        test = df[df[year_col] == test_year].dropna(
            subset=feature_cols + [target_col]
        )

        if len(train) < min_train_obs:
            logger.debug(
                "Skipping test_year=%d: %d train rows (need %d)",
                test_year,
                len(train),
                min_train_obs,
            )
            continue

        if len(test) == 0:
            logger.debug("Skipping test_year=%d: empty test set", test_year)
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        # Fit scaler on training data only (no leakage).
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # RidgeCV selects alpha via leave-one-out CV on the training set.
        model = RidgeCV(alphas=alphas)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

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
                "alpha": float(model.alpha_),
                "ic": float(ic) if not np.isnan(ic) else np.nan,
                "signal_sharpe": sharpe,
                "signal_hit_rate": hit_rate,
                "r2": float(r2),
            }
        )

        logger.debug(
            "year=%d | train=%d | test=%d | alpha=%.2f | IC=%.4f | Sharpe=%.4f | R2=%.4f",
            test_year,
            len(train),
            len(test),
            model.alpha_,
            ic,
            sharpe if not np.isnan(sharpe) else float("nan"),
            r2 if not np.isnan(r2) else float("nan"),
        )

    if not results:
        logger.warning("walk_forward_ridge: no valid folds produced")
        return pd.DataFrame()

    return pd.DataFrame(results)


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
    print(
        f"Dataset summary: {len(df):,} rows | "
        f"{df['pair'].nunique()} unique pairs | "
        f"{date_min} to {date_max}"
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

    print("\n=== WALK-FORWARD RESULTS (Ridge → ret_48b) ===")
    display_cols = [
        "year",
        "n_train",
        "n_test",
        "alpha",
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

    # Determine which feature columns are available in this dataset.
    feature_cols = select_features(df)

    if not feature_cols:
        print("ERROR: No valid feature columns found in dataset. Exiting.")
        sys.exit(1)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # Expanding-window Ridge walk-forward.
    wf_results = walk_forward_ridge(df, feature_cols)

    print_wf_summary(wf_results)


if __name__ == "__main__":
    main()
