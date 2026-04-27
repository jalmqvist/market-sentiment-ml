# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v8.py
========================
MODEL-BASED signal pipeline for FX sentiment research — **Version 8.3**.

Replaces handcrafted scoring with a learned alpha function: a LightGBM
regressor trained to predict ``ret_48b`` from sentiment and market features.
V8.1 extends V8 with **prediction ranking and top-k selection**: only the
strongest signals (by absolute prediction magnitude) are traded each fold,
reducing noise and improving Sharpe.
V8.2 extends V8.1 with **continuous position sizing**: instead of binary
sign-based positions, prediction magnitude is normalised and clipped to scale
each position continuously, improving Sharpe and smoothing PnL.
V8.3 extends V8.2 with **interaction features** that expose the conditional
signal structure (sentiment extremes × trend context × persistence) so that
LightGBM can learn it directly rather than approximating it from independent
raw inputs.

Pipeline overview
-----------------
1. **Features** – columns drawn from the dataset:

   Core (required):

   * ``net_sentiment``
   * ``abs_sentiment``
   * ``extreme_streak_70``
   * ``trend_strength_48b``

   Optional (used when present):

   * ``divergence``
   * ``signal_v2_raw``
   * ``sent_x_trend``      – net_sentiment × trend_strength_48b
   * ``extreme_x_trend``   – is_extreme × trend_strength_48b
   * ``streak_x_sent``     – extreme_streak_70 × net_sentiment
   * ``streak_x_trend``    – extreme_streak_70 × trend_strength_48b

2. **Walk-forward** (expanding window, minimum 3 years):

   For each test year:

   * ``train`` = all rows from all prior years
   * ``test``  = all rows in the current year

   Both splits are restricted to rows without NaN in any feature or target.

3. **Model** – LightGBM regressor trained on the train split to predict
   ``ret_48b`` directly.  Predictions on the test split drive positions.

4. **Signal construction (V8.3)**::

       pred      = model.predict(X_test)
       pred_std  = std(pred)
       scaled_pred = pred / pred_std  if pred_std > 1e-10 else pred
       scaled_pred = clip(scaled_pred, -3, 3)
       score     = abs(pred)
       threshold = quantile(score, 1 - top_frac)   # default top_frac=0.2
       position  = where(score >= threshold, scaled_pred, 0)
       pnl       = position * ret_48b   # only non-zero positions included

5. **Metrics** (per fold): n (traded), mean return, Sharpe, hit_rate, IC
   (Spearman correlation between predictions and ``ret_48b`` over all rows).

Required columns
----------------
``ret_48b``, ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``

Fold output schema
------------------
``["year", "n", "mean", "sharpe", "hit_rate", "ic"]``

Logging (per fold)
------------------
* Train size / test size
* Selection: top_frac, threshold, coverage
* Mean |pred| selected vs full
* Position stats: mean, std
* Sharpe, hit rate, IC

Usage::

    python experiments/regime_v8.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v8.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3 \\
        --log-level DEBUG
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
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

import config as cfg
from utils.io import read_csv
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Core features – always required.
CORE_FEATURES: list[str] = [
    "net_sentiment",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
]

#: Optional features – used when present in the dataset.
OPTIONAL_FEATURES: list[str] = [
    "divergence",
    "signal_v2_raw",
    "sent_x_trend",
    "extreme_x_trend",
    "streak_x_sent",
    "streak_x_trend",
]

#: Full candidate feature list (core + optional).
FEATURE_COLS: list[str] = CORE_FEATURES + OPTIONAL_FEATURES

#: Required columns for the pipeline (validated at entry points).
_REQUIRED_COLS: list[str] = [TARGET_COL] + CORE_FEATURES

#: Output fold columns.
_FOLD_COLS: list[str] = ["year", "n", "mean", "sharpe", "hit_rate", "ic"]

#: Absolute-sentiment threshold above which a reading is flagged as extreme (V8.3).
_EXTREME_SENTIMENT_THRESHOLD: int = 70

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# LightGBM model parameters
# ---------------------------------------------------------------------------

_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is ``None``, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        log_path = logs_dir / f"regime_v8_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logging.getLogger(__name__).info("File logging enabled: %s", log_path)


# ---------------------------------------------------------------------------
# Interaction feature engineering (V8.3)
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features that expose conditional signal structure.

    Creates four interaction feature columns (plus one intermediate) when their
    source columns are available:

    * ``sent_x_trend``    – net_sentiment × trend_strength_48b
    * ``is_extreme``      – 1 when abs_sentiment >= 70, else 0 (intermediate; not in OPTIONAL_FEATURES)
    * ``extreme_x_trend`` – is_extreme × trend_strength_48b
    * ``streak_x_sent``   – extreme_streak_70 × net_sentiment
    * ``streak_x_trend``  – extreme_streak_70 × trend_strength_48b

    Infinite values are replaced with NaN so that downstream ``dropna`` in
    the walk-forward step handles them cleanly.  No scaling is applied;
    LightGBM handles the magnitudes natively.

    Args:
        df: Dataset (must already have a ``year`` column from :func:`load_data`).

    Returns:
        The same DataFrame with interaction columns added in-place.
    """
    has_net = "net_sentiment" in df.columns
    has_abs = "abs_sentiment" in df.columns
    has_streak = "extreme_streak_70" in df.columns
    has_trend = "trend_strength_48b" in df.columns

    if has_net and has_trend:
        df["sent_x_trend"] = df["net_sentiment"] * df["trend_strength_48b"]
        df["sent_x_trend"] = df["sent_x_trend"].replace([np.inf, -np.inf], np.nan)

    if has_abs and has_trend:
        df["is_extreme"] = (df["abs_sentiment"] >= _EXTREME_SENTIMENT_THRESHOLD).astype(int)
        df["extreme_x_trend"] = df["is_extreme"] * df["trend_strength_48b"]
        df["extreme_x_trend"] = df["extreme_x_trend"].replace([np.inf, -np.inf], np.nan)

    if has_streak and has_net:
        df["streak_x_sent"] = df["extreme_streak_70"] * df["net_sentiment"]
        df["streak_x_sent"] = df["streak_x_sent"].replace([np.inf, -np.inf], np.nan)

    if has_streak and has_trend:
        df["streak_x_trend"] = df["extreme_streak_70"] * df["trend_strength_48b"]
        df["streak_x_trend"] = df["streak_x_trend"].replace([np.inf, -np.inf], np.nan)

    logger.info("Added interaction features")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the dataset for Regime V8.

    Reads the canonical research dataset CSV, parses the ``time`` column to
    derive a ``year`` column, and validates that all required columns are
    present.  Rows with NaN in any required column are **not** dropped here;
    that is deferred to the walk-forward step so callers see the full dataset.

    Args:
        path: Path to the master research dataset CSV.

    Returns:
        DataFrame with a ``year`` column added.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing after loading.
    """
    df = read_csv(path, required_columns=["time"] + _REQUIRED_COLS)
    df = parse_timestamps(df, "time", context="regime_v8.load_data")
    df["year"] = df["time"].dt.year

    logger.info(
        "load_data: %d rows, %d pairs, date_range=%s .. %s",
        len(df),
        df["pair"].nunique() if "pair" in df.columns else "N/A",
        df["time"].min(),
        df["time"].max(),
    )

    df = add_interaction_features(df)
    return df


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_cols: list[str] | None = None,
    year_col: str = "year",
    top_frac: float = 0.2,
) -> pd.DataFrame:
    """Regime-V8.3 walk-forward: LightGBM model-based signal pipeline with continuous position sizing.

    For each test year (from the third unique year onward):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any feature or target column.
    3. Train a LightGBM regressor on the train split to predict *target_col*.
    4. Predict on the test split; normalise and clip predictions to produce
       continuous positions, then zero out the bottom ``(1 - top_frac)``
       fraction by absolute prediction magnitude.
    5. Compute fold-level metrics using only traded (non-zero) positions.

    No test-period information enters model training.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            listed in ``_REQUIRED_COLS``.
        target_col: Forward-return column to predict.
        feature_cols: Feature columns used for training and prediction.
        year_col: Column containing calendar year.
        top_frac: Fraction of predictions to trade, ranked by absolute
            prediction magnitude.  Default is 0.2 (top 20%).

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
    if feature_cols is None:
        raise ValueError(
            "walk_forward: feature_cols must be provided explicitly. "
            "Build available_features from CORE_FEATURES + present OPTIONAL_FEATURES "
            "and pass it as feature_cols=available_features."
        )

    if year_col not in df.columns:
        logger.warning("walk_forward: year column '%s' not found", year_col)
        return pd.DataFrame(columns=_FOLD_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "walk_forward: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    all_cols = feature_cols + [target_col]
    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year].dropna(subset=all_cols)
        test_df = df[df[year_col] == test_year].dropna(subset=all_cols)

        if train_df.empty:
            logger.warning(
                "REGIME V8.1 [year=%d]: empty train set; skipping fold", test_year
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V8.1 [year=%d]: empty test set; skipping fold", test_year
            )
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values.astype(float)
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values.astype(float)

        logger.info(
            "REGIME V8.3 [year=%d] | train_size=%d | test_size=%d",
            test_year,
            len(train_df),
            len(test_df),
        )

        # Safety check: all feature columns must exist in df
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"walk_forward: feature column(s) missing from df: {missing_cols}"
            )

        # Train model on train split only (no forward leakage)
        model = LGBMRegressor(**_LGBM_PARAMS)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # Normalise prediction strength and clip extremes (V8.3)
        pred_std = np.std(pred)
        if pred_std > 1e-10:
            scaled_pred = pred / pred_std
        else:
            logger.warning(
                "REGIME V8.3 [year=%d]: pred_std=%.2e near zero; "
                "using unscaled predictions (clip will bound positions)",
                test_year,
                pred_std,
            )
            scaled_pred = pred
        scaled_pred = np.clip(scaled_pred, -3, 3)

        score = np.abs(pred)
        threshold = np.quantile(score, 1 - top_frac)
        position = np.where(score >= threshold, scaled_pred, 0)

        coverage = float(np.mean(score >= threshold))

        logger.info(
            "Selection: top_frac=%.2f | threshold=%.6f | coverage=%.2f%%",
            top_frac,
            threshold,
            100 * coverage,
        )
        if coverage > 0:
            mean_score_selected = float(np.mean(score[score >= threshold]))
            mean_score_full = float(np.mean(score))
            logger.info(
                "Mean |pred|: selected=%.6f | full=%.6f",
                mean_score_selected,
                mean_score_full,
            )
        else:
            logger.warning(
                "REGIME V8.3 [year=%d]: coverage=0; no positions taken, skipping fold",
                test_year,
            )
            continue

        logger.info(
            "Position stats: mean=%.4f | std=%.4f",
            np.mean(position),
            np.std(position),
        )

        active_mask = position != 0
        pnl = position[active_mask] * y_test[active_mask]

        metrics = _fold_metrics(pnl, pred, y_test)

        logger.info(
            "REGIME V8.3 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | ic=%+.4f",
            test_year,
            metrics["n"],
            metrics["mean"],
            metrics["sharpe"],
            metrics["hit_rate"],
            metrics["ic"],
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V8.3: no valid folds produced")
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    pnl: np.ndarray,
    pred: np.ndarray,
    ret: np.ndarray,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V8.

    Args:
        pnl: PnL series (``position * ret_48b``).
        pred: Raw model predictions.
        ret: Actual ``ret_48b`` values.

    Returns:
        Dict with keys: n, mean, sharpe, hit_rate, ic.
    """
    n = len(pnl)

    if n < 2:
        return {
            "n": n,
            "mean": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "ic": float("nan"),
        }

    mean_ret = float(np.mean(pnl))
    std_ret = float(np.std(pnl))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
    hit_rate = float(np.mean(pnl > 0))

    # IC: Spearman correlation between predictions and realized returns
    ic = float(spearmanr(pred, ret).statistic)

    return {
        "n": n,
        "mean": mean_ret,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "ic": ic,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with keys: folds, mean_sharpe, mean_hit_rate, mean_ic.
    """
    if fold_df.empty:
        return {
            "folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_ic": float("nan"),
        }
    return {
        "folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_ic": float(fold_df["ic"].dropna().mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V8.3 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V8.3: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | ic=%+.4f",
            row.year,
            row.n,
            row.mean,
            row.sharpe,
            row.hit_rate,
            row.ic,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the Regime V8 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        summary: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V8.3 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V8.3 SUMMARY: no results")
        return

    logger.info("Folds evaluated  : %d", summary["folds"])
    logger.info("Mean Sharpe      : %+.4f", summary["mean_sharpe"])
    logger.info("Mean hit rate    : %.4f", summary["mean_hit_rate"])
    logger.info("Mean IC          : %+.4f", summary["mean_ic"])
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | sharpe=%+.4f | hit_rate=%.4f | ic=%+.4f",
            row.year,
            row.sharpe,
            row.hit_rate,
            row.ic,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v8.py)
# ---------------------------------------------------------------------------

def _top_frac_arg(value: str) -> float:
    """Argparse type for ``--top-frac``: validates the value is in (0, 1]."""
    try:
        frac = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--top-frac must be a float, got: {value!r}")
    if not (0 < frac <= 1):
        raise argparse.ArgumentTypeError(
            f"--top-frac must be in (0, 1], got: {frac}"
        )
    return frac


def main(argv: list[str] | None = None) -> None:
    """Run the Regime V8 model-based signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V8.3: model-based signal pipeline with continuous position sizing "
            "and top-k filtering. "
            "Trains LightGBM to predict ret_48b from sentiment and market "
            "features using walk-forward validation, normalises prediction "
            "magnitude into continuous positions, and trades only the top "
            "top_frac of predictions ranked by absolute magnitude."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
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
        "--top-frac",
        type=_top_frac_arg,
        default=0.2,
        metavar="FRAC",
        help=(
            "Fraction of predictions to trade per fold, ranked by absolute "
            "prediction magnitude.  Must be in (0, 1].  Default is 0.2 "
            "(top 20%%)."
        ),
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    log = logging.getLogger(__name__)
    log.info("=== REGIME V8.3 ===")

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v8.main")
    log.info("Dataset ready: %d rows", len(df))

    # Build feature list dynamically: core (always used) + optional (if present)
    missing_optional = [c for c in OPTIONAL_FEATURES if c not in df.columns]
    if missing_optional:
        logger.warning("Missing optional features: %s", missing_optional)
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    if len(available_features) < 2:
        raise ValueError("Not enough features available for model")
    logger.info("Using features: %s", available_features)

    fold_df = walk_forward(df, feature_cols=available_features, top_frac=args.top_frac)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
