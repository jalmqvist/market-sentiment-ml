"""
experiments/regime_v8.py
========================
MODEL-BASED signal pipeline for FX sentiment research.

Replaces handcrafted scoring with a learned alpha function: a LightGBM
regressor trained to predict ``ret_48b`` from sentiment and market features.
Positions are taken as ``sign(prediction)``; walk-forward evaluation is used
to prevent forward leakage.

Pipeline overview
-----------------
1. **Features** – six columns already present in the dataset:

   * ``net_sentiment``
   * ``abs_sentiment``
   * ``extreme_streak_70``
   * ``trend_strength_48b``
   * ``divergence``
   * ``signal_v2_raw``

2. **Walk-forward** (expanding window, minimum 3 years):

   For each test year:

   * ``train`` = all rows from all prior years
   * ``test``  = all rows in the current year

   Both splits are restricted to rows without NaN in any feature or target.

3. **Model** – LightGBM regressor trained on the train split to predict
   ``ret_48b`` directly.  Predictions on the test split drive positions.

4. **Signal construction**::

       pred     = model.predict(X_test)
       position = sign(pred)          # −1, 0 (exactly zero pred), or +1
       pnl      = position * ret_48b

5. **Metrics** (per fold): n, mean return, Sharpe, hit_rate, IC
   (Spearman correlation between predictions and ``ret_48b``).

Required columns
----------------
``ret_48b``, ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``, ``divergence``, ``signal_v2_raw``

Fold output schema
------------------
``["year", "n", "mean", "sharpe", "hit_rate", "ic"]``

Logging (per fold)
------------------
* Train size / test size
* Sharpe
* Hit rate
* IC

Usage::

    python experiments/regime_v8.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v8.py \\
        --data data/output/master_research_dataset.csv \\
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

#: Features used to train and predict.
FEATURE_COLS: list[str] = [
    "net_sentiment",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
    "divergence",
    "signal_v2_raw",
]

#: Required columns for the pipeline (validated at entry points).
_REQUIRED_COLS: list[str] = [TARGET_COL] + FEATURE_COLS

#: Output fold columns.
_FOLD_COLS: list[str] = ["year", "n", "mean", "sharpe", "hit_rate", "ic"]

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
    return df


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_cols: list[str] = FEATURE_COLS,
    year_col: str = "year",
) -> pd.DataFrame:
    """Regime-V8 walk-forward: LightGBM model-based signal pipeline.

    For each test year (from the third unique year onward):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any feature or target column.
    3. Train a LightGBM regressor on the train split to predict *target_col*.
    4. Predict on the test split; compute positions as ``sign(pred)``.
    5. Compute fold-level metrics (n, mean, Sharpe, hit_rate, IC).

    No test-period information enters model training.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            listed in ``_REQUIRED_COLS``.
        target_col: Forward-return column to predict.
        feature_cols: Feature columns used for training and prediction.
        year_col: Column containing calendar year.

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
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
                "REGIME V8 [year=%d]: empty train set; skipping fold", test_year
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V8 [year=%d]: empty test set; skipping fold", test_year
            )
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values.astype(float)
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values.astype(float)

        logger.info(
            "REGIME V8 [year=%d] | train_size=%d | test_size=%d",
            test_year,
            len(train_df),
            len(test_df),
        )

        # Train model on train split only (no forward leakage)
        model = LGBMRegressor(**_LGBM_PARAMS)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        position = np.sign(pred)
        pnl = position * y_test

        metrics = _fold_metrics(pnl, pred, y_test)

        logger.info(
            "REGIME V8 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | ic=%+.4f",
            test_year,
            metrics["n"],
            metrics["mean"] if not np.isnan(metrics["mean"]) else float("nan"),
            metrics["sharpe"] if not np.isnan(metrics["sharpe"]) else float("nan"),
            metrics["hit_rate"] if not np.isnan(metrics["hit_rate"]) else float("nan"),
            metrics["ic"] if not np.isnan(metrics["ic"]) else float("nan"),
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V8: no valid folds produced")
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
    corr_result = spearmanr(pred, ret)
    ic = float(corr_result.statistic) if hasattr(corr_result, "statistic") else float(corr_result[0])

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
    logger.info("=== REGIME V8 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V8: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | ic=%+.4f",
            row.year,
            row.n,
            row.mean if not np.isnan(row.mean) else float("nan"),
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.ic if not np.isnan(row.ic) else float("nan"),
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
    logger.info("=== REGIME V8 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V8 SUMMARY: no results")
        return

    logger.info("Folds evaluated  : %d", summary["folds"])
    logger.info(
        "Mean Sharpe      : %+.4f",
        summary["mean_sharpe"]
        if not np.isnan(summary["mean_sharpe"])
        else float("nan"),
    )
    logger.info(
        "Mean hit rate    : %.4f",
        summary["mean_hit_rate"]
        if not np.isnan(summary["mean_hit_rate"])
        else float("nan"),
    )
    logger.info(
        "Mean IC          : %+.4f",
        summary["mean_ic"]
        if not np.isnan(summary["mean_ic"])
        else float("nan"),
    )
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | sharpe=%+.4f | hit_rate=%.4f | ic=%+.4f",
            row.year,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.ic if not np.isnan(row.ic) else float("nan"),
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v8.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V8 model-based signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V8: model-based signal pipeline. "
            "Trains LightGBM to predict ret_48b from sentiment and market "
            "features using walk-forward validation, and evaluates trading "
            "performance using sign(prediction)."
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
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    log = logging.getLogger(__name__)
    log.info("=== REGIME V8 ===")

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v8.main")
    log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(df)

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
