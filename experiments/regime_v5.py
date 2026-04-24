"""
experiments/regime_v5.py
========================
Continuous signal blending: Signal V2 × Regime weight × Behavioral score.

This experiment combines three multiplicative signal layers without any
filtering, thresholding, or discrete regime selection:

1. **Base signal** (Signal V2 raw)

   The composite signal from :mod:`experiments.signal_v2`::

       signal_v2_raw = 1.0 * divergence + 0.5 * shock - 0.75 * exhaustion
       base_signal   = np.tanh(signal_v2_raw)   # bounded to (-1, 1)

2. **Regime score** (from Regime V4 logic)

   Train-only Sharpe per 4-component regime key, converted to a smooth
   continuous weight::

       regime_weight = tanh(sharpe / std_sharpe)
       regime_score  = regime_weight_map.get(regime_key, 0)

   Unknown / ineligible regimes receive a score of 0.

3. **Behavioral score** (NEW)

   Derived from two columns already present in the dataset:

   * ``extreme_streak_70`` → persistence (crowd in extreme zone)
   * ``abs_sentiment``     → saturation (crowd intensity)

   Per-fold z-scores are computed on the **train set only** to avoid leakage.
   The train-set mean and standard deviation are then applied to the test set::

       persistence_z   = zscore_train_then_apply(extreme_streak_70)
       saturation_z    = zscore_train_then_apply(abs_sentiment)

       behavior_score_raw = 0.5 * persistence_z + 0.5 * saturation_z
       behavior_score     = np.tanh(behavior_score_raw)   # output in (-1, 1)

4. **Final position**::

       position = base_signal * regime_score * behavior_score

Pipeline design
---------------
* **No forward leakage**: all regime Sharpes and behavioral z-score parameters
  are computed on training data only; the test set is labelled using those
  training-derived statistics.
* **No filtering**: every test row receives a (possibly zero) position.
* **No thresholding**: no discrete selection or top-k regime logic.
* **Expanding walk-forward**: train on all prior years; test on each subsequent
  year.  Minimum of 3 unique years required.

Fold output schema
------------------
``["year", "n", "mean", "sharpe", "hit_rate", "coverage",
   "avg_regime_score", "avg_behavior_score", "avg_position", "corr_ret"]``

Logging (per fold)
------------------
* avg |regime_score|
* avg |behavior_score|
* avg |position|
* correlation(position, ret_48b)

Usage::

    python experiments/regime_v5.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v5.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 100 --window 96 --log-level DEBUG
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

# Reuse regime_v4 helpers: regime key + weight logic
from experiments.regime_v4 import (
    MIN_REGIME_N,
    SENT_BINS,
    SENT_LABELS,
    _build_regime_key,
    _build_weight_map,
    _compute_regime_stats,
    _compute_train_cuts,
)

# Reuse signal_v2 feature building and signal construction
from experiments.signal_v2 import (
    DEFAULT_WINDOW,
    build_features as _build_signal_v2_features,
    build_signal as _build_signal_v2,
    load_data as _load_signal_v2_data,
)

# Reuse regime_v3 build_features for vol_24b and interaction features
from experiments.regime_v3 import build_features as _build_regime_v3_features

from utils.validation import require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# Output fold columns
_FOLD_COLS: list[str] = [
    "year",
    "n",
    "mean",
    "sharpe",
    "hit_rate",
    "coverage",
    "avg_regime_score",
    "avg_behavior_score",
    "avg_position",
    "corr_ret",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
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
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v5_{timestamp}.log"
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

def load_data(path: str | Path, *, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Load and prepare the dataset for Regime V5.

    Combines the signal_v2 and regime_v3 feature pipelines:

    1. Load via ``signal_v2.load_data`` (adds ``year``, ``timestamp``).
    2. Detect and assign the price column.
    3. Build Signal V2 features (divergence, shock, exhaustion).
    4. Build Regime V3 features (``vol_24b``, interactions).

    Args:
        path: Path to the master research dataset CSV.
        window: Rolling z-score window size in bars (default 96).

    Returns:
        DataFrame ready for walk-forward evaluation.

    Raises:
        ValueError: If required columns are missing.
    """
    df = _load_signal_v2_data(path)

    # --- Robust price column detection (copied from signal_v2 main) ---
    _PRICE_CANDIDATES: list[str] = ["price", "price_end", "entry_close"]
    _VALID_RATIO_THRESHOLD: float = 0.99

    selected_col: str | None = None
    selected_series: pd.Series | None = None

    for candidate in _PRICE_CANDIDATES:
        if candidate not in df.columns:
            continue
        raw = df[candidate]
        converted = pd.to_numeric(raw, errors="coerce")
        total = len(converted)
        valid_ratio = converted.notna().sum() / total if total > 0 else 0.0
        if selected_col is None and valid_ratio >= _VALID_RATIO_THRESHOLD:
            selected_col = candidate
            selected_series = converted

    if selected_col is None:
        raise ValueError(
            "Regime V5: no valid numeric price column found among "
            f"{_PRICE_CANDIDATES}."
        )

    df["price"] = selected_series
    logger.info("load_data: using price column '%s'", selected_col)

    # Build Signal V2 features (divergence / shock / exhaustion)
    df = _build_signal_v2_features(df, window=window)
    logger.info("load_data: signal_v2 features built (%d rows)", len(df))

    # Build Signal V2 composite (creates signal_v2_raw column)
    df = _build_signal_v2(df)
    if "signal_v2_raw" not in df.columns:
        raise ValueError("signal_v2_raw not created in load_data")

    # Build Regime V3 features (vol_24b, interaction columns) —
    # regime_v3.build_features expects entry_time and entry_close.
    if "entry_time" in df.columns and "entry_close" in df.columns:
        df = _build_regime_v3_features(df)
        logger.info("load_data: regime_v3 features built (vol_24b added)")
    else:
        logger.warning(
            "load_data: 'entry_time' or 'entry_close' missing; "
            "vol_24b / interaction features will be absent — "
            "regime key components may be NaN"
        )

    return df


# ---------------------------------------------------------------------------
# Behavioral score (train-only z-score parameters → applied to test)
# ---------------------------------------------------------------------------

def _compute_behavior_params(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-column mean and std on the training set for behavioral z-scoring.

    Only ``extreme_streak_70`` and ``abs_sentiment`` are used.

    Args:
        train_df: Training-period DataFrame.

    Returns:
        Dict mapping column name → ``{"mean": float, "std": float}``.
        Columns that are absent or have zero std are skipped.
    """
    params: dict[str, dict[str, float]] = {}
    for col in ("extreme_streak_70", "abs_sentiment"):
        if col not in train_df.columns:
            logger.debug(
                "_compute_behavior_params: column '%s' not found; skipping", col
            )
            continue
        vals = train_df[col].dropna()
        if len(vals) < 2:
            logger.debug(
                "_compute_behavior_params: too few values for '%s'; skipping", col
            )
            continue
        std = float(vals.std())
        if std < 1e-10:
            logger.debug(
                "_compute_behavior_params: zero std for '%s'; skipping", col
            )
            continue
        params[col] = {"mean": float(vals.mean()), "std": std}
    return params


def _apply_behavior_score(
    test_df: pd.DataFrame,
    behavior_params: dict[str, dict[str, float]],
) -> np.ndarray:
    """Compute the behavioral score on the test set using training-derived parameters.

    Args:
        test_df: Test-period DataFrame (must not contain training data).
        behavior_params: Output of :func:`_compute_behavior_params`.

    Returns:
        1-D float array of behavioral scores in ``[-1, +1]``.
        If both behavioral columns are unavailable, returns zeros.
    """
    n = len(test_df)
    persistence_z = np.zeros(n, dtype=float)
    saturation_z = np.zeros(n, dtype=float)

    if "extreme_streak_70" in behavior_params:
        p = behavior_params["extreme_streak_70"]
        if "extreme_streak_70" in test_df.columns:
            raw = test_df["extreme_streak_70"].fillna(p["mean"]).values.astype(float)
            persistence_z = (raw - p["mean"]) / p["std"]

    if "abs_sentiment" in behavior_params:
        p = behavior_params["abs_sentiment"]
        if "abs_sentiment" in test_df.columns:
            raw = test_df["abs_sentiment"].fillna(p["mean"]).values.astype(float)
            saturation_z = (raw - p["mean"]) / p["std"]

    behavior_score_raw = 0.5 * persistence_z + 0.5 * saturation_z
    return np.tanh(behavior_score_raw)


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    positions: np.ndarray,
    returns: np.ndarray,
    regime_scores: np.ndarray,
    behavior_scores: np.ndarray,
    n_total_test: int,
) -> dict[str, float]:
    """Compute fold-level performance metrics for Regime V5.

    Args:
        positions: Full position array for all test rows.
        returns: Corresponding ``ret_48b`` values.
        regime_scores: Regime scores for all test rows.
        behavior_scores: Behavioral scores for all test rows.
        n_total_test: Total number of test rows.

    Returns:
        Dict with keys: n, mean, sharpe, hit_rate, coverage,
        avg_regime_score, avg_behavior_score, avg_position, corr_ret.
    """
    active_mask = positions != 0.0
    active_positions = positions[active_mask]
    active_returns = returns[active_mask]
    weighted_returns = active_positions * active_returns

    n = int(active_mask.sum())
    coverage = n / n_total_test if n_total_test > 0 else 0.0
    avg_regime_score = float(np.mean(np.abs(regime_scores)))
    avg_behavior_score = float(np.mean(np.abs(behavior_scores)))
    avg_position = float(np.mean(np.abs(positions)))

    if len(positions) > 1 and np.std(positions) > 1e-10 and np.std(returns) > 1e-10:
        corr_ret = float(np.corrcoef(positions, returns)[0, 1])
    else:
        corr_ret = float("nan")

    if n < 2:
        return {
            "n": n,
            "mean": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "coverage": coverage,
            "avg_regime_score": avg_regime_score,
            "avg_behavior_score": avg_behavior_score,
            "avg_position": avg_position,
            "corr_ret": corr_ret,
        }

    mean = float(np.mean(weighted_returns))
    std = float(np.std(weighted_returns))
    sharpe = mean / std if std > 1e-10 else float("nan")
    hit_rate = float(np.mean(weighted_returns > 0))

    return {
        "n": n,
        "mean": mean,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "coverage": coverage,
        "avg_regime_score": avg_regime_score,
        "avg_behavior_score": avg_behavior_score,
        "avg_position": avg_position,
        "corr_ret": corr_ret,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def regime_v5_walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_n: int = MIN_REGIME_N,
) -> pd.DataFrame:
    """Regime-V5 walk-forward: continuous signal blending.

    For each test year (from the third unique year onward):

    1. Split into train / test by year.
    2. Compute training-derived regime cuts (vol_regime, trend_strength_bin).
    3. Build 4-component regime keys on both slices.
    4. Compute per-regime return Sharpe on train only; build weight map
       ``regime_weight = tanh(sharpe / std_sharpe)``.
    5. Compute behavioral z-score parameters (mean / std) on train only.
    6. Build Signal V2 features and ``signal_v2_raw`` on test slice;
       ``base_signal = tanh(signal_v2_raw)``.
    7. Apply regime score and behavioral score to test::

           regime_score   = weight_map.get(regime_key, 0)
           behavior_score = tanh(0.5*persistence_z + 0.5*saturation_z)
           position       = base_signal * regime_score * behavior_score

    8. Compute per-fold metrics.

    No test-period information enters any weight or z-score computation.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all Signal V2
            feature columns, ``vol_24b``, ``trend_strength_48b``, ``abs_sentiment``,
            ``extreme_streak_70``, and *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        year_col: Column containing calendar year.
        min_n: Minimum training observations per regime for non-zero weight.

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
    if year_col not in df.columns:
        logger.warning(
            "regime_v5_walk_forward: year column '%s' not found", year_col
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "regime_v5_walk_forward: need at least 3 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year].copy()
        test_df = df[df[year_col] == test_year].copy()

        test_valid = test_df.dropna(subset=[target_col])
        if test_valid.empty:
            logger.warning(
                "REGIME V5 [year=%d]: no valid test rows; skipping fold",
                test_year,
            )
            continue

        n_total_test = len(test_valid)

        # ------------------------------------------------------------------
        # Step 1: Regime cuts and keys (training-derived, applied to test)
        # ------------------------------------------------------------------
        required_regime_cols = ["vol_24b", "trend_strength_48b", "abs_sentiment"]
        missing = [c for c in required_regime_cols if c not in train_df.columns]
        if missing:
            logger.warning(
                "Missing regime columns %s — regime_score will default to 0",
                missing,
            )
            use_regime = False
        else:
            use_regime = True

        if use_regime:
            cuts = _compute_train_cuts(train_df)
            train_labeled = _build_regime_key(train_df, cuts)
            test_labeled = _build_regime_key(test_valid, cuts)
        else:
            train_labeled = train_df.copy()
            test_labeled = test_valid.copy()
            test_labeled["regime_key"] = "NO_REGIME"

        # ------------------------------------------------------------------
        # Step 2: Regime Sharpe → weight map (train only)
        # ------------------------------------------------------------------
        regime_stats = _compute_regime_stats(
            train_labeled, target_col=target_col, min_n=min_n
        )

        if len(regime_stats) == 0:
            logger.warning("No valid regimes — regime_score = 0")
            regime_score = np.zeros(len(test_labeled))
        else:
            weight_map, std_sharpe = _build_weight_map(regime_stats)

            n_eligible = len(regime_stats)
            logger.info(
                "REGIME V5 [year=%d] | n_eligible_regimes=%d | std_sharpe=%.4f",
                test_year,
                n_eligible,
                std_sharpe,
            )

            if std_sharpe < 1e-10:
                logger.warning("std_sharpe ~ 0 — regime_score = 0")
                regime_score = np.zeros(len(test_labeled))
            else:
                regime_score = (
                    test_labeled["regime_key"]
                    .map(weight_map)
                    .fillna(0.0)
                    .values.astype(float)
                )

        # ------------------------------------------------------------------
        # Step 3: Behavioral z-score parameters (train only)
        # ------------------------------------------------------------------
        behavior_params = _compute_behavior_params(train_labeled)
        if not behavior_params:
            logger.warning("No behavioral features — using neutral score = 1")
            behavior_score = np.ones(len(test_labeled))
        else:
            behavior_score = _apply_behavior_score(test_labeled, behavior_params)

        # ------------------------------------------------------------------
        # Step 4: Base signal = tanh(signal_v2_raw)
        # ------------------------------------------------------------------
        if "signal_v2_raw" not in test_labeled.columns:
            # Build signal on the test fold (no training data leakage;
            # signal_v2 is purely feature-based)
            test_labeled = _build_signal_v2(test_labeled)

        if "signal_v2_raw" not in test_labeled.columns:
            logger.warning("signal_v2_raw missing in test fold — using zeros")
            base_signal = np.zeros(len(test_labeled))
        else:
            base_signal = np.tanh(
                test_labeled["signal_v2_raw"].fillna(0.0).values
            )

        # ------------------------------------------------------------------
        # Step 5: Final position
        # ------------------------------------------------------------------
        logger.debug(
            "Diagnostics | year=%d | base_mean=%.4f | regime_mean=%.4f | behavior_mean=%.4f",
            test_year,
            np.mean(np.abs(base_signal)),
            np.mean(np.abs(regime_score)),
            np.mean(np.abs(behavior_score)),
        )

        position = base_signal * regime_score * behavior_score

        if np.all(np.abs(position) < 1e-12):
            logger.warning(
                "All positions ~ 0 in fold year=%d — forcing base_signal fallback",
                test_year,
            )
            position = base_signal

        returns_arr = test_labeled[target_col].values.astype(float)

        if np.isnan(returns_arr).all():
            logger.warning("All returns NaN — skipping fold")
            continue

        m = _fold_metrics(
            position,
            returns_arr,
            regime_score,
            behavior_score,
            n_total_test,
        )

        logger.info(
            "REGIME V5 FOLD | year=%d"
            " | avg|regime_score|=%.4f"
            " | avg|behavior_score|=%.4f"
            " | avg|position|=%.4f"
            " | corr(pos,ret)=%+.4f",
            test_year,
            m["avg_regime_score"],
            m["avg_behavior_score"],
            m["avg_position"],
            m["corr_ret"] if not np.isnan(m["corr_ret"]) else float("nan"),
        )
        logger.info(
            "REGIME V5 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | coverage=%.1f%%",
            test_year,
            m["n"],
            m["mean"] if not np.isnan(m["mean"]) else float("nan"),
            m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
            m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
            m["coverage"] * 100,
        )

        fold_rows.append({"year": int(test_year), **m})

    if not fold_rows:
        logger.warning(
            "REGIME V5: no valid folds produced (min_n=%d)", min_n
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`regime_v5_walk_forward`.

    Returns:
        Dict with keys: n_folds, mean_sharpe, mean_hit_rate, mean_coverage,
        mean_avg_regime_score, mean_avg_behavior_score, mean_corr_ret.
    """
    if fold_df.empty:
        return {
            "n_folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
            "mean_avg_regime_score": float("nan"),
            "mean_avg_behavior_score": float("nan"),
            "mean_corr_ret": float("nan"),
        }
    return {
        "n_folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
        "mean_avg_regime_score": float(fold_df["avg_regime_score"].mean()),
        "mean_avg_behavior_score": float(fold_df["avg_behavior_score"].mean()),
        "mean_corr_ret": float(fold_df["corr_ret"].dropna().mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`regime_v5_walk_forward`.
    """
    logger.info("=== REGIME V5 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V5: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%%"
            " | avg|regime|=%.4f | avg|behavior|=%.4f"
            " | avg|pos|=%.4f | corr(ret)=%+.4f",
            row.year,
            row.n,
            row.mean if not np.isnan(row.mean) else float("nan"),
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
            row.avg_regime_score,
            row.avg_behavior_score,
            row.avg_position,
            row.corr_ret if not np.isnan(row.corr_ret) else float("nan"),
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    pooled: dict[str, float | int],
) -> None:
    """Log the consolidated final summary of the Regime V5 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`regime_v5_walk_forward`.
        pooled: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V5 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V5 SUMMARY: no results")
        return

    logger.info("Folds evaluated          : %d", pooled["n_folds"])
    logger.info(
        "Mean Sharpe              : %+.4f",
        pooled["mean_sharpe"]
        if not np.isnan(pooled["mean_sharpe"])
        else float("nan"),
    )
    logger.info(
        "Mean hit rate            : %.4f",
        pooled["mean_hit_rate"]
        if not np.isnan(pooled["mean_hit_rate"])
        else float("nan"),
    )
    logger.info("Mean coverage            : %.1f%%", pooled["mean_coverage"] * 100)
    logger.info(
        "Mean avg |regime_score|  : %.4f",
        pooled["mean_avg_regime_score"],
    )
    logger.info(
        "Mean avg |behavior_score|: %.4f",
        pooled["mean_avg_behavior_score"],
    )
    logger.info(
        "Mean corr(pos, ret_48b)  : %+.4f",
        pooled["mean_corr_ret"]
        if not np.isnan(pooled["mean_corr_ret"])
        else float("nan"),
    )
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | sharpe=%+.4f | hit_rate=%.4f | cov=%.1f%%"
            " | |regime|=%.4f | |beh|=%.4f | corr=%+.4f",
            row.year,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
            row.avg_regime_score,
            row.avg_behavior_score,
            row.corr_ret if not np.isnan(row.corr_ret) else float("nan"),
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v5.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V5 walk-forward pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V5: continuous signal blending of Signal V2, "
            "regime weights (V4 logic), and behavioral scoring.  "
            "No filtering, no discrete selection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_REGIME_N,
        metavar="N",
        help=(
            "Minimum training observations per regime for non-zero weight. "
            f"Default: {MIN_REGIME_N}."
        ),
    )
    p.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        metavar="N",
        help=(
            "Rolling z-score window size in bars for Signal V2 features. "
            f"Default: {DEFAULT_WINDOW}."
        ),
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

    _log = logging.getLogger(__name__)
    _log.info(
        "=== REGIME V5 === window=%d  min_n=%d",
        args.window,
        args.min_n,
    )

    df = load_data(args.data, window=args.window)

    require_columns(
        df,
        [TARGET_COL, "year"],
        context="regime_v5.main",
    )
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = regime_v5_walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
