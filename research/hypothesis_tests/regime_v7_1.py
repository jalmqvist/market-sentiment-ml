# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v7_1.py
==========================
CONTINUOUS EVENT SCORING pipeline for FX sentiment research.

Upgrades Regime V7 by replacing boolean event detection with continuous
*event scores* that are ranked per row so only the strongest signal drives
the position.

Pipeline overview
-----------------
1. **Scores** – three continuous scores per row (computed from rolling z-scores
   fitted on the training split only):

   * ``SATURATION_SCORE``   = tanh(z(abs_sentiment) + z(extreme_streak_70)
                                   + z(trend_strength_48b))
   * ``DIVERGENCE_SCORE``   = tanh(|z(divergence)|)
   * ``EXHAUSTION_SCORE``   = tanh(z(extreme_streak_70) - z(trend_strength_48b))

2. **Train phase** (per fold)

   For each score type, compute on the training split:

   * ``signal_ret = score * ret_48b``
   * mean_return, Sharpe, correlation(score, ret_48b)

   Store stats for diagnostics; do NOT hard-filter aggressively.

3. **Test phase** – for each test row:

   * Compute all scores (using train-only normalization stats).
   * Select the score with the greatest absolute magnitude.
   * If ``max(|scores|) < score_threshold`` → ``position = 0``.
   * Otherwise:
       ``base_signal = tanh(signal_v2_raw)``
       ``direction   = sign(best_score)``
       ``position    = base_signal * direction``
     or, with ``--use-score-weighting``:
       ``position    = base_signal * best_score``

4. **Metrics** – coverage, n, mean, Sharpe, hit_rate, avg_score_magnitude,
   plus per-score-type performance.

5. **Walk-forward** – expanding window, minimum 3 years, train on all prior
   years, test on next year.

Required columns
----------------
``ret_48b``, ``signal_v2_raw``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``, ``divergence``

Fold output schema
------------------
``["year", "n", "mean", "sharpe", "hit_rate", "coverage"]``

Logging (per fold)
------------------
* Stats for each score type (mean_return, Sharpe, correlation)
* Distribution of scores (mean, std, min, max)
* Chosen score type frequencies

Usage::

    python experiments/regime_v7_1.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v7_1.py \\
        --data data/output/master_research_dataset.csv \\
        --score-threshold 0.3 --use-score-weighting --log-level DEBUG
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

# Reuse signal_v2 loading and feature building
from experiments.signal_v2 import (
    DEFAULT_WINDOW,
    build_features as _build_signal_v2_features,
    build_signal as _build_signal_v2,
    load_data as _load_signal_v2_data,
)

# Reuse regime_v3 features for vol_24b
from experiments.regime_v3 import build_features as _build_regime_v3_features

from utils.validation import require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Default minimum training observations for diagnostics.
DEFAULT_MIN_N: int = 50

#: Default threshold below which the best score is ignored and position = 0.
DEFAULT_SCORE_THRESHOLD: float = 0.5

#: Rolling z-score window used when computing score normalisation.
_ZSCORE_WINDOW: int = 96

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

#: Required columns for the pipeline (validated at entry points).
_REQUIRED_COLS: list[str] = [
    TARGET_COL,
    "signal_v2_raw",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
    "divergence",
]

#: Output fold columns.
_FOLD_COLS: list[str] = ["year", "n", "mean", "sharpe", "hit_rate", "coverage"]

#: All score names.
_SCORE_NAMES: list[str] = [
    "SATURATION_SCORE",
    "DIVERGENCE_SCORE",
    "EXHAUSTION_SCORE",
]


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
        log_path = logs_dir / f"regime_v7_1_{timestamp}.log"
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
    """Load and prepare the dataset for Regime V7.1.

    Combines the signal_v2 and regime_v3 feature pipelines:

    1. Load via ``signal_v2.load_data`` (adds ``year``, ``timestamp``).
    2. Detect and assign the price column.
    3. Build Signal V2 features (divergence, shock, exhaustion).
    4. Build Signal V2 composite (creates ``signal_v2_raw``).
    5. Build Regime V3 features (``vol_24b``, interactions) if columns present.

    Args:
        path: Path to the master research dataset CSV.
        window: Rolling z-score window size in bars (default 96).

    Returns:
        DataFrame ready for walk-forward evaluation.

    Raises:
        ValueError: If required columns are missing after feature building.
    """
    df = _load_signal_v2_data(path)

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
            "Regime V7.1: no valid numeric price column found among "
            f"{_PRICE_CANDIDATES}."
        )

    df["price"] = selected_series
    logger.info("load_data: using price column '%s'", selected_col)

    df = _build_signal_v2_features(df, window=window)
    logger.info("load_data: signal_v2 features built (%d rows)", len(df))

    df = _build_signal_v2(df)
    if "signal_v2_raw" not in df.columns:
        raise ValueError("signal_v2_raw not created by build_signal")

    if "entry_time" in df.columns and "entry_close" in df.columns:
        df = _build_regime_v3_features(df)
        logger.info("load_data: regime_v3 features built (vol_24b added)")
    else:
        logger.warning(
            "load_data: 'entry_time' or 'entry_close' missing; "
            "vol_24b / interaction features will be absent"
        )

    return df


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Compute an expanding/rolling z-score with no forward leakage.

    Uses an *expanding* window so early rows still get a score (min_periods=2).
    For longer series the result stabilises after ``window`` rows.

    Args:
        series: Input series.
        window: Minimum window size (used as ``min_periods``).

    Returns:
        Z-scored series, NaN where insufficient data.
    """
    roll = series.expanding(min_periods=2)
    mean = roll.mean()
    std = roll.std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def _apply_train_zscore(
    series: pd.Series,
    train_mean: float,
    train_std: float,
) -> pd.Series:
    """Apply train-set mean/std to normalise a series without leakage.

    Args:
        series: Input series (train or test).
        train_mean: Mean computed on the training split only.
        train_std: Standard deviation computed on the training split only.

    Returns:
        Standardised series; zero-filled where std is degenerate.
    """
    if train_std < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - train_mean) / train_std


# ---------------------------------------------------------------------------
# Score building
# ---------------------------------------------------------------------------

def _compute_train_stats(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-feature mean and std on the training split.

    These statistics are later used to normalise both train and test data so
    that no test-period information leaks into the z-score computation.

    Args:
        train_df: Training-split DataFrame.

    Returns:
        Nested dict: ``{feature_name: {"mean": float, "std": float}}``.
    """
    features = [
        "abs_sentiment",
        "extreme_streak_70",
        "trend_strength_48b",
        "divergence",
    ]
    stats: dict[str, dict[str, float]] = {}
    for feat in features:
        col = train_df[feat].dropna()
        stats[feat] = {
            "mean": float(col.mean()) if len(col) > 0 else 0.0,
            "std": float(col.std()) if len(col) > 1 else 1.0,
        }
    return stats


def build_scores(
    df: pd.DataFrame,
    train_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compute continuous event scores for every row in *df*.

    Scores are derived from z-scores of raw features (using train-only
    mean/std to avoid leakage) and then bounded with ``tanh``.

    Score definitions
    -----------------
    **SATURATION_SCORE**::

        sat_score = tanh(
            z(abs_sentiment) + z(extreme_streak_70) + z(trend_strength_48b)
        )

    **DIVERGENCE_SCORE**::

        div_score = tanh(|z(divergence)|)

    **EXHAUSTION_SCORE**::

        exhaust_score = tanh(
            z(extreme_streak_70) - z(trend_strength_48b)
        )

    Args:
        df: DataFrame slice (train or test).
        train_stats: Output of :func:`_compute_train_stats` from the training
            split.  Used for both train and test normalisation.

    Returns:
        DataFrame with one float column per score type, same index as *df*.

    Raises:
        ValueError: If required feature columns are absent from *df*.
    """
    require_columns(
        df,
        ["abs_sentiment", "extreme_streak_70", "trend_strength_48b", "divergence"],
        context="build_scores",
    )

    def _z(col_name: str) -> pd.Series:
        s = df[col_name].fillna(0.0)
        return _apply_train_zscore(
            s,
            train_stats[col_name]["mean"],
            train_stats[col_name]["std"],
        )

    z_abs_sent = _z("abs_sentiment")
    z_streak = _z("extreme_streak_70")
    z_trend = _z("trend_strength_48b")
    z_div = _z("divergence")

    sat_raw = z_abs_sent + z_streak + z_trend
    div_raw = z_div.abs()
    exhaust_raw = z_streak - z_trend

    return pd.DataFrame(
        {
            "SATURATION_SCORE": np.tanh(sat_raw),
            "DIVERGENCE_SCORE": np.tanh(div_raw),
            "EXHAUSTION_SCORE": np.tanh(exhaust_raw),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Score evaluation (train phase)
# ---------------------------------------------------------------------------

def evaluate_scores(
    scores: pd.DataFrame,
    target: pd.Series,
    *,
    min_n: int = DEFAULT_MIN_N,
) -> dict[str, dict[str, float]]:
    """Evaluate each score type on the training split.

    For each score, computes:

    * ``signal_ret = score * ret_48b``
    * mean_return – mean of signal_ret
    * sharpe – mean / std of signal_ret
    * correlation – Pearson correlation between score and ret_48b

    Results are logged for diagnostics.  No hard filter is applied; all
    three scores are always returned (with NaN where computation fails).

    Args:
        scores: DataFrame from :func:`build_scores`.
        target: Forward-return series aligned to *scores*.
        min_n: Minimum number of non-NaN rows; logged as a warning if unmet.

    Returns:
        Dict mapping score name → ``{mean_return, sharpe, correlation, n}``.
    """
    stats: dict[str, dict[str, float]] = {}

    ret = target.reindex(scores.index).fillna(0.0).values.astype(float)

    for score_name in _SCORE_NAMES:
        sc = scores[score_name].fillna(0.0).values.astype(float)
        signal_ret = sc * ret

        n = int(np.sum(np.abs(sc) > 1e-12))

        if n < min_n:
            logger.warning(
                "evaluate_scores: %s has only %d active rows (< min_n=%d)",
                score_name,
                n,
                min_n,
            )

        mean_ret = float(np.mean(signal_ret))
        std_ret = float(np.std(signal_ret))
        sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")

        # Pearson correlation between raw score and returns
        score_std = float(np.std(sc))
        ret_std = float(np.std(ret))
        if score_std > 1e-10 and ret_std > 1e-10:
            corr = float(np.corrcoef(sc, ret)[0, 1])
        else:
            corr = float("nan")

        stats[score_name] = {
            "n": float(n),
            "mean_return": mean_ret,
            "sharpe": sharpe if not np.isnan(sharpe) else 0.0,
            "correlation": corr if not np.isnan(corr) else 0.0,
        }

        logger.info(
            "evaluate_scores: %-20s | n=%d | mean_ret=%+.6f"
            " | sharpe=%+.4f | corr=%+.4f",
            score_name,
            n,
            mean_ret,
            sharpe if not np.isnan(sharpe) else float("nan"),
            corr if not np.isnan(corr) else float("nan"),
        )

    return stats


# ---------------------------------------------------------------------------
# Score application (test phase)
# ---------------------------------------------------------------------------

def apply_scores(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_score_weighting: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate positions for a test split using continuous event scores.

    For each row:

    1. Compute ``best_score = argmax(|SATURATION|, |DIVERGENCE|, |EXHAUSTION|)``.
    2. If ``max(|scores|) < score_threshold`` → ``position = 0``.
    3. Otherwise:
       * ``base_signal = tanh(signal_v2_raw)``
       * With standard mode: ``position = base_signal * sign(best_score)``
       * With score-weighting: ``position = base_signal * best_score``

    Implemented with fully vectorised pandas / numpy operations.

    Args:
        df: Test DataFrame.  Must contain ``signal_v2_raw``.
        scores: Score DataFrame from :func:`build_scores`.
        score_threshold: Minimum best-score magnitude to take a position.
        use_score_weighting: If True, weight position by score magnitude instead
            of applying only direction.

    Returns:
        Tuple of three 1-D numpy arrays (all length = ``len(df)``):
        * ``positions`` – the final position array.
        * ``best_score_values`` – the chosen score value for each row.
        * ``chosen_score_idx`` – index (0-2) of the winning score per row
          (``-1`` where no position is taken).
    """
    n_rows = len(df)
    positions = np.zeros(n_rows, dtype=float)
    best_score_values = np.zeros(n_rows, dtype=float)
    chosen_score_idx = np.full(n_rows, -1, dtype=int)

    if "signal_v2_raw" not in df.columns:
        logger.warning("apply_scores: 'signal_v2_raw' missing; all positions = 0")
        return positions, best_score_values, chosen_score_idx

    base_signal = np.tanh(df["signal_v2_raw"].fillna(0.0).values.astype(float))

    # Stack scores: shape (n_rows, 3)
    score_matrix = np.column_stack(
        [scores[name].fillna(0.0).values for name in _SCORE_NAMES]
    )
    abs_matrix = np.abs(score_matrix)

    # Best score index and magnitude per row
    best_idx = np.argmax(abs_matrix, axis=1)
    best_abs = abs_matrix[np.arange(n_rows), best_idx]
    best_val = score_matrix[np.arange(n_rows), best_idx]

    # Apply threshold
    active = best_abs >= score_threshold
    chosen_score_idx[active] = best_idx[active]
    best_score_values[active] = best_val[active]

    if use_score_weighting:
        positions[active] = base_signal[active] * best_val[active]
    else:
        direction = np.sign(best_val[active])
        positions[active] = base_signal[active] * direction

    return positions, best_score_values, chosen_score_idx


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    positions: np.ndarray,
    returns: np.ndarray,
    n_total_test: int,
    best_score_values: np.ndarray,
    chosen_score_idx: np.ndarray,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V7.1.

    Args:
        positions: Full position array for all test rows.
        returns: Corresponding ``ret_48b`` values.
        n_total_test: Total number of test rows.
        best_score_values: Winning score value per row (0 where inactive).
        chosen_score_idx: Index of the winning score (-1 where inactive).

    Returns:
        Dict with keys: n, mean, sharpe, hit_rate, coverage,
        avg_score_magnitude, plus one ``score_freq_<name>`` key per score type.
    """
    active_mask = np.abs(positions) > 1e-12
    active_positions = positions[active_mask]
    active_returns = returns[active_mask]
    weighted_returns = active_positions * active_returns

    n = int(active_mask.sum())
    coverage = float(np.mean(active_mask)) if n_total_test > 0 else 0.0

    # Average score magnitude over active rows
    active_scores = np.abs(best_score_values[active_mask])
    avg_score_mag = float(np.mean(active_scores)) if n > 0 else float("nan")

    result: dict[str, Any] = {
        "coverage": coverage,
        "avg_score_magnitude": avg_score_mag,
    }

    # Per-score-type frequency in active rows
    active_chosen = chosen_score_idx[active_mask]
    for k, sname in enumerate(_SCORE_NAMES):
        freq = float(np.mean(active_chosen == k)) if n > 0 else float("nan")
        result[f"score_freq_{sname}"] = freq

    if n < 2:
        result.update(
            {
                "n": n,
                "mean": float("nan"),
                "sharpe": float("nan"),
                "hit_rate": float("nan"),
            }
        )
        return result

    mean_ret = float(np.mean(weighted_returns))
    std_ret = float(np.std(weighted_returns))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
    hit_rate = float(np.mean(weighted_returns > 0))

    result.update(
        {
            "n": n,
            "mean": mean_ret,
            "sharpe": sharpe,
            "hit_rate": hit_rate,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_n: int = DEFAULT_MIN_N,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_score_weighting: bool = False,
) -> pd.DataFrame:
    """Regime-V7.1 walk-forward: continuous score-based signal pipeline.

    For each test year (from the third unique year onward):

    1. Split into train / test by year (expanding window).
    2. Compute train-split normalisation statistics.
    3. Build scores on train and test (using train stats).
    4. Evaluate scores on **train only** (log diagnostics).
    5. Apply scores to test rows → positions.
    6. Compute fold-level metrics.

    No test-period information enters normalisation or score evaluation.

    Args:
        df: Full dataset (after :func:`load_data`).
        target_col: Forward-return column.
        year_col: Column containing calendar year.
        min_n: Minimum training rows for diagnostics warning.
        score_threshold: Minimum best-score magnitude to take a position.
        use_score_weighting: If True, position weighted by score magnitude.

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

    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year].copy()
        test_df = df[df[year_col] == test_year].copy()

        test_valid = test_df.dropna(subset=[target_col])
        if test_valid.empty:
            logger.warning(
                "REGIME V7.1 [year=%d]: no valid test rows; skipping fold",
                test_year,
            )
            continue

        n_total_test = len(test_valid)

        # ------------------------------------------------------------------
        # Step 1: Compute normalisation statistics on train split only
        # ------------------------------------------------------------------
        try:
            require_columns(
                train_df,
                ["abs_sentiment", "extreme_streak_70", "trend_strength_48b", "divergence"],
                context=f"walk_forward:train:{test_year}",
            )
        except ValueError as exc:
            logger.warning(
                "REGIME V7.1 [year=%d]: missing train columns (%s); skipping",
                test_year,
                exc,
            )
            continue

        train_stats = _compute_train_stats(train_df)

        # ------------------------------------------------------------------
        # Step 2: Build scores (train and test)
        # ------------------------------------------------------------------
        try:
            train_scores = build_scores(train_df, train_stats)
        except ValueError as exc:
            logger.warning(
                "REGIME V7.1 [year=%d]: build_scores failed on train (%s); skipping",
                test_year,
                exc,
            )
            continue

        try:
            test_scores = build_scores(test_valid, train_stats)
        except ValueError as exc:
            logger.warning(
                "REGIME V7.1 [year=%d]: build_scores failed on test (%s); skipping",
                test_year,
                exc,
            )
            continue

        # ------------------------------------------------------------------
        # Step 3: Evaluate scores on training data (diagnostics only)
        # ------------------------------------------------------------------
        logger.info("REGIME V7.1 [year=%d] — training score diagnostics:", test_year)
        score_stats = evaluate_scores(
            train_scores,
            train_df[target_col],
            min_n=min_n,
        )

        # Log score distributions (train)
        for sname in _SCORE_NAMES:
            sc = train_scores[sname]
            logger.debug(
                "  train score %-20s | mean=%+.4f | std=%.4f"
                " | min=%+.4f | max=%+.4f",
                sname,
                float(sc.mean()),
                float(sc.std()),
                float(sc.min()),
                float(sc.max()),
            )

        # Log test score distributions
        for sname in _SCORE_NAMES:
            sc = test_scores[sname]
            logger.debug(
                "  test  score %-20s | mean=%+.4f | std=%.4f"
                " | min=%+.4f | max=%+.4f",
                sname,
                float(sc.mean()),
                float(sc.std()),
                float(sc.min()),
                float(sc.max()),
            )

        # ------------------------------------------------------------------
        # Step 4: Apply scores to test rows
        # ------------------------------------------------------------------
        positions, best_score_values, chosen_score_idx = apply_scores(
            test_valid,
            test_scores,
            score_threshold=score_threshold,
            use_score_weighting=use_score_weighting,
        )

        returns_arr = test_valid[target_col].values.astype(float)
        if np.isnan(returns_arr).all():
            logger.warning(
                "REGIME V7.1 [year=%d]: all returns NaN; skipping", test_year
            )
            continue

        # ------------------------------------------------------------------
        # Step 5: Compute fold metrics
        # ------------------------------------------------------------------
        m = _fold_metrics(
            positions,
            returns_arr,
            n_total_test,
            best_score_values,
            chosen_score_idx,
        )

        coverage = m["coverage"]
        logger.info(
            "REGIME V7.1 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | coverage=%.1f%%"
            " | avg_score_mag=%.4f",
            test_year,
            m.get("n", 0),
            m.get("mean", float("nan"))
            if not np.isnan(m.get("mean", float("nan")))
            else float("nan"),
            m.get("sharpe", float("nan"))
            if not np.isnan(m.get("sharpe", float("nan")))
            else float("nan"),
            m.get("hit_rate", float("nan"))
            if not np.isnan(m.get("hit_rate", float("nan")))
            else float("nan"),
            coverage * 100,
            m.get("avg_score_magnitude", float("nan"))
            if not np.isnan(m.get("avg_score_magnitude", float("nan")))
            else float("nan"),
        )

        # Log score-type frequency in active rows
        for sname in _SCORE_NAMES:
            freq = m.get(f"score_freq_{sname}", float("nan"))
            if not np.isnan(freq):
                logger.info(
                    "  score_freq %-20s: %.1f%%",
                    sname,
                    freq * 100,
                )

        # Log per-score-type train performance
        for sname, ss in score_stats.items():
            logger.info(
                "  train_perf %-20s | sharpe=%+.4f | corr=%+.4f",
                sname,
                ss.get("sharpe", float("nan")),
                ss.get("correlation", float("nan")),
            )

        fold_rows.append(
            {
                "year": int(test_year),
                "n": m.get("n", 0),
                "mean": m.get("mean", float("nan")),
                "sharpe": m.get("sharpe", float("nan")),
                "hit_rate": m.get("hit_rate", float("nan")),
                "coverage": coverage,
            }
        )

    if not fold_rows:
        logger.warning(
            "REGIME V7.1: no valid folds produced (min_n=%d,"
            " score_threshold=%.4f)",
            min_n,
            score_threshold,
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with keys: n_folds, mean_sharpe, mean_hit_rate, mean_coverage.
    """
    if fold_df.empty:
        return {
            "n_folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
        }
    return {
        "n_folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V7.1 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V7.1: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%%",
            row.year,
            row.n,
            row.mean if not np.isnan(row.mean) else float("nan"),
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    pooled: dict[str, float | int],
) -> None:
    """Log the consolidated final summary of the Regime V7.1 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        pooled: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V7.1 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V7.1 SUMMARY: no results")
        return

    logger.info("Folds evaluated  : %d", pooled["n_folds"])
    logger.info(
        "Mean Sharpe      : %+.4f",
        pooled["mean_sharpe"]
        if not np.isnan(pooled["mean_sharpe"])
        else float("nan"),
    )
    logger.info(
        "Mean hit rate    : %.4f",
        pooled["mean_hit_rate"]
        if not np.isnan(pooled["mean_hit_rate"])
        else float("nan"),
    )
    logger.info("Mean coverage    : %.1f%%", pooled["mean_coverage"] * 100)
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | sharpe=%+.4f | hit_rate=%.4f | cov=%.1f%%",
            row.year,
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v7_1.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V7.1 continuous score-based signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V7.1: continuous event scoring pipeline. "
            "Computes ranked scores (saturation, divergence, exhaustion) "
            "per row and trades only when the strongest signal exceeds a "
            "configurable threshold."
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
        default=DEFAULT_MIN_N,
        metavar="N",
        help=(
            "Minimum training observations for diagnostics warning. "
            f"Default: {DEFAULT_MIN_N}."
        ),
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        metavar="T",
        help=(
            "Minimum best-score magnitude to take a position. "
            f"Default: {DEFAULT_SCORE_THRESHOLD}."
        ),
    )
    p.add_argument(
        "--use-score-weighting",
        action="store_true",
        default=False,
        help=(
            "Weight position by score magnitude instead of applying only "
            "direction.  Default: off (direction-only)."
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
        "=== REGIME V7.1 === window=%d  min_n=%d"
        "  score_threshold=%.4f  use_score_weighting=%s",
        args.window,
        args.min_n,
        args.score_threshold,
        args.use_score_weighting,
    )

    df = load_data(args.data, window=args.window)

    require_columns(df, _REQUIRED_COLS, context="regime_v7_1.main")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        score_threshold=args.score_threshold,
        use_score_weighting=args.use_score_weighting,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
