# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v7.py
========================
EVENT-BASED signal pipeline for FX sentiment research.

Replaces regime-based averaging with discrete event detection.  A position is
only taken when a statistically validated "event" fires on the test row.

Pipeline overview
-----------------
1. **Events** – three boolean masks per row:

   * ``SATURATION_EVENT``   – crowd at extreme sentiment AND trend confirmed
   * ``DIVERGENCE_EVENT``   – sentiment diverges strongly from price momentum
   * ``EXHAUSTION_EVENT``   – extreme persistence AND trend weakening

2. **Train phase** (per fold)

   For each event type, compute on the training split:

   * ``n`` – number of event rows
   * mean return, Sharpe, hit rate

   Retain only events where ``n >= min_n`` AND ``sharpe >= min_sharpe``.
   Store direction = ``sign(mean_return)`` and the highest Sharpe event wins
   when multiple events fire simultaneously.

3. **Test phase** – for each test row:

   * Check which *valid* events fire.
   * Select the event with the highest train-set Sharpe.
   * If no valid event fires → ``position = 0``.
   * If an event fires:  ``position = tanh(signal_v2_raw) * direction``

4. **Metrics** – coverage, n, mean, Sharpe, hit rate + per-event frequency.

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
* Number of valid events, per-event stats
* Coverage, performance metrics

Usage::

    python experiments/regime_v7.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v7.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 50 --min-sharpe 0.02 \\
        --streak-threshold 3 --trend-threshold 0.5 \\
        --divergence-threshold 1.0 --log-level DEBUG
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

#: Default minimum training observations per event for validation.
DEFAULT_MIN_N: int = 50

#: Default minimum train-set Sharpe for an event to be considered valid.
DEFAULT_MIN_SHARPE: float = 0.02

#: Default minimum extreme-streak count to trigger saturation / exhaustion events.
DEFAULT_STREAK_THRESHOLD: int = 3

#: Default minimum absolute trend strength for the saturation event.
DEFAULT_TREND_THRESHOLD: float = 0.5

#: Default minimum absolute divergence for the divergence event.
DEFAULT_DIVERGENCE_THRESHOLD: float = 1.0

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

#: All recognised event names.
_EVENT_NAMES: list[str] = [
    "SATURATION_EVENT",
    "DIVERGENCE_EVENT",
    "EXHAUSTION_EVENT",
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
        log_path = logs_dir / f"regime_v7_{timestamp}.log"
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
    """Load and prepare the dataset for Regime V7.

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
            "Regime V7: no valid numeric price column found among "
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
# Event definitions
# ---------------------------------------------------------------------------

def build_events(
    df: pd.DataFrame,
    *,
    streak_threshold: int = DEFAULT_STREAK_THRESHOLD,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """Build boolean event masks for all event types.

    Returns a DataFrame with one boolean column per event type aligned to
    *df*'s index.  Uses vectorised operations only.

    Event definitions
    -----------------
    **SATURATION_EVENT**
        Crowd at extreme absolute sentiment AND crowd has persisted there for
        at least ``streak_threshold`` bars AND a clear trend is present::

            abs_sentiment > 70
            extreme_streak_70 >= streak_threshold
            abs(trend_strength_48b) > trend_threshold

    **DIVERGENCE_EVENT**
        Sentiment diverges strongly from price momentum::

            abs(divergence) > divergence_threshold

    **EXHAUSTION_EVENT**
        Extreme sentiment persistence AND trend is weakening (low absolute
        trend strength relative to streak intensity)::

            extreme_streak_70 >= streak_threshold
            abs(trend_strength_48b) <= trend_threshold

    Args:
        df: DataFrame slice (train or test).
        streak_threshold: Minimum ``extreme_streak_70`` value.
        trend_threshold: Boundary for ``abs(trend_strength_48b)``; above →
            saturation, at-or-below → exhaustion.
        divergence_threshold: Minimum ``abs(divergence)`` for DIVERGENCE_EVENT.

    Returns:
        DataFrame of booleans, columns = ``_EVENT_NAMES``, same index as *df*.

    Raises:
        ValueError: If any of the four required feature columns are absent.
    """
    require_columns(
        df,
        ["abs_sentiment", "extreme_streak_70", "trend_strength_48b", "divergence"],
        context="build_events",
    )

    abs_sent = df["abs_sentiment"].values
    streak = df["extreme_streak_70"].values
    trend = df["trend_strength_48b"].values
    div = df["divergence"].values

    saturation = (
        (abs_sent > 70)
        & (streak >= streak_threshold)
        & (np.abs(trend) > trend_threshold)
    )

    divergence_event = np.abs(div) > divergence_threshold

    exhaustion = (streak >= streak_threshold) & (np.abs(trend) <= trend_threshold)

    return pd.DataFrame(
        {
            "SATURATION_EVENT": saturation,
            "DIVERGENCE_EVENT": divergence_event,
            "EXHAUSTION_EVENT": exhaustion,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Event statistics (train phase)
# ---------------------------------------------------------------------------

def compute_event_stats(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    min_n: int = DEFAULT_MIN_N,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
) -> dict[str, dict[str, Any]]:
    """Compute per-event statistics on a training split.

    For each event column in *events*, computes n, mean return, Sharpe, and
    hit rate using only the rows where the event fired.  Retains only events
    that pass both ``min_n`` and ``min_sharpe`` thresholds.

    Args:
        df: Training DataFrame with *target_col*.
        events: Boolean event mask DataFrame (columns from ``build_events``).
        target_col: Forward-return column.
        min_n: Minimum number of event rows required.
        min_sharpe: Minimum train-set Sharpe required.

    Returns:
        ``valid_events`` dict mapping event name → ``{direction, sharpe, n,
        mean, hit_rate}`` for every event that passed validation.
        Empty dict if no events qualify.
    """
    require_columns(df, [target_col], context="compute_event_stats")

    returns = df[target_col].values.astype(float)
    valid_events: dict[str, dict[str, Any]] = {}

    for event_name in events.columns:
        mask = events[event_name].values.astype(bool)
        event_returns = returns[mask]
        event_returns = event_returns[~np.isnan(event_returns)]

        n = len(event_returns)
        if n < min_n:
            logger.debug(
                "compute_event_stats: %s rejected (n=%d < min_n=%d)",
                event_name,
                n,
                min_n,
            )
            continue

        mean_ret = float(np.mean(event_returns))
        std_ret = float(np.std(event_returns))
        sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
        hit_rate = float(np.mean(event_returns > 0))

        if np.isnan(sharpe) or sharpe < min_sharpe:
            logger.debug(
                "compute_event_stats: %s rejected (sharpe=%.4f < min_sharpe=%.4f)",
                event_name,
                sharpe if not np.isnan(sharpe) else float("nan"),
                min_sharpe,
            )
            continue

        direction = 1 if mean_ret >= 0 else -1
        valid_events[event_name] = {
            "direction": direction,
            "sharpe": sharpe,
            "n": n,
            "mean": mean_ret,
            "hit_rate": hit_rate,
        }
        logger.info(
            "compute_event_stats: %s VALID | n=%d | mean=%.6f"
            " | sharpe=%.4f | hit_rate=%.4f | direction=%+d",
            event_name,
            n,
            mean_ret,
            sharpe,
            hit_rate,
            direction,
        )

    return valid_events


# ---------------------------------------------------------------------------
# Test phase: apply validated events
# ---------------------------------------------------------------------------

def apply_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    valid_events: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Generate positions for a test split using validated event rules.

    For each row:

    * Identify which validated events fire.
    * Select the event with the highest train-set Sharpe.
    * Compute ``position = tanh(signal_v2_raw) * direction``.
    * Rows where no validated event fires receive ``position = 0``.

    Implemented with vectorised operations; no Python-level row loops.

    Args:
        df: Test DataFrame.  Must contain ``signal_v2_raw``.
        events: Boolean event mask DataFrame aligned to *df*.
        valid_events: Dict from :func:`compute_event_stats`.

    Returns:
        1-D numpy array of float positions, length = ``len(df)``.
    """
    n_rows = len(df)
    positions = np.zeros(n_rows, dtype=float)

    if not valid_events or "signal_v2_raw" not in df.columns:
        return positions

    base_signal = np.tanh(df["signal_v2_raw"].fillna(0.0).values.astype(float))

    # Sort valid events by descending Sharpe so the first-match wins
    ranked_events = sorted(
        valid_events.items(), key=lambda kv: kv[1]["sharpe"], reverse=True
    )

    # Build a "best event" layer by assigning each row the highest-Sharpe
    # firing event — vectorised via a cascading boolean update.
    selected_direction = np.zeros(n_rows, dtype=float)
    assigned = np.zeros(n_rows, dtype=bool)

    for event_name, stats in ranked_events:
        if event_name not in events.columns:
            continue
        mask = events[event_name].values.astype(bool) & ~assigned
        selected_direction[mask] = float(stats["direction"])
        assigned[mask] = True

    positions = base_signal * selected_direction  # 0 where direction==0 (unassigned)
    return positions


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    positions: np.ndarray,
    returns: np.ndarray,
    n_total_test: int,
    valid_events: dict[str, dict[str, Any]],
    test_events: pd.DataFrame,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V7.

    Args:
        positions: Full position array for all test rows.
        returns: Corresponding ``ret_48b`` values.
        n_total_test: Total number of test rows.
        valid_events: Validated event dict from the train phase.
        test_events: Boolean event mask DataFrame for the test split.

    Returns:
        Dict with keys: n, mean, sharpe, hit_rate, coverage, plus one
        ``event_freq_<name>`` key per event type.
    """
    active_mask = np.abs(positions) > 1e-12
    active_positions = positions[active_mask]
    active_returns = returns[active_mask]
    weighted_returns = active_positions * active_returns

    n = int(active_mask.sum())
    coverage = float(np.mean(active_mask)) if n_total_test > 0 else 0.0

    result: dict[str, Any] = {"coverage": coverage}

    # Per-event frequency in test data
    for ename in _EVENT_NAMES:
        if ename in test_events.columns:
            freq = float(np.mean(test_events[ename].values))
        else:
            freq = float("nan")
        result[f"event_freq_{ename}"] = freq

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
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    streak_threshold: int = DEFAULT_STREAK_THRESHOLD,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """Regime-V7 walk-forward: event-based signal pipeline.

    For each test year (from the third unique year onward):

    1. Split into train / test by year (expanding window).
    2. Build event masks on both splits.
    3. Compute event stats on **train only**; select valid events.
    4. Apply valid events to test rows → positions.
    5. Compute fold-level metrics.

    No test-period information enters event validation.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            listed in ``_REQUIRED_COLS``.
        target_col: Forward-return column.
        year_col: Column containing calendar year.
        min_n: Minimum training event rows for validation.
        min_sharpe: Minimum training Sharpe for validation.
        streak_threshold: Minimum ``extreme_streak_70`` value.
        trend_threshold: Boundary for ``abs(trend_strength_48b)``.
        divergence_threshold: Minimum ``abs(divergence)`` for divergence event.

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
                "REGIME V7 [year=%d]: no valid test rows; skipping fold",
                test_year,
            )
            continue

        n_total_test = len(test_valid)

        # ------------------------------------------------------------------
        # Step 1: Build event masks on train and test
        # ------------------------------------------------------------------
        try:
            train_events = build_events(
                train_df,
                streak_threshold=streak_threshold,
                trend_threshold=trend_threshold,
                divergence_threshold=divergence_threshold,
            )
        except ValueError as exc:
            logger.warning(
                "REGIME V7 [year=%d]: build_events failed on train (%s); skipping",
                test_year,
                exc,
            )
            continue

        try:
            test_events = build_events(
                test_valid,
                streak_threshold=streak_threshold,
                trend_threshold=trend_threshold,
                divergence_threshold=divergence_threshold,
            )
        except ValueError as exc:
            logger.warning(
                "REGIME V7 [year=%d]: build_events failed on test (%s); skipping",
                test_year,
                exc,
            )
            continue

        # ------------------------------------------------------------------
        # Step 2: Validate events on train data only
        # ------------------------------------------------------------------
        valid_events = compute_event_stats(
            train_df,
            train_events,
            target_col=target_col,
            min_n=min_n,
            min_sharpe=min_sharpe,
        )

        logger.info(
            "REGIME V7 [year=%d] | valid_events=%d/%d: %s",
            test_year,
            len(valid_events),
            len(_EVENT_NAMES),
            list(valid_events.keys()),
        )

        if not valid_events:
            logger.warning(
                "REGIME V7 [year=%d]: no valid events — all positions = 0",
                test_year,
            )
            # Emit a zero-performance fold row so coverage is visible
            fold_rows.append(
                {
                    "year": int(test_year),
                    "n": 0,
                    "mean": float("nan"),
                    "sharpe": float("nan"),
                    "hit_rate": float("nan"),
                    "coverage": 0.0,
                }
            )
            continue

        for ename, stats in valid_events.items():
            logger.info(
                "  Event %-22s | n=%d | sharpe=%+.4f | hit_rate=%.4f"
                " | direction=%+d",
                ename,
                stats["n"],
                stats["sharpe"],
                stats["hit_rate"],
                stats["direction"],
            )

        # ------------------------------------------------------------------
        # Step 3: Apply validated events to test rows
        # ------------------------------------------------------------------
        positions = apply_events(test_valid, test_events, valid_events)

        returns_arr = test_valid[target_col].values.astype(float)
        if np.isnan(returns_arr).all():
            logger.warning("REGIME V7 [year=%d]: all returns NaN; skipping", test_year)
            continue

        # ------------------------------------------------------------------
        # Step 4: Compute fold metrics
        # ------------------------------------------------------------------
        m = _fold_metrics(
            positions,
            returns_arr,
            n_total_test,
            valid_events,
            test_events,
        )

        coverage = m["coverage"]
        logger.info(
            "REGIME V7 FOLD | year=%d | n=%5d | mean=%+.6f"
            " | sharpe=%+.4f | hit_rate=%.4f | coverage=%.1f%%",
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
        )

        for ename in _EVENT_NAMES:
            key = f"event_freq_{ename}"
            freq = m.get(key, float("nan"))
            logger.info(
                "  event_freq %-22s: %.2f%%",
                ename,
                freq * 100 if not np.isnan(freq) else float("nan"),
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
            "REGIME V7: no valid folds produced (min_n=%d, min_sharpe=%.4f)",
            min_n,
            min_sharpe,
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
    logger.info("=== REGIME V7 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V7: no fold results to display")
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
    """Log the consolidated final summary of the Regime V7 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        pooled: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V7 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V7 SUMMARY: no results")
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
# Module-level CLI (direct execution: python experiments/regime_v7.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V7 event-based signal pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V7: event-based signal pipeline. "
            "Detects validated sentiment events (saturation, divergence, "
            "exhaustion) on training data and trades only when those events "
            "fire on the test set."
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
            "Minimum training event observations for validation. "
            f"Default: {DEFAULT_MIN_N}."
        ),
    )
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=DEFAULT_MIN_SHARPE,
        metavar="S",
        help=(
            "Minimum train-set Sharpe for an event to be considered valid. "
            f"Default: {DEFAULT_MIN_SHARPE}."
        ),
    )
    p.add_argument(
        "--streak-threshold",
        type=int,
        default=DEFAULT_STREAK_THRESHOLD,
        metavar="K",
        help=(
            "Minimum extreme_streak_70 value for saturation / exhaustion events. "
            f"Default: {DEFAULT_STREAK_THRESHOLD}."
        ),
    )
    p.add_argument(
        "--trend-threshold",
        type=float,
        default=DEFAULT_TREND_THRESHOLD,
        metavar="T",
        help=(
            "Boundary for abs(trend_strength_48b): above → saturation, "
            "at-or-below → exhaustion. "
            f"Default: {DEFAULT_TREND_THRESHOLD}."
        ),
    )
    p.add_argument(
        "--divergence-threshold",
        type=float,
        default=DEFAULT_DIVERGENCE_THRESHOLD,
        metavar="D",
        help=(
            "Minimum abs(divergence) for the DIVERGENCE_EVENT. "
            f"Default: {DEFAULT_DIVERGENCE_THRESHOLD}."
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
        "=== REGIME V7 === window=%d  min_n=%d  min_sharpe=%.4f"
        "  streak_threshold=%d  trend_threshold=%.2f  divergence_threshold=%.2f",
        args.window,
        args.min_n,
        args.min_sharpe,
        args.streak_threshold,
        args.trend_threshold,
        args.divergence_threshold,
    )

    df = load_data(args.data, window=args.window)

    require_columns(df, _REQUIRED_COLS, context="regime_v7.main")
    _log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        min_sharpe=args.min_sharpe,
        streak_threshold=args.streak_threshold,
        trend_threshold=args.trend_threshold,
        divergence_threshold=args.divergence_threshold,
    )

    log_fold_results(fold_df)
    pooled = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, pooled)


if __name__ == "__main__":
    main()
