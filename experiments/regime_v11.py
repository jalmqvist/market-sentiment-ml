"""
experiments/regime_v11.py
=========================
EVENT-BASED signal pipeline for FX sentiment research — **Version 11**.

V11 is a structural upgrade of V10. Where V10 ranked events globally across
all regimes, V11 introduces **context-aware event ranking**: events are ranked
and selected WITHIN similar regime contexts, not mixed across them.

Key changes vs V10
------------------
* **Context key** — each row is assigned a 3-component regime/context key::

      context_key = vol_bucket + "_" + trend_bucket + "_" + sentiment_bucket

  where each bucket is a tertile label derived from **training-data quantiles
  only** (no leakage).

* **vol_bucket** — ``low`` / ``mid`` / ``high``: rolling standard deviation of
  ``net_sentiment`` (48-bar window, per-pair when ``pair`` column is present).

* **trend_bucket** — ``down`` / ``flat`` / ``up``: tertile discretization of
  ``trend_strength_48b``.

* **sentiment_bucket** — ``low`` / ``mid`` / ``high``: tertile discretization
  of ``abs_sentiment``.

* **Context-aware ranking** — events are ranked by normalized score WITHIN
  each context group independently.  Mixing across contexts is never done.

* **Min sample filter** — context groups with fewer than ``min_context_events``
  (default 30) events in the test fold are skipped.

* **Extended logging** — per-fold logs now also report number of active
  context groups and events-per-context distribution.

Pipeline overview
-----------------
1. **Event flag** — a row is an event when ALL of:

   * ``abs_sentiment >= 70``
   * ``extreme_streak_70 >= 2``

   Only event rows are ever eligible for trading.

2. **Raw score** (identical to V10)::

       score_raw = (
           (abs_sentiment / 100) ** 2
         - 0.5 * abs(net_sentiment * trend_strength_48b)
         + 0.3 * log1p(extreme_streak_70)
       )

3. **Rolling volatility proxy** — per-pair (or global) rolling std of
   ``net_sentiment`` over the most recent 48 rows (backward-looking only;
   no forward-looking bias).

4. **Context key** (per fold, using train-only tertile thresholds):

   a. Compute ``vol_q33``, ``vol_q67`` from train ``rolling_vol``.
   b. Compute ``trend_q33``, ``trend_q67`` from train ``trend_strength_48b``.
   c. Compute ``sent_q33``, ``sent_q67`` from train ``abs_sentiment``.
   d. Assign bucket labels to test rows using train thresholds.
   e. ``context_key = vol_bucket + "_" + trend_bucket + "_" + sentiment_bucket``

5. **Score normalization** — per fold, z-score parameters (mean, std) are
   estimated on the **train split only** and applied to the test split.
   No test information enters normalization.

6. **Context-aware ranking + selection** (per fold, test set only):

   a. Extract event rows from the test split.
   b. Group by ``context_key``.
   c. Skip context groups with ``n_events < min_context_events`` (default 30).
   d. Within each valid context group, rank events by normalized score
      (descending) and select the top ``top_frac`` fraction.
   e. All other events receive position = 0.

7. **Position** (contrarian, selected events only)::

       base_direction = -sign(net_sentiment)   # fade extreme sentiment
       position       = base_direction × score_normalized

8. **Walk-forward** — expanding window; minimum 2 prior years before the first
   test year.

9. **Metrics** (per fold):

   * ``n_total_events``    — events detected in the test split
   * ``n_selected_events`` — events traded (after context-aware selection)
   * ``selection_ratio``   — ``n_selected_events / n_total_events``
   * ``coverage``          — ``n_selected_events / total_test_rows``
   * ``mean_score``        — mean normalized score across *selected* events
   * ``sharpe``            — ``mean(pnl) / std(pnl)``  (selected events only)
   * ``hit_rate``          — fraction of selected events with ``pnl > 0``
   * ``n_contexts``        — number of context groups used (after min filter)

Required columns
----------------
``ret_48b``, ``net_sentiment``, ``abs_sentiment``, ``extreme_streak_70``,
``trend_strength_48b``

Optional columns
----------------
``pair`` — used for per-pair rolling volatility (falls back to global if absent)

Fold output schema
------------------
``["year", "n_total_events", "n_selected_events", "selection_ratio",
   "coverage", "mean_score", "sharpe", "hit_rate", "n_contexts"]``

Usage (direct)::

    python experiments/regime_v11.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v11.py \\
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

import config as cfg
from utils.io import read_csv
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Columns that must be present in the dataset.
_REQUIRED_COLS: list[str] = [
    TARGET_COL,
    "net_sentiment",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
]

#: Event definition thresholds (unchanged from V9/V10).
_EVENT_ABS_SENTIMENT_MIN: int = 70
_EVENT_STREAK_MIN: int = 2

#: Default fraction of top-ranked events to trade (within each context).
_DEFAULT_TOP_FRAC: float = 0.2

#: Minimum number of events in a context group to use it.
_MIN_CONTEXT_EVENTS: int = 30

#: Rolling window (bars) for the volatility proxy.
_ROLLING_VOL_WINDOW: int = 48

#: Output fold columns.
_FOLD_COLS: list[str] = [
    "year",
    "n_total_events",
    "n_selected_events",
    "selection_ratio",
    "coverage",
    "mean_score",
    "sharpe",
    "hit_rate",
    "n_contexts",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


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
        log_path = logs_dir / f"regime_v11_{timestamp}.log"
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
    """Load and prepare the dataset for Regime V11.

    Reads the canonical research dataset CSV, parses the ``time`` column to
    derive a ``year`` column, and validates that all required columns are
    present.  NaN rows are **not** dropped here; that is deferred to the
    walk-forward step.

    Args:
        path: Path to the master research dataset CSV.

    Returns:
        DataFrame with a ``year`` column added.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing after loading.
    """
    df = read_csv(path, required_columns=["time"] + _REQUIRED_COLS)
    df = parse_timestamps(df, "time", context="regime_v11.load_data")
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
# Event flag
# ---------------------------------------------------------------------------

def compute_event_flag(df: pd.DataFrame) -> pd.Series:
    """Compute the event flag column (identical to V9/V10).

    A row is flagged as an event when:

    * ``abs_sentiment >= 70``   (high-conviction extreme reading)
    * ``extreme_streak_70 >= 2`` (persistence of at least 2 consecutive bars)

    Args:
        df: DataFrame with ``abs_sentiment`` and ``extreme_streak_70`` columns.

    Returns:
        Integer Series (0/1) aligned with *df*.
    """
    return (
        (df["abs_sentiment"] >= _EVENT_ABS_SENTIMENT_MIN)
        & (df["extreme_streak_70"] >= _EVENT_STREAK_MIN)
    ).astype(int)


# ---------------------------------------------------------------------------
# Score formula (non-linear — same as V10)
# ---------------------------------------------------------------------------

def compute_raw_score(df: pd.DataFrame) -> pd.Series:
    """Compute the raw (un-normalized) event score for every row.

    V10/V11 non-linear formula::

        score_raw = (
            (abs_sentiment / 100) ** 2
          - 0.5 * abs(net_sentiment * trend_strength_48b)
          + 0.3 * log1p(extreme_streak_70)
        )

    Args:
        df: DataFrame with the required feature columns.

    Returns:
        Float Series aligned with *df*.
    """
    raw = (
        (df["abs_sentiment"] / 100.0) ** 2
        - 0.5 * (df["net_sentiment"] * df["trend_strength_48b"]).abs()
        + 0.3 * np.log1p(df["extreme_streak_70"])
    )
    return raw.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Rolling volatility proxy
# ---------------------------------------------------------------------------

def compute_rolling_vol(
    df: pd.DataFrame,
    window: int = _ROLLING_VOL_WINDOW,
) -> pd.Series:
    """Compute a rolling volatility proxy from ``net_sentiment``.

    Uses the backward-looking rolling standard deviation of ``net_sentiment``
    over ``window`` bars.  When a ``pair`` column is present the rolling window
    is computed independently for each pair, which prevents cross-pair
    contamination.

    Args:
        df: DataFrame sorted by ``time`` (must already be sorted).
        window: Number of bars for the rolling window.

    Returns:
        Float Series (same index as *df*) representing local sentiment
        volatility.  Rows with fewer than ``window // 2`` preceding observations
        will contain NaN.
    """
    min_periods = max(2, window // 2)
    if "pair" in df.columns:
        return df.groupby("pair", sort=False)["net_sentiment"].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
    return df["net_sentiment"].rolling(window, min_periods=min_periods).std()


# ---------------------------------------------------------------------------
# Context key helpers
# ---------------------------------------------------------------------------

def _tertile_thresholds(series: pd.Series) -> tuple[float, float]:
    """Return the 33rd and 67th percentile of *series* (dropping NaN).

    Args:
        series: Numeric pandas Series.

    Returns:
        Tuple ``(q33, q67)``.
    """
    clean = series.dropna()
    q33 = float(clean.quantile(0.333))
    q67 = float(clean.quantile(0.667))
    # Ensure strictly increasing thresholds to avoid degenerate bins
    if q67 <= q33:
        q67 = q33 + 1e-10
    return q33, q67


def _assign_tertile_bucket(
    series: pd.Series,
    q33: float,
    q67: float,
    labels: tuple[str, str, str] = ("low", "mid", "high"),
) -> pd.Series:
    """Assign tertile bucket labels to *series* using pre-computed thresholds.

    Rows where *series* is NaN are labelled ``"unknown"``.

    Args:
        series: Numeric pandas Series to bucket.
        q33: Lower tertile boundary (33rd-percentile from training data).
        q67: Upper tertile boundary (67th-percentile from training data).
        labels: Three string labels for the low / mid / high buckets.

    Returns:
        String Series aligned with *series*.
    """
    result = pd.Series(labels[1], index=series.index, dtype=object)
    result[series < q33] = labels[0]
    result[series >= q67] = labels[2]
    result[series.isna()] = "unknown"
    return result


def assign_context_keys(
    df: pd.DataFrame,
    *,
    vol_q33: float,
    vol_q67: float,
    trend_q33: float,
    trend_q67: float,
    sent_q33: float,
    sent_q67: float,
) -> pd.Series:
    """Assign a context key to every row based on pre-computed thresholds.

    The key is formed by concatenating three bucket labels::

        context_key = vol_bucket + "_" + trend_bucket + "_" + sentiment_bucket

    Bucket labels:
    * ``vol_bucket``       — ``low`` / ``mid`` / ``high``
    * ``trend_bucket``     — ``down`` / ``flat`` / ``up``
    * ``sentiment_bucket`` — ``low`` / ``mid`` / ``high``

    All thresholds must be derived from the training split only.

    Args:
        df: DataFrame containing ``rolling_vol``, ``trend_strength_48b``,
            and ``abs_sentiment`` columns.
        vol_q33: 33rd-percentile of ``rolling_vol`` in the training split.
        vol_q67: 67th-percentile of ``rolling_vol`` in the training split.
        trend_q33: 33rd-percentile of ``trend_strength_48b`` in the training
            split.
        trend_q67: 67th-percentile of ``trend_strength_48b`` in the training
            split.
        sent_q33: 33rd-percentile of ``abs_sentiment`` in the training split.
        sent_q67: 67th-percentile of ``abs_sentiment`` in the training split.

    Returns:
        String Series of context keys aligned with *df*.
    """
    vol_bucket = _assign_tertile_bucket(
        df["rolling_vol"], vol_q33, vol_q67, ("low", "mid", "high")
    )
    trend_bucket = _assign_tertile_bucket(
        df["trend_strength_48b"], trend_q33, trend_q67, ("down", "flat", "up")
    )
    sent_bucket = _assign_tertile_bucket(
        df["abs_sentiment"], sent_q33, sent_q67, ("low", "mid", "high")
    )
    return vol_bucket + "_" + trend_bucket + "_" + sent_bucket


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    top_frac: float = _DEFAULT_TOP_FRAC,
    min_context_events: int = _MIN_CONTEXT_EVENTS,
) -> pd.DataFrame:
    """Regime-V11 walk-forward: context-aware event ranking + selection.

    For each test year (from the third unique year onward, i.e. at least 2
    training years):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any required column or in ``score_raw``.
    3. Fit z-score normalization parameters (mean, std) on the **train** split.
    4. Normalize scores on the test split using train parameters.
    5. Compute tertile thresholds for ``rolling_vol``, ``trend_strength_48b``,
       and ``abs_sentiment`` from the train split.
    6. Assign context keys to test rows using train thresholds.
    7. Restrict to event rows in the test split.
    8. Group events by ``context_key``.
    9. Skip context groups with ``n_events < min_context_events``.
    10. Within each valid context group, rank events by normalized score
        (descending) and select the top ``top_frac`` fraction.
    11. Build contrarian positions for selected events (all others: position=0).
    12. Compute fold-level metrics.

    No test-period information enters score normalization or bucket thresholds.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            in ``_REQUIRED_COLS``.
        target_col: Forward-return column to evaluate positions against.
        year_col: Column containing calendar year.
        top_frac: Fraction (0, 1] of top-ranked events to trade per context
            group per fold.
        min_context_events: Minimum number of events required in a context group
            for it to be included.  Groups with fewer events are skipped.

    Returns:
        DataFrame with schema ``_FOLD_COLS``; one row per valid test fold.
    """
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1]; got {top_frac}")

    if year_col not in df.columns:
        logger.warning("walk_forward: year column '%s' not found", year_col)
        return pd.DataFrame(columns=_FOLD_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 3:
        logger.warning(
            "walk_forward: need at least 3 unique years, got %d", len(years)
        )
        return pd.DataFrame(columns=_FOLD_COLS)

    # Pre-compute rolling vol and raw score for all rows before splitting.
    # Rolling vol is backward-looking only — no leakage.
    df = df.copy()
    if "time" in df.columns:
        df = df.sort_values("time")
    df["rolling_vol"] = compute_rolling_vol(df)
    df["score_raw"] = compute_raw_score(df)
    df["is_event"] = compute_event_flag(df)

    score_cols = _REQUIRED_COLS + ["score_raw"]
    fold_rows: list[dict[str, Any]] = []

    for i in range(2, len(years)):
        test_year = years[i]
        train_df = df[df[year_col] < test_year].dropna(subset=score_cols)
        test_df = df[df[year_col] == test_year].dropna(subset=score_cols)

        if train_df.empty:
            logger.warning(
                "REGIME V11 [year=%d]: empty train set; skipping fold",
                test_year,
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V11 [year=%d]: empty test set; skipping fold",
                test_year,
            )
            continue

        logger.info(
            "REGIME V11 [year=%d] | train_size=%d | test_size=%d",
            test_year,
            len(train_df),
            len(test_df),
        )

        # ------------------------------------------------------------------
        # Fit z-score parameters on TRAIN only (no leakage)
        # ------------------------------------------------------------------
        score_mean = float(train_df["score_raw"].mean())
        score_std = float(train_df["score_raw"].std())

        if score_std < 1e-10:
            logger.warning(
                "REGIME V11 [year=%d]: train score_std=%.2e near zero; "
                "using unscaled scores",
                test_year,
                score_std,
            )
            score_std = 1.0

        # ------------------------------------------------------------------
        # Fit context bucket thresholds on TRAIN only (no leakage)
        # ------------------------------------------------------------------
        vol_q33, vol_q67 = _tertile_thresholds(train_df["rolling_vol"])
        trend_q33, trend_q67 = _tertile_thresholds(train_df["trend_strength_48b"])
        sent_q33, sent_q67 = _tertile_thresholds(train_df["abs_sentiment"])

        logger.debug(
            "REGIME V11 [year=%d] | vol thresholds=(%.4f, %.4f) "
            "| trend thresholds=(%.4f, %.4f) "
            "| sent thresholds=(%.4f, %.4f)",
            test_year,
            vol_q33, vol_q67,
            trend_q33, trend_q67,
            sent_q33, sent_q67,
        )

        # ------------------------------------------------------------------
        # Normalize test scores using TRAIN parameters
        # ------------------------------------------------------------------
        test_df = test_df.copy()
        test_df["score_norm"] = (test_df["score_raw"] - score_mean) / score_std

        # ------------------------------------------------------------------
        # Assign context keys to test rows using TRAIN thresholds
        # ------------------------------------------------------------------
        test_df["context_key"] = assign_context_keys(
            test_df,
            vol_q33=vol_q33,
            vol_q67=vol_q67,
            trend_q33=trend_q33,
            trend_q67=trend_q67,
            sent_q33=sent_q33,
            sent_q67=sent_q67,
        )

        # ------------------------------------------------------------------
        # Apply event filter on test set
        # ------------------------------------------------------------------
        n_total = len(test_df)
        event_df = test_df[test_df["is_event"] == 1].copy()
        n_total_events = len(event_df)

        logger.info(
            "REGIME V11 [year=%d] | total_events=%d",
            test_year,
            n_total_events,
        )

        if n_total_events == 0:
            logger.warning(
                "REGIME V11 [year=%d]: no events in test set; skipping fold",
                test_year,
            )
            continue

        # ------------------------------------------------------------------
        # Context-aware ranking: rank WITHIN each context group
        # ------------------------------------------------------------------
        selected_indices: list = []
        context_stats: dict[str, dict[str, int]] = {}

        for ctx_key, ctx_group in event_df.groupby("context_key", sort=False):
            n_ctx = len(ctx_group)
            if n_ctx < min_context_events:
                logger.debug(
                    "REGIME V11 [year=%d] | context=%s | n=%d < %d; skipped",
                    test_year,
                    ctx_key,
                    n_ctx,
                    min_context_events,
                )
                continue

            n_sel = max(1, int(np.ceil(top_frac * n_ctx)))
            top_indices = (
                ctx_group["score_norm"]
                .nlargest(n_sel)
                .index.tolist()
            )
            selected_indices.extend(top_indices)
            context_stats[str(ctx_key)] = {"n_events": n_ctx, "n_selected": n_sel}

        n_contexts = len(context_stats)

        if n_contexts == 0:
            logger.warning(
                "REGIME V11 [year=%d]: no valid context groups (all have "
                "n_events < %d); skipping fold",
                test_year,
                min_context_events,
            )
            continue

        # Log context breakdown
        logger.info(
            "REGIME V11 [year=%d] | n_contexts=%d",
            test_year,
            n_contexts,
        )
        for ctx_key, stats in sorted(context_stats.items()):
            logger.debug(
                "REGIME V11 [year=%d] | context=%s | n_events=%d | n_selected=%d",
                test_year,
                ctx_key,
                stats["n_events"],
                stats["n_selected"],
            )

        # ------------------------------------------------------------------
        # Build position for selected events
        # ------------------------------------------------------------------
        sel_df = event_df.loc[selected_indices]
        n_selected_events = len(sel_df)
        selection_ratio = n_selected_events / n_total_events
        coverage = n_selected_events / n_total if n_total > 0 else 0.0

        logger.info(
            "REGIME V11 [year=%d] | selected_events=%d | selection_ratio=%.4f"
            " | coverage=%.2f%%",
            test_year,
            n_selected_events,
            selection_ratio,
            100.0 * coverage,
        )

        sel_score = sel_df["score_norm"].to_numpy(dtype=float)
        sel_net_sent = sel_df["net_sentiment"].to_numpy(dtype=float)
        sel_ret = sel_df[target_col].to_numpy(dtype=float)

        mean_score = float(np.mean(sel_score))
        logger.info(
            "REGIME V11 [year=%d] | mean_score (normalized, selected)=%.4f",
            test_year,
            mean_score,
        )

        base_direction = -np.sign(sel_net_sent)
        position = base_direction * sel_score
        pnl = position * sel_ret

        metrics = _fold_metrics(
            pnl,
            n_total_events=n_total_events,
            n_selected_events=n_selected_events,
            selection_ratio=selection_ratio,
            coverage=coverage,
            mean_score=mean_score,
            n_contexts=n_contexts,
        )

        logger.info(
            "REGIME V11 FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%% | mean_score=%+.4f"
            " | sharpe=%+.4f | hit_rate=%.4f | n_contexts=%d",
            test_year,
            metrics["n_total_events"],
            metrics["n_selected_events"],
            metrics["selection_ratio"],
            100.0 * metrics["coverage"],
            metrics["mean_score"],
            metrics["sharpe"],
            metrics["hit_rate"],
            metrics["n_contexts"],
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V11: no valid folds produced")
        return pd.DataFrame(columns=_FOLD_COLS)

    return pd.DataFrame(fold_rows)[_FOLD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(
    pnl: np.ndarray,
    *,
    n_total_events: int,
    n_selected_events: int,
    selection_ratio: float,
    coverage: float,
    mean_score: float,
    n_contexts: int,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V11.

    Args:
        pnl: PnL series (``position × ret_48b``) for *selected* event rows.
        n_total_events: Total events detected in the test split.
        n_selected_events: Events traded (after context-aware selection).
        selection_ratio: ``n_selected_events / n_total_events``.
        coverage: ``n_selected_events / total_test_rows``.
        mean_score: Mean normalized score across *selected* event rows.
        n_contexts: Number of valid context groups used.

    Returns:
        Dict with keys matching ``_FOLD_COLS`` (minus ``year``).
    """
    n = len(pnl)

    if n < 2:
        return {
            "n_total_events": n_total_events,
            "n_selected_events": n_selected_events,
            "selection_ratio": selection_ratio,
            "coverage": coverage,
            "mean_score": mean_score,
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "n_contexts": n_contexts,
        }

    mean_ret = float(np.mean(pnl))
    std_ret = float(np.std(pnl))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
    hit_rate = float(np.mean(pnl > 0))

    return {
        "n_total_events": n_total_events,
        "n_selected_events": n_selected_events,
        "selection_ratio": selection_ratio,
        "coverage": coverage,
        "mean_score": mean_score,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "n_contexts": n_contexts,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with keys: folds, mean_sharpe, mean_hit_rate, mean_coverage,
        mean_selection_ratio, mean_n_contexts.
    """
    if fold_df.empty:
        return {
            "folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
            "mean_selection_ratio": float("nan"),
            "mean_n_contexts": float("nan"),
        }
    return {
        "folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
        "mean_selection_ratio": float(fold_df["selection_ratio"].mean()),
        "mean_n_contexts": float(fold_df["n_contexts"].mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V11 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V11: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%%"
            " | mean_score=%+.4f | sharpe=%+.4f | hit_rate=%.4f"
            " | n_contexts=%d",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100.0 * row.coverage,
            row.mean_score,
            row.sharpe,
            row.hit_rate,
            row.n_contexts,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the Regime V11 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        summary: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V11 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V11 SUMMARY: no results")
        return

    logger.info("Folds evaluated      : %d", summary["folds"])
    logger.info("Mean Sharpe          : %+.4f", summary["mean_sharpe"])
    logger.info("Mean hit rate        : %.4f", summary["mean_hit_rate"])
    logger.info("Mean coverage        : %.4f", summary["mean_coverage"])
    logger.info("Mean selection ratio : %.4f", summary["mean_selection_ratio"])
    logger.info("Mean contexts used   : %.1f", summary["mean_n_contexts"])
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | total=%5d | selected=%5d | sel_ratio=%.4f"
            " | coverage=%.2f%% | sharpe=%+.4f | hit_rate=%.4f"
            " | n_contexts=%d",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100.0 * row.coverage,
            row.sharpe,
            row.hit_rate,
            row.n_contexts,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v11.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V11 context-aware event-ranking pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V11: context-aware event ranking + selection pipeline. "
            "Detects high-conviction sentiment events, assigns each a regime "
            "context key (vol × trend × sentiment), and ranks + selects events "
            "WITHIN each context independently. "
            "Walk-forward expanding window with train-only normalization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--top-frac",
        type=float,
        default=_DEFAULT_TOP_FRAC,
        metavar="FRAC",
        help=(
            "Fraction of top-ranked events to trade per context group per fold "
            f"(0 < FRAC <= 1). Default: {_DEFAULT_TOP_FRAC}."
        ),
    )
    p.add_argument(
        "--min-context-events",
        type=int,
        default=_MIN_CONTEXT_EVENTS,
        metavar="N",
        help=(
            "Minimum number of events required in a context group for it to be "
            f"used. Groups below this threshold are skipped. Default: {_MIN_CONTEXT_EVENTS}."
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

    log = logging.getLogger(__name__)
    log.info("=== REGIME V11 (CONTEXT-AWARE EVENT RANKING) ===")
    log.info("top_frac=%.4f | min_context_events=%d", args.top_frac, args.min_context_events)

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v11.main")
    log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        top_frac=args.top_frac,
        min_context_events=args.min_context_events,
    )

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
