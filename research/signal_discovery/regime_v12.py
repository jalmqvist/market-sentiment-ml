# Legacy experiment — not part of current validated approach\n"""
experiments/regime_v12.py
=========================
EVENT-BASED signal pipeline for FX sentiment research — **Version 12**.

V12 is a structural upgrade of V11.  Where V11 ranks events within context
groups but treats all contexts equally, V12 adds a **context selection** step
before ranking: only contexts that demonstrate signal quality on training data
(above a minimum Sharpe threshold) are kept.  Noisy contexts are discarded
entirely.

Key changes vs V11
------------------
* **Context selection** — before ranking on the test set, each context is
  evaluated on training events.  A context is *selected* only when:

  * ``n_train_events >= min_context_events``
  * ``train_sharpe >= min_context_sharpe``

  All other contexts are *rejected* and their test events are ignored.

* **New parameter** — ``min_context_sharpe`` (default 0.02): minimum
  training Sharpe required for a context to be used.

* **Extended logging** — per-fold logs now report number of contexts
  selected vs rejected, and the top-5 per-context training Sharpe values.

Pipeline overview
-----------------
1. **Event flag** — a row is an event when ALL of:

   * ``abs_sentiment >= 70``
   * ``extreme_streak_70 >= 2``

   Only event rows are ever eligible for trading.

2. **Raw score** (identical to V10/V11)::

       score_raw = (
           (abs_sentiment / 100) ** 2
         - 0.5 * abs(net_sentiment * trend_strength_48b)
         + 0.3 * log1p(extreme_streak_70)
       )

3. **Rolling volatility proxy** — per-pair (or global) rolling std of
   ``net_sentiment`` over the most recent 48 rows (backward-looking only).

4. **Context key** (per fold, using train-only tertile thresholds):

   a. Compute ``vol_q33``, ``vol_q67`` from train ``rolling_vol``.
   b. Compute ``trend_q33``, ``trend_q67`` from train ``trend_strength_48b``.
   c. Compute ``sent_q33``, ``sent_q67`` from train ``abs_sentiment``.
   d. Assign bucket labels to all rows using train thresholds.
   e. ``context_key = vol_bucket + "_" + trend_bucket + "_" + sentiment_bucket``

5. **Score normalization** — per fold, z-score parameters (mean, std) are
   estimated on the **train split only** and applied to the test split.

6. **Context train statistics** (Step 1 from spec) — for each context key
   found in training events:

   * ``n_train_events`` — number of training events in the context
   * ``train_mean_ret``  — mean PnL across training events in the context
   * ``train_sharpe``    — ``mean_pnl / std_pnl``

   PnL is computed using the contrarian position::

       pnl = -sign(net_sentiment) × score_norm × ret_48b

7. **Context selection** (Step 2 from spec) — a context is selected if:

   * ``n_train_events >= min_context_events``
   * ``train_sharpe >= min_context_sharpe``

8. **Context-aware ranking + selection** (Step 3–4 from spec) — within each
   test fold:

   a. Extract event rows from the test split.
   b. Discard events whose ``context_key`` is not in the selected set.
   c. Group remaining events by ``context_key``.
   d. Within each group, rank events by normalized score (descending) and
      select the top ``top_frac`` fraction.

9. **Position** (contrarian, selected events only)::

       base_direction = -sign(net_sentiment)   # fade extreme sentiment
       position       = base_direction × score_normalized

10. **Walk-forward** — expanding window; minimum 2 prior years before the
    first test year.

11. **Metrics** (per fold):

    * ``n_total_events``     — events detected in the test split
    * ``n_selected_events``  — events traded (after context selection + ranking)
    * ``selection_ratio``    — ``n_selected_events / n_total_events``
    * ``coverage``           — ``n_selected_events / total_test_rows``
    * ``mean_score``         — mean normalized score across *selected* events
    * ``sharpe``             — ``mean(pnl) / std(pnl)`` (selected events only)
    * ``hit_rate``           — fraction of selected events with ``pnl > 0``
    * ``n_contexts``         — number of contexts used (selected + passing test filter)
    * ``n_contexts_selected`` — contexts passing both train filters
    * ``n_contexts_rejected`` — contexts failing at least one train filter

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
   "coverage", "mean_score", "sharpe", "hit_rate", "n_contexts",
   "n_contexts_selected", "n_contexts_rejected"]``

Usage (direct)::

    python experiments/regime_v12.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v12.py \\
        --data data/output/master_research_dataset.csv \\
        --top-frac 0.3 \\
        --min-context-sharpe 0.05 \\
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

#: Event definition thresholds (unchanged from V9/V10/V11).
_EVENT_ABS_SENTIMENT_MIN: int = 70
_EVENT_STREAK_MIN: int = 2

#: Default fraction of top-ranked events to trade (within each context).
_DEFAULT_TOP_FRAC: float = 0.2

#: Minimum number of training events in a context to consider it.
_MIN_CONTEXT_EVENTS: int = 30

#: Minimum training Sharpe a context must reach to be selected (V12 new).
_MIN_CONTEXT_SHARPE: float = 0.02

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
    "n_contexts_selected",
    "n_contexts_rejected",
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
        log_path = logs_dir / f"regime_v12_{timestamp}.log"
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
    """Load and prepare the dataset for Regime V12.

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
    df = parse_timestamps(df, "time", context="regime_v12.load_data")
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
    """Compute the event flag column (identical to V9/V10/V11).

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
# Score formula (non-linear — same as V10/V11)
# ---------------------------------------------------------------------------

def compute_raw_score(df: pd.DataFrame) -> pd.Series:
    """Compute the raw (un-normalized) event score for every row.

    V10/V11/V12 non-linear formula::

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
    q33 = float(clean.quantile(1 / 3))
    q67 = float(clean.quantile(2 / 3))
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
# Context train statistics (V12 new)
# ---------------------------------------------------------------------------

def compute_context_train_stats(
    train_events: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
) -> dict[str, dict[str, Any]]:
    """Compute per-context signal statistics from training events.

    For each context key found in ``train_events``, computes the contrarian
    PnL series and derives ``n_events``, ``mean_ret``, and ``sharpe``.

    PnL per event::

        pnl = -sign(net_sentiment) × score_norm × ret_48b

    Args:
        train_events: DataFrame of training event rows.  Must contain
            ``context_key``, ``net_sentiment``, ``score_norm``, and
            *target_col*.
        target_col: Forward-return column name.

    Returns:
        Dict mapping ``context_key`` → ``{n_events, mean_ret, sharpe}``.
    """
    stats: dict[str, dict[str, Any]] = {}

    for ctx_key, grp in train_events.groupby("context_key", sort=False):
        n = len(grp)
        if n < 2:
            # Cannot compute std with fewer than 2 observations.
            stats[str(ctx_key)] = {
                "n_events": n,
                "mean_ret": float("nan"),
                "sharpe": float("nan"),
            }
            continue

        base_dir = -np.sign(grp["net_sentiment"].to_numpy(dtype=float))
        score = grp["score_norm"].to_numpy(dtype=float)
        ret = grp[target_col].to_numpy(dtype=float)
        pnl = base_dir * score * ret

        mean_ret = float(np.mean(pnl))
        std_ret = float(np.std(pnl))
        sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")

        stats[str(ctx_key)] = {
            "n_events": n,
            "mean_ret": mean_ret,
            "sharpe": sharpe,
        }

    return stats


def select_contexts(
    context_train_stats: dict[str, dict[str, Any]],
    *,
    min_context_events: int,
    min_context_sharpe: float,
) -> tuple[set[str], set[str]]:
    """Partition contexts into selected and rejected sets.

    A context is *selected* when ALL conditions hold:

    * ``n_events >= min_context_events``
    * ``sharpe >= min_context_sharpe``

    Args:
        context_train_stats: Dict from :func:`compute_context_train_stats`.
        min_context_events: Minimum training events required.
        min_context_sharpe: Minimum training Sharpe required.

    Returns:
        Tuple ``(selected_contexts, rejected_contexts)`` of string sets.
    """
    selected: set[str] = set()
    rejected: set[str] = set()

    for ctx_key, s in context_train_stats.items():
        n = s["n_events"]
        sharpe = s["sharpe"]
        # A NaN sharpe (< 2 obs or zero std) never passes the filter.
        passes_n = n >= min_context_events
        passes_sharpe = (not np.isnan(sharpe)) and (sharpe >= min_context_sharpe)
        if passes_n and passes_sharpe:
            selected.add(ctx_key)
        else:
            rejected.add(ctx_key)

    return selected, rejected


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
    min_context_sharpe: float = _MIN_CONTEXT_SHARPE,
) -> pd.DataFrame:
    """Regime-V12 walk-forward: context selection + context-aware event ranking.

    For each test year (from the third unique year onward, i.e. at least 2
    training years):

    1. Split into train / test by year (expanding window).
    2. Drop rows with NaN in any required column or in ``score_raw``.
    3. Fit z-score normalization parameters (mean, std) on the **train** split.
    4. Normalize scores on the test split using train parameters.
    5. Compute tertile thresholds for ``rolling_vol``, ``trend_strength_48b``,
       and ``abs_sentiment`` from the train split.
    6. Assign context keys to all rows using train thresholds.
    7. Compute per-context train statistics (n, mean_ret, sharpe) on training
       event rows only.
    8. Select contexts: keep only contexts with ``n >= min_context_events``
       AND ``sharpe >= min_context_sharpe``.
    9. Restrict test events to selected contexts; discard the rest.
    10. Group remaining test events by ``context_key``.
    11. Within each context group, rank events by normalized score (descending)
        and select the top ``top_frac`` fraction.
    12. Build contrarian positions for selected events (all others: position=0).
    13. Compute fold-level metrics.

    No test-period information enters score normalization, bucket thresholds,
    or context selection.

    Args:
        df: Full dataset (after :func:`load_data`).  Must contain all columns
            in ``_REQUIRED_COLS``.
        target_col: Forward-return column to evaluate positions against.
        year_col: Column containing calendar year.
        top_frac: Fraction (0, 1] of top-ranked events to trade per context
            group per fold.
        min_context_events: Minimum training events required for a context to
            be selected.
        min_context_sharpe: Minimum training Sharpe required for a context to
            be selected.

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
                "REGIME V12 [year=%d]: empty train set; skipping fold",
                test_year,
            )
            continue

        if test_df.empty:
            logger.warning(
                "REGIME V12 [year=%d]: empty test set; skipping fold",
                test_year,
            )
            continue

        logger.info(
            "REGIME V12 [year=%d] | train_size=%d | test_size=%d",
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
                "REGIME V12 [year=%d]: train score_std=%.2e near zero; "
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
            "REGIME V12 [year=%d] | vol thresholds=(%.4f, %.4f) "
            "| trend thresholds=(%.4f, %.4f) "
            "| sent thresholds=(%.4f, %.4f)",
            test_year,
            vol_q33, vol_q67,
            trend_q33, trend_q67,
            sent_q33, sent_q67,
        )

        # ------------------------------------------------------------------
        # Normalize scores using TRAIN parameters
        # ------------------------------------------------------------------
        train_df = train_df.copy()
        train_df["score_norm"] = (train_df["score_raw"] - score_mean) / score_std

        test_df = test_df.copy()
        test_df["score_norm"] = (test_df["score_raw"] - score_mean) / score_std

        # ------------------------------------------------------------------
        # Assign context keys using TRAIN thresholds (to both splits)
        # ------------------------------------------------------------------
        ctx_kwargs = dict(
            vol_q33=vol_q33,
            vol_q67=vol_q67,
            trend_q33=trend_q33,
            trend_q67=trend_q67,
            sent_q33=sent_q33,
            sent_q67=sent_q67,
        )
        train_df["context_key"] = assign_context_keys(train_df, **ctx_kwargs)
        test_df["context_key"] = assign_context_keys(test_df, **ctx_kwargs)

        # ------------------------------------------------------------------
        # Step 1 — Context train statistics (V12 new)
        # Compute per-context n, mean_ret, sharpe from TRAIN events only.
        # ------------------------------------------------------------------
        train_events = train_df[train_df["is_event"] == 1]
        context_train_stats = compute_context_train_stats(
            train_events, target_col=target_col
        )

        logger.debug(
            "REGIME V12 [year=%d] | n_contexts_in_train=%d",
            test_year,
            len(context_train_stats),
        )

        # ------------------------------------------------------------------
        # Step 2 — Select contexts (V12 new)
        # ------------------------------------------------------------------
        selected_contexts, rejected_contexts = select_contexts(
            context_train_stats,
            min_context_events=min_context_events,
            min_context_sharpe=min_context_sharpe,
        )

        n_selected_ctx = len(selected_contexts)
        n_rejected_ctx = len(rejected_contexts)

        logger.info(
            "REGIME V12 [year=%d] | contexts_selected=%d | contexts_rejected=%d"
            " | min_n=%d | min_sharpe=%.4f",
            test_year,
            n_selected_ctx,
            n_rejected_ctx,
            min_context_events,
            min_context_sharpe,
        )

        # Log top-5 per-context training Sharpe
        if context_train_stats:
            top5 = sorted(
                [
                    (k, v["sharpe"], v["n_events"])
                    for k, v in context_train_stats.items()
                    if not np.isnan(v["sharpe"])
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for rank, (ctx_key, s, n) in enumerate(top5, 1):
                logger.info(
                    "REGIME V12 [year=%d] | top_context #%d | context=%s"
                    " | train_sharpe=%.4f | n_train_events=%d",
                    test_year,
                    rank,
                    ctx_key,
                    s,
                    n,
                )

        if n_selected_ctx == 0:
            logger.warning(
                "REGIME V12 [year=%d]: no contexts passed selection "
                "(min_n=%d, min_sharpe=%.4f); skipping fold",
                test_year,
                min_context_events,
                min_context_sharpe,
            )
            continue

        # ------------------------------------------------------------------
        # Apply event filter on test set
        # ------------------------------------------------------------------
        n_total = len(test_df)
        event_df = test_df[test_df["is_event"] == 1].copy()
        n_total_events = len(event_df)

        logger.info(
            "REGIME V12 [year=%d] | total_events=%d",
            test_year,
            n_total_events,
        )

        if n_total_events == 0:
            logger.warning(
                "REGIME V12 [year=%d]: no events in test set; skipping fold",
                test_year,
            )
            continue

        # ------------------------------------------------------------------
        # Step 3 — Filter test events to selected contexts (V12 new)
        # ------------------------------------------------------------------
        event_df = event_df[event_df["context_key"].isin(selected_contexts)]
        n_events_after_ctx_filter = len(event_df)

        logger.info(
            "REGIME V12 [year=%d] | events_after_context_filter=%d"
            " (discarded=%d)",
            test_year,
            n_events_after_ctx_filter,
            n_total_events - n_events_after_ctx_filter,
        )

        if n_events_after_ctx_filter == 0:
            logger.warning(
                "REGIME V12 [year=%d]: no test events remain after context "
                "filter; skipping fold",
                test_year,
            )
            continue

        # ------------------------------------------------------------------
        # Step 4 — Rank within selected contexts
        # ------------------------------------------------------------------
        selected_indices: list = []
        active_context_stats: dict[str, dict[str, int]] = {}

        for ctx_key, ctx_group in event_df.groupby("context_key", sort=False):
            n_ctx = len(ctx_group)

            n_sel = max(1, int(np.ceil(top_frac * n_ctx)))
            top_indices = (
                ctx_group["score_norm"]
                .nlargest(n_sel)
                .index.tolist()
            )
            selected_indices.extend(top_indices)
            active_context_stats[str(ctx_key)] = {
                "n_events": n_ctx,
                "n_selected": n_sel,
            }

        n_contexts = len(active_context_stats)

        # Log context breakdown
        logger.info(
            "REGIME V12 [year=%d] | n_contexts_active=%d",
            test_year,
            n_contexts,
        )
        for ctx_key, stats in sorted(active_context_stats.items()):
            logger.debug(
                "REGIME V12 [year=%d] | context=%s | n_events=%d | n_selected=%d",
                test_year,
                ctx_key,
                stats["n_events"],
                stats["n_selected"],
            )

        if n_contexts == 0:
            logger.warning(
                "REGIME V12 [year=%d]: no active context groups; skipping fold",
                test_year,
            )
            continue

        # ------------------------------------------------------------------
        # Step 5 — Build position for selected events
        # ------------------------------------------------------------------
        sel_df = event_df.loc[selected_indices]
        n_selected_events = len(sel_df)
        selection_ratio = n_selected_events / n_total_events
        coverage = n_selected_events / n_total if n_total > 0 else 0.0

        logger.info(
            "REGIME V12 [year=%d] | selected_events=%d | selection_ratio=%.4f"
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
            "REGIME V12 [year=%d] | mean_score (normalized, selected)=%.4f",
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
            n_contexts_selected=n_selected_ctx,
            n_contexts_rejected=n_rejected_ctx,
        )

        logger.info(
            "REGIME V12 FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%% | mean_score=%+.4f"
            " | sharpe=%+.4f | hit_rate=%.4f | n_contexts=%d"
            " | ctx_selected=%d | ctx_rejected=%d",
            test_year,
            metrics["n_total_events"],
            metrics["n_selected_events"],
            metrics["selection_ratio"],
            100.0 * metrics["coverage"],
            metrics["mean_score"],
            metrics["sharpe"],
            metrics["hit_rate"],
            metrics["n_contexts"],
            metrics["n_contexts_selected"],
            metrics["n_contexts_rejected"],
        )

        fold_rows.append({"year": int(test_year), **metrics})

    if not fold_rows:
        logger.warning("REGIME V12: no valid folds produced")
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
    n_contexts_selected: int,
    n_contexts_rejected: int,
) -> dict[str, Any]:
    """Compute fold-level performance metrics for Regime V12.

    Args:
        pnl: PnL series (``position × ret_48b``) for *selected* event rows.
        n_total_events: Total events detected in the test split.
        n_selected_events: Events traded (after context selection + ranking).
        selection_ratio: ``n_selected_events / n_total_events``.
        coverage: ``n_selected_events / total_test_rows``.
        mean_score: Mean normalized score across *selected* event rows.
        n_contexts: Number of context groups active in the test fold.
        n_contexts_selected: Contexts that passed train selection filters.
        n_contexts_rejected: Contexts that failed at least one train filter.

    Returns:
        Dict with keys matching ``_FOLD_COLS`` (minus ``year``).
    """
    n = len(pnl)

    base = {
        "n_total_events": n_total_events,
        "n_selected_events": n_selected_events,
        "selection_ratio": selection_ratio,
        "coverage": coverage,
        "mean_score": mean_score,
        "n_contexts": n_contexts,
        "n_contexts_selected": n_contexts_selected,
        "n_contexts_rejected": n_contexts_rejected,
    }

    if n < 2:
        return {**base, "sharpe": float("nan"), "hit_rate": float("nan")}

    mean_ret = float(np.mean(pnl))
    std_ret = float(np.std(pnl))
    sharpe = mean_ret / std_ret if std_ret > 1e-10 else float("nan")
    hit_rate = float(np.mean(pnl > 0))

    return {**base, "sharpe": sharpe, "hit_rate": hit_rate}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_pooled_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled aggregate metrics across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.

    Returns:
        Dict with aggregate keys across folds.
    """
    if fold_df.empty:
        return {
            "folds": 0,
            "mean_sharpe": float("nan"),
            "mean_hit_rate": float("nan"),
            "mean_coverage": float("nan"),
            "mean_selection_ratio": float("nan"),
            "mean_n_contexts": float("nan"),
            "mean_n_contexts_selected": float("nan"),
            "mean_n_contexts_rejected": float("nan"),
        }
    return {
        "folds": len(fold_df),
        "mean_sharpe": float(fold_df["sharpe"].dropna().mean()),
        "mean_hit_rate": float(fold_df["hit_rate"].dropna().mean()),
        "mean_coverage": float(fold_df["coverage"].mean()),
        "mean_selection_ratio": float(fold_df["selection_ratio"].mean()),
        "mean_n_contexts": float(fold_df["n_contexts"].mean()),
        "mean_n_contexts_selected": float(fold_df["n_contexts_selected"].mean()),
        "mean_n_contexts_rejected": float(fold_df["n_contexts_rejected"].mean()),
    }


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V12 — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V12: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | total_events=%5d | selected=%5d"
            " | sel_ratio=%.4f | coverage=%.2f%%"
            " | mean_score=%+.4f | sharpe=%+.4f | hit_rate=%.4f"
            " | n_contexts=%d | ctx_sel=%d | ctx_rej=%d",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100.0 * row.coverage,
            row.mean_score,
            row.sharpe,
            row.hit_rate,
            row.n_contexts,
            row.n_contexts_selected,
            row.n_contexts_rejected,
        )


def log_final_summary(
    fold_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Log the consolidated final summary of the Regime V12 pipeline.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
        summary: Dict returned by :func:`compute_pooled_summary`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V12 — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V12 SUMMARY: no results")
        return

    logger.info("Folds evaluated        : %d", summary["folds"])
    logger.info("Mean Sharpe            : %+.4f", summary["mean_sharpe"])
    logger.info("Mean hit rate          : %.4f", summary["mean_hit_rate"])
    logger.info("Mean coverage          : %.4f", summary["mean_coverage"])
    logger.info("Mean selection ratio   : %.4f", summary["mean_selection_ratio"])
    logger.info("Mean contexts used     : %.1f", summary["mean_n_contexts"])
    logger.info("Mean contexts selected : %.1f", summary["mean_n_contexts_selected"])
    logger.info("Mean contexts rejected : %.1f", summary["mean_n_contexts_rejected"])
    logger.info(sep)
    logger.info("Per-fold detail:")
    for row in fold_df.itertuples(index=False):
        logger.info(
            "  year=%d | total=%5d | selected=%5d | sel_ratio=%.4f"
            " | coverage=%.2f%% | sharpe=%+.4f | hit_rate=%.4f"
            " | n_contexts=%d | ctx_sel=%d | ctx_rej=%d",
            row.year,
            row.n_total_events,
            row.n_selected_events,
            row.selection_ratio,
            100.0 * row.coverage,
            row.sharpe,
            row.hit_rate,
            row.n_contexts,
            row.n_contexts_selected,
            row.n_contexts_rejected,
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Module-level CLI (direct execution: python experiments/regime_v12.py)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the Regime V12 context-selection + event-ranking pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=(
            "Regime V12: context selection + context-aware event ranking pipeline. "
            "Detects high-conviction sentiment events, evaluates each context on "
            "training data (n + Sharpe filter), keeps only signal-bearing contexts, "
            "then ranks + selects events WITHIN each surviving context. "
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
            "Minimum number of training events required in a context for it to "
            f"be selected. Default: {_MIN_CONTEXT_EVENTS}."
        ),
    )
    p.add_argument(
        "--min-context-sharpe",
        type=float,
        default=_MIN_CONTEXT_SHARPE,
        metavar="S",
        help=(
            "Minimum training Sharpe required for a context to be selected. "
            f"Default: {_MIN_CONTEXT_SHARPE}."
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
    log.info("=== REGIME V12 (CONTEXT SELECTION + EVENT RANKING) ===")
    log.info(
        "top_frac=%.4f | min_context_events=%d | min_context_sharpe=%.4f",
        args.top_frac,
        args.min_context_events,
        args.min_context_sharpe,
    )

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS, context="regime_v12.main")
    log.info("Dataset ready: %d rows", len(df))

    fold_df = walk_forward(
        df,
        top_frac=args.top_frac,
        min_context_events=args.min_context_events,
        min_context_sharpe=args.min_context_sharpe,
    )

    log_fold_results(fold_df)
    summary = compute_pooled_summary(fold_df)
    log_final_summary(fold_df, summary)


if __name__ == "__main__":
    main()
