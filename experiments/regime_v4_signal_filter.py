"""
experiments/regime_v4_signal_filter.py
=======================================
Signal V2 × Regime Filter hybrid pipeline (Regime V4 Signal Filter).

Uses **Signal V2** as the base signal and applies **regime keys** purely as a
conditional filter and optional direction modifier.  No regime-based weights
are applied — a regime is either selected (trade) or not (no trade).

Pipeline design
---------------
1. **Build Signal V2** (causal rolling features; no look-ahead) using
   :func:`experiments.signal_v2.build_features` and
   :func:`experiments.signal_v2.build_signal`.

2. **Define regime key** per row using four components:

   * ``vol_regime``         – training-derived tertile of ``vol_24b``
   * ``trend_dir``          – sign of ``trend_strength_48b``
   * ``trend_strength_bin`` – training-derived tertile of
                              ``abs(trend_strength_48b)``
   * ``sent_regime``        – fixed-threshold bin of ``abs_sentiment``

   Regime key format::

       regime_key = f"{vol_regime}__{trend_dir}__{trend_strength_bin}__{sent_regime}"

3. **Walk-forward loop** – strict expanding window:

   ``train_df = df[df.year < test_year]``
   ``test_df  = df[df.year == test_year]``

4. **Training** (train_df only):

   Per-regime statistics based on ``position * ret_48b`` (signal-weighted
   returns):

   * ``n``           – number of training observations in the regime
   * ``mean_return`` – mean(position × ret_48b)
   * ``sharpe``      – mean_return / std(position × ret_48b)

   Regime is *selected* when:

   * ``n >= min_n``          (default 100)
   * ``sharpe >= min_sharpe`` (default 0.05)

5. **Apply to test set**:

   Filter:
     If ``regime_key`` ∉ selected → ``position = 0``

   Direction (optional, ``--with-direction``):
     If ``mean_return >  threshold`` → keep signal as-is
     If ``mean_return < -threshold`` → flip signal (multiply by −1)
     Otherwise                       → ``position = 0``

6. **Per-fold metrics**:

   * ``n``         – active positions
   * ``mean``      – mean(position × ret_48b)
   * ``sharpe``    – mean / std
   * ``hit_rate``  – fraction of active positions with positive return
   * ``coverage``  – fraction of test rows with non-zero position

7. **Logging** (per fold):

   * ``n_selected_regimes``
   * ``coverage``
   * Top regimes (n, sharpe, mean_return)

   Final summary:
   * ``mean_sharpe``
   * ``mean_coverage``

No leakage: all regime selection is derived exclusively from training data.
Signal V2 itself is left unmodified.

Usage::

    python experiments/regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv

    python experiments/regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 100 --min-sharpe 0.05 --with-direction \\
        --direction-threshold 0.0002 --log-level DEBUG
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
from experiments.regime_v4 import (
    SENT_BINS,
    SENT_LABELS,
    _build_regime_key,
    _compute_train_cuts,
)
from experiments.signal_v2 import build_features as signal_v2_build_features
from experiments.signal_v2 import build_signal
from utils.io import read_csv
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "ret_48b"

#: Minimum training observations per regime to be eligible for selection.
MIN_N: int = 100

#: Minimum training Sharpe (mean(pos*ret) / std(pos*ret)) to select a regime.
MIN_SHARPE: float = 0.05

#: Absolute mean-return threshold for direction logic.
DIRECTION_THRESHOLD: float = 0.0002

#: Default rolling z-score window forwarded to Signal V2 feature builder.
DEFAULT_WINDOW: int = 96

#: Number of top regimes (by training Sharpe) to log per fold.
TOP_N_LOG: int = 5

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with file-only output (no stdout).

    If *log_file* is ``None``, a timestamped file is created automatically
    in ``logs/``.

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
        log_path = logs_dir / f"regime_v4_signal_filter_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logger.info("File logging enabled: %s", log_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_REQUIRED_LOAD_COLS: list[str] = [
    "pair",
    "time",
    "entry_time",
    "entry_close",
    "net_sentiment",
    "sentiment_change",
    "abs_sentiment",
    "side_streak",
    TARGET_COL,
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the research dataset for the hybrid pipeline.

    Combines the column requirements of both Signal V2 and Regime V4:

    * Signal V2 features: ``net_sentiment``, ``sentiment_change``,
      ``abs_sentiment``, ``side_streak``, ``ret_48b``
    * Regime V4 features: ``entry_time``, ``entry_close``
      (→ ``vol_24b``, ``trend_strength_48b``)

    Args:
        path: Path to the master research dataset CSV.

    Returns:
        DataFrame with ``year`` column added.

    Raises:
        ValueError: If required columns are missing.
    """
    df = read_csv(path, required_columns=_REQUIRED_LOAD_COLS)

    df = parse_timestamps(df, "time", context="regime_v4_signal_filter.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    df = parse_timestamps(df, "entry_time", context="regime_v4_signal_filter.load_data")

    df["pair_group"] = np.where(
        df["pair"].str.contains(cfg.JPY_PAIR_PATTERN, case=False, na=False),
        "JPY_cross",
        "other",
    )

    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    logger.info(
        "Dataset loaded: rows=%d | pairs=%d | date_range=%s .. %s",
        len(df),
        df["pair"].nunique(),
        date_min,
        date_max,
    )
    return df


# ---------------------------------------------------------------------------
# Regime feature building (vol_24b, trend_strength_48b)
# ---------------------------------------------------------------------------


def _build_regime_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``vol_24b`` and ensure ``trend_strength_48b`` is present.

    Computes a causal per-pair rolling 24-bar volatility of ``entry_close``
    returns (mirroring ``experiments.regime_v3.build_features``).  The column
    ``trend_strength_48b`` is expected to be present in the raw dataset; if
    absent it is skipped with a warning.

    Args:
        df: Dataset sorted appropriately; must contain ``pair`` and
            ``entry_close``.

    Returns:
        Copy of *df* with ``vol_24b`` added (and ``trend_strength_48b``
        validated if present).
    """
    out = df.copy()
    out = out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    out["_bar_ret"] = out.groupby("pair")["entry_close"].pct_change()
    out["vol_24b"] = out.groupby("pair")["_bar_ret"].transform(
        lambda s: s.rolling(24, min_periods=5).std()
    )
    out = out.drop(columns=["_bar_ret"])

    logger.debug("vol_24b: %d non-null values", out["vol_24b"].notna().sum())

    if "trend_strength_48b" not in out.columns:
        logger.warning(
            "_build_regime_base_features: 'trend_strength_48b' not found "
            "in dataset; regime_key components trend_dir and "
            "trend_strength_bin will be NaN"
        )

    return out


# ---------------------------------------------------------------------------
# Signal V2 preparation
# ---------------------------------------------------------------------------


def _prepare_signal_v2(df: pd.DataFrame, window: int, threshold: float | None) -> pd.DataFrame:
    """Build Signal V2 features and positions on the full dataset.

    Maps ``entry_close`` → ``price`` (used by
    :func:`experiments.signal_v2.build_features`) and delegates to the
    Signal V2 feature and signal builders.

    Signal V2 is purely feature-based (no training required) so positions
    can be computed on the full dataset without any look-ahead risk.

    Args:
        df: Dataset after :func:`_build_regime_base_features`.
        window: Rolling z-score window forwarded to Signal V2.
        threshold: Optional position threshold forwarded to Signal V2.

    Returns:
        DataFrame with signal_v2 feature columns (``divergence``,
        ``shock``, ``exhaustion``, ``signal_v2_raw``, ``position``)
        and NaN rows dropped.
    """
    out = df.copy()

    # Signal V2 uses "price" and sorts by "time" per pair.
    # "entry_close" is the canonical price column; map it here.
    if "price" not in out.columns:
        converted = pd.to_numeric(out["entry_close"], errors="coerce")
        valid_ratio = converted.notna().mean()
        if valid_ratio < 0.99:
            logger.warning(
                "_prepare_signal_v2: entry_close has only %.1f%% valid numeric values",
                valid_ratio * 100,
            )
        out["price"] = converted

    out = signal_v2_build_features(out, window=window)
    out = build_signal(out, threshold=threshold)
    return out


# ---------------------------------------------------------------------------
# Regime statistics from training data
# ---------------------------------------------------------------------------


def _compute_regime_stats(
    train_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    min_n: int = MIN_N,
) -> dict[str, dict[str, float]]:
    """Compute per-regime signal-weighted return statistics on training data.

    Groups *train_df* by ``regime_key`` and computes signal-weighted
    statistics (``position * ret_48b``) for each regime.  Only regimes with
    ``n >= min_n`` and a non-NaN Sharpe are returned.

    Args:
        train_df: Training slice with ``regime_key``, ``position``, and
            *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        min_n: Minimum observations for a regime to be eligible.

    Returns:
        Dict mapping ``regime_key`` → stat dict
        ``{"n", "mean_return", "std", "sharpe"}``.
    """
    if "regime_key" not in train_df.columns:
        logger.warning("_compute_regime_stats: 'regime_key' not found in train data")
        return {}

    # Only consider rows with an active position in training data
    valid = train_df.dropna(subset=["regime_key", "position", target_col])
    valid = valid[valid["position"] != 0.0]
    if valid.empty:
        logger.warning("_compute_regime_stats: no valid rows in training data")
        return {}

    valid = valid.copy()
    valid["_signal_ret"] = valid["position"] * valid[target_col]

    stats: dict[str, dict[str, float]] = {}

    for regime_label, grp in valid.groupby("regime_key"):
        signal_rets = grp["_signal_ret"].values
        n = len(signal_rets)
        if n < min_n:
            logger.debug(
                "_compute_regime_stats: regime=%s skipped (n=%d < min_n=%d)",
                regime_label,
                n,
                min_n,
            )
            continue

        mean_return = float(np.mean(signal_rets))
        std = float(np.std(signal_rets))
        sharpe = mean_return / std if std > 1e-10 else np.nan
        if np.isnan(sharpe):
            logger.debug(
                "_compute_regime_stats: regime=%s skipped (sharpe=NaN, std~0)",
                regime_label,
            )
            continue

        stats[str(regime_label)] = {
            "n": float(n),
            "mean_return": mean_return,
            "std": std,
            "sharpe": sharpe,
        }

    logger.debug(
        "_compute_regime_stats: %d eligible regimes (n>=%d)",
        len(stats),
        min_n,
    )
    return stats


# ---------------------------------------------------------------------------
# Regime selection
# ---------------------------------------------------------------------------


def _select_regimes(
    regime_stats: dict[str, dict[str, float]],
    *,
    min_sharpe: float = MIN_SHARPE,
) -> set[str]:
    """Select regime keys that meet the Sharpe threshold.

    Args:
        regime_stats: Output of :func:`_compute_regime_stats`.
        min_sharpe: Minimum Sharpe to include a regime.

    Returns:
        Set of ``regime_key`` strings that pass the filter.
    """
    selected = {
        k
        for k, v in regime_stats.items()
        if not np.isnan(v["sharpe"]) and v["sharpe"] >= min_sharpe
    }
    logger.debug(
        "_select_regimes: %d / %d regimes selected (min_sharpe=%.4f)",
        len(selected),
        len(regime_stats),
        min_sharpe,
    )
    return selected


# ---------------------------------------------------------------------------
# Apply filter + optional direction logic to test fold
# ---------------------------------------------------------------------------


def _apply_filter(
    test_df: pd.DataFrame,
    selected: set[str],
    regime_stats: dict[str, dict[str, float]],
    *,
    with_direction: bool = False,
    direction_threshold: float = DIRECTION_THRESHOLD,
) -> pd.DataFrame:
    """Apply regime filter (and optional direction modifier) to a test fold.

    Steps:
    1. Set ``position = 0`` for rows whose ``regime_key`` is not in *selected*.
    2. If *with_direction*:
       - ``mean_return >  direction_threshold`` → keep position
       - ``mean_return < -direction_threshold`` → flip position
       - Otherwise                              → ``position = 0``

    Direction logic uses training-derived ``mean_return`` from *regime_stats*.

    Args:
        test_df: Test-fold DataFrame with ``position`` and ``regime_key``.
        selected: Set of selected regime keys (from training).
        regime_stats: Training-derived regime statistics.
        with_direction: Whether to apply direction modification.
        direction_threshold: Absolute mean-return threshold for direction.

    Returns:
        Copy of *test_df* with ``position`` updated in-place.
    """
    out = test_df.copy()

    # Step 1: filter — zero out unselected regimes
    unselected_mask = ~out["regime_key"].isin(selected) | out["regime_key"].isna()
    out.loc[unselected_mask, "position"] = 0.0

    if with_direction:
        # Step 2: direction modification for selected rows
        for key in selected:
            stats = regime_stats.get(key)
            if stats is None:
                continue

            row_mask = out["regime_key"] == key
            mean_ret = stats["mean_return"]

            if mean_ret > direction_threshold:
                pass  # keep position as-is
            elif mean_ret < -direction_threshold:
                out.loc[row_mask, "position"] *= -1.0  # flip
            else:
                out.loc[row_mask, "position"] = 0.0  # zero out

    return out


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------


def _fold_metrics(
    test_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
) -> dict[str, float | int]:
    """Compute performance metrics for a single walk-forward test fold.

    Args:
        test_df: Test fold with ``position`` and *target_col*.
        target_col: Forward-return column.

    Returns:
        Dict with keys: ``n``, ``mean``, ``sharpe``, ``hit_rate``,
        ``coverage``.
    """
    n_total = len(test_df)
    active = test_df[test_df["position"] != 0.0].copy()
    n_active = len(active)
    coverage = float(n_active / n_total) if n_total > 0 else 0.0

    if n_active < 2:
        return {
            "n": n_active,
            "mean": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
            "coverage": coverage,
        }

    pnl = active["position"] * active[target_col]
    mean = float(pnl.mean())
    std = float(pnl.std())
    sharpe = mean / std if std > 1e-10 else np.nan
    hit_rate = float((pnl > 0).mean())

    return {
        "n": n_active,
        "mean": mean,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def walk_forward(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    year_col: str = "year",
    min_n: int = MIN_N,
    min_sharpe: float = MIN_SHARPE,
    with_direction: bool = False,
    direction_threshold: float = DIRECTION_THRESHOLD,
    top_n_log: int = TOP_N_LOG,
) -> pd.DataFrame:
    """Signal V2 × Regime Filter hybrid walk-forward evaluation.

    Strict expanding-window walk-forward:

    1. For each test year (from the second unique year onward):
    2. Compute quantile cut points for ``vol_regime`` and
       ``trend_strength_bin`` from ``train_df`` only.
    3. Build 4-component regime keys on both ``train_df`` and ``test_df``
       using training-derived cuts.
    4. Compute per-regime signal-weighted statistics (``position × ret_48b``)
       on ``train_df``; only regimes with ``n >= min_n`` are considered.
    5. Select regimes with ``sharpe >= min_sharpe``.
    6. Apply filter (and optional direction logic) to ``test_df``.
    7. Compute fold metrics on the resulting active positions.

    No test-period return information enters the regime selection.

    Args:
        df: Full dataset after feature building and signal computation.
            Must contain ``position``, ``regime_key`` prerequisites
            (``vol_24b``, ``trend_strength_48b``, ``abs_sentiment``),
            and *target_col*.
        target_col: Forward-return column (default ``ret_48b``).
        year_col: Column containing calendar year.
        min_n: Minimum training observations per regime.
        min_sharpe: Minimum training Sharpe to select a regime.
        with_direction: Apply direction modification when ``True``.
        direction_threshold: Absolute mean-return threshold for direction.
        top_n_log: Number of top regimes (by Sharpe) to log per fold.

    Returns:
        DataFrame with per-fold columns:
        ``["year", "n", "mean", "sharpe", "hit_rate", "coverage",
        "n_selected_regimes"]``.
    """
    _COLS = ["year", "n", "mean", "sharpe", "hit_rate", "coverage", "n_selected_regimes"]

    if year_col not in df.columns:
        logger.warning("walk_forward: year column '%s' not found", year_col)
        return pd.DataFrame(columns=_COLS)

    years = sorted(df[year_col].unique())
    if len(years) < 2:
        logger.warning(
            "walk_forward: need at least 2 unique years, got %d",
            len(years),
        )
        return pd.DataFrame(columns=_COLS)

    fold_rows: list[dict[str, Any]] = []

    for i, test_year in enumerate(years):
        if i == 0:
            # Need at least one prior year for an expanding train set
            continue

        train_df = df[df[year_col] < test_year]
        test_df = df[df[year_col] == test_year]

        if test_df.empty:
            logger.debug("walk_forward: empty test fold year=%d", test_year)
            continue

        # --- Step 1: quantile cuts from training data only ---
        cuts = _compute_train_cuts(train_df)

        # --- Step 2: build 4-component regime keys ---
        train_labeled = _build_regime_key(train_df, cuts)
        test_labeled = _build_regime_key(test_df, cuts)

        n_unique_train_regimes = int(train_labeled["regime_key"].dropna().nunique())

        # --- Step 3: compute regime stats on training data ---
        regime_stats = _compute_regime_stats(
            train_labeled,
            target_col=target_col,
            min_n=min_n,
        )
        n_eligible = len(regime_stats)

        # --- Step 4: select regimes ---
        selected = _select_regimes(regime_stats, min_sharpe=min_sharpe)
        n_selected = len(selected)

        logger.info(
            "REGIME V4 SIGNAL FILTER [year=%d] | train_regimes=%d"
            " | eligible(n>=%d)=%d | selected(sharpe>=%.2f)=%d",
            test_year,
            n_unique_train_regimes,
            min_n,
            n_eligible,
            min_sharpe,
            n_selected,
        )

        # Log top regimes by sharpe
        if regime_stats:
            top_regimes = sorted(
                regime_stats.items(),
                key=lambda kv: kv[1]["sharpe"],
                reverse=True,
            )[:top_n_log]
            for regime_label, stats in top_regimes:
                is_sel = "✓" if regime_label in selected else "✗"
                logger.info(
                    "  [%s] REGIME %-60s | n=%5.0f | sharpe=%+.4f"
                    " | mean_ret=%+.6f",
                    is_sel,
                    regime_label,
                    stats["n"],
                    stats["sharpe"],
                    stats["mean_return"],
                )

        # --- Step 5: apply filter (+ optional direction) to test set ---
        test_valid = test_labeled.dropna(subset=[target_col, "position"])
        if test_valid.empty:
            logger.warning(
                "walk_forward [year=%d]: no valid test rows; skipping fold",
                test_year,
            )
            continue

        test_filtered = _apply_filter(
            test_valid,
            selected,
            regime_stats,
            with_direction=with_direction,
            direction_threshold=direction_threshold,
        )

        # --- Step 6: compute fold metrics ---
        m = _fold_metrics(test_filtered, target_col=target_col)

        fold_rows.append(
            {
                "year": int(test_year),
                "n": m["n"],
                "mean": m["mean"],
                "sharpe": m["sharpe"],
                "hit_rate": m["hit_rate"],
                "coverage": m["coverage"],
                "n_selected_regimes": n_selected,
            }
        )

        logger.info(
            "FOLD [year=%d] | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%% | n_selected=%d",
            test_year,
            m["n"],
            m["mean"] if not np.isnan(m["mean"]) else float("nan"),
            m["sharpe"] if not np.isnan(m["sharpe"]) else float("nan"),
            m["hit_rate"] if not np.isnan(m["hit_rate"]) else float("nan"),
            m["coverage"] * 100,
            n_selected,
        )

    if not fold_rows:
        logger.warning("walk_forward: no valid folds produced")
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame(fold_rows)[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def log_fold_results(fold_df: pd.DataFrame) -> None:
    """Log per-fold walk-forward results.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    logger.info("=== REGIME V4 SIGNAL FILTER — PER-FOLD RESULTS ===")
    if fold_df.empty:
        logger.warning("REGIME V4 SIGNAL FILTER: no fold results to display")
        return
    for row in fold_df.itertuples(index=False):
        logger.info(
            "FOLD | year=%d | n=%5d | mean=%+.6f | sharpe=%+.4f"
            " | hit_rate=%.4f | coverage=%.1f%% | n_selected=%d",
            row.year,
            row.n,
            row.mean if not np.isnan(row.mean) else float("nan"),
            row.sharpe if not np.isnan(row.sharpe) else float("nan"),
            row.hit_rate if not np.isnan(row.hit_rate) else float("nan"),
            row.coverage * 100,
            row.n_selected_regimes,
        )


def log_final_summary(fold_df: pd.DataFrame) -> None:
    """Log the consolidated final summary across all walk-forward folds.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward`.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("=== REGIME V4 SIGNAL FILTER — FINAL SUMMARY ===")
    logger.info(sep)

    if fold_df.empty:
        logger.warning("REGIME V4 SIGNAL FILTER SUMMARY: no results")
        return

    mean_sharpe = float(fold_df["sharpe"].dropna().mean())
    mean_coverage = float(fold_df["coverage"].mean())
    mean_hit_rate = float(fold_df["hit_rate"].dropna().mean())
    mean_n_selected = float(fold_df["n_selected_regimes"].mean())

    logger.info("  folds              : %d", len(fold_df))
    logger.info(
        "  mean_sharpe        : %+.4f",
        mean_sharpe if not np.isnan(mean_sharpe) else float("nan"),
    )
    logger.info("  mean_coverage      : %.1f%%", mean_coverage * 100)
    logger.info(
        "  mean_hit_rate      : %.4f",
        mean_hit_rate if not np.isnan(mean_hit_rate) else float("nan"),
    )
    logger.info("  mean_n_selected    : %.1f regimes/fold", mean_n_selected)
    logger.info(sep)
    logger.info("Per-fold results:\n%s", fold_df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI (direct execution)
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V4 Signal Filter: Signal V2 base signal filtered "
            "by training-derived regime selection.  Evaluates the hybrid "
            "via an expanding walk-forward."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        metavar="N",
        help="Rolling z-score window size for Signal V2 features.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Optional signal threshold for Signal V2.  Positions with "
            "|signal_v2_raw| <= T are set to zero.  Omit for pure sign signal."
        ),
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_N,
        metavar="N",
        help="Minimum training observations per regime for selection.",
    )
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=MIN_SHARPE,
        metavar="S",
        help="Minimum training Sharpe for a regime to be selected.",
    )
    p.add_argument(
        "--direction-threshold",
        type=float,
        default=DIRECTION_THRESHOLD,
        metavar="D",
        help=(
            "Absolute mean-return threshold for direction logic.  "
            "Only used when --with-direction is set."
        ),
    )
    direction_group = p.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--with-direction",
        action="store_true",
        default=False,
        help="Enable direction modification: flip signal for negative-mean regimes.",
    )
    direction_group.add_argument(
        "--no-direction",
        action="store_false",
        dest="with_direction",
        help="Disable direction modification (default).",
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
        "=== REGIME V4 SIGNAL FILTER ==="
        " | window=%d | threshold=%s | min_n=%d | min_sharpe=%.4f"
        " | with_direction=%s | direction_threshold=%s",
        args.window,
        args.threshold,
        args.min_n,
        args.min_sharpe,
        args.with_direction,
        args.direction_threshold if args.with_direction else "N/A",
    )

    # Load and prepare data
    df = load_data(args.data)

    # Build regime base features (vol_24b)
    df = _build_regime_base_features(df)

    # Build Signal V2 features and positions
    df = _prepare_signal_v2(df, window=args.window, threshold=args.threshold)

    _log.info("Dataset prepared: %d rows after feature build", len(df))

    # Walk-forward evaluation
    fold_df = walk_forward(
        df,
        target_col=TARGET_COL,
        min_n=args.min_n,
        min_sharpe=args.min_sharpe,
        with_direction=args.with_direction,
        direction_threshold=args.direction_threshold,
    )

    log_fold_results(fold_df)
    log_final_summary(fold_df)


if __name__ == "__main__":
    main()
