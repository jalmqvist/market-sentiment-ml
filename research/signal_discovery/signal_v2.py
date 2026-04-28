# Legacy experiment — not part of current validated approach\n"""
experiments/signal_v2.py
========================
Signal V2.1: sentiment divergence, shock, and exhaustion.

Builds a composite signal from three components:

* **Divergence**  – sentiment z-score minus **causal** price-momentum z-score
  (price momentum is the 48-bar cumulative past return ``mom_48b``, computed
  from ``price.pct_change()``, never from forward returns)
* **Shock**       – z-score of raw sentiment change
* **Exhaustion**  – sum of z-scored absolute sentiment and z-scored side streak

Signal formula::

    signal_v2_raw = 1.0 * divergence + 0.5 * shock - 0.75 * exhaustion
    position      = sign(signal_v2_raw)

An optional threshold filters low-conviction positions::

    position = where(|signal_v2_raw| > threshold, sign(signal_v2_raw), 0)

Walk-forward evaluation uses an expanding window (train on all prior years,
test on each subsequent year).  No model fitting is required because the
signal is purely feature-based.

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    python experiments/signal_v2.py \\
        --data data/output/master_research_dataset.csv

    python experiments/signal_v2.py \\
        --data data/output/master_research_dataset.csv \\
        --window 96 --threshold 0.5 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from evaluation.metrics import compute_stats
from utils.io import read_csv
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW: int = 96
TARGET_COL: str = "ret_48b"

_REQUIRED_INPUT_COLS: list[str] = [
    "price",
    "net_sentiment",
    TARGET_COL,
    "sentiment_change",
    "abs_sentiment",
    "side_streak",
]

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
        log_path = logs_dir / f"signal_v2_{timestamp}.log"
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
    """Load and prepare the research dataset.

    Args:
        path: Path to the canonical research dataset CSV.

    Returns:
        DataFrame with ``year`` column added.

    Raises:
        ValueError: If required columns are missing.
    """
    df = read_csv(
        path,
        required_columns=[
            "pair",
            "time",
            "net_sentiment",
            "sentiment_change",
            "abs_sentiment",
            "side_streak",
            "ret_48b",
        ],
    )
    df = parse_timestamps(df, "time", context="signal_v2.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    logger.info(
        "Dataset loaded: rows=%d, pairs=%d, date_range=%s .. %s",
        len(df), df["pair"].nunique(), date_min, date_max,
    )
    return df


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a causal (past-only) rolling z-score.

    Args:
        series: Input series.
        window: Rolling window size in bars.

    Returns:
        Z-scored series with the same index.  Values within the first
        ``window - 1`` rows are NaN (no leakage).
    """
    roll = series.rolling(window, min_periods=window)
    return (series - roll.mean()) / roll.std()


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Construct signal features using only past data (causal rolling stats).

    The function sorts by the ``time`` column within each pair before computing
    rolling statistics to guarantee correct temporal ordering.

    Args:
        df: Research dataset; must contain the columns in
            ``_REQUIRED_INPUT_COLS`` plus ``pair`` and ``time``.
        window: Rolling window size in bars (default 96).

    Returns:
        DataFrame with new feature columns appended; NaN rows are dropped.

    Raises:
        ValueError: If required columns are missing.
    """
    if "price" not in df.columns:
        raise ValueError(
            "Missing required column 'price'. "
            "Expected caller to map 'price_end' → 'price'."
        )

    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise TypeError("'price' must be numeric before feature computation")

    missing = [c for c in _REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"build_features: missing required columns: {missing}")

    df = df.sort_values(["pair", "time"]).copy()

    # Compute features per pair to avoid cross-pair contamination in rolling windows
    feature_frames: list[pd.DataFrame] = []
    for _, grp in df.groupby("pair", sort=False):
        grp = grp.copy()

        grp["sentiment_z"] = rolling_zscore(grp["net_sentiment"], window)

        grp["ret_1b"] = grp["price"].pct_change()
        grp["mom_48b"] = (
            grp["ret_1b"]
            .rolling(48, min_periods=48)
            .sum()
        )
        logger.debug(
            "mom_48b stats: mean=%.6f std=%.6f",
            grp["mom_48b"].mean(),
            grp["mom_48b"].std(),
        )

        grp["price_mom_z"] = rolling_zscore(grp["mom_48b"], window)
        grp["divergence"] = grp["sentiment_z"] - grp["price_mom_z"]
        grp["shock"] = rolling_zscore(grp["sentiment_change"], window)
        grp["exhaustion"] = (
            rolling_zscore(grp["abs_sentiment"], window)
            + rolling_zscore(grp["side_streak"], window)
        )

        feature_frames.append(grp)

    result = pd.concat(feature_frames, ignore_index=True)
    before = len(result)
    result = result.dropna(
        subset=["divergence", "shock", "exhaustion", TARGET_COL]
    )
    logger.debug(
        "build_features: %d rows before dropna, %d after (window=%d)",
        before, len(result), window,
    )
    return result


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_signal(
    df: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Compute ``signal_v2_raw`` and ``position``.

    Args:
        df: DataFrame with ``divergence``, ``shock``, and ``exhaustion``
            columns (produced by :func:`build_features`).
        threshold: Optional absolute-value threshold.  Positions with
            ``|signal_v2_raw| <= threshold`` are set to zero.  ``None``
            means no thresholding (pure sign signal).

    Returns:
        DataFrame with ``signal_v2_raw`` and ``position`` columns added.
    """
    df = df.copy()
    df["signal_v2_raw"] = (
        1.0 * df["divergence"]
        + 0.5 * df["shock"]
        - 0.75 * df["exhaustion"]
    )

    if threshold is not None:
        df["position"] = np.where(
            np.abs(df["signal_v2_raw"]) > threshold,
            np.sign(df["signal_v2_raw"]),
            0.0,
        )
    else:
        df["position"] = np.sign(df["signal_v2_raw"])

    return df


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------

def _fold_metrics(test: pd.DataFrame) -> dict:
    """Compute PnL metrics for a single walk-forward fold.

    PnL is defined as ``position * ret_48b``.  Only rows with a non-zero
    position are included in the computation.

    Args:
        test: Test-fold DataFrame with ``position`` and ``ret_48b``.

    Returns:
        Dict with keys: n, mean, sharpe, hit_rate.
    """
    active = test[test["position"] != 0].copy()
    if active.empty:
        return {"n": 0, "mean": np.nan, "sharpe": np.nan, "hit_rate": np.nan}

    active["pnl"] = active["position"] * active[TARGET_COL]
    return compute_stats(active, "pnl")


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_signal_v2(
    df: pd.DataFrame,
    *,
    threshold: float | None = None,
    year_col: str = "year",
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation for Signal V2.

    For each test year the expanding train set contains all prior years.
    Because Signal V2 is feature-based no model fitting is required; the
    function simply applies :func:`build_signal` to each test fold and logs
    per-fold diagnostics.

    Args:
        df: DataFrame with pre-built features (from :func:`build_features`)
            and the ``year`` column.
        threshold: Optional signal threshold forwarded to :func:`build_signal`.
        year_col: Column containing the calendar year.

    Returns:
        DataFrame with per-fold columns: year, n, mean, sharpe, hit_rate.
    """
    years = sorted(df[year_col].unique())
    if len(years) < 2:
        logger.warning("walk_forward_signal_v2: fewer than 2 years; skipping")
        return pd.DataFrame()

    results = []

    for i, test_year in enumerate(years):
        if i == 0:
            # Need at least one prior year for an expanding train set
            continue

        test_df = df[df[year_col] == test_year].copy()
        if test_df.empty:
            logger.debug("walk_forward_signal_v2: empty test fold year=%d", test_year)
            continue

        test_df = build_signal(test_df, threshold=threshold)

        # Per-fold signal diagnostics (computed on test fold)
        sig = test_df["signal_v2_raw"]
        corr = sig.corr(test_df[TARGET_COL])
        logger.info(
            "Fold year=%d | signal_v2_raw: mean=%.4f  std=%.4f  corr(ret_48b)=%.4f",
            test_year,
            sig.mean(),
            sig.std(),
            corr if not np.isnan(corr) else 0.0,
        )

        metrics = _fold_metrics(test_df)
        metrics["year"] = int(test_year)
        results.append(metrics)

        logger.info(
            "Fold year=%d | n=%d  mean=%.6f  sharpe=%.4f  hit_rate=%.4f",
            test_year,
            metrics["n"],
            metrics["mean"] if not np.isnan(metrics["mean"]) else 0.0,
            metrics["sharpe"] if not np.isnan(metrics["sharpe"]) else 0.0,
            metrics["hit_rate"] if not np.isnan(metrics["hit_rate"]) else 0.0,
        )

    if not results:
        return pd.DataFrame()

    cols = ["year", "n", "mean", "sharpe", "hit_rate"]
    fold_df = pd.DataFrame(results)
    return fold_df[[c for c in cols if c in fold_df.columns]]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def log_final_summary(fold_df: pd.DataFrame) -> None:
    """Log pooled walk-forward summary statistics.

    Args:
        fold_df: DataFrame returned by :func:`walk_forward_signal_v2`.
    """
    if fold_df.empty:
        logger.warning("walk_forward_signal_v2: no valid folds")
        return

    mean_sharpe = fold_df["sharpe"].mean()
    mean_hit_rate = fold_df["hit_rate"].mean()

    logger.info("=" * 60)
    logger.info("SIGNAL V2 — WALK-FORWARD SUMMARY")
    logger.info("  folds        : %d", len(fold_df))
    logger.info("  mean_sharpe  : %.4f", mean_sharpe if not np.isnan(mean_sharpe) else 0.0)
    logger.info(
        "  mean_hit_rate: %.4f",
        mean_hit_rate if not np.isnan(mean_hit_rate) else 0.0,
    )
    logger.info("=" * 60)
    logger.info("Per-fold results:\n%s", fold_df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Signal V2: sentiment divergence, shock, and exhaustion. "
            "Evaluates a composite signal via expanding walk-forward testing."
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
        help="Rolling z-score window size in bars.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Optional signal threshold.  Positions with |signal_v2_raw| <= T "
            "are set to zero.  Omit for pure sign signal."
        ),
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
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    _log = logging.getLogger(__name__)
    _log.info(
        "=== SIGNAL V2 === window=%d  threshold=%s",
        args.window,
        args.threshold,
    )

    df = load_data(args.data)

    # --- ROBUST PRICE COLUMN DETECTION ---
    _PRICE_CANDIDATES: list[str] = ["price", "price_end", "entry_close"]
    _VALID_RATIO_THRESHOLD: float = 0.99

    _log.debug("PRICE COLUMN VALIDATION:")
    selected_col: str | None = None
    selected_series: pd.Series | None = None
    candidate_diagnostics: list[str] = []

    for candidate in _PRICE_CANDIDATES:
        if candidate not in df.columns:
            _log.debug("  candidate=%s | not present in dataset", candidate)
            continue

        raw = df[candidate]
        converted = pd.to_numeric(raw, errors="coerce")
        total = len(converted)
        valid_count = converted.notna().sum()
        valid_ratio = valid_count / total if total > 0 else 0.0
        sample = raw.dropna().head(5).tolist()

        _log.debug(
            "  candidate=%s | dtype=%s | valid_ratio=%.2f | sample=%s",
            candidate,
            raw.dtype,
            valid_ratio,
            sample,
        )

        candidate_diagnostics.append(
            f"candidate={candidate} | dtype={raw.dtype} | valid_ratio={valid_ratio:.2f} "
            f"| first_5_raw={raw.head(5).tolist()}"
        )

        if selected_col is None and valid_ratio >= _VALID_RATIO_THRESHOLD:
            selected_col = candidate
            selected_series = converted

    if selected_col is None:
        diagnostics_str = "\n  ".join(candidate_diagnostics) if candidate_diagnostics else "(none checked)"
        raise ValueError(
            f"Signal V2: no valid numeric price column found among candidates "
            f"{_PRICE_CANDIDATES}.\n  {diagnostics_str}"
        )

    _log.debug("Selected price column: %s", selected_col)
    df["price"] = selected_series

    nan_count = df["price"].isna().sum()
    if nan_count > 0:
        _log.warning(
            "Selected price column '%s' has %d NaN(s); these will be dropped during feature construction",
            selected_col,
            nan_count,
        )

    df = build_features(df, window=args.window)

    _log.info("Features built: %d rows remaining after NaN drop", len(df))

    fold_df = walk_forward_signal_v2(df, threshold=args.threshold)
    log_final_summary(fold_df)


if __name__ == "__main__":
    main()
