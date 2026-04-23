"""
experiments/discovery.py
========================
Per-pair behavioral signal discovery.

Refactored from ``discover_behavioral_signal.py``.  All core logic is
preserved; structure and safety have been improved.

Usage::

    python -m experiments.discovery --data data/output/master_research_dataset.csv
    python experiments/discovery.py --data data/output/master_research_dataset.csv \\
                                    --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution (python experiments/discovery.py)
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from evaluation.holdout import holdout_test
from evaluation.metrics import compute_metrics
from evaluation.walk_forward import walk_forward_yearly  # noqa: F401 – imported for downstream use
from pipeline.filters import enforce_non_overlap
from pipeline.signal import apply_behavioral_signal
from utils.io import read_csv, setup_logging
from utils.validation import parse_timestamps, warn_if_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and prepare the research dataset for discovery.

    Args:
        path: Path to the canonical research dataset CSV.  Required; no default.

    Raises:
        ValueError: If required columns are missing.
    """
    df = read_csv(
        path,
        required_columns=["pair", "time"],
    )

    df = parse_timestamps(df, "time", context="discovery.load_data")
    df["timestamp"] = df["time"]
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    # Dataset summary
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    print(f"Dataset summary: {len(df):,} rows | {df['pair'].nunique()} unique pairs | {date_min} to {date_max}")

    logger.info("Dataset loaded: %d rows, %d pairs", len(df), df["pair"].nunique())
    return df


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def analyze_pairs(signal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute per-pair performance metrics.

    Args:
        signal: Signal rows from :func:`apply_behavioral_signal`.
        horizon: Return horizon in bars.

    Returns:
        DataFrame sorted by Sharpe (descending).  Empty if no valid pairs.
    """
    results = []

    for pair, df_pair in signal.groupby("pair"):
        df_pair = enforce_non_overlap(df_pair, horizon)
        metrics = compute_metrics(df_pair, horizon)
        if metrics is None:
            continue

        hold = holdout_test(df_pair, horizon)

        results.append(
            {
                "pair": pair,
                "n": metrics["n"],
                "mean": metrics["mean"],
                "sharpe": metrics["sharpe"],
                "hit_rate": metrics["hit_rate"],
                "train_sharpe": hold["train"]["sharpe"] if hold["train"] else np.nan,
                "test_sharpe": hold["test"]["sharpe"] if hold["test"] else np.nan,
            }
        )

    if not results:
        logger.warning("analyze_pairs: no valid pairs for horizon=%d", horizon)
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Per-pair behavioral signal discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to master research dataset CSV (canonical dataset).",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    logger.info("Loading dataset...")
    df = load_data(args.data)
    print(f"Dataset size: {len(df):,}")

    signal = apply_behavioral_signal(df)
    print(f"Raw signal count: {len(signal):,}")

    if warn_if_empty(signal, context="discovery.main"):
        print("No signals found.")
        return

    print("\n=== SIGNAL DISTRIBUTION ===")
    print(signal["pair"].value_counts().head(10).to_string())

    for horizon in cfg.EVAL_HORIZONS:
        print(f"\n{'=' * 80}")
        print(f"PER-PAIR DISCOVERY (horizon={horizon})")
        print("=" * 80)

        results = analyze_pairs(signal, horizon)

        if results.empty:
            print("No valid pairs.")
            continue

        print(results.to_string(index=False))
        print("\nTop 5:")
        print(results.head(5).to_string(index=False))
        print("\nBottom 5:")
        print(results.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
