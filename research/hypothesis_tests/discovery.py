# Legacy experiment — not part of current validated approach\n"""
experiments/discovery.py
========================
Per-pair behavioral signal discovery.

Refactored from ``discover_behavioral_signal.py``.  All core logic is
preserved; structure and safety have been improved.

Usage::

    python -m experiments.discovery --data data/output/master_research_dataset.csv
    python experiments/discovery.py --data data/output/master_research_dataset.csv \\
                                    --log-level DEBUG

Artifact output::

    By default, writes a JSON artifact to data/output/discovery_results.json
    containing selected pairs, thresholds used, horizon, and selection metrics.
    Use --out-artifact to override the output path.
"""

from __future__ import annotations

import argparse
import json
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
from pipeline.filters import enforce_non_overlap, select_survivor_pairs
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
    logger.info(
        "Dataset loaded: rows=%d, pairs=%d, date_range=%s .. %s",
        len(df), df["pair"].nunique(), date_min, date_max,
    )
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
        "--out-artifact",
        type=Path,
        default=Path("data/output/discovery_results.json"),
        help="Path to write the discovery artifact JSON (selected pairs + metrics).",
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
    logger.info("Dataset size: %d rows", len(df))

    signal = apply_behavioral_signal(df)
    raw_signal_count = len(signal)
    logger.info("Raw signal count: %d", raw_signal_count)

    if warn_if_empty(signal, context="discovery.main"):
        logger.info("No signals found.")
        return

    logger.info("=== SIGNAL DISTRIBUTION ===\n%s", signal["pair"].value_counts().head(10).to_string())

    # Artifact: will accumulate per-horizon results
    artifact: dict = {
        "thresholds": {
            "SIGNAL_EXTREME_STREAK_MIN": cfg.SIGNAL_EXTREME_STREAK_MIN,
            "SIGNAL_PERSISTENCE_BUCKETS": cfg.SIGNAL_PERSISTENCE_BUCKETS,
            "SURVIVOR_MIN_SIGNALS": cfg.SURVIVOR_MIN_SIGNALS,
            "SURVIVOR_MIN_AFTER_DEDUP": cfg.SURVIVOR_MIN_AFTER_DEDUP,
            "SURVIVOR_MIN_SHARPE": cfg.SURVIVOR_MIN_SHARPE,
            "HOLDOUT_SPLIT_YEAR": cfg.HOLDOUT_SPLIT_YEAR,
        },
        "raw_signal_count": raw_signal_count,
        "horizons": {},
    }

    for horizon in cfg.EVAL_HORIZONS:
        logger.info("=" * 80)
        logger.info("PER-PAIR DISCOVERY (horizon=%d)", horizon)
        logger.info("=" * 80)

        results = analyze_pairs(signal, horizon)

        # Determine survivor pairs using the same logic as portfolio_builder
        selected_pairs = select_survivor_pairs(signal, horizon)

        if results.empty:
            logger.info("No valid pairs for horizon=%d.", horizon)
            artifact["horizons"][str(horizon)] = {
                "selected_pairs": [],
                "selection_metrics": {},
            }
            continue

        logger.info("Per-pair results (horizon=%d):\n%s", horizon, results.to_string(index=False))
        logger.info("Top 5:\n%s", results.head(5).to_string(index=False))
        logger.info("Bottom 5:\n%s", results.tail(5).to_string(index=False))

        # Non-overlapping count
        non_overlapping = enforce_non_overlap(signal, horizon)
        non_overlap_count = len(non_overlapping)
        logger.info(
            "[summary] discovery(h=%d): raw_signals=%d, non_overlapping=%d, selected_pairs=%d: %s",
            horizon, raw_signal_count, non_overlap_count, len(selected_pairs), selected_pairs,
        )

        # Build per-pair metrics dict for artifact
        metrics_by_pair = {}
        for _, row in results.iterrows():
            metrics_by_pair[str(row["pair"])] = {
                k: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None)
                for k, v in row.items()
                if k != "pair"
            }

        artifact["horizons"][str(horizon)] = {
            "selected_pairs": selected_pairs,
            "non_overlapping_count": non_overlap_count,
            "selection_metrics": metrics_by_pair,
        }

    # Write artifact JSON
    artifact_path: Path = args.out_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2))
    logger.info("Discovery artifact written: %s", artifact_path)


if __name__ == "__main__":
    main()
