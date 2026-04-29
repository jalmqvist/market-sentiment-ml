"""
research/abm/sweep.py
=====================
Parameter sweep pipeline for the retail FX sentiment ABM.

Runs a grid search over three axes:
    - trend_ratios:         fraction of non-noise agents that are trend-followers
    - persistence_weights:  _PERSISTENCE_WEIGHT injected into agents.py
    - inertia_thresholds:   _INERTIA_THRESHOLD injected into agents.py

Results are sorted by score (ascending, lower is better) and saved to
``logs/abm_sweep_{timestamp}.csv``.

Usage::

    python research/abm/sweep.py \\
        --version 1.2.0 \\
        --pair eur-usd \\
        --steps 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config as cfg
from research.abm import agents as agents_module
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.scoring import compute_score, extract_metric
from research.abm.simulation import FXSentimentSimulation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

_TREND_RATIOS = [0.0, 0.5, 1.0]
_PERSISTENCE_WEIGHTS = [0.0, 0.1, 0.2]
_INERTIA_THRESHOLDS = [0.02, 0.05, 0.1]

# Fixed non-noise / noise split (mirrors run_abm defaults: 40+40+20)
_N_NON_NOISE = 80
_N_NOISE = 20


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------

def run_sweep(
    df: pd.DataFrame,
    pair: str,
    n_steps: int,
    seed: int = 42,
    momentum_window: int = 12,
) -> pd.DataFrame:
    """Run parameter sweep and return results sorted by score ascending.

    Args:
        df: Full dataset DataFrame (as returned by :func:`run_abm._load_real_data`).
        pair: FX pair to simulate (e.g. ``"eur-usd"``).
        n_steps: Maximum number of simulation steps per run.
        seed: Random seed for reproducibility.
        momentum_window: Momentum window passed to trend/contrarian agents.

    Returns:
        DataFrame with one row per parameter combination, sorted by ``score``
        ascending.  Columns: ``trend_ratio``, ``persistence``, ``threshold``,
        ``score``, ``std_diff``, ``autocorr_diff``.
    """
    _result_columns = [
        "trend_ratio", "persistence", "threshold",
        "score", "std_diff", "autocorr_diff",
    ]

    # Compute calibration targets once — reused for all runs
    targets = calibrate_from_dataset(df, pair=pair)

    # Prepare price / sentiment series once
    sub = df[df["pair"] == pair].copy().sort_values("entry_time")
    price_series = sub["entry_close"].values
    timestamps = sub["entry_time"].values
    real_sentiment = sub["net_sentiment"].values

    # Save original module constants so we can restore them
    orig_persistence = agents_module._PERSISTENCE_WEIGHT
    orig_threshold = agents_module._INERTIA_THRESHOLD

    results: list[dict] = []

    try:
        for trend_ratio, persistence, threshold in product(
            _TREND_RATIOS, _PERSISTENCE_WEIGHTS, _INERTIA_THRESHOLDS
        ):
            agents_module._PERSISTENCE_WEIGHT = persistence
            agents_module._INERTIA_THRESHOLD = threshold

            n_trend = int(round(trend_ratio * _N_NON_NOISE))
            n_contrarian = _N_NON_NOISE - n_trend

            rng = np.random.default_rng(seed)
            agents = _build_agents(rng, n_trend, n_contrarian, _N_NOISE, momentum_window)

            sim = FXSentimentSimulation(agents, rng=rng)

            max_steps = len(price_series) - sim.warmup_steps - 1
            if max_steps <= 0:
                logger.warning(
                    "Not enough price data for pair=%s, skipping run", pair
                )
                continue

            effective_steps = min(n_steps, max_steps)

            try:
                sim_df = sim.run(
                    n_steps=effective_steps,
                    price_series=price_series,
                    timestamps=timestamps,
                )
                warmup = sim.warmup_steps
                sim_df["real_net_sentiment"] = (
                    real_sentiment[warmup + 1 : warmup + 1 + len(sim_df)]
                )

                comparison = compare_to_data(sim_df, targets)
                score = compute_score(comparison)

                results.append({
                    "trend_ratio": trend_ratio,
                    "persistence": persistence,
                    "threshold": threshold,
                    "score": score,
                    "std_diff": extract_metric(comparison, "std"),
                    "autocorr_diff": extract_metric(comparison, "autocorr"),
                })

                logger.info(
                    "trend_ratio=%.1f persistence=%.2f threshold=%.3f score=%.4f",
                    trend_ratio, persistence, threshold, score,
                )

            except Exception as exc:
                logger.warning(
                    "Run failed (trend_ratio=%.1f persistence=%.2f threshold=%.3f): %s",
                    trend_ratio, persistence, threshold, exc,
                )

    finally:
        # Always restore module constants
        agents_module._PERSISTENCE_WEIGHT = orig_persistence
        agents_module._INERTIA_THRESHOLD = orig_threshold

    if not results:
        return pd.DataFrame(columns=_result_columns)

    return (
        pd.DataFrame(results)
        .sort_values("score")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parameter sweep for the retail FX sentiment ABM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", required=True, help="Dataset version (e.g. '1.2.0')")
    p.add_argument(
        "--variant",
        default="core",
        choices=["full", "core", "extended"],
        help="Dataset variant",
    )
    p.add_argument("--pair", required=True, help="FX pair (e.g. 'eur-usd')")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--momentum-window", type=int, default=12)
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("=== ABM Parameter Sweep ===")
    logger.info(
        "pair=%s version=%s steps=%d seed=%d momentum_window=%d",
        args.pair, args.version, args.steps, args.seed, args.momentum_window,
    )

    df, _ = _load_real_data(args.version, args.variant)

    result_df = run_sweep(
        df=df,
        pair=args.pair,
        n_steps=args.steps,
        seed=args.seed,
        momentum_window=args.momentum_window,
    )

    log_dir = cfg.REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = log_dir / f"abm_sweep_{timestamp}.csv"

    result_df.to_csv(out_path, index=False)
    logger.info("Sweep results saved: %s  rows=%d", out_path, len(result_df))

    if not result_df.empty:
        best = result_df.iloc[0]
        logger.info(
            "Best: trend_ratio=%.1f persistence=%.2f threshold=%.3f score=%.4f",
            best["trend_ratio"], best["persistence"], best["threshold"], best["score"],
        )


if __name__ == "__main__":
    main()
