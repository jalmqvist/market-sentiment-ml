"""
sweep.py

ABM parameter sweep with research-grade scoring.

Upgrades:
- volatility clustering
- kurtosis (fat tails)
- stronger autocorrelation penalty
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, UTC

import numpy as np
import pandas as pd

from research.abm.agents import (
    build_agents,
    _PERSISTENCE_WEIGHT,
    _INERTIA_THRESHOLD,
)
from research.abm.simulation import FXSentimentSimulation


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # build synthetic price series from returns
    if "ret_1b" in df.columns:
        r = df["ret_1b"].fillna(0.0).values
        price = np.cumprod(1.0 + r)
        df["price"] = price

    if "price" not in df.columns:
        raise ValueError("Dataset must contain 'price' or 'ret_1b'")

    return df


# ---------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------

def compute_stats(returns: np.ndarray) -> dict:
    returns = np.asarray(returns)

    mean = np.mean(returns)
    std = np.std(returns)
    abs_mean = np.mean(np.abs(returns))

    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0.0

    # NEW: volatility clustering
    if len(returns) > 1:
        vol_clust = np.corrcoef(np.abs(returns[:-1]), np.abs(returns[1:]))[0, 1]
    else:
        vol_clust = 0.0

    # NEW: kurtosis (fat tails)
    if std > 0:
        kurt = np.mean(((returns - mean) / std) ** 4)
    else:
        kurt = 3.0

    return {
        "mean": float(mean),
        "std": float(std),
        "abs_mean": float(abs_mean),
        "autocorr": float(autocorr),
        "vol_clust": float(vol_clust),
        "kurtosis": float(kurt),
    }


# ---------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------

def score_stats(sim: dict, real: dict) -> float:
    """
    Lower is better
    """

    score = 0.0

    score += abs(sim["mean"] - real["mean"])
    score += abs(sim["std"] - real["std"])
    score += abs(sim["abs_mean"] - real["abs_mean"])

    # stronger penalty (important)
    score += 3.0 * abs(sim["autocorr"] - real["autocorr"])

    # NEW: volatility clustering
    score += 2.0 * abs(sim["vol_clust"] - real["vol_clust"])

    # NEW: fat tails
    score += 1.0 * abs(sim["kurtosis"] - real["kurtosis"])

    return float(score)


# ---------------------------------------------------------------------
# SWEEP
# ---------------------------------------------------------------------

def run_sweep(price_series, real_stats, steps, seed):
    global _PERSISTENCE_WEIGHT, _INERTIA_THRESHOLD

    rng = np.random.default_rng(seed)

    trend_grid = [0.0, 0.5, 1.0]
    persistence_grid = [0.0, 0.1, 0.2]
    threshold_grid = [0.02, 0.05, 0.1]

    results = []

    for trend_ratio in trend_grid:
        for persistence in persistence_grid:
            for threshold in threshold_grid:
                try:
                    _PERSISTENCE_WEIGHT = persistence
                    _INERTIA_THRESHOLD = threshold

                    agents = build_agents(
                        n_agents=100,
                        trend_ratio=trend_ratio,
                        rng=rng,
                    )

                    sim = FXSentimentSimulation(agents, rng=rng)
                    df = sim.run(price_series, steps=steps)

                    sim_stats = compute_stats(df["returns"].values)
                    score = score_stats(sim_stats, real_stats)

                    logger.info(
                        f"trend_ratio={trend_ratio} persistence={persistence} "
                        f"threshold={threshold} score={score:.4f}"
                    )

                    results.append({
                        "trend_ratio": trend_ratio,
                        "persistence": persistence,
                        "threshold": threshold,
                        "score": score,
                    })

                except Exception as e:
                    logger.warning(
                        f"Run failed (trend_ratio={trend_ratio} "
                        f"persistence={persistence} threshold={threshold}): {e}"
                    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    logger.info("=== ABM Parameter Sweep ===")
    logger.info(f"pair={args.pair} version={args.version} steps={args.steps} seed=42")

    path = f"data/output/{args.version}/master_research_dataset_core.csv"
    df = load_dataset(path)

    price_series = df["price"].values

    real_stats = compute_stats(np.diff(price_series) / price_series[:-1])
    logger.info(f"Real stats: {real_stats}")

    result_df = run_sweep(
        price_series=price_series,
        real_stats=real_stats,
        steps=args.steps,
        seed=42,
    )

    if result_df.empty:
        raise RuntimeError("All sweep runs failed")

    result_df.sort_values("score", inplace=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"logs/abm_sweep_{args.pair}_{args.version}_{ts}.csv"

    result_df.to_csv(out_path, index=False)

    best = result_df.iloc[0]

    logger.info(f"Sweep results saved: {out_path}  rows={len(result_df)}")
    logger.info(
        f"Best: trend_ratio={best.trend_ratio} "
        f"persistence={best.persistence} "
        f"threshold={best.threshold} "
        f"score={best.score:.4f}"
    )


if __name__ == "__main__":
    main()