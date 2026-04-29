"""
research/abm/run_abm.py
=======================
CLI entry-point for running the retail FX sentiment ABM.

This version enforces:
- real data only (no synthetic price)
- per-pair simulation
- strict causal alignment
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from utils.io import setup_logging
from research.abm.agents import Contrarian, NoiseTrader, TrendFollower
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.simulation import FXSentimentSimulation

logger = logging.getLogger(__name__)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the retail FX sentiment agent-based model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--version", required=True, help="Dataset version (e.g. '1.1.0')")
    p.add_argument(
        "--variant",
        default="core",
        choices=["full", "core", "extended"],
        help="Dataset variant",
    )
    p.add_argument("--pair", required=True, help="FX pair (e.g. 'eur-usd')")

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-trend", type=int, default=40)
    p.add_argument("--n-contrarian", type=int, default=40)
    p.add_argument("--n-noise", type=int, default=20)

    p.add_argument("--momentum-window", type=int, default=12)

    p.add_argument("--output", type=Path, default=None)

    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return p.parse_args(argv)


# ============================================================
# Helpers
# ============================================================

def _build_agents(
    rng: np.random.Generator,
    n_trend: int,
    n_contrarian: int,
    n_noise: int,
    momentum_window: int,
):
    agents = []

    agents.extend(
        TrendFollower(rng, momentum_window=momentum_window)
        for _ in range(n_trend)
    )
    agents.extend(
        Contrarian(rng, momentum_window=momentum_window)
        for _ in range(n_contrarian)
    )
    agents.extend(NoiseTrader(rng) for _ in range(n_noise))

    return agents


def _load_real_data(version: str, variant: str) -> pd.DataFrame:
    suffix = "" if variant == "full" else f"_{variant}"
    filename = f"master_research_dataset{suffix}.csv"
    path = cfg.OUTPUT_DIR / version / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset: %s", path)

    return pd.read_csv(
        path,
        parse_dates=["snapshot_time", "entry_time"],
    )


# ============================================================
# Main
# ============================================================

def main(argv=None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    logger.info("=== FX Sentiment ABM ===")
    logger.info(
        "seed=%d steps=%d trend=%d contrarian=%d noise=%d",
        args.seed,
        args.steps,
        args.n_trend,
        args.n_contrarian,
        args.n_noise,
    )

    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = _load_real_data(args.version, args.variant)

    sub = df[df["pair"] == args.pair].copy()

    if sub.empty:
        logger.error("No data for pair %s", args.pair)
        sys.exit(1)

    sub = sub.sort_values("entry_time")

    price_series = sub["entry_close"].values
    timestamps = sub["entry_time"].values
    real_sentiment = sub["net_sentiment"].values

    # --------------------------------------------------------
    # Build agents
    # --------------------------------------------------------

    total_agents = args.n_trend + args.n_contrarian + args.n_noise
    if total_agents == 0:
        logger.error("Total agent count is 0")
        sys.exit(1)

    agents = _build_agents(
        rng,
        args.n_trend,
        args.n_contrarian,
        args.n_noise,
        args.momentum_window,
    )

    logger.info(
        "Population: %d trend + %d contrarian + %d noise = %d",
        args.n_trend,
        args.n_contrarian,
        args.n_noise,
        total_agents,
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    sim = FXSentimentSimulation(agents, rng=rng)

    max_steps = len(price_series) - sim._warmup_steps - 1
    if max_steps <= 0:
        logger.error("Not enough data for simulation after warmup")
        sys.exit(1)

    n_steps = min(args.steps, max_steps)

    sim_df = sim.run(
        n_steps=n_steps,
        price_series=price_series,
        timestamps=timestamps,
    )

    # Align real sentiment (tail-aligned)
    sim_df["real_net_sentiment"] = real_sentiment[-len(sim_df):]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("--- Simulation summary ---")
    logger.info(
        "sim mean=%.2f std=%.2f abs_mean=%.2f",
        sim_df["net_sentiment"].mean(),
        sim_df["net_sentiment"].std(),
        sim_df["net_sentiment"].abs().mean(),
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    targets = calibrate_from_dataset(df, pair=args.pair)
    comparison = compare_to_data(sim_df, targets)

    logger.info("--- Calibration comparison ---\n%s", comparison.to_string(index=False))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(args.output, index=False)
        logger.info("Saved to %s", args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()