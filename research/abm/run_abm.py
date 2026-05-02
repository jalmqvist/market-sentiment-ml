"""
research/abm/run_abm.py
=======================
CLI entry-point for running the retail FX sentiment ABM.

This version enforces:
- real data only (no synthetic price)
- per-pair simulation
- strict causal alignment
- reproducible config snapshots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from utils.logging import setup_experiment_logging
from research.abm.agents import Contrarian, NoiseTrader, TrendFollower
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.simulation import FXSentimentSimulation

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = ["timestamp", "price", "net_sentiment", "real_net_sentiment"]


# ============================================================
# CLI
# ============================================================

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the retail FX sentiment agent-based model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--version", required=True)
    p.add_argument("--variant", default="core", choices=["full", "core", "extended"])
    p.add_argument("--pair", required=True)

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

    p.add_argument("--no-log-file", action="store_true")

    return p.parse_args(argv)


# ============================================================
# Helpers
# ============================================================

def _build_agents(
    rng: np.random.Generator,
    pair: str,
    n_trend: int,
    n_contrarian: int,
    n_noise: int,
    momentum_window: int,
):
    agents = []

    agents.extend(
        TrendFollower(rng, pair=pair, momentum_window=momentum_window)
        for _ in range(n_trend)
    )
    agents.extend(
        Contrarian(rng, pair=pair, momentum_window=momentum_window)
        for _ in range(n_contrarian)
    )
    agents.extend(
        NoiseTrader(rng, pair=pair)
        for _ in range(n_noise)
    )

    return agents


def _load_real_data(version: str, variant: str) -> tuple[pd.DataFrame, Path]:
    suffix = "" if variant == "full" else f"_{variant}"
    filename = f"master_research_dataset{suffix}.csv"
    path = cfg.OUTPUT_DIR / version / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset: %s", path)

    df = pd.read_csv(
        path,
        parse_dates=["snapshot_time", "entry_time"],
    )
    return df, path


def _write_config_snapshot(
    log_file: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    n_steps: int,
) -> Path:
    config_file = log_file.with_suffix(".json")
    payload = {
        "experiment_type": "abm",
        "cli_command": " ".join(sys.argv),
        "dataset_path": str(dataset_path),
        "dataset_version": args.version,
        "variant": args.variant,
        "pair": args.pair,
        "seed": args.seed,
        "steps": args.steps,
        "n_trend": args.n_trend,
        "n_contrarian": args.n_contrarian,
        "n_noise": args.n_noise,
        "momentum_window": args.momentum_window,
        "total_agents": args.n_trend + args.n_contrarian + args.n_noise,
        "effective_steps": n_steps,
    }
    config_file.write_text(json.dumps(payload, indent=2))
    return config_file


# ============================================================
# Main
# ============================================================

def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    log_file = setup_experiment_logging(
        experiment_type="abm",
        tag=args.pair,
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=cfg.REPO_ROOT / "logs",
    )

    if log_file is not None:
        logging.getLogger().info("Logging to %s", log_file)

    logger.info("=== FX Sentiment ABM ===")
    logger.info("cli_command: %s", " ".join(sys.argv))
    logger.info(
        "version=%s variant=%s pair=%s seed=%d steps=%d "
        "n_trend=%d n_contrarian=%d n_noise=%d momentum_window=%d",
        args.version,
        args.variant,
        args.pair,
        args.seed,
        args.steps,
        args.n_trend,
        args.n_contrarian,
        args.n_noise,
        args.momentum_window,
    )

    df, dataset_path = _load_real_data(args.version, args.variant)
    logger.info("Dataset loaded: %d rows", len(df))

    sub = df[df["pair"] == args.pair].copy()

    if sub.empty:
        logger.error("No data for pair %s", args.pair)
        sys.exit(1)

    sub = sub.sort_values("entry_time")

    price_series = sub["entry_close"].values
    timestamps = sub["entry_time"].values
    real_sentiment = sub["net_sentiment"].values

    total_agents = args.n_trend + args.n_contrarian + args.n_noise
    if total_agents == 0:
        logger.error("Total agent count is 0")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    agents = _build_agents(
        rng=rng,
        pair=args.pair,
        n_trend=args.n_trend,
        n_contrarian=args.n_contrarian,
        n_noise=args.n_noise,
        momentum_window=args.momentum_window,
    )

    logger.info(
        "Population: %d trend + %d contrarian + %d noise = %d",
        args.n_trend,
        args.n_contrarian,
        args.n_noise,
        total_agents,
    )

    sim = FXSentimentSimulation(agents, rng=rng)

    max_steps = len(price_series) - sim._warmup_steps - 1
    if max_steps <= 0:
        logger.error("Not enough data for simulation after warmup")
        sys.exit(1)

    n_steps = min(args.steps, max_steps)

    if log_file is not None:
        config_file = _write_config_snapshot(log_file, args, dataset_path, n_steps)
        logger.info("Config snapshot: %s", config_file)

    sim_df = sim.run(
        n_steps=n_steps,
        price_series=price_series,
        timestamps=timestamps,
    )

    warmup = sim._warmup_steps
    sim_df["real_net_sentiment"] = real_sentiment[
        warmup + 1 : warmup + 1 + len(sim_df)
    ]

    logger.info("--- Simulation summary ---")
    logger.info("steps_run=%d n_agents=%d", len(sim_df), sim.n_agents)
    logger.info(
        "sim mean=%.2f std=%.2f abs_mean=%.2f",
        sim_df["net_sentiment"].mean(),
        sim_df["net_sentiment"].std(),
        sim_df["net_sentiment"].abs().mean(),
    )

    targets = calibrate_from_dataset(df, pair=args.pair)
    comparison = compare_to_data(sim_df, targets)

    logger.info("--- Calibration comparison ---\n%s", comparison.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        assert set(_OUTPUT_COLUMNS).issubset(sim_df.columns)
        sim_df[_OUTPUT_COLUMNS].to_csv(args.output, index=False)
        logger.info("Saved output CSV: %s  rows=%d", args.output, len(sim_df))

    logger.info("Done.")


if __name__ == "__main__":
    main()