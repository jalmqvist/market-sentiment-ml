"""
research/abm/sweep.py
=====================
Parameter sweep pipeline for the retail FX sentiment ABM.

Runs a grid search over three axes:
    - trend_ratios:         fraction of non-noise agents that are trend-followers
    - persistence_weights:  _PERSISTENCE_WEIGHT injected into agents.py
    - inertia_thresholds:   _INERTIA_THRESHOLD injected into agents.py

Results are sorted by score (ascending, lower is better) and saved to
``logs/abm_sweep_{pair}_{version}_{timestamp}.csv``.

Usage::

    python research/abm/sweep.py \\
        --version 1.2.0 \\
        --pair eur-usd \\
        --steps 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config as cfg
from utils.logging import setup_experiment_logging
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
    seed_base: int | None = None,
) -> pd.DataFrame:
    """Run parameter sweep and return results sorted by score ascending.

    Args:
        df: Full dataset DataFrame (as returned by :func:`run_abm._load_real_data`).
        pair: FX pair to simulate (e.g. ``"eur-usd"``).
        n_steps: Maximum number of simulation steps per run.
        seed: Random seed for reproducibility (used when *seed_base* is ``None``).
        momentum_window: Momentum window passed to trend/contrarian agents.
        seed_base: When provided, each run uses ``seed_base + run_index`` as its
            seed instead of the fixed *seed*.

    Returns:
        DataFrame with one row per parameter combination, sorted by ``score``
        ascending.  Columns: ``trend_ratio``, ``persistence``, ``threshold``,
        ``score``, ``std_diff``, ``autocorr_diff``, ``abs_mean_diff``,
        ``extreme_diff``.
    """
    _result_columns = [
        "trend_ratio", "persistence", "threshold",
        "score", "std_diff", "autocorr_diff", "abs_mean_diff", "extreme_diff",
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
        for run_index, (trend_ratio, persistence, threshold) in enumerate(product(
            _TREND_RATIOS, _PERSISTENCE_WEIGHTS, _INERTIA_THRESHOLDS
        )):
            agents_module._PERSISTENCE_WEIGHT = persistence
            agents_module._INERTIA_THRESHOLD = threshold

            n_trend = int(round(trend_ratio * _N_NON_NOISE))
            n_contrarian = _N_NON_NOISE - n_trend

            run_seed = (seed_base + run_index) if seed_base is not None else seed
            rng = np.random.default_rng(run_seed)
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
                    "abs_mean_diff": extract_metric(comparison, "abs_mean"),
                    "extreme_diff": extract_metric(comparison, "extreme_freq"),
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
    p.add_argument("--seed-base", type=int, default=None,
                   help="When set, each run uses seed_base + run_index as its seed")
    p.add_argument("--momentum-window", type=int, default=12)
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging; write to stdout only",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    pair = args.pair

    log_file = setup_experiment_logging(
        experiment_type="abm",
        tag=f"sweep-{pair}",
        log_level=args.log_level,
        no_log_file=args.no_log_file,
        log_dir=cfg.REPO_ROOT / "logs",
    )

    # Derive shared timestamp from the log filename so all outputs are aligned
    if log_file is not None:
        timestamp = log_file.stem.rsplit("_", 1)[-1]
        logging.getLogger().info("Logging to %s", log_file)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    logger.info("=== ABM Parameter Sweep ===")
    logger.info("cli_command: %s", " ".join(sys.argv))
    logger.info(
        "pair=%s version=%s steps=%d seed=%d momentum_window=%d",
        args.pair, args.version, args.steps, args.seed, args.momentum_window,
    )

    df, dataset_path = _load_real_data(args.version, args.variant)

    # Write config snapshot alongside the log file
    if log_file is not None:
        config_payload = {
            "experiment_type": "abm_sweep",
            "cli_command": " ".join(sys.argv),
            "dataset_path": str(dataset_path),
            "dataset_version": args.version,
            "pair": args.pair,
            "steps": args.steps,
            "seed": args.seed,
            "seed_base": args.seed_base,
            "momentum_window": args.momentum_window,
            "trend_ratios": _TREND_RATIOS,
            "persistence_weights": _PERSISTENCE_WEIGHTS,
            "inertia_thresholds": _INERTIA_THRESHOLDS,
        }
        config_file = log_file.with_suffix(".json")
        config_file.write_text(json.dumps(config_payload, indent=2))
        logger.info("Config snapshot: %s", config_file)

    result_df = run_sweep(
        df=df,
        pair=args.pair,
        n_steps=args.steps,
        seed=args.seed,
        momentum_window=args.momentum_window,
        seed_base=args.seed_base,
    )

    log_dir = cfg.REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"abm_sweep_{pair}_{args.version}_{timestamp}.csv"

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
