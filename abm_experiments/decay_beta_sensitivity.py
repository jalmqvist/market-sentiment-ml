"""abm_experiments/decay_beta_sensitivity.py

Fixed-configuration sensitivity harness for Stage-2 decay (release).

Goal
----
Vary ONLY decay_volatility_scale (beta) while keeping the ABM configuration
fixed, then report timeseries-derived diagnostics.

Constraints
-----------
- Single file
- No refactors / no shared utilities
- No changes to existing pipeline
- One beta per invocation (no internal loops)

Fixed ABM configuration (per runbook / baseline):
- trend_ratio = 1.0
- persistence = 0.20
- threshold = 0.100

Metrics
-------
- pct_time_saturated: fraction of steps with |net_sentiment| >= 90
- sign_flips: number of sign changes in net_sentiment
- autocorr_lag1: lag-1 autocorrelation of net_sentiment

Output
------
By default, print a single table row to stdout (backward compatible):

    beta | pct_saturated | sign_flips | autocorr

If --verbose is passed, include identifying columns first:

    pair | seed | beta | pct_saturated | sign_flips | autocorr

Notes
-----
- --seed is optional; default remains 42.
- --verbose only changes the printed output (no files are written).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Match sweep.py behavior: allow repo-root imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.abm import agents as agents_module
from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.simulation import FXSentimentSimulation


def _autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-config sensitivity harness for Stage-2 decay beta.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", required=True, help="Dataset version (e.g. '1.2.0')")
    p.add_argument("--pair", required=True, help="FX pair (e.g. 'eur-usd')")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--beta", type=float, required=True, help="decay_volatility_scale")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Include pair and seed columns in stdout output.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Fixed configuration (do not expose as CLI)
    trend_ratio = 1.0
    persistence = 0.20
    threshold = 0.100
    seed = int(args.seed)
    momentum_window = 12

    # Agent population composition (mirrors run_abm defaults: 40+40+20)
    n_non_noise = 80
    n_noise = 20

    # With trend_ratio=1.0, all non-noise are trend followers
    n_trend = int(round(trend_ratio * n_non_noise))
    n_contrarian = n_non_noise - n_trend

    # Save original module constants so we can restore them
    orig_persistence = agents_module._PERSISTENCE_WEIGHT
    orig_threshold = agents_module._INERTIA_THRESHOLD

    orig_decay_base = getattr(agents_module, "_DECAY_BASE", 0.0)
    orig_decay_vol_scale = getattr(agents_module, "_DECAY_VOLATILITY_SCALE", 0.0)
    orig_decay_clip_max = getattr(agents_module, "_DECAY_CLIP_MAX", 0.2)

    # Fixed decay parameters (per experiment definition)
    decay_base = 0.0
    decay_clip_max = 0.2

    try:
        # Apply fixed ABM parameters
        agents_module._PERSISTENCE_WEIGHT = float(persistence)
        agents_module._INERTIA_THRESHOLD = float(threshold)

        # Apply decay parameters
        agents_module._DECAY_BASE = float(decay_base)
        agents_module._DECAY_VOLATILITY_SCALE = float(args.beta)
        agents_module._DECAY_CLIP_MAX = float(decay_clip_max)

        df, _dataset_path = _load_real_data(args.version, variant="core")
        sub = df[df["pair"] == args.pair].copy().sort_values("entry_time")
        if sub.empty:
            raise ValueError(f"No data found for pair={args.pair}")

        price_series = sub["entry_close"].to_numpy(dtype=float)
        timestamps = sub["entry_time"].values

        rng = np.random.default_rng(seed)
        agents = _build_agents(
            rng,
            pair=args.pair,
            n_trend=n_trend,
            n_contrarian=n_contrarian,
            n_noise=n_noise,
            momentum_window=momentum_window,
        )

        sim = FXSentimentSimulation(agents, rng=rng)
        max_steps = len(price_series) - sim.warmup_steps - 1
        if max_steps <= 0:
            raise ValueError(f"Not enough price data for pair={args.pair}")

        effective_steps = min(int(args.steps), int(max_steps))

        sim_df = sim.run(
            n_steps=effective_steps,
            price_series=price_series,
            timestamps=timestamps,
        )

        s = sim_df["net_sentiment"].to_numpy(dtype=float)
        pct_saturated = float((np.abs(s) >= 90.0).mean())

        sign = np.sign(s)
        sign_flips = int(((sign[1:] * sign[:-1]) < 0).sum()) if len(sign) > 1 else 0

        ac1 = _autocorr_lag1(s)

        if args.verbose:
            print(
                f"{args.pair} | {seed:d} | {float(args.beta):.6g} | "
                f"{pct_saturated:.6g} | {sign_flips:d} | {ac1:.6g}"
            )
        else:
            # Backward compatible output format
            print(f"{float(args.beta):.6g} | {pct_saturated:.6g} | {sign_flips:d} | {ac1:.6g}")

    finally:
        # Restore module constants
        agents_module._PERSISTENCE_WEIGHT = orig_persistence
        agents_module._INERTIA_THRESHOLD = orig_threshold

        agents_module._DECAY_BASE = orig_decay_base
        agents_module._DECAY_VOLATILITY_SCALE = orig_decay_vol_scale
        agents_module._DECAY_CLIP_MAX = orig_decay_clip_max


if __name__ == "__main__":
    main()
