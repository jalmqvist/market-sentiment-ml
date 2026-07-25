"""abm_experiments/reactive_jpy_episode_calibration.py
================================================
Stage 2 — Episode-calibrated ABM harness for Reactive-JPY.

Replaces the statistical calibration objective (mean, std, autocorr)
with an episode-structure objective: does the ABM generate consensus
episodes matching the empirical BSVE calibration?

Fixed configuration (post-PR5 USDJPY unlock regime):
- Population: n_trend=50, n_contrarian=50, n_noise=0
- Momentum window: 3
- Persistence: 0.10, threshold: 0.05
- Decay clip max: 0.5 (increased from 0.2 for observability)

Sweeps one mechanism parameter per invocation. Pair with shell loops
for grid exploration.

Usage:
    # Stage 2.2: Anchor sweep (unlock dynamics)
    for anchor in 0.00 0.05 0.10 0.15 0.25; do
        python abm_experiments/reactive_jpy_episode_calibration.py \
            --pair usd-jpy \
            --steps 2000 \
            --anchor-strength $anchor \
            --beta 0.0 \
            --calibration-artifact bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
            --verbose
    done

    # Stage 2.3: Beta sweep (hazard structure)
    for beta in 0.00 0.01 0.02 0.05 0.10; do
        python abm_experiments/reactive_jpy_episode_calibration.py \
            --pair usd-jpy \
            --steps 2000 \
            --anchor-strength 0.10 \
            --beta $beta \
            --calibration-artifact bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
            --verbose
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm_experiments.episode_utils import (
    ConsensusEpisode,
    compute_hazard_by_maturity,
    episode_summary,
    extract_consensus_episodes,
    load_calibration_artifact,
    score_episode_structure,
)
from research.abm import agents as agents_module
from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.simulation import FXSentimentSimulation




# Fixed configuration (USDJPY unlock regime per post-PR85 calibration)
_PERSISTENCE = 0.10
_THRESHOLD = 0.05
_N_TREND = 50
_N_CONTRARIAN = 50
_N_NOISE = 0
_MOMENTUM_WINDOW = 3
_DECAY_BASE = 0.0
_DECAY_CLIP_MAX = 0.5  # Increased from 0.2 for JPY observability

# Episode extraction parameters
_MIN_EPISODE_STEPS = 2  # Same as BSVE calibration




def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Episode-calibrated ABM harness for Reactive-JPY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        default="1.6.1",
        help="Dataset version (e.g. '1.6.1')",
    )
    p.add_argument(
        "--variant",
        default="core",
        choices=["full", "core", "extended"],
        help="Dataset variant",
    )
    p.add_argument("--pair", required=True, help="FX pair (e.g. 'usd-jpy')")
    p.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")

    # Mechanism parameters
    p.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="decay_volatility_scale (Stage-2 decay)",
    )
    p.add_argument(
        "--anchor-strength",
        type=float,
        default=None,
        help="Override ABM_ANCHOR_STRENGTH (default: module default, 2.0)",
    )
    p.add_argument(
        "--decay-base",
        type=float,
        default=_DECAY_BASE,
        help="Base decay rate (default: 0.0)",
    )
    p.add_argument(
        "--decay-clip-max",
        type=float,
        default=_DECAY_CLIP_MAX,
        help="Maximum decay per step (default: 0.5)",
    )

    # Calibration target
    p.add_argument(
        "--calibration-artifact",
        required=True,
        help="Path to BSVE calibration artifact JSON (provides thresholds)",
    )

    # Output control
    p.add_argument("--verbose", action="store_true", help="Full diagnostics")
    return p.parse_args(argv)




def _run_abm_episode_simulation(
    args: argparse.Namespace,
    extreme_threshold: float,
) -> tuple[list[ConsensusEpisode], pd.DataFrame, pd.DataFrame, int]:
    """
    Run ABM simulation and extract consensus episodes.

    Returns:
        episodes: List of ConsensusEpisode from simulation
        sim_df: Full simulation DataFrame
        hazard_df: Hazard analysis DataFrame
        n_total_steps: Total simulation steps (for frequency calculation)
    """
    # Save original module constants
    orig_persistence = agents_module._PERSISTENCE_WEIGHT
    orig_threshold = agents_module._INERTIA_THRESHOLD
    orig_decay_base = getattr(agents_module, "_DECAY_BASE", 0.0)
    orig_decay_vol_scale = getattr(agents_module, "_DECAY_VOLATILITY_SCALE", 0.0)
    orig_decay_clip_max = getattr(agents_module, "_DECAY_CLIP_MAX", 0.2)
    orig_anchor = getattr(agents_module, "_SWITCHING_ANCHOR_STRENGTH", 2.0)

    episodes: list[ConsensusEpisode] = []
    sim_df: pd.DataFrame = pd.DataFrame()
    hazard_df: pd.DataFrame = pd.DataFrame()
    n_total_steps: int = 0

    try:
        # Apply fixed configuration
        agents_module._PERSISTENCE_WEIGHT = float(_PERSISTENCE)
        agents_module._INERTIA_THRESHOLD = float(_THRESHOLD)
        agents_module._DECAY_BASE = float(args.decay_base)
        agents_module._DECAY_VOLATILITY_SCALE = float(args.beta)
        agents_module._DECAY_CLIP_MAX = float(args.decay_clip_max)

        if args.anchor_strength is not None:
            agents_module._SWITCHING_ANCHOR_STRENGTH = float(args.anchor_strength)

        # Load data
        df, _dataset_path = _load_real_data(args.version, args.variant)
        sub = df[df["pair"] == args.pair].copy().sort_values("entry_time")
        if sub.empty:
            raise ValueError(f"No data found for pair={args.pair}")

        price_series = sub["entry_close"].to_numpy(dtype=float)
        timestamps = sub["entry_time"].values

        # Build agents
        rng = np.random.default_rng(args.seed)
        agents = _build_agents(
            rng,
            pair=args.pair,
            n_trend=_N_TREND,
            n_contrarian=_N_CONTRARIAN,
            n_noise=_N_NOISE,
            momentum_window=_MOMENTUM_WINDOW,
        )

        # Run simulation
        sim = FXSentimentSimulation(agents, rng=rng)
        max_steps = len(price_series) - sim.warmup_steps - 1
        if max_steps <= 0:
            raise ValueError(f"Not enough price data for pair={args.pair}")

        effective_steps = min(int(args.steps), int(max_steps))
        n_total_steps = effective_steps

        sim_df = sim.run(
            n_steps=effective_steps,
            price_series=price_series,
            timestamps=timestamps,
        )

        # Extract episodes from net_sentiment
        net_sentiment = sim_df["net_sentiment"].to_numpy()
        episodes = extract_consensus_episodes(
            net_sentiment,
            extreme_threshold=extreme_threshold,
            min_episode_steps=_MIN_EPISODE_STEPS,
            pair=args.pair,
        )

        # Compute hazard curve
        if len(episodes) >= 10:
            hazard_df = compute_hazard_by_maturity(episodes)
        else:
            hazard_df = pd.DataFrame(columns=[
                "maturity_step", "n_at_risk", "n_reversals",
                "hazard_rate", "cumulative_survival"
            ])

    finally:
        # Restore module constants
        agents_module._PERSISTENCE_WEIGHT = orig_persistence
        agents_module._INERTIA_THRESHOLD = orig_threshold
        agents_module._DECAY_BASE = orig_decay_base
        agents_module._DECAY_VOLATILITY_SCALE = orig_decay_vol_scale
        agents_module._DECAY_CLIP_MAX = orig_decay_clip_max
        if args.anchor_strength is not None:
            agents_module._SWITCHING_ANCHOR_STRENGTH = orig_anchor

    return episodes, sim_df, hazard_df, n_total_steps




def main(argv=None) -> None:
    args = _parse_args(argv)

    # Load calibration artifact for thresholds and targets
    artifact = load_calibration_artifact(args.calibration_artifact)
    thresholds = artifact.get("thresholds", {})

    extreme_threshold = thresholds.get("extreme_threshold_net_pct")
    young_boundary = thresholds.get("young_boundary_bars")
    mature_boundary = thresholds.get("mature_boundary_bars")

    if extreme_threshold is None:
        print("ERROR: calibration artifact missing extreme_threshold", file=sys.stderr)
        sys.exit(1)

    # Run ABM and extract episodes
    episodes, sim_df, hazard_df, n_total_steps = _run_abm_episode_simulation(
        args, extreme_threshold
    )

    # Score episode structure against calibration artifact
    score = score_episode_structure(
        episodes=episodes,
        hazard_df=hazard_df,
        calibration_artifact=artifact,
        n_total_steps=n_total_steps,
        young_fraction=0.4,
        mature_fraction=1.6,
    )

    # Compute summary diagnostics
    summary = episode_summary(
        episodes,
        n_total_steps=n_total_steps,
        young_boundary=young_boundary,
        mature_boundary=mature_boundary,
    )

    # Output
    anchor_disp = args.anchor_strength if args.anchor_strength is not None else "default"
    beta_disp = args.beta

    if args.verbose:
        # Full multi-line output for single-run inspection
        print(f"# Reactive-JPY Episode Calibration")
        print(f"# pair={args.pair} seed={args.seed} steps={n_total_steps}")
        print(f"# anchor={anchor_disp} beta={beta_disp}")
        print(f"#")
        print(f"# Episode structure score: {score.total_score:.4f}")
        print(f"#   duration_error: {score.duration_ratio_error:.4f}")
        print(f"#   reversal_error: {score.reversal_gradient_error:.4f}")
        print(f"#   crossover_error: {score.hazard_crossover_error:.4f}")
        print(f"#   frequency_error: {score.frequency_ratio_error:.4f}")
        print(f"#")
        print(f"# Simulated vs empirical:")
        print(f"#   median_duration: {score.sim_median_duration:.1f} vs {score.emp_median_duration:.1f}")
        print(f"#   reversal_young: {score.sim_reversal_young:.3f} vs {score.emp_reversal_young:.3f}")
        print(f"#   reversal_mature: {score.sim_reversal_mature:.3f} vs {score.emp_reversal_mature:.3f}")
        print(f"#   hazard_crossover: {score.sim_hazard_crossover:.1f} vs {score.emp_hazard_crossover:.1f}")
        print(f"#   episode_freq_1000: {score.sim_episode_frequency:.1f} vs {score.emp_episode_frequency:.1f}")
        print(f"#")
        print(f"# Summary diagnostics:")
        print(f"#   n_episodes: {summary['episode_count']}")
        print(f"#   censoring_rate: {summary['censoring_rate']:.4f}")
        print(f"#   survival_8: {summary['survival_counts'].get('8', 0)}")
        print(f"#   survival_16: {summary['survival_counts'].get('16', 0)}")
        print(f"#   survival_24: {summary['survival_counts'].get('24', 0)}")
    else:
        # Compact single-line output for shell-loop sweeps
        # Format: pair | seed | anchor | beta | score | n_ep | med_dur | rev_y | rev_m | surv_8
        anchor_str = f"{args.anchor_strength:.2f}" if args.anchor_strength is not None else "def"
        print(
            f"{args.pair} | {args.seed} | {anchor_str} | {beta_disp:.2f} | "
            f"{score.total_score:.4f} | {summary['episode_count']} | "
            f"{summary['median_episode_duration_steps'] or 0:.1f} | "
            f"{score.sim_reversal_young:.3f} | {score.sim_reversal_mature:.3f} | "
            f"{summary['survival_counts'].get('8', 0)}"
        )

    sys.exit(0)




if __name__ == "__main__":
    main()
