"""abm_experiments/sweep_with_volatility.py
================================================
Single-file ABM experiment that perturbs the *environment* volatility by
scaling returns proportionally to rolling realized volatility.

Constraints (per repository request):
- Do NOT modify research/abm/*
- Do NOT modify research/abm/sweep.py
- No refactors, no new modules, no new dependencies
- One alpha (volatility_scale) per run (no loops)

Baseline command (unchanged pipeline reference):

    python research/abm/sweep.py --version 1.2.0 --pair eur-usd --steps 500

This experiment command (new):

    python abm_experiments/sweep_with_volatility.py \
        --version 1.2.0 \
        --pair eur-usd \
        --steps 500 \
        --volatility-scale 1.0

Mechanism:
- returns = diff(price)
- vol_t = rolling_std(returns, window=24)
- vol_norm_t = clip(vol_t / mean(vol), 0, 5)
- adjusted_returns_t = returns_t * (1 + alpha * vol_norm_t)
- adjusted_price_t = price_0 + cumsum(adjusted_returns)

The ABM is then run as usual but with adjusted_price injected in place of
entry_close-derived price series.

Outputs:
- log file + config snapshot JSON in logs/ (same schema as sweep.py, plus
  volatility_scale)
- sweep CSV in logs/: abm_sweep_vol_{pair}_{version}_{timestamp}.csv
- best-path timeseries CSV in logs/: abm_sweep_vol_bestpath_{pair}_{version}_{timestamp}.csv

Note:
This experiment logs additional diagnostics to confirm that alpha is
actually perturbing the environment:
- std of original returns
- std of adjusted returns
- ratio adjusted/original
- max(vol_norm)
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

# Match sweep.py behavior: allow repo-root imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.logging import setup_experiment_logging
from research.abm import agents as agents_module
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.scoring import compute_score, extract_metric
from research.abm.simulation import FXSentimentSimulation

logger = logging.getLogger(__name__)

# Keep parameter grid identical to research/abm/sweep.py
_TREND_RATIOS = [0.0, 0.5, 1.0]
_PERSISTENCE_WEIGHTS = [0.0, 0.1, 0.2]
_INERTIA_THRESHOLDS = [0.02, 0.05, 0.1]

# Fixed non-noise / noise split (mirrors run_abm defaults: 40+40+20)
_N_NON_NOISE = 80
_N_NOISE = 20

_VOL_WINDOW = 24
_VOL_NORM_CLIP_MAX = 5.0


def _volatility_adjust_price(
    price: np.ndarray,
    alpha: float,
    window: int = _VOL_WINDOW,
) -> tuple[np.ndarray, dict]:
    """Return adjusted price series and diagnostics.

    The adjustment scales returns by (1 + alpha * vol_norm), where vol_norm is
    rolling std of returns normalized by its mean and clipped to [0, 5].

    Returns:
        adjusted_price: numpy array same length as price
        diag: dict diagnostics, including mean/std of vol and return scaling
    """
    if price.ndim != 1:
        raise ValueError("price must be 1D")
    if len(price) == 0:
        raise ValueError("price is empty")

    # returns: first element is 0.0 (since prepend is price[0])
    returns = np.diff(price, prepend=price[0]).astype(float)

    # Rolling std; ddof=0 for population std; min_periods=1 for early values
    vol = (
        pd.Series(returns)
        .rolling(window=window, min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    vol_mean = float(np.mean(vol))
    vol_std = float(np.std(vol))

    if vol_mean > 0.0 and np.isfinite(vol_mean):
        vol_norm = vol / vol_mean
    else:
        vol_norm = np.zeros_like(vol)

    vol_norm = np.clip(vol_norm, 0.0, _VOL_NORM_CLIP_MAX)

    adjusted_returns = returns * (1.0 + alpha * vol_norm)
    adjusted_price = float(price[0]) + np.cumsum(adjusted_returns)

    # Safety checks
    if len(adjusted_price) != len(price):
        raise RuntimeError("Adjusted price length mismatch")
    if not np.isfinite(adjusted_price).all():
        raise RuntimeError("Adjusted price contains NaN/inf")

    # Ensure first value matches original (within tolerance)
    if not np.isclose(adjusted_price[0], price[0], rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Adjusted price[0] != original price[0] ({adjusted_price[0]} vs {price[0]})"
        )

    # Additional diagnostics to verify alpha effect
    returns_std = float(np.std(returns))
    adjusted_returns_std = float(np.std(adjusted_returns))
    std_ratio = (adjusted_returns_std / returns_std) if returns_std > 0 else float("nan")

    diag = {
        "vol_window": int(window),
        "vol_mean": vol_mean,
        "vol_std": vol_std,
        "vol_norm_mean": float(np.mean(vol_norm)),
        "vol_norm_max": float(np.max(vol_norm)) if len(vol_norm) else 0.0,
        "vol_norm_clip_max": float(_VOL_NORM_CLIP_MAX),
        "returns_std": returns_std,
        "adjusted_returns_std": adjusted_returns_std,
        "returns_std_ratio": float(std_ratio),
        "alpha": float(alpha),
    }
    return adjusted_price, diag


def run_sweep_with_price_series(
    df: pd.DataFrame,
    pair: str,
    n_steps: int,
    price_series_override: np.ndarray,
    seed: int = 42,
    momentum_window: int = 12,
    seed_base: int | None = None,
    bestpath_out_path: Path | None = None,
) -> pd.DataFrame:
    """Run the same sweep as research/abm/sweep.py but with injected price_series.

    If bestpath_out_path is provided, the script additionally runs the best
    parameter combination again (deterministically) and saves the simulation
    timeseries (sim_df) to that path.
    """
    _result_columns = [
        "trend_ratio",
        "persistence",
        "threshold",
        "score",
        "std_diff",
        "autocorr_diff",
        "abs_mean_diff",
        "extreme_diff",
    ]

    targets = calibrate_from_dataset(df, pair=pair)

    # We still need timestamps + real_sentiment aligned to pair data
    sub = df[df["pair"] == pair].copy().sort_values("entry_time")
    timestamps = sub["entry_time"].values
    real_sentiment = sub["net_sentiment"].values

    price_series = np.asarray(price_series_override, dtype=float)
    if len(price_series) != len(timestamps):
        raise ValueError(
            f"Injected price_series length {len(price_series)} != timestamps length {len(timestamps)}"
        )

    # Save original module constants so we can restore them
    orig_persistence = agents_module._PERSISTENCE_WEIGHT
    orig_threshold = agents_module._INERTIA_THRESHOLD

    results: list[dict] = []

    # Track best run for best-path output
    best: dict | None = None
    best_run_index: int | None = None

    try:
        for run_index, (trend_ratio, persistence, threshold) in enumerate(
            product(_TREND_RATIOS, _PERSISTENCE_WEIGHTS, _INERTIA_THRESHOLDS)
        ):
            agents_module._PERSISTENCE_WEIGHT = persistence
            agents_module._INERTIA_THRESHOLD = threshold

            n_trend = int(round(trend_ratio * _N_NON_NOISE))
            n_contrarian = _N_NON_NOISE - n_trend

            run_seed = (seed_base + run_index) if seed_base is not None else seed
            rng = np.random.default_rng(run_seed)
            agents = _build_agents(
                rng,
                pair=pair,
                n_trend=n_trend,
                n_contrarian=n_contrarian,
                n_noise=_N_NOISE,
                momentum_window=momentum_window,
            )

            sim = FXSentimentSimulation(agents, rng=rng)

            max_steps = len(price_series) - sim.warmup_steps - 1
            if max_steps <= 0:
                logger.warning("Not enough price data for pair=%s, skipping run", pair)
                continue

            effective_steps = min(n_steps, max_steps)

            try:
                sim_df = sim.run(
                    n_steps=effective_steps,
                    price_series=price_series,
                    timestamps=timestamps,
                )
                warmup = sim.warmup_steps
                sim_df["real_net_sentiment"] = real_sentiment[
                    warmup + 1 : warmup + 1 + len(sim_df)
                ]

                comparison = compare_to_data(sim_df, targets)
                score = compute_score(comparison)

                row = {
                    "trend_ratio": trend_ratio,
                    "persistence": persistence,
                    "threshold": threshold,
                    "score": score,
                    "std_diff": extract_metric(comparison, "std"),
                    "autocorr_diff": extract_metric(comparison, "autocorr"),
                    "abs_mean_diff": extract_metric(comparison, "abs_mean"),
                    "extreme_diff": extract_metric(comparison, "extreme_freq"),
                }
                results.append(row)

                if best is None or score < best["score"]:
                    best = row
                    best_run_index = run_index

                logger.info(
                    "trend_ratio=%.1f persistence=%.2f threshold=%.3f score=%.4f",
                    trend_ratio,
                    persistence,
                    threshold,
                    score,
                )

            except Exception as exc:
                logger.warning(
                    "Run failed (trend_ratio=%.1f persistence=%.2f threshold=%.3f): %s",
                    trend_ratio,
                    persistence,
                    threshold,
                    exc,
                )

    finally:
        agents_module._PERSISTENCE_WEIGHT = orig_persistence
        agents_module._INERTIA_THRESHOLD = orig_threshold

    if not results:
        return pd.DataFrame(columns=_result_columns)

    result_df = pd.DataFrame(results).sort_values("score").reset_index(drop=True)

    # Optional: dump best-path sim_df
    if bestpath_out_path is not None and not result_df.empty and best is not None:
        assert best_run_index is not None

        # Re-run the best combination deterministically and write its time series
        best_trend_ratio = float(best["trend_ratio"])
        best_persistence = float(best["persistence"])
        best_threshold = float(best["threshold"])

        agents_module._PERSISTENCE_WEIGHT = best_persistence
        agents_module._INERTIA_THRESHOLD = best_threshold

        try:
            n_trend = int(round(best_trend_ratio * _N_NON_NOISE))
            n_contrarian = _N_NON_NOISE - n_trend

            run_seed = (seed_base + best_run_index) if seed_base is not None else seed
            rng = np.random.default_rng(run_seed)

            agents = _build_agents(
                rng,
                pair=pair,
                n_trend=n_trend,
                n_contrarian=n_contrarian,
                n_noise=_N_NOISE,
                momentum_window=momentum_window,
            )

            sim = FXSentimentSimulation(agents, rng=rng)
            max_steps = len(price_series) - sim.warmup_steps - 1
            effective_steps = min(n_steps, max_steps)

            sim_df = sim.run(
                n_steps=effective_steps,
                price_series=price_series,
                timestamps=timestamps,
            )
            warmup = sim.warmup_steps
            sim_df["real_net_sentiment"] = real_sentiment[
                warmup + 1 : warmup + 1 + len(sim_df)
            ]

            bestpath_out_path.parent.mkdir(parents=True, exist_ok=True)
            sim_df.to_csv(bestpath_out_path, index=False)

            logger.info(
                "Best-path saved: %s  rows=%d (trend_ratio=%.1f persistence=%.2f threshold=%.3f)",
                bestpath_out_path,
                len(sim_df),
                best_trend_ratio,
                best_persistence,
                best_threshold,
            )

        finally:
            agents_module._PERSISTENCE_WEIGHT = orig_persistence
            agents_module._INERTIA_THRESHOLD = orig_threshold

    return result_df


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ABM sweep with volatility-scaled price environment.",
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
    p.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="When set, each run uses seed_base + run_index as its seed",
    )
    p.add_argument("--momentum-window", type=int, default=12)
    p.add_argument(
        "--volatility-scale",
        type=float,
        default=0.0,
        help="Alpha scaling for volatility-conditioned return amplification",
    )
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
        tag=f"sweep-vol-{pair}",
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

    logger.info("=== ABM Parameter Sweep (Volatility Environment Scaling) ===")
    logger.info("cli_command: %s", " ".join(sys.argv))
    logger.info(
        "pair=%s version=%s steps=%d seed=%d momentum_window=%d volatility_scale=%.4f",
        args.pair,
        args.version,
        args.steps,
        args.seed,
        args.momentum_window,
        args.volatility_scale,
    )

    df, dataset_path = _load_real_data(args.version, args.variant)

    # Create a copy of the filtered pair subset; do not mutate df in-place.
    sub = df[df["pair"] == pair].copy().sort_values("entry_time")
    if sub.empty:
        raise ValueError(f"No data found for pair={pair}")

    price = sub["entry_close"].to_numpy(dtype=float)

    adjusted_price, vol_diag = _volatility_adjust_price(
        price=price, alpha=float(args.volatility_scale), window=_VOL_WINDOW
    )

    # Required diagnostics (mean/std of rolling vol)
    logger.info(
        "rolling_vol diagnostics: window=%d vol_mean=%.8g vol_std=%.8g vol_norm_mean=%.6g vol_norm_max=%.6g clip_max=%.1f",
        vol_diag["vol_window"],
        vol_diag["vol_mean"],
        vol_diag["vol_std"],
        vol_diag["vol_norm_mean"],
        vol_diag["vol_norm_max"],
        vol_diag["vol_norm_clip_max"],
    )

    # Additional sanity check diagnostics proving alpha has an effect
    logger.info(
        "return_scaling diagnostics: alpha=%.4g returns_std=%.8g adjusted_returns_std=%.8g std_ratio=%.6g",
        vol_diag["alpha"],
        vol_diag["returns_std"],
        vol_diag["adjusted_returns_std"],
        vol_diag["returns_std_ratio"],
    )

    # Write config snapshot alongside the log file
    if log_file is not None:
        config_payload = {
            "experiment_type": "abm_sweep_vol",
            "cli_command": " ".join(sys.argv),
            "dataset_path": str(dataset_path),
            "dataset_version": args.version,
            "dataset_variant": args.variant,
            "pair": args.pair,
            "steps": args.steps,
            "seed": args.seed,
            "seed_base": args.seed_base,
            "momentum_window": args.momentum_window,
            "trend_ratios": _TREND_RATIOS,
            "persistence_weights": _PERSISTENCE_WEIGHTS,
            "inertia_thresholds": _INERTIA_THRESHOLDS,
            "volatility_scale": float(args.volatility_scale),
            "vol_window": int(_VOL_WINDOW),
            "vol_norm_clip_max": float(_VOL_NORM_CLIP_MAX),
            "vol_mean": vol_diag["vol_mean"],
            "vol_std": vol_diag["vol_std"],
            "vol_norm_mean": vol_diag["vol_norm_mean"],
            "vol_norm_max": vol_diag["vol_norm_max"],
            "returns_std": vol_diag["returns_std"],
            "adjusted_returns_std": vol_diag["adjusted_returns_std"],
            "returns_std_ratio": vol_diag["returns_std_ratio"],
        }
        config_file = log_file.with_suffix(".json")
        config_file.write_text(json.dumps(config_payload, indent=2))
        logger.info("Config snapshot: %s", config_file)

    log_dir = cfg.REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bestpath_out_path = log_dir / f"abm_sweep_vol_bestpath_{pair}_{args.version}_{timestamp}.csv"

    result_df = run_sweep_with_price_series(
        df=df,
        pair=args.pair,
        n_steps=args.steps,
        price_series_override=adjusted_price,
        seed=args.seed,
        momentum_window=args.momentum_window,
        seed_base=args.seed_base,
        bestpath_out_path=bestpath_out_path,
    )

    out_path = log_dir / f"abm_sweep_vol_{pair}_{args.version}_{timestamp}.csv"

    result_df.to_csv(out_path, index=False)
    logger.info("Sweep results saved: %s  rows=%d", out_path, len(result_df))

    if not result_df.empty:
        best = result_df.iloc[0]
        logger.info(
            "Best: trend_ratio=%.1f persistence=%.2f threshold=%.3f score=%.4f",
            best["trend_ratio"],
            best["persistence"],
            best["threshold"],
            best["score"],
        )


if __name__ == "__main__":
    main()
