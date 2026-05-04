"""
ABM parameter sweep with regime-aware scoring.

The sweep varies:
  - trend_ratio     : fraction of trend-following agents
  - persistence     : written to agents._PERSISTENCE_WEIGHT (regime smoothing)
  - threshold       : written to agents._INERTIA_THRESHOLD  (volatile trigger)

Module constants are saved before the sweep and restored unconditionally
afterwards (even on error), so they are always left in their original state.

Public API
----------
run_sweep(df, pair, n_steps, seed) → pd.DataFrame  (sorted ascending by score)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, UTC

import numpy as np
import pandas as pd

import research.abm.agents as _agents_mod
from research.abm.agents import build_agents
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.scoring import compute_score
from research.abm.simulation import FXSentimentSimulation


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

_TREND_GRID = [0.0, 0.5, 1.0]
_PERSISTENCE_GRID = [0.0, 0.1, 0.2]
_THRESHOLD_GRID = [0.02, 0.05, 0.1]

_N_AGENTS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sweep(
    df: pd.DataFrame,
    pair: str,
    n_steps: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a parameter sweep and return results sorted by score (lower = better)."""

    # Filter and sort pair data
    sub = df[df["pair"] == pair].copy()
    sort_col = next(
        (c for c in ("entry_time", "snapshot_time") if c in sub.columns), None
    )
    if sort_col is not None:
        sub = sub.sort_values(sort_col)

    price_series = sub["entry_close"].values

    # Calibration targets from real data
    targets = calibrate_from_dataset(df, pair=pair)

    # Save module constants – restored unconditionally in `finally`
    _orig_persistence = _agents_mod._PERSISTENCE_WEIGHT
    _orig_threshold = _agents_mod._INERTIA_THRESHOLD

    results: list[dict] = []

    try:
        for trend_ratio in _TREND_GRID:
            for persistence in _PERSISTENCE_GRID:
                for threshold in _THRESHOLD_GRID:

                    # ------------------------------------------------------
                    # FIX: independent RNG streams per run
                    # ------------------------------------------------------
                    rng_agents = np.random.default_rng(seed)
                    rng_sim = np.random.default_rng(seed + 1)

                    try:
                        # Mutate module globals (read by simulation at runtime)
                        _agents_mod._PERSISTENCE_WEIGHT = persistence
                        _agents_mod._INERTIA_THRESHOLD = threshold

                        agents = build_agents(
                            n_agents=_N_AGENTS,
                            trend_ratio=trend_ratio,
                            rng=rng_agents,
                        )

                        sim = FXSentimentSimulation(agents, rng=rng_sim)
                        sim_df = sim.run(n_steps=n_steps, price_series=price_series)

                        comparison = compare_to_data(sim_df, targets)
                        score = compute_score(comparison)

                        std_diff = _extract_abs_diff(comparison, "std")
                        autocorr_diff = _extract_abs_diff(comparison, "autocorr")

                        logger.info(
                            "trend_ratio=%.1f persistence=%.2f threshold=%.2f score=%.4f",
                            trend_ratio, persistence, threshold, score,
                        )

                        results.append(
                            {
                                "trend_ratio": trend_ratio,
                                "persistence": persistence,
                                "threshold": threshold,
                                "score": score,
                                "std_diff": std_diff,
                                "autocorr_diff": autocorr_diff,
                            }
                        )

                    except Exception as exc:
                        logger.warning(
                            "Run failed (trend_ratio=%.1f persistence=%.2f threshold=%.2f): %s",
                            trend_ratio, persistence, threshold, exc,
                        )

    finally:
        # Always restore module constants
        _agents_mod._PERSISTENCE_WEIGHT = _orig_persistence
        _agents_mod._INERTIA_THRESHOLD = _orig_threshold

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("score").reset_index(drop=True)

    return result_df


def _extract_abs_diff(comparison: pd.DataFrame, name: str) -> float:
    mask = comparison["statistic"] == name
    if not mask.any():
        return float("nan")
    val = comparison.loc[mask, "abs_diff"].values[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "ret_1b" in df.columns:
        r = df["ret_1b"].fillna(0.0).values
        price = np.cumprod(1.0 + r)
        df["price"] = price

    if "price" not in df.columns:
        raise ValueError("Dataset must contain 'price' or 'ret_1b'")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    logger.info("=== ABM Parameter Sweep ===")
    logger.info(
        "pair=%s version=%s steps=%d seed=42",
        args.pair,
        args.version,
        args.steps,
    )

    path = f"data/output/{args.version}/master_research_dataset_core.csv"
    df = load_dataset(path)

    result_df = run_sweep(df, pair=args.pair, n_steps=args.steps, seed=42)

    if result_df.empty:
        raise RuntimeError("All sweep runs failed")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"logs/abm_sweep_{args.pair}_{args.version}_{ts}.csv"

    result_df.to_csv(out_path, index=False)

    best = result_df.iloc[0]
    logger.info("Sweep results saved: %s  rows=%d", out_path, len(result_df))
    logger.info(
        "Best: trend_ratio=%.1f persistence=%.2f threshold=%.2f score=%.4f",
        best.trend_ratio,
        best.persistence,
        best.threshold,
        best.score,
    )


if __name__ == "__main__":
    main()