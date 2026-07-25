"""
regime_hierarchy_test.py — Stage 3: BSVE State-Label Injection
==============================================================
Drop-in replacement for the Stage 2 regime_hierarchy_test.py.

New behaviour when --use-bsve-states is set:
  - Loads the BSVE augmented dataset (state_id column: ENTRY/MATURING/MATURE)
  - Aligns ABM-generated net_sentiment to empirical dataset timesteps
  - Computes per-state forward-return correlation (Spearman + Pearson)
  - Tests hypothesis H4: MATURING > ENTRY > MATURE correlation gradient

Without --use-bsve-states the script falls back to the original price-only
LVTF/HVTF/LVR/HVR regime classification (Stage 2 behaviour, unchanged).

Constraints:
  - Single file, no modifications to research/abm/
  - Calibrated parameter point: anchor=0.25, beta=0.02
  - Reads sweep.py / agents.py through standard import path only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path bootstrap — insert REPO ROOT so 'research.abm' is importable as a
# package. Mirrors the pattern in old_regime_hierarchy_test.py and sweep.py.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from research.abm.agents import TrendFollower, Contrarian, NoiseTrader
    import research.abm.agents as agents_module
    import config as cfg
    from research.abm.simulation import FXSentimentSimulation
except ImportError as exc:
    raise ImportError(
        f"Cannot import from research/abm. Check REPO_ROOT={REPO_ROOT}\n{exc}"
    ) from exc
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Replace the BSVE_STATES and H4_ORDER constants:
BSVE_STATES = ("ENTRY", "MATURING", "MATURE")
H4_ORDER    = ["MATURING", "ENTRY", "MATURE"]

# Add this mapping constant — dataset labels → canonical short labels
STATE_ID_MAP = {
    "JPY_CONSENSUS_YOUNG":    "ENTRY",
    "JPY_CONSENSUS_MATURING": "MATURING",
    "JPY_CONSENSUS_MATURE":   "MATURE",
    # Direct short-form pass-through (future-proofing)
    "ENTRY":    "ENTRY",
    "MATURING": "MATURING",
    "MATURE":   "MATURE",
}

# Rows with state_id not in STATE_ID_MAP (e.g. JPY_NON_EXTREME) are
# excluded from the H4 analysis — they are non-episode background rows.
NON_EPISODE_STATES = {"JPY_NON_EXTREME"}
PRICE_REGIMES = ("LVTF", "HVTF", "LVR", "HVR")  # Stage 2 fallback labels

# Calibrated parameter point (locked 2026-07-25)
CALIBRATED_PARAMS = dict(
    anchor_strength = 0.25,
    beta            = 0.02,   # decay_vol_scale
    decay_base      = 0.00,
    decay_clip_max  = 0.50,
    n_trend         = 50,
    n_contrarian    = 50,
    n_noise         = 0,
    momentum_window = 3,
    persistence     = 0.10,
    threshold       = 0.05,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regime hierarchy test with optional BSVE state injection (Stage 3)."
    )
    p.add_argument("--pair", default="usd-jpy",
                   help="Currency pair slug (e.g. usd-jpy, eur-jpy, gbp-jpy)")
    p.add_argument("--steps", type=int, default=1000,
                   help="ABM simulation steps per run")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed")
    p.add_argument("--runs", type=int, default=5,
                   help="Number of independent runs to average over")
    p.add_argument("--beta", type=float,
                   default=CALIBRATED_PARAMS["beta"],
                   help="decay_vol_scale (beta)")
    p.add_argument("--anchor-strength", type=float,
                   default=CALIBRATED_PARAMS["anchor_strength"],
                   help="ABM anchor strength")
    # BSVE injection
    p.add_argument("--use-bsve-states", action="store_true",
                   help="Replace price-regime classification with BSVE state_id labels")
    p.add_argument("--bsve-states-path", type=str, default=None,
                   help="Path to BSVE augmented dataset CSV")
    p.add_argument("--forward-horizon", type=int, default=24,
                   help="Forward-return horizon in bars (default: 24 = ret_24b)")
    p.add_argument("--calibration-artifact", type=str, default=None,
                   help="Path to reactive_jpy_calibration_v1.json (metadata echo)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write result dict to this JSON path")
    return p


# ---------------------------------------------------------------------------
# ABM runner
# ---------------------------------------------------------------------------

def run_abm_series(
    steps: int,
    seed: int,
    params: dict,
    pair: str,
    verbose: bool = False,
) -> np.ndarray:
    """
    Run one ABM episode using FXSentimentSimulation directly.
    All parameter injection uses direct module-attribute patching,
    mirroring old_regime_hierarchy_test.py. Environment variable
    injection is NOT used because agents.py reads env vars only at
    module import time, not per-call.
    """
    from research.abm.simulation import FXSentimentSimulation

    # --- resolve price data ---
    import config as cfg
    dataset_path = (
        cfg.OUTPUT_DIR / "1.6.1"
        / "master_research_dataset_reactive_jpy_v1_core.csv"
    )
    if not dataset_path.exists():
        dataset_path = cfg.OUTPUT_DIR / "1.6.1" / "master_research_dataset_core.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cannot locate dataset for ABM price series. Tried: {dataset_path}"
        )

    df_full = pd.read_csv(dataset_path, parse_dates=["entry_time"], low_memory=False)

    def _norm(s):
        return s.lower().replace("-", "").replace("/", "")

    df_full["_pn"] = df_full["pair"].apply(_norm)
    sub = (
        df_full[df_full["_pn"] == _norm(pair)]
        .drop(columns=["_pn"])
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if sub.empty:
        raise ValueError(f"No price data found for pair='{pair}' in {dataset_path}")

    price_series = sub["entry_close"].values
    timestamps   = sub["entry_time"].values

    # --- save all module constants that we will patch ---
    orig = {
        "_PERSISTENCE_WEIGHT":       agents_module._PERSISTENCE_WEIGHT,
        "_INERTIA_THRESHOLD":        agents_module._INERTIA_THRESHOLD,
        "_DECAY_BASE":               agents_module._DECAY_BASE,
        "_DECAY_VOLATILITY_SCALE":   agents_module._DECAY_VOLATILITY_SCALE,
        "_DECAY_CLIP_MAX":           agents_module._DECAY_CLIP_MAX,
        "_SWITCHING_ANCHOR_STRENGTH":agents_module._SWITCHING_ANCHOR_STRENGTH,
    }

    try:
        # --- patch all parameters directly (env vars are import-time only) ---
        agents_module._PERSISTENCE_WEIGHT        = float(params["persistence"])
        agents_module._INERTIA_THRESHOLD         = float(params["threshold"])
        agents_module._DECAY_BASE                = float(params["decay_base"])
        agents_module._DECAY_VOLATILITY_SCALE    = float(params["beta"])
        agents_module._DECAY_CLIP_MAX            = float(params["decay_clip_max"])
        agents_module._SWITCHING_ANCHOR_STRENGTH = float(params["anchor_strength"])

        rng = np.random.default_rng(seed)

        agent_list = []
        agent_list.extend(
            TrendFollower(rng, pair=pair, momentum_window=params["momentum_window"])
            for _ in range(params["n_trend"])
        )
        agent_list.extend(
            Contrarian(rng, pair=pair, momentum_window=params["momentum_window"])
            for _ in range(params["n_contrarian"])
        )
        agent_list.extend(
            NoiseTrader(rng, pair=pair)
            for _ in range(params["n_noise"])
        )

        sim = FXSentimentSimulation(agent_list, rng=rng)

        max_steps = len(price_series) - sim.warmup_steps - 1
        if max_steps <= 0:
            raise ValueError(
                f"Price series too short for pair='{pair}' "
                f"(need > {sim.warmup_steps + 1} rows, got {len(price_series)})"
            )
        effective_steps = min(steps, max_steps)

        if verbose:
            print(f"[ABM] pair={pair} seed={seed} agents={len(agent_list)} "
                  f"steps={effective_steps} (requested {steps})  "
                  f"anchor={agents_module._SWITCHING_ANCHOR_STRENGTH}  "
                  f"beta={agents_module._DECAY_VOLATILITY_SCALE}")

        sim_df = sim.run(
            n_steps      = effective_steps,
            price_series = price_series,
            timestamps   = timestamps,
        )

    finally:
        # Always restore — even if sim.run() raises
        for attr, val in orig.items():
            setattr(agents_module, attr, val)

    sentiment = sim_df["net_sentiment"].values

    # Sanity check — warn if the series is constant (degenerate run)
    if np.std(sentiment) < 1e-6:
        warnings.warn(
            f"[ABM] seed={seed} pair={pair}: net_sentiment is constant "
            f"(std={np.std(sentiment):.2e}). "
            "Check parameter patch — anchor or decay may not have applied.",
            stacklevel=2,
        )

    return sentiment


# ---------------------------------------------------------------------------
# BSVE helpers
# ---------------------------------------------------------------------------

def load_bsve_dataset(csv_path: str, pair: str) -> pd.DataFrame:
    """Load and filter the BSVE augmented dataset for one currency pair."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"BSVE dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["entry_time"], low_memory=False)

    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("/", "").replace("_", "")

    pair_norm = _norm(pair)
    df["_pair_norm"] = df["pair"].apply(_norm)
    df_pair = df[df["_pair_norm"] == pair_norm].copy().drop(columns=["_pair_norm"])

    if df_pair.empty:
        available = sorted(df["pair"].unique().tolist())
        raise ValueError(
            f"No rows for pair='{pair}' after normalisation ('{pair_norm}'). "
            f"Available: {available}"
        )

    required = {"state_id", "net_sentiment", "ret_24b", "episode_id",
                "maturity_bars", "entry_time"}
    missing = required - set(df_pair.columns)
    if missing:
        raise ValueError(f"BSVE dataset missing required columns: {missing}")

    # --- Map ontology state labels to canonical short labels ---
    # Rows not in STATE_ID_MAP (e.g. JPY_NON_EXTREME) are non-episode
    # background rows and are excluded from H4 analysis.
    df_pair["state_id"] = df_pair["state_id"].map(STATE_ID_MAP)
    df_pair = df_pair[df_pair["state_id"].notna()].copy()

    if df_pair.empty:
        raise ValueError(
            f"No episode rows remain for pair='{pair}' after state_id mapping. "
            f"Check STATE_ID_MAP against dataset values."
        )

    # Validate only canonical labels remain
    bad_states = set(df_pair["state_id"].unique()) - set(BSVE_STATES)
    if bad_states:
        warnings.warn(f"Unexpected state_id values after mapping: {bad_states}")

    df_pair = df_pair.sort_values("entry_time").reset_index(drop=True)
    return df_pair


def align_abm_to_bsve(
    abm_sentiment: np.ndarray,
    bsve_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Align the ABM synthetic sentiment series to BSVE row indices.

    The ABM produces a structural surrogate — we test distributional
    properties per lifecycle state, not point-in-time prediction.
    Alignment is by row index; mod-wrapping if ABM is shorter than BSVE.
    """
    n_abm  = len(abm_sentiment)
    n_bsve = len(bsve_df)

    if verbose:
        print(f"[align] ABM steps={n_abm}, BSVE rows={n_bsve}")

    if n_abm < n_bsve:
        warnings.warn(
            f"ABM series ({n_abm}) shorter than BSVE rows ({n_bsve}). "
            "Using mod-wrap. Consider --steps >= BSVE row count.",
            stacklevel=2,
        )
        indices = np.arange(n_bsve) % n_abm
    else:
        indices = np.arange(n_bsve)

    aligned = bsve_df.copy()
    aligned["abm_net_sentiment"] = abm_sentiment[indices]
    return aligned


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_state_correlations(
    aligned_df: pd.DataFrame,
    forward_col: str = "ret_24b",
    sentiment_col: str = "abm_net_sentiment",
    verbose: bool = False,
) -> Dict[str, dict]:
    """
    Pearson + Spearman correlation of ABM net_sentiment vs forward returns,
    computed separately for each BSVE lifecycle state.
    """
    results: Dict[str, dict] = {}

    for state in BSVE_STATES:
        subset = (
            aligned_df[aligned_df["state_id"] == state]
            .dropna(subset=[sentiment_col, forward_col])
        )
        n = len(subset)

        if n < 5:
            warnings.warn(
                f"State '{state}': only {n} usable rows — "
                "correlations unreliable, reporting NaN."
            )
            results[state] = {
                "n": n,
                "pearson_r":  np.nan, "pearson_p":  np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
            }
            continue

        x = subset[sentiment_col].values
        y = subset[forward_col].values

        pearson_r,  pearson_p  = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        results[state] = {
            "n":          n,
            "pearson_r":  float(pearson_r),
            "pearson_p":  float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

        if verbose:
            sig_p = "**" if pearson_p  < 0.05 else "  "
            sig_s = "**" if spearman_p < 0.05 else "  "
            print(
                f"  [{state:10s}] n={n:5d}  "
                f"Pearson r={pearson_r:+.4f}{sig_p}(p={pearson_p:.3f})  "
                f"Spearman r={spearman_r:+.4f}{sig_s}(p={spearman_p:.3f})"
            )

    return results


def test_h4_hypothesis(
    state_corrs: Dict[str, dict],
    metric: str = "spearman_r",
    min_n_reliable: int = 100,
) -> dict:
    """
    Test H4: |corr(MATURING)| > |corr(ENTRY)| > |corr(MATURE)|.
    Verdicts are flagged CAUTIOUS when any state has n < min_n_reliable.
    """
    vals = {
        s: state_corrs[s].get(metric, math.nan)
        for s in H4_ORDER
    }
    ns = {s: state_corrs[s].get("n", 0) for s in H4_ORDER}

    if sum(math.isnan(v) for v in vals.values()) > 0:
        missing = [s for s, v in vals.items() if math.isnan(v)]
        return {
            "h4_supported":         None,
            "h4_partial_supported": None,
            "low_n_states":         [],
            "cautious":             False,
            "reason":               f"NaN correlation for states: {missing}",
            "metric":               metric,
            **{s: vals[s] for s in H4_ORDER},
        }

    maturing = vals["MATURING"]
    entry    = vals["ENTRY"]
    mature   = vals["MATURE"]

    abs_m = abs(maturing)
    abs_e = abs(entry)
    abs_t = abs(mature)

    h4_full    = abs_m > abs_e > abs_t
    h4_partial = abs_m > abs_t

    ranked = sorted(H4_ORDER, key=lambda s: abs(vals[s]), reverse=True)

    # Flag states with unreliable n
    low_n_states = [s for s in H4_ORDER if ns[s] < min_n_reliable]
    cautious = len(low_n_states) > 0

    return {
        "h4_supported":         h4_full,
        "h4_partial_supported": h4_partial,
        "cautious":             cautious,
        "low_n_states":         low_n_states,
        "metric":               metric,
        "MATURING":             maturing,
        "ENTRY":                entry,
        "MATURE":               mature,
        "abs_MATURING":         abs_m,
        "abs_ENTRY":            abs_e,
        "abs_MATURE":           abs_t,
        "empirical_rank_order": ranked,
        "expected_rank_order":  H4_ORDER[:],
    }

# ---------------------------------------------------------------------------
# Multi-
# ---------------------------------------------------------------------------
# Stage 2 fallback: price-only regime classification (unchanged behaviour)
# ---------------------------------------------------------------------------

def classify_price_regimes(
    prices: np.ndarray,
    vol_window: int = 20,
    trend_window: int = 50,
) -> np.ndarray:
    """
    Reproduce the original Stage 2 LVTF/HVTF/LVR/HVR classification.
    Returns a string array of regime labels, length == len(prices).
    """
    n = len(prices)
    labels = np.full(n, "UNKNOWN", dtype=object)

    # Rolling volatility (std of log returns)
    log_ret = np.diff(np.log(np.where(prices > 0, prices, np.nan)))
    vol = np.full(n, np.nan)
    for i in range(vol_window, n):
        vol[i] = np.std(log_ret[i - vol_window: i])

    vol_median = np.nanmedian(vol)

    # Trend proxy: price vs rolling mean
    trend_ma = np.full(n, np.nan)
    for i in range(trend_window, n):
        trend_ma[i] = np.mean(prices[i - trend_window: i])

    for i in range(trend_window, n):
        if np.isnan(vol[i]):
            continue
        high_vol  = vol[i] > vol_median
        trending  = abs(prices[i] - trend_ma[i]) / trend_ma[i] > 0.005

        if high_vol and trending:
            labels[i] = "HVTF"
        elif high_vol and not trending:
            labels[i] = "HVR"
        elif not high_vol and trending:
            labels[i] = "LVTF"
        else:
            labels[i] = "LVR"

    return labels


def compute_price_regime_correlations(
    sentiment: np.ndarray,
    prices: np.ndarray,
    forward_horizon: int = 24,
    verbose: bool = False,
) -> Dict[str, dict]:
    """
    Stage 2 fallback: compute forward-return correlations per price regime.
    Forward returns are computed directly from the ABM price series.
    """
    n = len(prices)
    labels = classify_price_regimes(prices)

    # Forward returns
    fwd_ret = np.full(n, np.nan)
    for i in range(n - forward_horizon):
        if prices[i] > 0:
            fwd_ret[i] = (prices[i + forward_horizon] - prices[i]) / prices[i]

    results: Dict[str, dict] = {}
    for regime in PRICE_REGIMES:
        mask = (labels == regime) & ~np.isnan(fwd_ret) & ~np.isnan(sentiment)
        x = sentiment[mask]
        y = fwd_ret[mask]
        n_r = len(x)

        if n_r < 5:
            results[regime] = {
                "n": n_r,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
            }
            continue

        pearson_r,  pearson_p  = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        results[regime] = {
            "n":          n_r,
            "pearson_r":  float(pearson_r),
            "pearson_p":  float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

        if verbose:
            print(
                f"  [{regime:6s}] n={n_r:5d}  "
                f"Pearson r={pearson_r:+.4f}  Spearman r={spearman_r:+.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(
    run_results: List[Dict[str, dict]],
    states: Tuple[str, ...],
) -> Dict[str, dict]:
    """
    Average Pearson/Spearman correlations across independent runs.
    Also reports std-dev across runs as a stability indicator.
    """
    aggregated: Dict[str, dict] = {}

    for state in states:
        pearson_rs  = [r[state]["pearson_r"]  for r in run_results
                       if not math.isnan(r[state].get("pearson_r", math.nan))]
        spearman_rs = [r[state]["spearman_r"] for r in run_results
                       if not math.isnan(r[state].get("spearman_r", math.nan))]
        ns          = [r[state]["n"]          for r in run_results]

        aggregated[state] = {
            "n_mean":          float(np.mean(ns)),
            "pearson_r_mean":  float(np.mean(pearson_rs))  if pearson_rs  else math.nan,
            "pearson_r_std":   float(np.std(pearson_rs))   if pearson_rs  else math.nan,
            "spearman_r_mean": float(np.mean(spearman_rs)) if spearman_rs else math.nan,
            "spearman_r_std":  float(np.std(spearman_rs))  if spearman_rs else math.nan,
            "runs_used":       len(pearson_rs),
        }

    return aggregated


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_bsve_report(
    pair: str,
    aggregated: Dict[str, dict],
    h4_verdict: dict,
    calibration_meta: Optional[dict],
    runs: int,
    steps: int,
    forward_horizon: int,
) -> None:
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  Stage 3 — BSVE State Injection  |  {pair.upper()}")
    print(f"  Runs: {runs}  |  Steps/run: {steps}  |  Forward horizon: {forward_horizon}b")
    if calibration_meta:
        print(f"  Artifact: extreme_thresh={calibration_meta.get('extreme_threshold_net_pct')}  "
              f"young={calibration_meta.get('young_boundary_bars')}  "
              f"mature={calibration_meta.get('mature_boundary_bars')}")
    print(sep)
    print(f"  {'State':<12} {'n':>6}  {'Pearson r':>10}  {'±std':>7}  "
          f"{'Spearman r':>11}  {'±std':>7}")
    print(f"  {'-'*12} {'-'*6}  {'-'*10}  {'-'*7}  {'-'*11}  {'-'*7}")

    for state in BSVE_STATES:
        a = aggregated[state]
        pr  = f"{a['pearson_r_mean']:+.4f}"  if not math.isnan(a['pearson_r_mean'])  else "   NaN"
        ps  = f"{a['pearson_r_std']:.4f}"    if not math.isnan(a['pearson_r_std'])    else "   NaN"
        sr  = f"{a['spearman_r_mean']:+.4f}" if not math.isnan(a['spearman_r_mean']) else "   NaN"
        ss  = f"{a['spearman_r_std']:.4f}"   if not math.isnan(a['spearman_r_std'])  else "   NaN"
        print(f"  {state:<12} {int(a['n_mean']):>6}  {pr:>10}  {ps:>7}  {sr:>11}  {ss:>7}")

    print(sep)
    print(f"\n  H4 Verdict  ({h4_verdict['metric']})")
    print(f"  Expected rank : {' > '.join(H4_ORDER)}")
    print(f"  Empirical rank: {' > '.join(h4_verdict.get('empirical_rank_order', ['?', '?', '?']))}")

    supported = h4_verdict.get("h4_supported")
    partial   = h4_verdict.get("h4_partial_supported")

    if supported is None:
        print(f"  Result  : INCONCLUSIVE — {h4_verdict.get('reason', '')}")
    elif supported:
        print(f"  Result  : H4 SUPPORTED (full gradient confirmed)")
    elif partial:
        print(f"  Result  : H4 PARTIALLY SUPPORTED "
              f"(MATURING > MATURE, but ENTRY ordering not strict)")
    else:
        print(f"  Result  : H4 NOT SUPPORTED")
        
    if h4_verdict.get("cautious"):
        low = ", ".join(
            f"{s} (n={aggregated[s]['n_mean']:.0f})"
            for s in h4_verdict["low_n_states"]
        )
        print(f"  ⚠ CAUTIOUS: low-n states may inflate |corr| — {low}")

    print(f"\n  |corr| values — "
          f"MATURING={h4_verdict.get('abs_MATURING', math.nan):.4f}  "
          f"ENTRY={h4_verdict.get('abs_ENTRY', math.nan):.4f}  "
          f"MATURE={h4_verdict.get('abs_MATURE', math.nan):.4f}")
    print(sep + "\n")


def print_price_regime_report(
    pair: str,
    aggregated: Dict[str, dict],
    runs: int,
    steps: int,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Stage 2 (fallback) — Price Regime Hierarchy  |  {pair.upper()}")
    print(f"  Runs: {runs}  |  Steps/run: {steps}")
    print(sep)
    print(f"  {'Regime':<8} {'n':>6}  {'Pearson r':>10}  {'Spearman r':>11}")
    print(f"  {'-'*8} {'-'*6}  {'-'*10}  {'-'*11}")

    for regime in PRICE_REGIMES:
        a = aggregated[regime]
        pr = f"{a['pearson_r_mean']:+.4f}"  if not math.isnan(a['pearson_r_mean'])  else "   NaN"
        sr = f"{a['spearman_r_mean']:+.4f}" if not math.isnan(a['spearman_r_mean']) else "   NaN"
        print(f"  {regime:<8} {int(a['n_mean']):>6}  {pr:>10}  {sr:>11}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Override calibrated params from CLI where provided
    params = CALIBRATED_PARAMS.copy()
    params["beta"]            = args.beta
    params["anchor_strength"] = args.anchor_strength

    # Optional calibration artifact (metadata echo only)
    calibration_meta: Optional[dict] = None
    if args.calibration_artifact:
        art_path = Path(args.calibration_artifact)
        if art_path.exists():
            with open(art_path) as f:
                calibration_meta = json.load(f)
        else:
            warnings.warn(f"Calibration artifact not found: {art_path}")

    # Resolve forward-return column name from horizon
    forward_col = f"ret_{args.forward_horizon}b"

    # ------------------------------------------------------------------
    # BSVE STATE INJECTION PATH
    # ------------------------------------------------------------------
    if args.use_bsve_states:
        if not args.bsve_states_path:
            parser.error("--bsve-states-path is required when --use-bsve-states is set")

        print(f"\n[Stage 3] Loading BSVE dataset for pair={args.pair} ...")
        bsve_df = load_bsve_dataset(args.bsve_states_path, args.pair)

        # Validate forward-return column exists
        if forward_col not in bsve_df.columns:
            available_ret = [c for c in bsve_df.columns if c.startswith("ret_")]
            raise ValueError(
                f"Column '{forward_col}' not found in BSVE dataset. "
                f"Available return columns: {available_ret}"
            )

        if args.verbose:
            state_counts = bsve_df["state_id"].value_counts().to_dict()
            print(f"[Stage 3] BSVE rows: {len(bsve_df)}  state counts: {state_counts}")

        # Run ABM multiple times, collect per-run correlations
        run_results: List[Dict[str, dict]] = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            if args.verbose:
                print(f"\n[Run {run_idx + 1}/{args.runs}] seed={seed}")

            abm_sentiment = run_abm_series(
                steps   = args.steps,
                seed    = seed,
                params  = params,
                pair    = args.pair,
                verbose = args.verbose,
            )

            aligned_df = align_abm_to_bsve(abm_sentiment, bsve_df, verbose=args.verbose)

            run_corrs = compute_state_correlations(
                aligned_df,
                forward_col   = forward_col,
                sentiment_col = "abm_net_sentiment",
                verbose       = args.verbose,
            )
            run_results.append(run_corrs)

        # Aggregate across runs
        aggregated = aggregate_runs(run_results, states=BSVE_STATES)

        # Build a flat corr dict from means for H4 test
        mean_corrs = {
            state: {
                "spearman_r": aggregated[state]["spearman_r_mean"],
                "pearson_r":  aggregated[state]["pearson_r_mean"],
                "n":          int(aggregated[state]["n_mean"]),
                "pearson_p":  math.nan,   # p-values not averaged (use run-level)
                "spearman_p": math.nan,
            }
            for state in BSVE_STATES
        }

        h4_verdict = test_h4_hypothesis(mean_corrs, metric="spearman_r")

        print_bsve_report(
            pair              = args.pair,
            aggregated        = aggregated,
            h4_verdict        = h4_verdict,
            calibration_meta  = calibration_meta,
            runs              = args.runs,
            steps             = args.steps,
            forward_horizon   = args.forward_horizon,
        )

        result_payload = {
            "mode":            "bsve_state_injection",
            "pair":            args.pair,
            "runs":            args.runs,
            "steps":           args.steps,
            "forward_horizon": args.forward_horizon,
            "forward_col":     forward_col,
            "anchor_strength": params["anchor_strength"],
            "beta":            params["beta"],
            "aggregated":      aggregated,
            "h4_verdict":      h4_verdict,
            "calibration_meta": calibration_meta,
        }

    # ------------------------------------------------------------------
    # STAGE 2 FALLBACK PATH (price-only regime classification)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STAGE 2 FALLBACK PATH (price-only regime classification)
    # ------------------------------------------------------------------
    else:
        if args.verbose:
            print(f"\n[Stage 2 fallback] Price-regime hierarchy for pair={args.pair}")

        # Load price data once — same source as run_abm_series
        import config as cfg
        dataset_path = (
            cfg.OUTPUT_DIR
            / "1.6.1"
            / "master_research_dataset_reactive_jpy_v1_core.csv"
        )
        if not dataset_path.exists():
            dataset_path = cfg.OUTPUT_DIR / "1.6.1" / "master_research_dataset_core.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Cannot locate dataset for price series. Tried: {dataset_path}"
            )

        df_full = pd.read_csv(dataset_path, parse_dates=["entry_time"])

        def _norm_pair(s: str) -> str:
            return s.lower().replace("-", "").replace("/", "")

        pair_norm = _norm_pair(args.pair)
        df_full["_pn"] = df_full["pair"].apply(_norm_pair)
        sub = (
            df_full[df_full["_pn"] == pair_norm]
            .drop(columns=["_pn"])
            .sort_values("entry_time")
            .reset_index(drop=True)
        )

        if sub.empty:
            raise ValueError(
                f"No price data found for pair='{args.pair}' in {dataset_path}"
            )

        price_series_full = sub["entry_close"].values

        run_results: List[Dict[str, dict]] = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            if args.verbose:
                print(f"\n[Run {run_idx + 1}/{args.runs}] seed={seed}")

            abm_sentiment = run_abm_series(
                steps   = args.steps,
                seed    = seed,
                params  = params,
                pair    = args.pair,
                verbose = args.verbose,
            )

            # Align price series length to sentiment series length.
            # run_abm_series returns effective_steps rows (may be < args.steps
            # if price data is the binding constraint). Take the matching window
            # starting after warmup, mirroring FXSentimentSimulation.run().
            from research.abm.simulation import FXSentimentSimulation
            _warmup = FXSentimentSimulation.__init__.__defaults__  # (48,) for warmup_steps
            warmup = 48  # matches FXSentimentSimulation default
            n_sent = len(abm_sentiment)
            price_window = price_series_full[warmup + 1 : warmup + 1 + n_sent]

            # Pad or trim to match sentiment length (safety guard)
            if len(price_window) < n_sent:
                price_window = np.pad(
                    price_window,
                    (0, n_sent - len(price_window)),
                    mode="edge",
                )
            else:
                price_window = price_window[:n_sent]

            run_corrs = compute_price_regime_correlations(
                sentiment       = abm_sentiment,
                prices          = price_window,
                forward_horizon = args.forward_horizon,
                verbose         = args.verbose,
            )
            run_results.append(run_corrs)

        aggregated = aggregate_runs(run_results, states=PRICE_REGIMES)

        print_price_regime_report(
            pair       = args.pair,
            aggregated = aggregated,
            runs       = args.runs,
            steps      = args.steps,
        )

        result_payload = {
            "mode":            "price_regime_fallback",
            "pair":            args.pair,
            "runs":            args.runs,
            "steps":           args.steps,
            "forward_horizon": args.forward_horizon,
            "anchor_strength": params["anchor_strength"],
            "beta":            params["beta"],
            "aggregated":      aggregated,
        }

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_payload, f, indent=2, default=str)
        print(f"[output] Results written to {out_path}")


if __name__ == "__main__":
    main()