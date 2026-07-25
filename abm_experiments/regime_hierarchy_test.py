"""abm_experiments/regime_hierarchy_test.py

Experiment 1: Can ABM reproduce the DL regime hierarchy structurally?

Tests whether the ABM -- with no explicit predictive mechanism -- produces
a stronger sentiment -> forward-return relationship in "trend + low-vol"
(LVTF) conditions than in "trend + high-vol" (HVTF), and how "range" regimes
(LVR, HVR) compare.

IMPORTANT: regime classification (trend vs. range, low-vol vs. high-vol) is
computed ONLY from the exogenous price series, never from agent/sentiment
state. This avoids circularity (using the model's own saturation as the
definition of "trend").

Fixed ABM configuration mirrors decay_beta_sensitivity.py (USDJPY unlock
regime) so results are comparable across the two harnesses. Anchor strength
and disagree-hold-prob are exposed as optional overrides so this can be run
in both the "locked" (default) and "unlocked" (reduced anchor) regimes.

Constraints
-----------
- Single file
- No refactors / no shared utilities
- No changes to existing pipeline
- One (beta, anchor, hold) combination per invocation

Output (verbose)
-----------------
pair | seed | beta | anchor | hold |
LVTF_corr_fwd | HVTF_corr_fwd | LVR_corr_fwd | HVR_corr_fwd |
LVTF_autocorr | HVTF_autocorr | LVR_autocorr | HVR_autocorr |
LVTF_n | HVTF_n | LVR_n | HVR_n |
overall_autocorr | sign_flips | pct_saturated

Output (default, non-verbose)
------------------------------
beta | LVTF | HVTF | LVR | HVR | overall_autocorr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Match sweep.py / decay_beta_sensitivity.py behavior for repo-root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.abm import agents as agents_module
from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.simulation import FXSentimentSimulation


# ---------------------------------------------------------------------------
# Regime classification (PRICE-ONLY -- never uses agent/sentiment state)
# ---------------------------------------------------------------------------

_VOL_LOW_PCTILE = 33.0
_VOL_HIGH_PCTILE = 67.0
_TREND_LOW_PCTILE = 33.0
_TREND_HIGH_PCTILE = 67.0

_VOL_WINDOW = 24        # bars, matches simulation.py _VOL_WINDOW
_TREND_WINDOW = 24      # bars, lookback for efficiency ratio


def _rolling_realized_vol(price: np.ndarray, window: int) -> np.ndarray:
    """Rolling std of returns, price-only, backward-looking."""
    returns = np.diff(price)
    returns = np.concatenate([[0.0], returns])  # align length with price
    vol = np.zeros_like(price)
    for i in range(len(price)):
        start = max(0, i - window + 1)
        seg = returns[start:i + 1]
        vol[i] = float(np.std(seg)) if len(seg) > 1 else 0.0
    return vol


def _rolling_efficiency_ratio(price: np.ndarray, window: int) -> np.ndarray:
    """Kaufman-style efficiency ratio: |net displacement| / sum(|abs moves|).

    Bounded [0, 1]. High = clean directional trend. Low = choppy/range.
    Price-only, backward-looking.
    """
    er = np.zeros_like(price)
    for i in range(len(price)):
        start = max(0, i - window)
        seg = price[start:i + 1]
        if len(seg) < 2:
            er[i] = 0.0
            continue
        net_move = abs(seg[-1] - seg[0])
        path_len = float(np.sum(np.abs(np.diff(seg))))
        er[i] = net_move / (path_len + 1e-12)
    return er


def _classify_regimes(price: np.ndarray) -> dict[str, np.ndarray]:
    """Classify each step into LVTF / HVTF / LVR / HVR using price only.

    Trend/range split: efficiency ratio terciles.
    Vol split: rolling realized vol terciles.
    Middle tercile on either axis is excluded from all four buckets (kept
    out to avoid diluting the hierarchy test with ambiguous regime steps).
    """
    vol = _rolling_realized_vol(price, _VOL_WINDOW)
    er = _rolling_efficiency_ratio(price, _TREND_WINDOW)

    vol_low_thresh = np.percentile(vol, _VOL_LOW_PCTILE)
    vol_high_thresh = np.percentile(vol, _VOL_HIGH_PCTILE)
    er_low_thresh = np.percentile(er, _TREND_LOW_PCTILE)
    er_high_thresh = np.percentile(er, _TREND_HIGH_PCTILE)

    is_low_vol = vol <= vol_low_thresh
    is_high_vol = vol >= vol_high_thresh
    is_range = er <= er_low_thresh      # choppy / low efficiency
    is_trend = er >= er_high_thresh     # clean directional / high efficiency

    return {
        "LVTF": is_low_vol & is_trend,
        "HVTF": is_high_vol & is_trend,
        "LVR": is_low_vol & is_range,
        "HVR": is_high_vol & is_range,
    }


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------

def _regime_forward_correlation(
    sentiment: np.ndarray,
    price: np.ndarray,
    mask: np.ndarray,
    forward_steps: int = 1,
    min_n: int = 15,
) -> float:
    """sentiment[t] vs cumulative forward return over [t, t+forward_steps]."""
    returns = np.diff(price)
    if forward_steps == 1:
        fwd_ret = returns
        aligned_sent = sentiment[:-1]
        aligned_mask = mask[:-1]
    else:
        n_fwd = len(returns) - forward_steps + 1
        if n_fwd <= 0:
            return float("nan")
        fwd_ret = np.array(
            [np.sum(returns[i:i + forward_steps]) for i in range(n_fwd)]
        )
        aligned_sent = sentiment[:n_fwd]
        aligned_mask = mask[:n_fwd]

    rs = aligned_sent[aligned_mask]
    rr = fwd_ret[aligned_mask]

    if len(rs) < min_n or np.std(rs) == 0 or np.std(rr) == 0:
        return float("nan")
    return float(np.corrcoef(rs, rr)[0, 1])


def _regime_autocorr(sentiment: np.ndarray, mask: np.ndarray, min_n: int = 15) -> float:
    rs = sentiment[mask]
    if len(rs) < min_n:
        return float("nan")
    a, b = rs[:-1], rs[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test whether ABM reproduces the DL regime hierarchy "
                     "(LVTF > HVR > LVR > HVTF) using price-only regime "
                     "classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        default="1.6.1",
        help="Dataset version (e.g. '1.6.1')",
    )
    p.add_argument("--pair", required=True, help="FX pair (e.g. 'eur-usd')")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--beta", type=float, default=0.0, help="decay_volatility_scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--forward-steps",
        type=int,
        default=1,
        help="Bars ahead for forward-return correlation",
    )
    p.add_argument(
        "--anchor-strength",
        type=float,
        default=None,
        help="Override ABM_ANCHOR_STRENGTH (default: module default, 2.0)",
    )
    p.add_argument(
        "--disagree-hold",
        type=float,
        default=None,
        help="Override ABM_DISAGREE_HOLD_PROB (default: module default, 0.7)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Fixed configuration (mirrors decay_beta_sensitivity.py USDJPY unlock regime,
    # so results are directly comparable across harnesses).
    persistence = 0.10
    threshold = 0.05
    momentum_window = 3
    n_trend = 50
    n_contrarian = 50
    n_noise = 0

    decay_base = 0.0
    decay_clip_max = 0.5

    orig_persistence = agents_module._PERSISTENCE_WEIGHT
    orig_threshold = agents_module._INERTIA_THRESHOLD
    orig_decay_base = getattr(agents_module, "_DECAY_BASE", 0.0)
    orig_decay_vol_scale = getattr(agents_module, "_DECAY_VOLATILITY_SCALE", 0.0)
    orig_decay_clip_max = getattr(agents_module, "_DECAY_CLIP_MAX", 0.2)
    orig_anchor = getattr(agents_module, "_SWITCHING_ANCHOR_STRENGTH", 2.0)
    orig_hold = getattr(agents_module, "_DISAGREE_HOLD_PROB", 0.7)

    try:
        agents_module._PERSISTENCE_WEIGHT = float(persistence)
        agents_module._INERTIA_THRESHOLD = float(threshold)
        agents_module._DECAY_BASE = float(decay_base)
        agents_module._DECAY_VOLATILITY_SCALE = float(args.beta)
        agents_module._DECAY_CLIP_MAX = float(decay_clip_max)

        if args.anchor_strength is not None:
            agents_module._SWITCHING_ANCHOR_STRENGTH = float(args.anchor_strength)
        if args.disagree_hold is not None:
            agents_module._DISAGREE_HOLD_PROB = float(np.clip(args.disagree_hold, 0.0, 1.0))

        if args.verbose:
            print(
                "[regime_hierarchy_test] knobs:",
                f"beta={agents_module._DECAY_VOLATILITY_SCALE}",
                f"anchor={agents_module._SWITCHING_ANCHOR_STRENGTH}",
                f"hold={agents_module._DISAGREE_HOLD_PROB}",
                file=sys.stderr,
                flush=True,
            )

        df, _dataset_path = _load_real_data(args.version, variant="core")
        sub = df[df["pair"] == args.pair].copy().sort_values("entry_time")
        if sub.empty:
            raise ValueError(f"No data found for pair={args.pair}")

        price_series = sub["entry_close"].to_numpy(dtype=float)
        timestamps = sub["entry_time"].values

        rng = np.random.default_rng(args.seed)
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
        p = sim_df["price"].to_numpy(dtype=float)

        # --- Regime classification: PRICE ONLY, never sentiment/position state ---
        regimes = _classify_regimes(p)

        corr_fwd = {}
        autocorr_regime = {}
        n_regime = {}
        for name in ("LVTF", "HVTF", "LVR", "HVR"):
            mask = regimes[name]
            corr_fwd[name] = _regime_forward_correlation(
                s, p, mask, forward_steps=args.forward_steps
            )
            autocorr_regime[name] = _regime_autocorr(s, mask)
            n_regime[name] = int(np.sum(mask))

        # Overall diagnostics (comparable to decay_beta_sensitivity.py output)
        overall_autocorr = _regime_autocorr(s, np.ones_like(s, dtype=bool))
        sign = np.sign(s)
        sign_flips = int(((sign[1:] * sign[:-1]) < 0).sum()) if len(sign) > 1 else 0
        pct_saturated = float((np.abs(s) >= 90.0).mean())

        if args.verbose:
            anchor_val = agents_module._SWITCHING_ANCHOR_STRENGTH
            hold_val = agents_module._DISAGREE_HOLD_PROB
            print(
                f"{args.pair} | {args.seed} | {float(args.beta):.6g} | "
                f"{anchor_val:.6g} | {hold_val:.6g} | "
                f"{corr_fwd['LVTF']:.6g} | {corr_fwd['HVTF']:.6g} | "
                f"{corr_fwd['LVR']:.6g} | {corr_fwd['HVR']:.6g} | "
                f"{autocorr_regime['LVTF']:.6g} | {autocorr_regime['HVTF']:.6g} | "
                f"{autocorr_regime['LVR']:.6g} | {autocorr_regime['HVR']:.6g} | "
                f"{n_regime['LVTF']} | {n_regime['HVTF']} | "
                f"{n_regime['LVR']} | {n_regime['HVR']} | "
                f"{overall_autocorr:.6g} | {sign_flips} | {pct_saturated:.6g}"
            )
        else:
            print(
                f"{float(args.beta):.6g} | "
                f"LVTF={corr_fwd['LVTF']:.4g} | HVTF={corr_fwd['HVTF']:.4g} | "
                f"LVR={corr_fwd['LVR']:.4g} | HVR={corr_fwd['HVR']:.4g} | "
                f"autocorr={overall_autocorr:.6g}"
            )

    finally:
        agents_module._PERSISTENCE_WEIGHT = orig_persistence
        agents_module._INERTIA_THRESHOLD = orig_threshold
        agents_module._DECAY_BASE = orig_decay_base
        agents_module._DECAY_VOLATILITY_SCALE = orig_decay_vol_scale
        agents_module._DECAY_CLIP_MAX = orig_decay_clip_max
        if args.anchor_strength is not None:
            agents_module._SWITCHING_ANCHOR_STRENGTH = orig_anchor
        if args.disagree_hold is not None:
            agents_module._DISAGREE_HOLD_PROB = orig_hold

if __name__ == "__main__":
    main()