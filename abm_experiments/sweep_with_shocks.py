"""
sweep_with_shocks.py — Stage 3: Shock-Driven Episode Formation
==============================================================
Tests hypothesis H3: exogenous crowd-alignment shocks improve episode
formation rate and speed to better match empirical BSVE diagnostics.

Tests hypothesis H4: the shock mechanism, combined with persistence +
decay, produces the MATURING > ENTRY > MATURE forward-return correlation
gradient that persistence + decay alone cannot reproduce.

Design:
  - Subclasses FXSentimentSimulation to inject shocks at the step level
  - No modifications to research/abm/ (single-file constraint)
  - Reuses correlation logic from regime_hierarchy_test.py pattern
  - Reports both episode structure metrics (H3) and H4 gradient in one run

Shock mechanism:
  - Trigger: volatility-based (EMA vol > percentile threshold) or periodic
  - Effect: fraction of agents pushed toward recent price direction
  - Cooldown: minimum bars between shock events

Constraints:
  - Single file, no modifications to research/abm/
  - Calibrated parameter point: anchor=0.25, beta=0.02
  - Dataset version: 1.6.1
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
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from research.abm.agents import TrendFollower, Contrarian, NoiseTrader
    import research.abm.agents as agents_module
    import config as cfg
    from research.abm.simulation import FXSentimentSimulation
    from research.abm.simulation import _VOL_WINDOW          # ADD THIS LINE
except ImportError as exc:
    raise ImportError(
        f"Cannot import from research/abm. Check REPO_ROOT={REPO_ROOT}\n{exc}"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BSVE_STATES = ("ENTRY", "MATURING", "MATURE")
H4_ORDER    = ["MATURING", "ENTRY", "MATURE"]

STATE_ID_MAP = {
    "JPY_CONSENSUS_YOUNG":    "ENTRY",
    "JPY_CONSENSUS_MATURING": "MATURING",
    "JPY_CONSENSUS_MATURE":   "MATURE",
    "ENTRY":    "ENTRY",
    "MATURING": "MATURING",
    "MATURE":   "MATURE",
}

# Calibrated parameter point (locked 2026-07-25)
CALIBRATED_PARAMS = dict(
    anchor_strength = 0.25,
    beta            = 0.02,
    decay_base      = 0.00,
    decay_clip_max  = 0.50,
    n_trend         = 50,
    n_contrarian    = 50,
    n_noise         = 0,
    momentum_window = 3,
    persistence     = 0.10,
    threshold       = 0.05,
)

# Shock defaults
SHOCK_DEFAULTS = dict(
    shock_enable        = False,
    shock_trigger       = "volatility",   # "volatility" | "periodic"
    shock_vol_threshold = 0.80,           # EMA vol percentile
    shock_fraction      = 0.30,           # fraction of agents pushed
    shock_direction     = "price",        # "price" | "random"
    shock_cooldown      = 20,             # min bars between shocks
    shock_period        = 50,             # bars between periodic shocks
)


# ---------------------------------------------------------------------------
# Shock-aware simulation subclass
# ---------------------------------------------------------------------------

class ShockSimulation(FXSentimentSimulation):
    """
    Extends FXSentimentSimulation with a crowd-alignment shock mechanism.

    Shocks are injected at the step level by overriding run(). All other
    simulation logic is inherited unchanged from research/abm/simulation.py.

    Shock effect: a fraction of agents have their position directly set
    toward the recent price direction, bypassing their normal update logic
    for that step. Normal dynamics resume on the next step — persistence
    and anchoring then sustain the newly-formed consensus.

    This is the minimal intervention: we push agents, then let the
    existing mechanism do the rest.
    """

    def __init__(
        self,
        agents: list,
        rng: np.random.Generator,
        # Shock parameters
        shock_enable:        bool  = False,
        shock_trigger:       str   = "volatility",
        shock_vol_threshold: float = 0.80,
        shock_fraction:      float = 0.30,
        shock_direction:     str   = "price",
        shock_cooldown:      int   = 20,
        shock_period:        int   = 50,
    ):
        super().__init__(agents, rng=rng)
        self.shock_enable        = shock_enable
        self.shock_trigger       = shock_trigger
        self.shock_vol_threshold = shock_vol_threshold
        self.shock_fraction      = shock_fraction
        self.shock_direction     = shock_direction
        self.shock_cooldown      = shock_cooldown
        self.shock_period        = shock_period

        # Runtime state
        self._shock_cooldown_remaining = 0
        self._shock_events: List[int] = []   # step indices where shocks fired

    def _should_shock_volatility(
        self,
        step: int,
        vol_history: np.ndarray,
        vol_percentile_threshold: float,
    ) -> bool:
        """Fire on volatility spike: current EMA vol > threshold percentile
        of all vol values seen so far."""
        if self._shock_cooldown_remaining > 0:
            return False
        if len(vol_history) < 10:
            return False
        threshold = float(np.percentile(vol_history, vol_percentile_threshold * 100))
        return float(vol_history[-1]) > threshold

    def _should_shock_periodic(self, step: int) -> bool:
        """Fire every shock_period bars after warmup, subject to cooldown."""
        if self._shock_cooldown_remaining > 0:
            return False
        return (step > 0) and (step % self.shock_period == 0)

    def _apply_shock(
        self,
        agents: list,
        price_direction: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Push shock_fraction of agents toward the shock direction.

        Mechanism: set agent.position to +1 or -1 (full conviction) for
        the chosen agents. The existing persistence/anchoring/decay logic
        then operates on these positions from the next step onward,
        naturally sustaining or dissolving the shock-induced consensus.

        Price direction convention: +1 = price rising (push long),
        -1 = price falling (push short).
        """
        n_agents = len(agents)
        n_shock  = max(1, int(n_agents * self.shock_fraction))

        # Which agents are shocked
        shocked_indices = rng.choice(n_agents, size=n_shock, replace=False)

        # Direction
        if self.shock_direction == "price":
            direction = 1.0 if price_direction >= 0 else -1.0
        else:
            # random: flip a coin per shock event (all shocked agents same side)
            direction = 1.0 if rng.random() > 0.5 else -1.0

        for idx in shocked_indices:
            # Set position directly — agents.py stores position as float
            agents[idx].position = direction

        self._shock_cooldown_remaining = self.shock_cooldown
        return shocked_indices

    def run(
        self,
        n_steps: int,
        price_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """
        Override run() to inject shocks at the step level.

        If shock_enable is False, delegates entirely to the parent
        implementation — zero overhead, identical output.
        """
        if not self.shock_enable:
            return super().run(
                n_steps      = n_steps,
                price_series = price_series,
                timestamps   = timestamps,
            )

        # --- shock-enabled path ---
        # We replicate the parent's step loop with shock injection.
        # This relies on the parent exposing _step() or equivalent.
        # If FXSentimentSimulation does not expose a per-step method,
        # we call the full parent run() and post-process.
        #
        # Preferred: call super().run() but intercept per-step state.
        # Since we can't easily hook into the parent loop without
        # modifying simulation.py, we use a pre-run vol calibration pass
        # to identify shock steps, then inject by patching agent positions
        # between a warmup run and the main run.
        #
        # Strategy:
        #   1. Run parent once (warmup pass) to collect vol_ema series
        #   2. Identify shock steps from vol series
        #   3. Run step-by-step using parent's internal machinery,
        #      applying position patches at shock steps
        #
        # If parent does not expose step-level control, fall back to
        # the position-seeding approach (pre-set positions before run,
        # let dynamics evolve). This is documented below.

        return self._run_with_shocks(n_steps, price_series, timestamps)

    def _run_with_shocks(
        self,
        n_steps: int,
        price_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """
        Step-level shock injection.

        Attempts to use FXSentimentSimulation._run_step() if available.
        Falls back to the pre-calibration + position-seeding approach
        if the parent does not expose per-step control.
        """
        # Check if parent exposes a per-step method
        has_step_method = hasattr(self, '_run_step') or hasattr(self, 'step')

        if has_step_method:
            return self._run_with_shocks_step_method(
                n_steps, price_series, timestamps
            )
        else:
            return self._run_with_shocks_intercept(
                n_steps, price_series, timestamps
            )

    def _run_with_shocks_step_method(
        self,
        n_steps: int,
        price_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """
        Step-by-step shock injection using parent._run_step() if available.
        """
        step_fn = getattr(self, '_run_step', None) or getattr(self, 'step', None)
        records = []
        vol_history = []

        warmup = self.warmup_steps
        self._shock_cooldown_remaining = 0
        self._shock_events = []

        for i in range(warmup + n_steps):
            price_idx = i + 1
            if price_idx >= len(price_series):
                break

            price_prev = price_series[price_idx - 1]
            price_curr = price_series[price_idx]
            ret        = price_curr - price_prev

            # Update vol EMA (mirrors simulation.py logic)
            if not vol_history:
                vol_ema = abs(ret)
            else:
                alpha   = 2.0 / (24 + 1)   # vol_window=24, matches simulation.py
                vol_ema = alpha * abs(ret) + (1 - alpha) * vol_history[-1]
            vol_history.append(vol_ema)

            # Check for shock (post-warmup only)
            if i >= warmup and self.shock_enable:
                fire = False
                if self.shock_trigger == "volatility":
                    fire = self._should_shock_volatility(
                        i, np.array(vol_history), self.shock_vol_threshold
                    )
                elif self.shock_trigger == "periodic":
                    fire = self._should_shock_periodic(i - warmup)

                if fire:
                    price_direction = price_curr - price_series[max(0, price_idx - 5):price_idx].mean()
                    self._apply_shock(self.agents, price_direction, self.rng)
                    self._shock_events.append(i - warmup)

            if self._shock_cooldown_remaining > 0:
                self._shock_cooldown_remaining -= 1

            # Run one parent step
            record = step_fn(
                price_prev = price_prev,
                price_curr = price_curr,
                timestamp  = timestamps[price_idx] if price_idx < len(timestamps) else None,
                vol_norm   = vol_ema / (np.mean(vol_history) + 1e-8),
            )
            if i >= warmup and record is not None:
                records.append(record)

        return pd.DataFrame(records)

    def _run_with_shocks_intercept(
        self,
        n_steps: int,
        price_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fallback shock injection when parent does not expose per-step control.

        Strategy: run a calibration pass (parent.run()) to extract the
        vol_ema series, identify shock steps, then run again with agent
        positions pre-seeded at shock steps.

        This works by saving/restoring agent state around each shock
        injection point. Since we cannot pause the parent run() mid-stream,
        we instead run the full simulation in segments:
          segment 1: steps 0 to first_shock-1  (parent.run())
          shock injection: set positions
          segment 2: steps first_shock to next_shock-1 (parent.run())
          ... and so on.

        Each segment uses the agent state left by the previous segment.
        This correctly propagates shock effects through persistence and
        anchoring dynamics.
        """
        # --- pass 1: calibration run to identify shock steps ---
        # Run full parent simulation to get vol_ema series
        calib_df = super().run(
            n_steps      = n_steps,
            price_series = price_series,
            timestamps   = timestamps,
        )

        # Extract vol series if available, else recompute from price
        if "vol_norm" in calib_df.columns:
            vol_series = calib_df["vol_norm"].values
        else:
            # Recompute EMA vol from price series
            alpha      = 2.0 / (24 + 1)
            warmup     = self.warmup_steps
            prices     = price_series[warmup + 1: warmup + 1 + len(calib_df)]
            vol_ema    = np.zeros(len(prices))
            for j in range(1, len(prices)):
                ret        = abs(prices[j] - prices[j-1])
                vol_ema[j] = alpha * ret + (1 - alpha) * vol_ema[j-1]
            baseline   = np.mean(vol_ema[vol_ema > 0]) if np.any(vol_ema > 0) else 1.0
            vol_series = vol_ema / (baseline + 1e-8)

        # Identify shock steps
        shock_steps = self._identify_shock_steps(vol_series, n_steps)

        if not shock_steps:
            # No shocks triggered — return calibration run result
            return calib_df

        self._shock_events = shock_steps

        # --- pass 2: segmented run with shock injection ---
        # Reset agents to the state after the calibration pass warmup.
        # We patch _warmup_steps=0 for all segment calls so the parent
        # does not re-run its own warmup — agents are already stabilised.
        # Each segment receives a price slice starting one bar before its
        # first recorded step so the parent's ret_t calculation is correct.
        self._reset_agents()

        all_frames:   list = []
        segment_start = 0
        orig_warmup   = self._warmup_steps

        try:
            self._warmup_steps = 0   # agents stabilised by calibration pass

            for shock_step in shock_steps + [n_steps]:
                seg_len = shock_step - segment_start
                if seg_len > 0:
                    # Absolute index into price_series for this segment.
                    # With _warmup_steps=0, parent needs: len >= 0 + seg_len + 1
                    # We provide seg_len + 1 prices starting one bar before
                    # segment_start (so the first ret_t is valid).
                    p_start   = orig_warmup + 1 + segment_start
                    seg_prices = price_series[p_start - 1: p_start + seg_len]
                    # seg_prices has length seg_len + 1 — exactly what parent
                    # needs for warmup=0: total_required = 0 + seg_len + 1

                    seg_ts = (
                        timestamps[p_start - 1: p_start + seg_len]
                        if timestamps is not None else None
                    )

                    seg_df = super().run(
                        n_steps      = seg_len,
                        price_series = seg_prices,
                        timestamps   = seg_ts,
                    )
                    all_frames.append(seg_df)

                # Apply shock (skip terminal sentinel)
                if shock_step < n_steps:
                    p_idx           = orig_warmup + 1 + shock_step
                    recent          = price_series[max(0, p_idx - 5): p_idx + 1]
                    price_direction = float(price_series[p_idx]) - float(recent.mean())
                    self._apply_shock(self._agents, price_direction, self._rng)

                segment_start = shock_step

        finally:
            self._warmup_steps = orig_warmup   # always restore

        if not all_frames:
            return calib_df

        result = pd.concat(all_frames, ignore_index=True)
        result["step"] = np.arange(len(result))
        return result

    def _identify_shock_steps(
            self,
            vol_series: np.ndarray,
            n_steps: int,
    ) -> List[int]:
        """
        Identify which steps should trigger a shock, given the vol series.
        Returns a sorted list of step indices (0-indexed within the
        post-warmup simulation window).
        Respects cooldown between events.
        """
        shock_steps = []
        cooldown_remaining = 0

        for step in range(n_steps):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            fire = False
            if self.shock_trigger == "volatility":
                if step >= 10:
                    threshold = np.percentile(
                        vol_series[:step], self.shock_vol_threshold * 100
                    )
                    fire = vol_series[step] > threshold
            elif self.shock_trigger == "periodic":
                fire = (step > 0) and (step % self.shock_period == 0)

            if fire:
                shock_steps.append(step)
                cooldown_remaining = self.shock_cooldown

        return shock_steps

    def _reset_agents(self) -> None:
        """Reset all agent positions to zero for a fresh segmented run."""
        for agent in self._agents:  # was: self.agents
            if hasattr(agent, "position"):
                agent.position = 0.0
            if hasattr(agent, "_momentum_buffer"):
                agent._momentum_buffer = []
            if hasattr(agent, "_price_history"):
                agent._price_history = []
            if hasattr(agent, "_accumulation"):
                agent._accumulation = 0.0
            if hasattr(agent, "_inertia_count"):
                agent._inertia_count = 0

    @property
    def shock_event_count(self) -> int:
        return len(self._shock_events)

    @property
    def shock_event_steps(self) -> List[int]:
        return list(self._shock_events)


# ---------------------------------------------------------------------------
# ABM runner (shock-aware)
# ---------------------------------------------------------------------------

def run_abm_series_with_shocks(
        steps: int,
        seed: int,
        params: dict,
        shock_params: dict,
        pair: str,
        verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Run one ABM episode with optional shock injection.
    Returns (net_sentiment array, n_shocks_fired).
    All parameter injection uses direct module-attribute patching.
    """
    # --- resolve price data ---
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

    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("/", "")

    df_full["_pn"] = df_full["pair"].apply(_norm)
    sub = (
        df_full[df_full["_pn"] == _norm(pair)]
        .drop(columns=["_pn"])
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if sub.empty:
        raise ValueError(f"No price data for pair='{pair}' in {dataset_path}")

    price_series = sub["entry_close"].values
    timestamps = sub["entry_time"].values

    # --- patch module parameters ---
    orig = {
        "_PERSISTENCE_WEIGHT": agents_module._PERSISTENCE_WEIGHT,
        "_INERTIA_THRESHOLD": agents_module._INERTIA_THRESHOLD,
        "_DECAY_BASE": agents_module._DECAY_BASE,
        "_DECAY_VOLATILITY_SCALE": agents_module._DECAY_VOLATILITY_SCALE,
        "_DECAY_CLIP_MAX": agents_module._DECAY_CLIP_MAX,
        "_SWITCHING_ANCHOR_STRENGTH": agents_module._SWITCHING_ANCHOR_STRENGTH,
    }

    try:
        agents_module._PERSISTENCE_WEIGHT = float(params["persistence"])
        agents_module._INERTIA_THRESHOLD = float(params["threshold"])
        agents_module._DECAY_BASE = float(params["decay_base"])
        agents_module._DECAY_VOLATILITY_SCALE = float(params["beta"])
        agents_module._DECAY_CLIP_MAX = float(params["decay_clip_max"])
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

        sim = ShockSimulation(
            agents=agent_list,
            rng=rng,
            shock_enable=shock_params["shock_enable"],
            shock_trigger=shock_params["shock_trigger"],
            shock_vol_threshold=shock_params["shock_vol_threshold"],
            shock_fraction=shock_params["shock_fraction"],
            shock_direction=shock_params["shock_direction"],
            shock_cooldown=shock_params["shock_cooldown"],
            shock_period=shock_params["shock_period"],
        )

        max_steps = len(price_series) - sim.warmup_steps - 1
        effective = min(steps, max_steps)

        if verbose:
            print(
                f"[ABM] pair={pair} seed={seed} agents={len(agent_list)} "
                f"steps={effective}  anchor={params['anchor_strength']}  "
                f"beta={params['beta']}  shock={shock_params['shock_enable']}  "
                f"trigger={shock_params['shock_trigger']}"
            )

        sim_df = sim.run(
            n_steps=effective,
            price_series=price_series,
            timestamps=timestamps,
        )
        n_shocks = sim.shock_event_count

    finally:
        for attr, val in orig.items():
            setattr(agents_module, attr, val)

    sentiment = sim_df["net_sentiment"].values

    if np.std(sentiment) < 1e-6:
        warnings.warn(
            f"[ABM] seed={seed} pair={pair}: net_sentiment is constant. "
            "Check parameter patch.",
            stacklevel=2,
        )

    if verbose and shock_params["shock_enable"]:
        print(f"[ABM] Shocks fired: {n_shocks}  "
              f"(steps: {sim.shock_event_steps[:10]}"
              f"{'...' if n_shocks > 10 else ''})")

    return sentiment, n_shocks


# ---------------------------------------------------------------------------
# BSVE helpers (mirrors regime_hierarchy_test.py)
# ---------------------------------------------------------------------------

def load_bsve_dataset(csv_path: str, pair: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"BSVE dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["entry_time"], low_memory=False)

    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("/", "").replace("_", "")

    pair_norm = _norm(pair)
    df["_pn"] = df["pair"].apply(_norm)
    df_pair = df[df["_pn"] == pair_norm].copy().drop(columns=["_pn"])

    if df_pair.empty:
        raise ValueError(
            f"No rows for pair='{pair}'. "
            f"Available: {sorted(df['pair'].unique().tolist())}"
        )

    required = {"state_id", "net_sentiment", "ret_24b", "entry_time"}
    missing = required - set(df_pair.columns)
    if missing:
        raise ValueError(f"BSVE dataset missing columns: {missing}")

    df_pair["state_id"] = df_pair["state_id"].map(STATE_ID_MAP)
    df_pair = df_pair[df_pair["state_id"].notna()].copy()
    df_pair = df_pair.sort_values("entry_time").reset_index(drop=True)
    return df_pair


def align_abm_to_bsve(
        abm_sentiment: np.ndarray,
        bsve_df: pd.DataFrame,
        verbose: bool = False,
) -> pd.DataFrame:
    n_abm = len(abm_sentiment)
    n_bsve = len(bsve_df)

    if verbose:
        print(f"[align] ABM steps={n_abm}, BSVE rows={n_bsve}")

    if n_abm < n_bsve:
        warnings.warn(
            f"ABM series ({n_abm}) shorter than BSVE rows ({n_bsve}). "
            "Using mod-wrap.",
            stacklevel=2,
        )
        indices = np.arange(n_bsve) % n_abm
    else:
        indices = np.arange(n_bsve)

    aligned = bsve_df.copy()
    aligned["abm_net_sentiment"] = abm_sentiment[indices]
    return aligned


# ---------------------------------------------------------------------------
# Correlation and H4 test (mirrors regime_hierarchy_test.py)
# ---------------------------------------------------------------------------

def compute_state_correlations(
        aligned_df: pd.DataFrame,
        forward_col: str = "ret_24b",
        sentiment_col: str = "abm_net_sentiment",
        verbose: bool = False,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    for state in BSVE_STATES:
        subset = (
            aligned_df[aligned_df["state_id"] == state]
            .dropna(subset=[sentiment_col, forward_col])
        )
        n = len(subset)

        if n < 5:
            results[state] = {
                "n": n,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
            }
            continue

        x = subset[sentiment_col].values
        y = subset[forward_col].values

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        results[state] = {
            "n": n,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

        if verbose:
            sig = "**" if spearman_p < 0.05 else "  "
            print(
                f"  [{state:10s}] n={n:5d}  "
                f"Spearman r={spearman_r:+.4f}{sig}(p={spearman_p:.3f})"
            )

    return results


def test_h4_hypothesis(
        state_corrs: Dict[str, dict],
        metric: str = "spearman_r",
        min_n_reliable: int = 100,
) -> dict:
    vals = {s: state_corrs[s].get(metric, math.nan) for s in H4_ORDER}
    ns = {s: state_corrs[s].get("n", 0) for s in H4_ORDER}

    if any(math.isnan(v) for v in vals.values()):
        missing = [s for s, v in vals.items() if math.isnan(v)]
        return {
            "h4_supported": None, "h4_partial_supported": None,
            "cautious": False, "low_n_states": [],
            "reason": f"NaN for states: {missing}", "metric": metric,
            **{s: vals[s] for s in H4_ORDER},
        }

    abs_m = abs(vals["MATURING"])
    abs_e = abs(vals["ENTRY"])
    abs_t = abs(vals["MATURE"])

    ranked = sorted(H4_ORDER, key=lambda s: abs(vals[s]), reverse=True)
    low_n_states = [s for s in H4_ORDER if ns[s] < min_n_reliable]

    return {
        "h4_supported": abs_m > abs_e > abs_t,
        "h4_partial_supported": abs_m > abs_t,
        "cautious": len(low_n_states) > 0,
        "low_n_states": low_n_states,
        "metric": metric,
        "MATURING": vals["MATURING"],
        "ENTRY": vals["ENTRY"],
        "MATURE": vals["MATURE"],
        "abs_MATURING": abs_m,
        "abs_ENTRY": abs_e,
        "abs_MATURE": abs_t,
        "empirical_rank_order": ranked,
        "expected_rank_order": H4_ORDER[:],
    }


def aggregate_runs(
        run_results: List[Dict[str, dict]],
        states: Tuple[str, ...],
) -> Dict[str, dict]:
    aggregated: Dict[str, dict] = {}
    for state in states:
        prs = [r[state]["pearson_r"] for r in run_results
               if not math.isnan(r[state].get("pearson_r", math.nan))]
        srs = [r[state]["spearman_r"] for r in run_results
               if not math.isnan(r[state].get("spearman_r", math.nan))]
        ns = [r[state]["n"] for r in run_results]

        aggregated[state] = {
            "n_mean": float(np.mean(ns)),
            "pearson_r_mean": float(np.mean(prs)) if prs else math.nan,
            "pearson_r_std": float(np.std(prs)) if prs else math.nan,
            "spearman_r_mean": float(np.mean(srs)) if srs else math.nan,
            "spearman_r_std": float(np.std(srs)) if srs else math.nan,
            "runs_used": len(prs),
        }

    return aggregated

# ---------------------------------------------------------------------------
# Episode structure metrics (H3 target)
# ---------------------------------------------------------------------------

def compute_episode_metrics(
        sentiment: np.ndarray,
        extreme_threshold_pct: float = 70.0,
        young_boundary: int = 8,
        mature_boundary: int = 24,
        steps: int = 1000,
) -> dict:
    """
    Compute episode structure metrics from a net_sentiment series.
    Mirrors the BSVE calibration sign-off conditions so H3 can be
    evaluated against empirical targets.

    Returns a dict with:
      - n_episodes
      - ep_freq_per_1000        (episodes per 1000 steps)
      - median_duration_bars
      - reversal_rate_young     (fraction reversing before young_boundary)
      - reversal_rate_mature    (fraction reversing after mature_boundary)
      - rev_gradient_correct    (reversal_rate_young > reversal_rate_mature)
      - censoring_rate          (fraction active at end of window)
    """
    if len(sentiment) == 0:
        return _empty_episode_metrics()

    # Extreme threshold from percentile of abs(sentiment)
    threshold = float(np.percentile(np.abs(sentiment), extreme_threshold_pct))
    if threshold < 1e-6:
        return _empty_episode_metrics()

    # Extract episodes: contiguous runs where abs(sentiment) >= threshold
    in_episode = np.abs(sentiment) >= threshold
    episodes = []
    start = None
    entry_side = None

    for i, val in enumerate(in_episode):
        if val and start is None:
            start = i
            entry_side = np.sign(sentiment[i])
        elif not val and start is not None:
            duration = i - start
            exit_side = np.sign(sentiment[i - 1])
            exit_type = "REVERSAL" if exit_side != entry_side else "THRESHOLD"
            episodes.append({
                "start": start,
                "end": i,
                "duration": duration,
                "exit_type": exit_type,
                "censored": False,
            })
            start = None

    # Handle episode still active at end
    if start is not None:
        duration = len(sentiment) - start
        episodes.append({
            "start": start,
            "end": len(sentiment),
            "duration": duration,
            "exit_type": "CENSORED",
            "censored": True,
        })

    n_ep = len(episodes)
    if n_ep == 0:
        return _empty_episode_metrics()

    completed = [e for e in episodes if not e["censored"]]
    n_complete = len(completed)

    durations = [e["duration"] for e in completed] if completed else [0]
    censored_n = sum(1 for e in episodes if e["censored"])

    # Reversal rates by maturity zone
    young_eps = [e for e in completed if e["duration"] < young_boundary]
    mature_eps = [e for e in completed if e["duration"] >= mature_boundary]

    rev_young = (
        sum(1 for e in young_eps if e["exit_type"] == "REVERSAL") / len(young_eps)
        if young_eps else math.nan
    )
    rev_mature = (
        sum(1 for e in mature_eps if e["exit_type"] == "REVERSAL") / len(mature_eps)
        if mature_eps else math.nan
    )

    gradient_ok = (
        (rev_young > rev_mature)
        if not (math.isnan(rev_young) or math.isnan(rev_mature))
        else None
    )

    ep_freq = (n_ep / max(len(sentiment), 1)) * 1000

    return {
        "n_episodes": n_ep,
        "n_complete": n_complete,
        "ep_freq_per_1000": float(ep_freq),
        "median_duration_bars": float(np.median(durations)),
        "reversal_rate_young": rev_young,
        "reversal_rate_mature": rev_mature,
        "rev_gradient_correct": gradient_ok,
        "censoring_rate": censored_n / n_ep if n_ep > 0 else math.nan,
    }

def _empty_episode_metrics() -> dict:
    return {
        "n_episodes": 0,
        "n_complete": 0,
        "ep_freq_per_1000": 0.0,
        "median_duration_bars": math.nan,
        "reversal_rate_young": math.nan,
        "reversal_rate_mature": math.nan,
        "rev_gradient_correct": None,
        "censoring_rate": math.nan,
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_run_report(
        pair: str,
        shock_params: dict,
        aggregated_corrs: Dict[str, dict],
        h4_verdict: dict,
        episode_metrics: dict,
        runs: int,
        steps: int,
        n_shocks_mean: float,
) -> None:
    sep = "=" * 74
    sep2 = "-" * 74

    shock_str = (
        f"trigger={shock_params['shock_trigger']}  "
        f"frac={shock_params['shock_fraction']}  "
        f"vol_thresh={shock_params['shock_vol_threshold']}  "
        f"cooldown={shock_params['shock_cooldown']}"
        if shock_params["shock_enable"]
        else "DISABLED"
    )

    print(f"\n{sep}")
    print(f"  Stage 3 Shock Test  |  {pair.upper()}")
    print(f"  Runs: {runs}  |  Steps/run: {steps}  |  Shocks/run (mean): "
          f"{n_shocks_mean:.1f}")
    print(f"  Shock: {shock_str}")
    print(sep)

    # --- Episode structure (H3) ---
    print(f"\n  Episode structure (H3 targets: freq~45-56/1k, med_dur~4, "
          f"rev_young>rev_mature)")
    print(f"  {'Metric':<26} {'Value':>10}")
    print(f"  {'-' * 26} {'-' * 10}")
    for key, label in [
        ("n_episodes", "n_episodes"),
        ("ep_freq_per_1000", "freq/1000 steps"),
        ("median_duration_bars", "median duration (bars)"),
        ("reversal_rate_young", "reversal_rate_young"),
        ("reversal_rate_mature", "reversal_rate_mature"),
        ("rev_gradient_correct", "rev gradient correct?"),
        ("censoring_rate", "censoring_rate"),
    ]:
        val = episode_metrics.get(key, math.nan)
        if isinstance(val, float) and not math.isnan(val):
            print(f"  {label:<26} {val:>10.4f}")
        else:
            print(f"  {label:<26} {str(val):>10}")

    print(f"\n{sep2}")

    # --- H4 correlations ---
    print(f"\n  H4 correlations (Spearman |r|, mean ± std across {runs} runs)")
    print(f"  {'State':<12} {'n':>6}  {'Spearman r':>11}  {'±std':>7}  "
          f"{'Pearson r':>10}  {'±std':>7}")
    print(f"  {'-' * 12} {'-' * 6}  {'-' * 11}  {'-' * 7}  {'-' * 10}  {'-' * 7}")

    for state in BSVE_STATES:
        a = aggregated_corrs[state]
        sr = f"{a['spearman_r_mean']:+.4f}" if not math.isnan(a['spearman_r_mean']) else "   NaN"
        ss = f"{a['spearman_r_std']:.4f}" if not math.isnan(a['spearman_r_std']) else "   NaN"
        pr = f"{a['pearson_r_mean']:+.4f}" if not math.isnan(a['pearson_r_mean']) else "   NaN"
        ps = f"{a['pearson_r_std']:.4f}" if not math.isnan(a['pearson_r_std']) else "   NaN"
        print(f"  {state:<12} {int(a['n_mean']):>6}  {sr:>11}  {ss:>7}  "
              f"{pr:>10}  {ps:>7}")

    print(f"\n{sep2}")
    print(f"\n  H4 Verdict  ({h4_verdict['metric']})")
    print(f"  Expected : {' > '.join(H4_ORDER)}")
    print(f"  Empirical: {' > '.join(h4_verdict.get('empirical_rank_order', ['?', '?', '?']))}")

    supported = h4_verdict.get("h4_supported")
    partial = h4_verdict.get("h4_partial_supported")

    if supported is None:
        print(f"  Result   : INCONCLUSIVE — {h4_verdict.get('reason', '')}")
    elif supported:
        print(f"  Result   : H4 SUPPORTED ✓")
    elif partial:
        print(f"  Result   : H4 PARTIALLY SUPPORTED (MATURING > MATURE, "
              f"ENTRY ordering not strict)")
    else:
        print(f"  Result   : H4 NOT SUPPORTED")

    if h4_verdict.get("cautious"):
        low = ", ".join(
            f"{s} (n={aggregated_corrs[s]['n_mean']:.0f})"
            for s in h4_verdict["low_n_states"]
        )
        print(f"  ⚠ CAUTIOUS: low-n — {low}")

    print(f"\n  |corr| — "
          f"MATURING={h4_verdict.get('abs_MATURING', math.nan):.4f}  "
          f"ENTRY={h4_verdict.get('abs_ENTRY', math.nan):.4f}  "
          f"MATURE={h4_verdict.get('abs_MATURE', math.nan):.4f}")
    print(sep + "\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3: Shock-driven episode formation (H3 + H4 test)."
    )
    # Core
    p.add_argument("--pair", default="usd-jpy")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs", type=int, default=20)
    # Calibrated params (overridable)
    p.add_argument("--anchor-strength", type=float,
                   default=CALIBRATED_PARAMS["anchor_strength"])
    p.add_argument("--beta", type=float,
                   default=CALIBRATED_PARAMS["beta"])
    # BSVE
    p.add_argument("--bsve-states-path", type=str, required=True,
                   help="Path to BSVE augmented dataset CSV")
    p.add_argument("--forward-horizon", type=int, default=24)
    p.add_argument("--calibration-artifact", type=str, default=None)
    # Shock parameters
    p.add_argument("--shock-enable", action="store_true",
                   help="Activate shock mechanism")
    p.add_argument("--shock-trigger", default="volatility",
                   choices=["volatility", "periodic"],
                   help="Shock trigger type")
    p.add_argument("--shock-vol-threshold", type=float, default=0.80,
                   help="EMA vol percentile trigger (0-1, volatility mode)")
    p.add_argument("--shock-fraction", type=float, default=0.30,
                   help="Fraction of agents pushed per shock (0-1)")
    p.add_argument("--shock-direction", default="price",
                   choices=["price", "random"],
                   help="Shock direction: follow recent price or random")
    p.add_argument("--shock-cooldown", type=int, default=20,
                   help="Minimum bars between shocks")
    p.add_argument("--shock-period", type=int, default=50,
                   help="Bars between shocks in periodic mode")
    # Output
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = CALIBRATED_PARAMS.copy()
    params["anchor_strength"] = args.anchor_strength
    params["beta"] = args.beta

    shock_params = {
        "shock_enable": args.shock_enable,
        "shock_trigger": args.shock_trigger,
        "shock_vol_threshold": args.shock_vol_threshold,
        "shock_fraction": args.shock_fraction,
        "shock_direction": args.shock_direction,
        "shock_cooldown": args.shock_cooldown,
        "shock_period": args.shock_period,
    }

    forward_col = f"ret_{args.forward_horizon}b"

    # Load calibration artifact for episode metric targets
    calibration_meta: Optional[dict] = None
    if args.calibration_artifact:
        art_path = Path(args.calibration_artifact)
        if art_path.exists():
            with open(art_path) as f:
                calibration_meta = json.load(f)
        else:
            warnings.warn(f"Calibration artifact not found: {art_path}")

    # Load BSVE dataset
    bsve_df = load_bsve_dataset(args.bsve_states_path, args.pair)
    if forward_col not in bsve_df.columns:
        available = [c for c in bsve_df.columns if c.startswith("ret_")]
        raise ValueError(
            f"Column '{forward_col}' not in BSVE dataset. Available: {available}"
        )

    if args.verbose:
        counts = bsve_df["state_id"].value_counts().to_dict()
        print(f"\n[Stage 3] pair={args.pair}  BSVE rows={len(bsve_df)}  "
              f"states={counts}")
        print(f"[Stage 3] shock_enable={args.shock_enable}  "
              f"trigger={args.shock_trigger}  "
              f"frac={args.shock_fraction}  "
              f"vol_thresh={args.shock_vol_threshold}  "
              f"cooldown={args.shock_cooldown}")

        # --- multi-run loop ---
    run_results: List[Dict[str, dict]] = []
    episode_metrics_list: List[dict] = []
    n_shocks_list: List[int] = []

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        if args.verbose:
            print(f"\n[Run {run_idx + 1}/{args.runs}] seed={seed}")

        sentiment, n_shocks = run_abm_series_with_shocks(
            steps=args.steps,
            seed=seed,
            params=params,
            shock_params=shock_params,
            pair=args.pair,
            verbose=args.verbose,
        )

        n_shocks_list.append(n_shocks)

        # Episode structure (H3)
        ep_meta = compute_episode_metrics(
            sentiment=sentiment,
            extreme_threshold_pct=70.0,
            young_boundary=8,
            mature_boundary=24,
            steps=len(sentiment),
        )
        episode_metrics_list.append(ep_meta)

        # H4 correlations
        aligned_df = align_abm_to_bsve(sentiment, bsve_df, verbose=args.verbose)
        run_corrs = compute_state_correlations(
            aligned_df,
            forward_col=forward_col,
            sentiment_col="abm_net_sentiment",
            verbose=args.verbose,
        )
        run_results.append(run_corrs)

    # --- aggregate ---
    aggregated_corrs = aggregate_runs(run_results, states=BSVE_STATES)

    mean_corrs = {
        state: {
            "spearman_r": aggregated_corrs[state]["spearman_r_mean"],
            "pearson_r": aggregated_corrs[state]["pearson_r_mean"],
            "n": int(aggregated_corrs[state]["n_mean"]),
            "pearson_p": math.nan,
            "spearman_p": math.nan,
        }
        for state in BSVE_STATES
    }
    h4_verdict = test_h4_hypothesis(mean_corrs, metric="spearman_r")

    # Average episode metrics across runs
    episode_metrics_mean = {}
    for key in episode_metrics_list[0].keys():
        vals = [
            m[key] for m in episode_metrics_list
            if m[key] is not None and not (
                    isinstance(m[key], float) and math.isnan(m[key])
            )
        ]
        if vals:
            if isinstance(vals[0], bool):
                # rev_gradient_correct — majority vote
                episode_metrics_mean[key] = sum(vals) > len(vals) / 2
            else:
                episode_metrics_mean[key] = float(np.mean(vals))
        else:
            episode_metrics_mean[key] = math.nan

    n_shocks_mean = float(np.mean(n_shocks_list))

    # --- report ---
    print_run_report(
        pair=args.pair,
        shock_params=shock_params,
        aggregated_corrs=aggregated_corrs,
        h4_verdict=h4_verdict,
        episode_metrics=episode_metrics_mean,
        runs=args.runs,
        steps=args.steps,
        n_shocks_mean=n_shocks_mean,
    )

    # --- JSON output ---
    result_payload = {
        "mode": "shock_test",
        "pair": args.pair,
        "runs": args.runs,
        "steps": args.steps,
        "forward_horizon": args.forward_horizon,
        "forward_col": forward_col,
        "anchor_strength": params["anchor_strength"],
        "beta": params["beta"],
        "shock_params": shock_params,
        "n_shocks_mean": n_shocks_mean,
        "n_shocks_list": n_shocks_list,
        "aggregated_corrs": aggregated_corrs,
        "h4_verdict": h4_verdict,
        "episode_metrics_mean": episode_metrics_mean,
        "episode_metrics_runs": episode_metrics_list,
        "calibration_meta": calibration_meta,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_payload, f, indent=2, default=str)
        print(f"[output] Results written to {out_path}")

if __name__ == "__main__":
    main()