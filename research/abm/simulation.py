"""
simulation.py

ABM simulation engine for FX sentiment with latent regime persistence.

Regime state
------------
regime_state ∈ {"trend", "neutral", "volatile"}

Transitions (driven by smoothed agent signals):
  high disagreement  → volatile  (high vol, low impact)
  strong alignment   → trend     (low vol, high impact)
  otherwise          → neutral   (baseline)

The smoothing speed and disagreement trigger threshold are read at runtime
from ``research.abm.agents._PERSISTENCE_WEIGHT`` and
``research.abm.agents._INERTIA_THRESHOLD`` so that sweep.py can vary them
between runs without rebuilding the simulation.

Volatility feedback
-------------------
Each step's crowd volatility is passed back to agents via
``agent.update(..., volatility=vol)``, creating a feedback loop: in a
volatile regime, agents receive higher volatility, scale up their noise,
and produce more disagreement – sustaining the volatile regime.

Interface
---------
The public API is unchanged:
  FXSentimentSimulation(agents, rng=None, seed=42, warmup_steps=50)
  .run(n_steps, price_series, timestamps=None) → pd.DataFrame

Output columns: step, price, net_sentiment, abs_sentiment,
                crowd_side, n_long, n_short, n_flat
                (+ timestamp when timestamps is supplied)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research.abm.agents as _agents_mod


class FXSentimentSimulation:
    def __init__(
        self,
        agents,
        rng: np.random.Generator | None = None,
        seed: int = 42,
        warmup_steps: int = 50,
    ) -> None:
        """
        Parameters
        ----------
        agents : list[BaseAgent]
            Non-empty list of agent objects.
        rng : np.random.Generator, optional
            Shared random generator.  When *None* a new generator is created
            from ``seed``.
        seed : int
        warmup_steps : int
            Number of price observations fed to agents before recording begins.
        """
        if not agents:
            raise ValueError("agents list must not be empty")

        self.agents = agents

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.warmup_steps = int(warmup_steps)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def run(
        self,
        n_steps: int,
        price_series,
        timestamps=None,
    ) -> pd.DataFrame:
        """Run the simulation and return a per-step DataFrame.

        Parameters
        ----------
        n_steps : int
            Number of recorded steps (> 0).
        price_series : array-like
            Observed price sequence.  Must have at least
            ``warmup_steps + n_steps`` entries.
        timestamps : array-like, optional
            If supplied, a ``timestamp`` column is added to the output aligned
            to ``price_series[warmup_steps : warmup_steps + n_steps]``.

        Returns
        -------
        pd.DataFrame
            Columns: step, price, net_sentiment, abs_sentiment,
            crowd_side, n_long, n_short, n_flat  (+ timestamp).
        """
        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")

        price_series = np.asarray(price_series, dtype=float)

        min_len = self.warmup_steps + n_steps
        if len(price_series) < min_len:
            raise ValueError(
                f"price_series too short: need {min_len} entries "
                f"(warmup_steps={self.warmup_steps} + n_steps={n_steps}), "
                f"got {len(price_series)}"
            )

        n_agents = self.n_agents

        # ------------------------------------------------------------------
        # Warmup: feed agents the initial price history without recording.
        # ------------------------------------------------------------------
        for t in range(self.warmup_steps):
            ph = price_series[: t + 1]
            for agent in self.agents:
                agent.update(ph, crowd_sentiment=0.0, volatility=0.002)

        # ------------------------------------------------------------------
        # Regime state – smoothed indicators for persistence
        # ------------------------------------------------------------------
        regime_state = "neutral"
        smooth_disagree = 0.0
        smooth_align = 0.0

        # ------------------------------------------------------------------
        # Main simulation loop
        # ------------------------------------------------------------------
        records: list[dict] = []

        for t in range(n_steps):
            # Price history available at this step
            ph = price_series[: self.warmup_steps + t + 1]
            current_price = float(price_series[self.warmup_steps + t])

            # --- Read regime parameters from agents module (modified by sweep) ---
            persistence_weight = float(_agents_mod._PERSISTENCE_WEIGHT)
            inertia_thresh = float(_agents_mod._INERTIA_THRESHOLD)

            # --- Crowd state from current agent positions ---
            n_long = sum(1 for a in self.agents if a.position == 1)
            n_short = sum(1 for a in self.agents if a.position == -1)
            n_flat = n_agents - n_long - n_short

            net_sentiment = float((n_long - n_short) / n_agents * 100.0)
            crowd_s = net_sentiment / 100.0  # normalised to [-1, 1]

            # --- Disagreement: divergence among active agents ---
            active = n_long + n_short
            if active > 0:
                agreement = abs(n_long - n_short) / active
                disagreement = 1.0 - agreement
            else:
                disagreement = 0.0

            # --- EMA smoothing (alpha controlled by persistence_weight) ---
            # Higher persistence_weight → slower alpha → regime is stickier
            smooth_alpha = 1.0 / (1.0 + 10.0 * persistence_weight)
            smooth_disagree = (
                (1.0 - smooth_alpha) * smooth_disagree + smooth_alpha * disagreement
            )
            smooth_align = (
                (1.0 - smooth_alpha) * smooth_align + smooth_alpha * abs(crowd_s)
            )

            # --- Regime transition ---
            # inertia_thresh scales the volatile trigger: larger threshold
            # requires stronger disagreement to enter volatile regime.
            vol_trigger = 0.2 + 3.0 * inertia_thresh

            if smooth_disagree > vol_trigger:
                regime_state = "volatile"
            elif smooth_align > 0.4:
                regime_state = "trend"
            else:
                regime_state = "neutral"

            # --- Regime-dependent crowd volatility ---
            if regime_state == "trend":
                vol = 0.001   # low vol, high persistence
            elif regime_state == "volatile":
                vol = 0.008   # high vol, low persistence
            else:
                vol = 0.003   # baseline

            # --- Volatility feedback: update agents with current vol ---
            for agent in self.agents:
                agent.update(ph, crowd_sentiment=crowd_s, volatility=vol)

            records.append(
                {
                    "step": t,
                    "price": current_price,
                    "net_sentiment": net_sentiment,
                    "abs_sentiment": abs(net_sentiment),
                    "crowd_side": int(np.sign(net_sentiment)),
                    "n_long": n_long,
                    "n_short": n_short,
                    "n_flat": n_flat,
                }
            )

        df = pd.DataFrame(records)

        if timestamps is not None:
            ts_arr = np.asarray(timestamps)
            df["timestamp"] = ts_arr[
                self.warmup_steps: self.warmup_steps + n_steps
            ]

        return df