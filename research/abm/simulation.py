"""
research/abm/simulation.py
==========================
Core simulation runner for the retail FX sentiment ABM.

FXSentimentSimulation
---------------------
Manages a heterogeneous population of :class:`~research.abm.agents.RetailTrader`
agents and an exogenous real price series only (no internal price generation).
At each step the simulation:

1. Advances the price by one bar from the exogenous series.
2. Computes the current aggregate net sentiment from agent positions.
3. Asks each agent to update its position given the updated price history and
   current crowd sentiment.
4. Records the step-level state.

The output DataFrame mirrors the column conventions of the research dataset
(``net_sentiment``, ``abs_sentiment``, ``crowd_side``) so that downstream
analysis tools work without modification.

Usage::

    from research.abm.simulation import FXSentimentSimulation
    from research.abm.agents import TrendFollower, Contrarian, NoiseTrader
    import numpy as np

    rng = np.random.default_rng(42)
    agents = (
        [TrendFollower(rng) for _ in range(40)]
        + [Contrarian(rng) for _ in range(40)]
        + [NoiseTrader(rng) for _ in range(20)]
    )
    sim = FXSentimentSimulation(agents, rng=rng)
    results = sim.run(n_steps=500)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from research.abm.agents import RetailTrader

logger = logging.getLogger(__name__)

# Volatility proxy parameters (used to compute per-timestep vol_norm)
_VOL_WINDOW = 24

# When agent accumulation state is continuous (float), treat near-zero positions
# as neutral when aggregating to dataset-scale net_sentiment.
# This preserves the dataset semantics: net_sentiment is the net fraction long
# minus fraction short, scaled to [-100, +100].
_AGGREGATION_EPS = 1e-6


class FXSentimentSimulation:
    """Agent-based simulation of retail FX crowd sentiment.

    Args:
        agents: Sequence of :class:`~research.abm.agents.RetailTrader` instances
            forming the simulated population.
        rng: NumPy ``Generator`` for the price process.  Defaults to a fresh
            ``default_rng()`` if not supplied.
        warmup_steps: Number of initial steps run before recording begins.
            Allows agent positions to stabilise before measurement.
    """

    def __init__(
            self,
            agents: Sequence[RetailTrader],
            rng: np.random.Generator | None = None,
            warmup_steps: int = 48,
    ) -> None:
        if not agents:
            raise ValueError("agents must be a non-empty sequence")

        self._agents = list(agents)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._warmup_steps = warmup_steps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        """Total number of agents in the population."""
        return len(self._agents)

    @property
    def warmup_steps(self) -> int:
        """Number of warmup steps run before recording begins."""
        return self._warmup_steps

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
            self,
            n_steps: int,
            price_series: np.ndarray,
            timestamps: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Run simulation using EXOGENOUS price series ONLY.

        Args:
            n_steps: number of steps to record
            price_series: REQUIRED real price series
            timestamps: optional timestamps aligned with price_series

        Returns:
            DataFrame aligned with input time
        """
        if price_series is None:
            raise ValueError("price_series must be provided (no GBM allowed)")

        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        total_required = self._warmup_steps + n_steps + 1
        if len(price_series) < total_required:
            raise ValueError(
                f"price_series too short: need {total_required}, got {len(price_series)}"
            )

        logger.info(
            "Running ABM on real price: n_agents=%d warmup=%d steps=%d",
            self.n_agents,
            self._warmup_steps,
            n_steps,
        )

        records = []
        price_history = [price_series[0]]

        # EMA-based volatility proxy
        ema_alpha = 2.0 / (_VOL_WINDOW + 1.0)
        vol_ema = 0.0
        baseline_vol = 0.0
        baseline_alpha = ema_alpha
        prev_price = float(price_series[0])

        for t in range(1, total_required):
            price = float(price_series[t])
            price_history.append(price)
            ph = np.array(price_history, dtype=np.float64)

            # Volatility proxy from absolute returns (EMA)
            ret_t = price - prev_price
            prev_price = price

            vol_ema = ema_alpha * abs(ret_t) + (1.0 - ema_alpha) * vol_ema
            baseline_vol = baseline_alpha * vol_ema + (1.0 - baseline_alpha) * baseline_vol

            vol_norm = vol_ema / (baseline_vol + 1e-8)

            # sentiment BEFORE update
            crowd_sentiment = self._aggregate_sentiment(normalised=True)

            for agent in self._agents:
                agent.update(ph, crowd_sentiment, volatility=vol_norm)

            if t > self._warmup_steps:
                idx = t - self._warmup_steps - 1

                net_sent = self._aggregate_sentiment(normalised=False)

                row = {
                    "step": idx,
                    "price": price,
                    "net_sentiment": net_sent,
                    "abs_sentiment": abs(net_sent),
                    "crowd_side": int(np.sign(net_sent)),
                    # NOTE: counts below are currently based on exact equality
                    # against {+1, 0, -1}. With continuous positions, these are
                    # not meaningful and are preserved only for backward
                    # compatibility in the output schema.
                    "n_long": sum(a.position == 1 for a in self._agents),
                    "n_short": sum(a.position == -1 for a in self._agents),
                    "n_flat": sum(a.position == 0 for a in self._agents),
                }

                if timestamps is not None:
                    row["timestamp"] = timestamps[t]

                records.append(row)

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_sentiment(self, *, normalised: bool) -> float:
        """Compute aggregate net sentiment from agent positions.

        Args:
            normalised: When ``True`` return a value in [-1, +1] (fraction
                long minus fraction short).  When ``False`` scale to [-100, +100]
                to match the research dataset convention.

        Returns:
            Net sentiment value.
        """
        positions = np.array([a.position for a in self._agents], dtype=np.float64)
        n = len(positions)

        # Preserve dataset semantics even with continuous internal accumulation
        # state: each agent contributes a bounded vote in {-1, 0, +1}.
        votes = np.zeros_like(positions)
        votes[positions > _AGGREGATION_EPS] = 1.0
        votes[positions < -_AGGREGATION_EPS] = -1.0

        net_fraction = votes.sum() / n  # in [-1, 1]
        out = float(net_fraction if normalised else net_fraction * 100.0)
        lo, hi = (-1.0, 1.0) if normalised else (-100.0, 100.0)

        if out < lo or out > hi:
            clamped = float(np.clip(out, lo, hi))
            logger.debug(
                "Clamped aggregate sentiment from %.8f to %.8f (bounds=[%.1f, %.1f])",
                out,
                clamped,
                lo,
                hi,
            )
            return clamped

        return out
