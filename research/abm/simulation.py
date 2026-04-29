"""
research/abm/simulation.py
==========================
Core simulation runner for the retail FX sentiment ABM.

FXSentimentSimulation
---------------------
Manages a heterogeneous population of :class:`~research.abm.agents.RetailTrader`
agents and an exogenous price process.  At each step the simulation:

1. Advances the price by one bar (geometric Brownian motion or externally
   supplied series).
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

# Default GBM parameters for the endogenous price process.
_DEFAULT_DRIFT = 0.0
_DEFAULT_VOLATILITY = 0.001  # per-step log-return std (≈ hourly FX volatility)
_DEFAULT_INITIAL_PRICE = 1.0


class FXSentimentSimulation:
    """Agent-based simulation of retail FX crowd sentiment.

    Args:
        agents: Sequence of :class:`~research.abm.agents.RetailTrader` instances
            forming the simulated population.
        rng: NumPy ``Generator`` for the price process.  Defaults to a fresh
            ``default_rng()`` if not supplied.
        drift: Per-step log-return drift for the GBM price process.
        volatility: Per-step log-return standard deviation (GBM).
        initial_price: Starting price level.  Ignored when ``price_series``
            is supplied to :meth:`run`.
        warmup_steps: Number of initial steps run before recording begins.
            Allows agent positions to stabilise before measurement.
    """

    def __init__(
        self,
        agents: Sequence[RetailTrader],
        rng: np.random.Generator | None = None,
        drift: float = _DEFAULT_DRIFT,
        volatility: float = _DEFAULT_VOLATILITY,
        initial_price: float = _DEFAULT_INITIAL_PRICE,
        warmup_steps: int = 48,
    ) -> None:
        if not agents:
            raise ValueError("agents must be a non-empty sequence")
        if volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {volatility}")
        if initial_price <= 0:
            raise ValueError(f"initial_price must be > 0, got {initial_price}")

        self._agents = list(agents)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._drift = drift
        self._volatility = volatility
        self._initial_price = initial_price
        self._warmup_steps = warmup_steps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        """Total number of agents in the population."""
        return len(self._agents)

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self,
        n_steps: int,
        price_series: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run the simulation and return per-step results.

        Args:
            n_steps: Number of recorded steps (excluding warmup).
            price_series: Optional externally supplied price series.  When
                provided it must have length >= ``warmup_steps + n_steps``;
                the GBM process is not used.  When ``None`` a GBM price
                path is generated internally.

        Returns:
            DataFrame with one row per recorded step and the following columns:

            - ``step``: integer step index (0-based)
            - ``price``: exogenous price level
            - ``net_sentiment``: aggregate net positioning in [-100, +100]
              (positive = crowd net long, negative = crowd net short)
            - ``abs_sentiment``: absolute value of ``net_sentiment``
            - ``crowd_side``: +1, -1, or 0 (sign of ``net_sentiment``)
            - ``n_long``: number of long agents
            - ``n_short``: number of short agents
            - ``n_flat``: number of flat agents

        Raises:
            ValueError: If ``n_steps`` < 1.
            ValueError: If ``price_series`` is too short.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        total_steps = self._warmup_steps + n_steps
        prices = self._build_price_path(price_series, total_steps)

        logger.info(
            "Running ABM: n_agents=%d  warmup=%d  recorded_steps=%d",
            self.n_agents,
            self._warmup_steps,
            n_steps,
        )

        records: list[dict] = []
        price_history: list[float] = [prices[0]]

        for t in range(1, total_steps + 1):
            price = float(prices[t]) if t < len(prices) else price_history[-1]
            price_history.append(price)
            ph = np.array(price_history, dtype=np.float64)

            # Compute crowd sentiment BEFORE agents update (uses previous step).
            crowd_sentiment = self._aggregate_sentiment(normalised=True)

            # All agents update simultaneously (synchronous update).
            for agent in self._agents:
                agent.update(ph, crowd_sentiment)

            # Record only after warmup.
            if t > self._warmup_steps:
                step_idx = t - self._warmup_steps - 1
                net_sent = self._aggregate_sentiment(normalised=False)
                records.append(
                    {
                        "step": step_idx,
                        "price": price,
                        "net_sentiment": net_sent,
                        "abs_sentiment": abs(net_sent),
                        "crowd_side": int(np.sign(net_sent)),
                        "n_long": sum(a.position == 1 for a in self._agents),
                        "n_short": sum(a.position == -1 for a in self._agents),
                        "n_flat": sum(a.position == 0 for a in self._agents),
                    }
                )

        df = pd.DataFrame(records)
        logger.info("Simulation complete: %d rows recorded", len(df))
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
        net_fraction = positions.sum() / n  # in [-1, 1]
        if normalised:
            return float(net_fraction)
        return float(net_fraction * 100.0)

    def _build_price_path(
        self,
        price_series: np.ndarray | None,
        total_steps: int,
    ) -> np.ndarray:
        """Return a price array of length ``total_steps + 1``.

        If ``price_series`` is provided it is used directly (after validation).
        Otherwise a GBM path is generated.
        """
        required = total_steps + 1
        if price_series is not None:
            arr = np.asarray(price_series, dtype=np.float64)
            if len(arr) < required:
                raise ValueError(
                    f"price_series length {len(arr)} is too short; "
                    f"need at least {required} (warmup + n_steps + 1)"
                )
            return arr[:required]

        # Geometric Brownian Motion
        log_returns = self._rng.normal(
            self._drift, self._volatility, size=total_steps
        )
        log_prices = np.concatenate([[np.log(self._initial_price)], log_returns.cumsum() + np.log(self._initial_price)])
        return np.exp(log_prices)
