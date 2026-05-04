"""
agents.py

Agent definitions for the retail FX sentiment ABM.

Agent hierarchy
---------------
BaseAgent       – abstract; discrete position ∈ {-1, 0, +1}
  TrendFollower – follows price momentum
  Contrarian    – fades price momentum
  RetailTrader  – abstract retail base (_price_signal raises NotImplementedError)
    NoiseTrader – no price signal; driven by crowd + noise

Module-level sweep parameters
------------------------------
_PERSISTENCE_WEIGHT and _INERTIA_THRESHOLD are read by simulation.py to
control regime-transition smoothing and the disagreement trigger threshold.
sweep.py mutates these between runs and restores them afterwards.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Module-level parameters read by simulation.py and varied by sweep.py
# ---------------------------------------------------------------------------

_PERSISTENCE_WEIGHT: float = 0.0   # regime-smoothing factor ∈ [0, 1)
_INERTIA_THRESHOLD: float = 0.02   # disagreement trigger threshold ∈ (0, 1)

# Amplification of market volatility into agent noise:
#   effective_noise = noise_scale * (1 + _VOL_FEEDBACK_SCALE * volatility)
# A value of 100 means a 1% vol spike doubles the agent's noise.
_VOL_FEEDBACK_SCALE: float = 100.0


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract agent with a discrete position."""

    def __init__(
        self,
        rng: np.random.Generator,
        noise_scale: float = 0.02,
        crowd_weight: float = 0.1,
    ) -> None:
        self.rng = rng
        self.noise_scale = float(noise_scale)
        self.crowd_weight = float(crowd_weight)
        self.position: int = 0  # discrete: -1, 0, or +1

    def _price_signal(self, price_history: np.ndarray) -> float | None:
        """Return a price-based signal in [-1, 1], or None when data is insufficient."""
        raise NotImplementedError

    def update(
        self,
        price_history: np.ndarray,
        crowd_sentiment: float = 0.0,
        volatility: float = 0.002,
    ) -> None:
        """Update discrete position given price history, crowd sentiment, and volatility."""
        price_sig = self._price_signal(price_history)
        if price_sig is None:
            return  # insufficient data – keep current position

        # Volatility feedback: scale agent noise by current market volatility
        effective_noise = self.noise_scale * (1.0 + _VOL_FEEDBACK_SCALE * volatility)
        noise = self.rng.normal(0.0, effective_noise)
        raw = float(price_sig) + self.crowd_weight * float(crowd_sentiment) + noise
        if raw > 0.05:
            self.position = 1
        elif raw < -0.05:
            self.position = -1
        else:
            self.position = 0


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------

class TrendFollower(BaseAgent):
    """Trend-following agent: goes long (short) after an up (down) move."""

    def __init__(
        self,
        rng: np.random.Generator,
        momentum_window: int = 12,
        noise_scale: float = 0.02,
        crowd_weight: float = 0.1,
        pair: str | None = None,
    ) -> None:
        if int(momentum_window) <= 0:
            raise ValueError(f"momentum_window must be > 0, got {momentum_window!r}")
        super().__init__(rng, noise_scale, crowd_weight)
        self.momentum_window = int(momentum_window)

    def _price_signal(self, price_history: np.ndarray) -> float | None:
        if len(price_history) < self.momentum_window + 1:
            return None
        p0 = price_history[-self.momentum_window - 1]
        p1 = price_history[-1]
        if p0 <= 0:
            return 0.0
        return float(np.tanh((p1 / p0 - 1.0) * 10.0))


class Contrarian(BaseAgent):
    """Contrarian agent: fades momentum by taking the opposite side."""

    def __init__(
        self,
        rng: np.random.Generator,
        momentum_window: int = 12,
        noise_scale: float = 0.02,
        crowd_weight: float = 0.1,
        pair: str | None = None,
    ) -> None:
        if int(momentum_window) <= 0:
            raise ValueError(f"momentum_window must be > 0, got {momentum_window!r}")
        super().__init__(rng, noise_scale, crowd_weight)
        self.momentum_window = int(momentum_window)

    def _price_signal(self, price_history: np.ndarray) -> float | None:
        if len(price_history) < self.momentum_window + 1:
            return None
        p0 = price_history[-self.momentum_window - 1]
        p1 = price_history[-1]
        if p0 <= 0:
            return 0.0
        return float(-np.tanh((p1 / p0 - 1.0) * 10.0))


class RetailTrader(BaseAgent):
    """Abstract base for retail traders.

    Subclasses must implement ``_price_signal``.  Calling ``_price_signal``
    directly on a ``RetailTrader`` instance raises ``NotImplementedError``.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        noise_scale: float = 0.02,
        crowd_weight: float = 0.1,
        pair: str | None = None,
    ) -> None:
        super().__init__(rng, noise_scale, crowd_weight)

    def _price_signal(self, price_history: np.ndarray) -> float | None:
        raise NotImplementedError


class NoiseTrader(RetailTrader):
    """Pure noise trader: no directional price signal.

    Position is driven entirely by crowd sentiment and random noise.
    ``_price_signal`` always returns 0.0 (overrides RetailTrader).
    """

    def _price_signal(self, price_history: np.ndarray) -> float | None:
        return 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_agents(
    n_agents: int,
    trend_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
    momentum_window: int = 12,
) -> list[BaseAgent]:
    """Build a heterogeneous population of agents.

    Parameters
    ----------
    n_agents : int
    trend_ratio : float
        Fraction of trend-following agents (remainder split contrarian/noise).
    rng : np.random.Generator | None
    momentum_window : int
        Default look-back window for trend/contrarian agents.
    """
    rng = rng or np.random.default_rng()
    agents: list[BaseAgent] = []

    n_trend = int(n_agents * trend_ratio)
    n_remaining = n_agents - n_trend
    n_contrarian = n_remaining // 2
    n_noise = n_remaining - n_contrarian

    for _ in range(n_trend):
        mw = int(rng.integers(6, 24))
        agents.append(TrendFollower(rng, momentum_window=mw))

    for _ in range(n_contrarian):
        mw = int(rng.integers(6, 24))
        agents.append(Contrarian(rng, momentum_window=mw))

    for _ in range(n_noise):
        agents.append(NoiseTrader(rng))

    rng.shuffle(agents)
    return agents