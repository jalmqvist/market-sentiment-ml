"""
research/abm/agents.py
======================
Heterogeneous retail FX trader agents.

Each agent holds a position (+1 long, -1 short, 0 flat) and updates it at
every simulation step based on observed price history and current crowd
sentiment.

Agent types
-----------
TrendFollower
    Aligns with recent price momentum: buys after up-moves, sells after
    down-moves.  Represents traders who chase trends.

Contrarian
    Fades recent price momentum: buys after down-moves, sells after up-moves.
    Represents traders who expect mean-reversion.

NoiseTrader
    Updates position randomly, independent of price or crowd.  Represents
    background liquidity and behaviorally inconsistent traders.

All agents accept an optional ``crowd_weight`` parameter.  When non-zero,
they partially align with (or against) the current crowd sentiment, allowing
herding dynamics to emerge.
"""

from __future__ import annotations

import numpy as np


class RetailTrader:
    """Abstract base class for a retail FX trader agent.

    Attributes:
        position: Current market position: +1 (long), -1 (short), or 0 (flat).
        rng: NumPy ``Generator`` used for all stochastic updates.
        crowd_weight: Weight applied to crowd sentiment when updating position.
            Positive → herding, negative → contrarian-to-crowd.
        noise_scale: Standard deviation of Gaussian noise added at each step.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        crowd_weight: float = 0.0,
        noise_scale: float = 0.0,
    ) -> None:
        self.rng = rng
        self.crowd_weight = crowd_weight
        self.noise_scale = noise_scale
        # Initialize at a random position.
        self.position: int = int(rng.choice([-1, 0, 1]))

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _price_signal(self, price_history: np.ndarray) -> float:
        """Compute a raw directional signal from price history.

        Args:
            price_history: 1-D array of past price levels (most recent last).

        Returns:
            A float in roughly [-1, 1] representing directional preference.
            Positive → lean long, negative → lean short.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public update method
    # ------------------------------------------------------------------

    def update(self, price_history: np.ndarray, crowd_sentiment: float) -> None:
        if len(price_history) < 2:
            return

        signal = self._price_signal(price_history)
        herd = self.crowd_weight * crowd_sentiment
        noise = self.rng.normal(0.0, self.noise_scale)

        score = signal + herd + noise

        # --- NEW: inertia ---
        threshold = 0.2

        if abs(score) < threshold:
            # do nothing → keep position
            return

        if score > 0:
            self.position = 1
        else:
            self.position = -1


class TrendFollower(RetailTrader):
    """Trend-following retail trader.

    Computes momentum as the sign of the recent price return over a rolling
    window.  Goes long when momentum is positive, short when negative.

    Args:
        rng: NumPy ``Generator``.
        momentum_window: Number of past bars to compute momentum over.
        crowd_weight: Herding weight (see :class:`RetailTrader`).
        noise_scale: Noise standard deviation (see :class:`RetailTrader`).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        momentum_window: int = 12,
        crowd_weight: float = 0.1,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__(rng, crowd_weight=crowd_weight, noise_scale=noise_scale)
        if momentum_window < 1:
            raise ValueError(f"momentum_window must be >= 1, got {momentum_window}")
        self.momentum_window = momentum_window

    def _price_signal(self, price_history: np.ndarray) -> float:
        window = min(self.momentum_window, len(price_history) - 1)
        ret = (price_history[-1] - price_history[-(window + 1)]) / (
            abs(price_history[-(window + 1)]) + 1e-12
        )
        return float(np.sign(ret))


class Contrarian(RetailTrader):
    """Contrarian (mean-reverting) retail trader.

    Computes momentum exactly as :class:`TrendFollower` but bets against it.
    Represents traders who expect recent moves to reverse.

    Args:
        rng: NumPy ``Generator``.
        momentum_window: Number of past bars to compute momentum over.
        crowd_weight: Herding weight (see :class:`RetailTrader`).
        noise_scale: Noise standard deviation (see :class:`RetailTrader`).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        momentum_window: int = 12,
        crowd_weight: float = -0.05,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__(rng, crowd_weight=crowd_weight, noise_scale=noise_scale)
        if momentum_window < 1:
            raise ValueError(f"momentum_window must be >= 1, got {momentum_window}")
        self.momentum_window = momentum_window

    def _price_signal(self, price_history: np.ndarray) -> float:
        window = min(self.momentum_window, len(price_history) - 1)
        ret = (price_history[-1] - price_history[-(window + 1)]) / (
            abs(price_history[-(window + 1)]) + 1e-12
        )
        return -float(np.sign(ret))  # fade the trend


class NoiseTrader(RetailTrader):
    """Noise trader with no systematic price view.

    Updates position based solely on Gaussian noise and crowd sentiment.
    Provides background randomness in the aggregate sentiment signal.

    Args:
        rng: NumPy ``Generator``.
        crowd_weight: Herding weight (see :class:`RetailTrader`).
        noise_scale: Noise standard deviation (see :class:`RetailTrader`).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        crowd_weight: float = 0.05,
        noise_scale: float = 0.5,
    ) -> None:
        super().__init__(rng, crowd_weight=crowd_weight, noise_scale=noise_scale)

    def _price_signal(self, price_history: np.ndarray) -> float:  # noqa: ARG002
        return 0.0  # no systematic price signal
