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


# Tunable parameters for RetailTrader.update()
_SIGNAL_AMPLIFICATION = 5.0   # tanh gain applied to raw price signal
_PERSISTENCE_WEIGHT = 0.1     # fraction of current position fed back into score
_INERTIA_THRESHOLD = 0.05     # minimum |score| required to change position


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
        # start small, but allow accumulation later
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

        signal = np.tanh(_SIGNAL_AMPLIFICATION * self._price_signal(price_history))
        # amplify crowd influence nonlinearly
        herd = self.crowd_weight * np.tanh(3 * crowd_sentiment)
        noise = self.rng.normal(0.0, self.noise_scale)
        persistence = _PERSISTENCE_WEIGHT * self.position

        score = signal + herd + noise + persistence


        # --- asymmetric behavior: hold losers, reinforce winners ---

        if self.position != 0:
            # Case 1: losing → resist change (hope)
            if np.sign(score) != self.position:
                if self.rng.random() < 0.7:
                    return

            # Case 2: winning → reinforce conviction
            else:
                score += 3.0 * self.position

        # --- anchoring parameters (module-level tuning knobs) ---
        _ANCHOR_STRENGTH = 2.0  # how strongly agents resist flipping
        _SWITCH_BASE_PROB = 1.0  # baseline probability of switching

        # If signal is too weak → do nothing
        if abs(score) < _INERTIA_THRESHOLD:
            return


        direction = 1 if score > 0 else -1

        # If flat → enter immediately (no anchoring yet)
        if self.position == 0:
            if self.rng.random() < 0.7:
                self.position = direction
            else:
                self.position = -direction
            return

        # --- accumulation: add to winning positions ---
        if np.sign(self.position) == direction:
            # increase conviction (bounded)
            if abs(self.position) < 5:
                self.position += direction
            return

        # --- anchoring (this is the key change) ---
        # Existing position creates resistance to switching
        anchor_bias = _ANCHOR_STRENGTH * self.position

        # Adjust switching score
        switch_score = score - anchor_bias

        # Convert to probability via logistic function
        switch_prob = 1.0 / (1.0 + np.exp(-switch_score))

        # Damp switching overall
        switch_prob *= _SWITCH_BASE_PROB

        # Probabilistic switch
        if self.rng.random() < switch_prob:
            # flip direction but reduce magnitude (not full reset)
            self.position = direction


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
