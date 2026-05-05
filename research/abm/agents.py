from __future__ import annotations

import numpy as np


# Tunable parameters for RetailTrader.update()
_SIGNAL_AMPLIFICATION = 5.0
_PERSISTENCE_WEIGHT = 0.1
_INERTIA_THRESHOLD = 0.05

# Stage-2 release mechanism (backward compatible defaults: 0.0)
_DECAY_BASE = 0.0
_DECAY_VOLATILITY_SCALE = 0.0
_DECAY_CLIP_MAX = 0.2


class RetailTrader:
    def __init__(
        self,
        rng: np.random.Generator,
        pair: str,
        crowd_weight: float = 0.0,
        noise_scale: float = 0.0,
    ) -> None:
        self.rng = rng
        self.crowd_weight = crowd_weight
        self.noise_scale = noise_scale

        self.pair = pair.lower()
        base, quote = self.pair.split("-")

        # Normalize to "non-USD strength"
        if base == "usd":
            self.signal_sign = -1
        else:
            self.signal_sign = 1

        self.position: int = int(rng.choice([-1, 0, 1]))

    def _price_signal(self, price_history: np.ndarray) -> float:
        raise NotImplementedError

    def update(
        self,
        price_history: np.ndarray,
        crowd_sentiment: float,
        volatility: float = 0.0,
    ) -> None:
        """Update agent position given price history, crowd sentiment, and volatility.

        Args:
            price_history: price history up to current timestep.
            crowd_sentiment: aggregate crowd sentiment (normalised to [-1, 1]).
            volatility: normalised realised volatility proxy for the current timestep.
                Defaults to 0.0 for backward compatibility.
        """
        if len(price_history) < 2:
            return

        raw_signal = self._price_signal(price_history)
        normalized_signal = self.signal_sign * raw_signal
        signal = np.tanh(_SIGNAL_AMPLIFICATION * normalized_signal)

        herd = self.crowd_weight * np.tanh(3 * crowd_sentiment)
        noise = self.rng.normal(0.0, self.noise_scale)
        persistence = _PERSISTENCE_WEIGHT * self.position

        score = signal + herd + noise + persistence

        # --- asymmetric behavior ---
        if self.position != 0:
            if np.sign(score) != self.position:
                if self.rng.random() < 0.7:
                    return
            else:
                score += 3.0 * self.position

        _ANCHOR_STRENGTH = 2.0
        _SWITCH_BASE_PROB = 1.0

        if abs(score) < _INERTIA_THRESHOLD:
            return

        direction = 1 if score > 0 else -1

        # Entry
        if self.position == 0:
            if self.rng.random() < 0.7:
                self.position = direction
            else:
                self.position = -direction
            return

        # Accumulation (with volatility-conditioned decay / release)
        if np.sign(self.position) == direction:
            # Decay reduces the magnitude of accumulated position before accumulating
            lam = _DECAY_BASE + _DECAY_VOLATILITY_SCALE * float(volatility)
            lam = float(np.clip(lam, 0.0, _DECAY_CLIP_MAX))

            # Apply decay (release) to current accumulation state
            if lam > 0.0 and self.position != 0:
                decayed = (1.0 - lam) * float(self.position)
                # Keep position as integer state: round toward zero
                self.position = int(np.trunc(decayed))

            if abs(self.position) < 5:
                self.position += direction
            return

        # Switching with anchoring
        anchor_bias = _ANCHOR_STRENGTH * self.position
        switch_score = score - anchor_bias

        switch_prob = 1.0 / (1.0 + np.exp(-switch_score))
        switch_prob *= _SWITCH_BASE_PROB

        if self.rng.random() < switch_prob:
            self.position = direction


class TrendFollower(RetailTrader):
    def __init__(
        self,
        rng: np.random.Generator,
        pair: str,
        momentum_window: int = 12,
        crowd_weight: float = 0.1,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__(rng, pair, crowd_weight=crowd_weight, noise_scale=noise_scale)
        self.momentum_window = momentum_window

    def _price_signal(self, price_history: np.ndarray) -> float:
        window = min(self.momentum_window, len(price_history) - 1)
        ret = (price_history[-1] - price_history[-(window + 1)]) / (
            abs(price_history[-(window + 1)]) + 1e-12
        )
        return float(np.sign(ret))


class Contrarian(RetailTrader):
    def __init__(
        self,
        rng: np.random.Generator,
        pair: str,
        momentum_window: int = 12,
        crowd_weight: float = -0.05,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__(rng, pair, crowd_weight=crowd_weight, noise_scale=noise_scale)
        self.momentum_window = momentum_window

    def _price_signal(self, price_history: np.ndarray) -> float:
        window = min(self.momentum_window, len(price_history) - 1)
        ret = (price_history[-1] - price_history[-(window + 1)]) / (
            abs(price_history[-(window + 1)]) + 1e-12
        )
        return -float(np.sign(ret))


class NoiseTrader(RetailTrader):
    def __init__(
        self,
        rng: np.random.Generator,
        pair: str,
        crowd_weight: float = 0.05,
        noise_scale: float = 0.5,
    ) -> None:
        super().__init__(rng, pair, crowd_weight=crowd_weight, noise_scale=noise_scale)

    def _price_signal(self, price_history: np.ndarray) -> float:
        return 0.0
