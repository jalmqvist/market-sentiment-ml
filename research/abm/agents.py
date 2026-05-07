from __future__ import annotations

import os

import numpy as np


# Tunable parameters for RetailTrader.update()
_SIGNAL_AMPLIFICATION = 5.0
_PERSISTENCE_WEIGHT = 0.1
_INERTIA_THRESHOLD = 0.05

# Stage-2 release mechanism (backward compatible defaults: 0.0)
_DECAY_BASE = 0.0
_DECAY_VOLATILITY_SCALE = 0.0
_DECAY_CLIP_MAX = 0.2

# Stage-3 regime escape / de-alignment (backward compatible defaults: disabled)
# These parameters implement a minimal "escape" mechanism to prevent long-lived
# one-sided herd states. When the crowd is sufficiently one-sided (|crowd_sentiment|
# above a threshold), agents have a small probability of partially de-aligning
# (shrinking their accumulated position magnitude). Defaults keep this off.
#
# For experiment convenience, these can be overridden via environment variables:
# - ABM_ESCAPE_PROB_SAT
# - ABM_ESCAPE_SAT_THRESHOLD
# - ABM_ESCAPE_SHRINK_FACTOR
# - ABM_ESCAPE_FLIP_PROB
_ESCAPE_PROB_SAT = 0.0
_ESCAPE_SAT_THRESHOLD = 0.7
_ESCAPE_SHRINK_FACTOR = 0.5
_ESCAPE_FLIP_PROB = 0.0

if "ABM_ESCAPE_PROB_SAT" in os.environ:
    _ESCAPE_PROB_SAT = float(os.environ["ABM_ESCAPE_PROB_SAT"])
if "ABM_ESCAPE_SAT_THRESHOLD" in os.environ:
    _ESCAPE_SAT_THRESHOLD = float(os.environ["ABM_ESCAPE_SAT_THRESHOLD"])
if "ABM_ESCAPE_SHRINK_FACTOR" in os.environ:
    _ESCAPE_SHRINK_FACTOR = float(os.environ["ABM_ESCAPE_SHRINK_FACTOR"])
if "ABM_ESCAPE_FLIP_PROB" in os.environ:
    _ESCAPE_FLIP_PROB = float(os.environ["ABM_ESCAPE_FLIP_PROB"])


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

        # Continuous accumulation state (float) to avoid quantization of decay
        self.position: float = float(rng.choice([-1, 0, 1]))

    def _price_signal(self, price_history: np.ndarray) -> float:
        raise NotImplementedError

    def _maybe_escape_regime(self, crowd_sentiment: float) -> None:
        """Optional regime escape / de-alignment mechanism.

        Trigger: only when the crowd is sufficiently one-sided.

        Action (minimal, configurable):
        - Primary: shrink the magnitude of accumulated position.
        - Optional: with small probability, flip sign to break sign-lock.

        This is designed to be minimal and backward compatible:
        - off by default (probability 0.0)
        - only depends on crowd_sentiment (already passed into update)
        - does not require refactors or pipeline changes

        Note: parameters can be overridden via environment variables.
        """
        if _ESCAPE_PROB_SAT <= 0.0:
            return
        if abs(float(crowd_sentiment)) <= float(_ESCAPE_SAT_THRESHOLD):
            return
        if self.position == 0.0:
            return

        if self.rng.random() >= float(_ESCAPE_PROB_SAT):
            return

        # Optional sign de-alignment (off by default)
        if _ESCAPE_FLIP_PROB > 0.0 and self.rng.random() < float(_ESCAPE_FLIP_PROB):
            self.position = -float(self.position)
            return

        # Partial de-alignment: reduce magnitude but keep sign.
        self.position = float(self.position) * float(_ESCAPE_SHRINK_FACTOR)

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

        # Optional saturation-conditioned escape (Stage-3). Applied before the
        # main decision logic so it can prevent long-lived lock-in.
        self._maybe_escape_regime(crowd_sentiment)

        raw_signal = self._price_signal(price_history)
        normalized_signal = self.signal_sign * raw_signal
        signal = np.tanh(_SIGNAL_AMPLIFICATION * normalized_signal)

        herd = self.crowd_weight * np.tanh(3 * crowd_sentiment)
        noise = self.rng.normal(0.0, self.noise_scale)
        persistence = _PERSISTENCE_WEIGHT * float(self.position)

        score = signal + herd + noise + persistence

        # --- asymmetric behavior ---
        if self.position != 0:
            if np.sign(score) != np.sign(self.position):
                if self.rng.random() < 0.7:
                    return
            else:
                score += 3.0 * np.sign(self.position)

        _ANCHOR_STRENGTH = 2.0
        _SWITCH_BASE_PROB = 1.0

        if abs(score) < _INERTIA_THRESHOLD:
            return

        direction = 1.0 if score > 0 else -1.0

        # Entry
        if self.position == 0:
            if self.rng.random() < 0.7:
                self.position = direction
            else:
                self.position = -direction
            return

        # Accumulation (with volatility-conditioned decay / release)
        if np.sign(self.position) == np.sign(direction):
            # Decay reduces the magnitude of accumulated position before accumulating
            lam = _DECAY_BASE + _DECAY_VOLATILITY_SCALE * float(volatility)
            lam = float(np.clip(lam, 0.0, _DECAY_CLIP_MAX))

            # Apply decay (release) to current accumulation state (continuous)
            if lam > 0.0 and self.position != 0:
                self.position = (1.0 - lam) * float(self.position)

            if abs(self.position) < 5:
                self.position = float(self.position) + direction
            return

        # Switching with anchoring
        anchor_bias = _ANCHOR_STRENGTH * float(self.position)
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
