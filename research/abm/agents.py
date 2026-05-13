from __future__ import annotations

import os

import numpy as np


# Tunable parameters for RetailTrader.update()
_SIGNAL_AMPLIFICATION = 5.0
_PERSISTENCE_WEIGHT = 0.1
_INERTIA_THRESHOLD = 0.05

# Asymmetric reinforcement strength (positive feedback when already aligned).
# Backward-compatible default = 3.0.
_REINFORCE_STRENGTH = 3.0

# When score disagrees with current position, agents may "hold" and ignore the
# evidence with some probability (ratchet). Backward-compatible default = 0.7.
_DISAGREE_HOLD_PROB = 0.7

# Switching anchor strength (resistance to switching direction).
# Backward-compatible default = 2.0.
_ANCHOR_STRENGTH = 2.0

# Stage-2 release mechanism (backward compatible defaults: 0.0)
_DECAY_BASE = 0.0
_DECAY_VOLATILITY_SCALE = 0.0
_DECAY_CLIP_MAX = 0.2

# Stage-3 regime escape / de-alignment (backward compatible defaults: disabled)
_ESCAPE_PROB_SAT = 0.0
_ESCAPE_SAT_THRESHOLD = 0.7
_ESCAPE_SHRINK_FACTOR = 0.5
_ESCAPE_FLIP_PROB = 0.0
_ESCAPE_ZERO_PROB = 0.0
_ESCAPE_ZERO_COOLDOWN = 6


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return int(default)
    try:
        return int(float(v))
    except ValueError:
        return int(default)


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


# Environment overrides (optional)
_ESCAPE_PROB_SAT = _env_float("ABM_ESCAPE_PROB_SAT", _ESCAPE_PROB_SAT)
_ESCAPE_SAT_THRESHOLD = _env_float("ABM_ESCAPE_SAT_THRESHOLD", _ESCAPE_SAT_THRESHOLD)
_ESCAPE_SHRINK_FACTOR = _env_float("ABM_ESCAPE_SHRINK_FACTOR", _ESCAPE_SHRINK_FACTOR)
_ESCAPE_FLIP_PROB = _env_float("ABM_ESCAPE_FLIP_PROB", _ESCAPE_FLIP_PROB)
_ESCAPE_ZERO_PROB = _env_float("ABM_ESCAPE_ZERO_PROB", _ESCAPE_ZERO_PROB)
_ESCAPE_ZERO_COOLDOWN = _env_int("ABM_ESCAPE_ZERO_COOLDOWN", _ESCAPE_ZERO_COOLDOWN)

_REINFORCE_STRENGTH = _env_float("ABM_REINFORCE_STRENGTH", _REINFORCE_STRENGTH)
_DISAGREE_HOLD_PROB = _clip01(_env_float("ABM_DISAGREE_HOLD_PROB", _DISAGREE_HOLD_PROB))
_ANCHOR_STRENGTH = _env_float("ABM_ANCHOR_STRENGTH", _ANCHOR_STRENGTH)


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

        # Optional cooldown after forced neutralization escape, to keep the agent
        # neutral for a few steps (prevents immediate re-entry).
        self.escape_cooldown: int = 0

    def _price_signal(self, price_history: np.ndarray) -> float:
        raise NotImplementedError

    def _maybe_escape_regime(self, crowd_sentiment: float) -> None:
        if _ESCAPE_PROB_SAT <= 0.0:
            return
        if abs(float(crowd_sentiment)) <= float(_ESCAPE_SAT_THRESHOLD):
            return
        if self.position == 0.0:
            return

        if self.rng.random() >= float(_ESCAPE_PROB_SAT):
            return

        # Optional neutralization (off by default)
        if _ESCAPE_ZERO_PROB > 0.0 and self.rng.random() < float(_ESCAPE_ZERO_PROB):
            self.position = 0.0
            self.escape_cooldown = max(int(_ESCAPE_ZERO_COOLDOWN), 0)
            return

        # Optional sign de-alignment (off by default)
        if _ESCAPE_FLIP_PROB > 0.0 and self.rng.random() < float(_ESCAPE_FLIP_PROB):
            self.position = -1.0 * float(np.sign(self.position))
            return

        # Partial de-alignment: reduce magnitude but keep sign.
        self.position = float(self.position) * float(_ESCAPE_SHRINK_FACTOR)

    def update(
        self,
        price_history: np.ndarray,
        crowd_sentiment: float,
        volatility: float = 0.0,
    ) -> None:
        if len(price_history) < 2:
            return

        # If recently neutralized, hold out of the market for a few steps.
        if getattr(self, "escape_cooldown", 0) > 0:
            self.escape_cooldown -= 1
            return

        # Optional saturation-conditioned escape (Stage-3).
        self._maybe_escape_regime(crowd_sentiment)

        if self.position == 0.0 and getattr(self, "escape_cooldown", 0) > 0:
            return

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
                # Ratchet: ignore disagreeing evidence with some probability.
                if self.rng.random() < float(_DISAGREE_HOLD_PROB):
                    return
            else:
                # Positive feedback when aligned.
                score += float(_REINFORCE_STRENGTH) * np.sign(self.position)

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
            lam = _DECAY_BASE + _DECAY_VOLATILITY_SCALE * float(volatility)
            lam = float(np.clip(lam, 0.0, _DECAY_CLIP_MAX))

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

        # Implement contrarian behavior in the *normalized* signal frame by flipping
        # signal_sign after pair-level normalization. This avoids “double flip”
        # cancellation on USD-base pairs (e.g. usd-jpy).
        self.signal_sign *= -1

    def _price_signal(self, price_history: np.ndarray) -> float:
        # Same raw signal as TrendFollower; contrarian-ness is handled via signal_sign.
        window = min(self.momentum_window, len(price_history) - 1)
        ret = (price_history[-1] - price_history[-(window + 1)]) / (
            abs(price_history[-(window + 1)]) + 1e-12
        )
        return float(np.sign(ret))


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
