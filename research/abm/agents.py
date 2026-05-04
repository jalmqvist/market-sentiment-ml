"""
agents.py

Stable agent definitions for ABM.
"""

from __future__ import annotations
import numpy as np


_PERSISTENCE_WEIGHT = 0.0
_INERTIA_THRESHOLD = 0.02


class BaseAgent:
    def __init__(self):
        self.last_signal = 0.0

    def apply_inertia(self, raw_signal: float) -> float:
        signal = (
            (1.0 - _PERSISTENCE_WEIGHT) * raw_signal
            + _PERSISTENCE_WEIGHT * self.last_signal
        )

        delta = signal - self.last_signal

        if _INERTIA_THRESHOLD > 0:
            scale = np.tanh(abs(delta) / (_INERTIA_THRESHOLD + 1e-8))
            signal = self.last_signal + scale * delta

        signal = np.clip(signal, -0.5, 0.5)

        self.last_signal = signal
        return signal

    def update(self, price_history, returns=None, sentiment=0.0):
        raise NotImplementedError


class TrendFollower(BaseAgent):
    def __init__(self, momentum_window=12):
        super().__init__()
        self.momentum_window = momentum_window

    def update(self, price_history, returns=None, sentiment=0.0):
        if len(price_history) < self.momentum_window + 1:
            return self.last_signal

        p0 = price_history[-self.momentum_window - 1]
        p1 = price_history[-1]

        if p0 <= 0:
            momentum = 0.0
        else:
            momentum = p1 / p0 - 1.0

        signal = np.tanh(momentum * 5.0) * 0.3
        return self.apply_inertia(signal)


class Contrarian(BaseAgent):
    def update(self, price_history, returns=None, sentiment=0.0):
        if returns is None or len(returns) == 0:
            return self.last_signal

        r = returns[-1]
        signal = -np.tanh(r * 8.0) * 0.1
        return self.apply_inertia(signal)


class NoiseTrader(BaseAgent):
    def __init__(self, rng=None):
        super().__init__()
        self.rng = rng or np.random.default_rng()

    def update(self, price_history, returns=None, sentiment=0.0):
        noise = self.rng.normal(0.0, 0.02)
        signal = np.tanh(0.1 * sentiment + noise)
        return self.apply_inertia(signal)


class RetailTrader(NoiseTrader):
    """
    Slightly more sentiment-sensitive version of NoiseTrader.
    Exists for compatibility + mild behavioral variation.
    """
    def update(self, price_history, returns=None, sentiment=0.0):
        noise = self.rng.normal(0.0, 0.02)
        signal = np.tanh(0.15 * sentiment + noise)
        return self.apply_inertia(signal)


def build_agents(n_agents, trend_ratio=0.5, rng=None):
    rng = rng or np.random.default_rng()
    agents = []

    n_trend = int(n_agents * trend_ratio)
    n_remaining = n_agents - n_trend
    n_contrarian = n_remaining // 2
    n_noise = n_remaining - n_contrarian

    for _ in range(n_trend):
        mw = int(rng.integers(6, 24))
        agents.append(TrendFollower(momentum_window=mw))

    for _ in range(n_contrarian):
        agents.append(Contrarian())

    for _ in range(n_noise):
        # mix Noise + Retail for heterogeneity
        if rng.random() < 0.5:
            agents.append(NoiseTrader(rng))
        else:
            agents.append(RetailTrader(rng))

    rng.shuffle(agents)
    return agents