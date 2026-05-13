from __future__ import annotations

import importlib
import os

import numpy as np

import research.abm.agents as agents_module
from research.abm.simulation import FXSentimentSimulation

_TEST_SEED = 7
_SIMULATION_STEPS = 500


def _run_sim(reinforce_strength: float | None) -> np.ndarray:
    prev = os.environ.get("ABM_REINFORCE_STRENGTH")
    try:
        if reinforce_strength is None:
            os.environ.pop("ABM_REINFORCE_STRENGTH", None)
        else:
            os.environ["ABM_REINFORCE_STRENGTH"] = str(reinforce_strength)

        importlib.reload(agents_module)

        rng = np.random.default_rng(_TEST_SEED)
        agents = [
            agents_module.TrendFollower(rng, pair="usd-jpy", momentum_window=3, noise_scale=0.0)
            for _ in range(50)
        ] + [
            agents_module.Contrarian(rng, pair="usd-jpy", momentum_window=3, noise_scale=0.0)
            for _ in range(50)
        ]

        for agent in agents:
            agent.position = 5e-7

        sim = FXSentimentSimulation(agents, rng=rng)
        prices = np.full(_SIMULATION_STEPS + sim.warmup_steps + 1, 110.0, dtype=float)
        return sim.run(n_steps=_SIMULATION_STEPS, price_series=prices)["net_sentiment"].to_numpy(float)
    finally:
        if prev is None:
            os.environ.pop("ABM_REINFORCE_STRENGTH", None)
        else:
            os.environ["ABM_REINFORCE_STRENGTH"] = prev
        importlib.reload(agents_module)


def test_zero_reinforcement_enables_sentiment_variation() -> None:
    sentiments = _run_sim(reinforce_strength=0.0)
    assert np.any(sentiments < 100.0)
    assert np.any(sentiments == 0.0)


def test_default_reinforcement_backward_compatible_runs() -> None:
    sentiments = _run_sim(reinforce_strength=None)
    assert len(sentiments) == _SIMULATION_STEPS
    assert np.all(sentiments <= 100.0)
    assert np.all(sentiments >= -100.0)
