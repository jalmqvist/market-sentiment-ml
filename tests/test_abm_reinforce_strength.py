from __future__ import annotations

import importlib

import numpy as np

import research.abm.agents as agents_module
from research.abm.simulation import FXSentimentSimulation


def _run_sim(reinforce_strength: float | None) -> np.ndarray:
    if reinforce_strength is None:
        agents_module.os.environ.pop("ABM_REINFORCE_STRENGTH", None)
    else:
        agents_module.os.environ["ABM_REINFORCE_STRENGTH"] = str(reinforce_strength)

    importlib.reload(agents_module)

    rng = np.random.default_rng(7)
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
    n_steps = 500
    prices = np.full(n_steps + sim.warmup_steps + 1, 110.0, dtype=float)
    return sim.run(n_steps=n_steps, price_series=prices)["net_sentiment"].to_numpy(float)


def test_reduced_reinforcement_breaks_plus100_lock() -> None:
    sentiments = _run_sim(reinforce_strength=0.0)
    assert not np.all(sentiments == 100.0)
    assert float(np.mean(sentiments > 0.0)) < 1.0


def test_default_reinforcement_backward_compatible_runs() -> None:
    sentiments = _run_sim(reinforce_strength=None)
    assert len(sentiments) == 500
    assert np.all(sentiments <= 100.0)
    assert np.all(sentiments >= -100.0)
