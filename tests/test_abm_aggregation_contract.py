from __future__ import annotations

import numpy as np
import pytest

from research.abm.simulation import FXSentimentSimulation, _AGGREGATION_EPS


class _DummyAgent:
    def __init__(self, position: float) -> None:
        self.position = float(position)

    def update(self, price_history: np.ndarray, crowd_sentiment: float, volatility: float = 0.0) -> None:
        return


def _run_once(positions: list[float]) -> tuple[float, float]:
    agents = [_DummyAgent(p) for p in positions]
    sim = FXSentimentSimulation(agents=agents, rng=np.random.default_rng(0), warmup_steps=0)
    df = sim.run(n_steps=1, price_series=np.array([1.0, 1.0], dtype=np.float64))
    row = df.iloc[0]
    assert -100.0 <= row["net_sentiment"] <= 100.0
    assert row["abs_sentiment"] == pytest.approx(abs(row["net_sentiment"]))
    return float(row["crowd_side"]), float(row["net_sentiment"])


def _expected_net(positions: list[float]) -> float:
    arr = np.array(positions, dtype=np.float64)
    votes = np.zeros_like(arr)
    votes[arr > _AGGREGATION_EPS] = 1.0
    votes[arr < -_AGGREGATION_EPS] = -1.0
    return float(votes.mean() * 100.0)


@pytest.mark.parametrize(
    ("positions", "expected_side"),
    [
        ([1e9, 1e6, -2.0, 0.0], 1),
        ([-1e9, -1e6, 2.0, 0.0], -1),
        ([1e-9, -1e-9, 0.0, _AGGREGATION_EPS / 2], 0),
    ],
)
def test_abm_output_contract_for_continuous_positions(positions: list[float], expected_side: int):
    crowd_side, net_sentiment = _run_once(positions)
    assert net_sentiment == pytest.approx(_expected_net(positions))
    assert int(crowd_side) == expected_side
