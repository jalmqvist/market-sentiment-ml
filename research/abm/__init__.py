"""
research/abm
============
Minimal agent-based model (ABM) for retail FX sentiment simulation.

The ABM reproduces the observed dynamics of retail crowd positioning
(net_sentiment) by simulating a population of heterogeneous traders.

Public API
----------
FXSentimentSimulation
    Core simulation class.  Instantiate with a mix of agents, then call
    ``run(n_steps)`` to obtain a per-step sentiment time series.

TrendFollower, Contrarian, NoiseTrader
    Concrete agent types.

calibrate_from_dataset, compare_to_data
    Helpers for comparing simulated output to the real research dataset.
"""

from research.abm.agents import Contrarian, NoiseTrader, RetailTrader, TrendFollower
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.simulation import FXSentimentSimulation

__all__ = [
    "RetailTrader",
    "TrendFollower",
    "Contrarian",
    "NoiseTrader",
    "FXSentimentSimulation",
    "calibrate_from_dataset",
    "compare_to_data",
]
