from __future__ import annotations

import pandas as pd

from bsve.features.consensus import compute_consensus_maturity
from bsve.features.registry import FeatureDefinition, FeatureRegistry, build_default_registry
from bsve.features.volatility import compute_volatility_regime


def _feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": ["usd-jpy", "usd-jpy", "usd-jpy", "eur-chf"],
            "net_sentiment": [80.0, 85.0, 20.0, -30.0],
            "sentiment_extreme_flag": [True, True, False, False],
            "crowd_side": [1, 1, -1, -1],
            "volatility_regime": ["low", "low", "medium", "high"],
        }
    )


def test_consensus_maturity_placeholder_computation() -> None:
    maturity = compute_consensus_maturity(_feature_df())
    assert maturity.tolist()[:3] == [1, 2, 0]


def test_volatility_regime_returns_unclassified_without_thresholds() -> None:
    s = pd.Series([0.1, 0.2, 0.3])
    regime = compute_volatility_regime(s)
    assert regime.tolist() == ["unclassified", "unclassified", "unclassified"]


def test_registry_register_and_compute() -> None:
    registry = FeatureRegistry()
    registry.register(
        FeatureDefinition(
            name="demo",
            compute=lambda df: df["net_sentiment"],
            required_columns=("net_sentiment",),
        )
    )

    result = registry.compute("demo", _feature_df())
    assert len(result) == 4


def test_default_registry_exposes_foundation_features() -> None:
    registry = build_default_registry()
    names = registry.list_features()

    assert "consensus_maturity" in names
    assert "persistence_duration" in names
    assert "volatility_regime_persistence" in names
