"""Behavioral feature registry for shared, ontology-agnostic feature access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import consensus, persistence, volatility

FeatureComputer = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class FeatureDefinition:
    """Declarative feature definition for registry-based lookup."""

    name: str
    compute: FeatureComputer
    required_columns: tuple[str, ...] = ()
    description: str = ""


@dataclass
class FeatureRegistry:
    """Central registry of reusable behavioral feature computations."""

    _features: dict[str, FeatureDefinition] = field(default_factory=dict)

    def register(self, definition: FeatureDefinition) -> None:
        if definition.name in self._features:
            raise ValueError(f"feature already registered: {definition.name}")
        self._features[definition.name] = definition

    def get(self, name: str) -> FeatureDefinition:
        if name not in self._features:
            raise KeyError(f"unknown feature: {name}")
        return self._features[name]

    def list_features(self) -> list[str]:
        return sorted(self._features.keys())

    def compute(self, name: str, df: pd.DataFrame, **kwargs: object) -> pd.Series | pd.DataFrame:
        definition = self.get(name)
        missing = [c for c in definition.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"feature '{name}' missing required columns: {missing}")
        return definition.compute(df, **kwargs)


def build_default_registry() -> FeatureRegistry:
    """Build the default BSVE feature registry."""
    registry = FeatureRegistry()
    registry.register(
        FeatureDefinition(
            name="consensus_maturity",
            compute=consensus.compute_consensus_maturity,
            required_columns=("pair",),
            description="Consecutive-bar maturity within extreme consensus states.",
        )
    )
    registry.register(
        FeatureDefinition(
            name="consensus_velocity",
            compute=consensus.compute_consensus_velocity,
            required_columns=("pair", "net_sentiment"),
            description="First-difference sentiment velocity per pair.",
        )
    )
    registry.register(
        FeatureDefinition(
            name="persistence_duration",
            compute=persistence.compute_persistence_duration,
            required_columns=("pair", "crowd_side"),
            description="Consecutive-bar duration of a persistence signal.",
        )
    )
    registry.register(
        FeatureDefinition(
            name="transition_flag",
            compute=persistence.compute_transition_flag,
            required_columns=("pair", "crowd_side"),
            description="Transition indicator when the persistence signal changes.",
        )
    )
    registry.register(
        FeatureDefinition(
            name="volatility_regime_persistence",
            compute=volatility.compute_volatility_regime_persistence,
            required_columns=("pair", "volatility_regime"),
            description="Consecutive-bar persistence within volatility regimes.",
        )
    )
    return registry
