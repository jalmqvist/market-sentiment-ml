"""Protocol contracts for behavioral surface ontology plugins."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypedDict, runtime_checkable


class CalibrationArtifact(TypedDict, total=False):
    """Top-level calibration artifact fields used by state-machine components."""

    calibration_id: str
    artifact_hash: str
    thresholds: Mapping[str, Any]


Observation = Mapping[str, Any]


@runtime_checkable
class BehavioralOntologyPlugin(Protocol):
    """Contract for deterministic ontology classifiers."""

    ontology_id: str
    ontology_version: str

    def is_consensus_active(
        self,
        observation: Observation,
        calibration_artifact: CalibrationArtifact,
    ) -> bool:
        """Return whether the current row is inside an active consensus episode."""
        ...

    def classify(
        self,
        observation: Observation,
        running_maturity: int,
        calibration_artifact: CalibrationArtifact,
    ) -> str:
        """Classify exactly one behavioral state for this row."""
        ...
