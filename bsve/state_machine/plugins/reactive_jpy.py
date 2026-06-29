"""Reactive-JPY ontology plugin for deterministic behavioral state classification."""

from __future__ import annotations

from typing import Any, Mapping

from bsve.state_machine.protocol import CalibrationArtifact, Observation


class ReactiveJPYPlugin:
    """Pure classifier for the frozen Reactive-JPY ontology."""

    ontology_id = "reactive_jpy"
    ontology_version = "1.0.0"

    _NON_EXTREME = "JPY_NON_EXTREME"
    _YOUNG = "JPY_CONSENSUS_YOUNG"
    _MATURING = "JPY_CONSENSUS_MATURING"
    _MATURE = "JPY_CONSENSUS_MATURE"

    def __init__(
        self,
        *,
        sentiment_col: str = "net_sentiment",
        crowd_side_col: str = "crowd_side",
    ) -> None:
        self.sentiment_col = sentiment_col
        self.crowd_side_col = crowd_side_col

    def _thresholds(self, calibration_artifact: CalibrationArtifact) -> tuple[float, int, int]:
        thresholds = calibration_artifact.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("calibration artifact thresholds are missing")

        extreme = thresholds.get("extreme_threshold_net_pct")
        young = thresholds.get("young_boundary_bars")
        mature = thresholds.get("mature_boundary_bars")
        if extreme is None or young is None or mature is None:
            raise ValueError(
                "calibration artifact missing one or more required thresholds: "
                "extreme_threshold_net_pct, young_boundary_bars, mature_boundary_bars"
            )
        return float(extreme), int(young), int(mature)

    def _sentiment(self, observation: Observation) -> float:
        value = observation.get(self.sentiment_col)
        if value is None:
            raise ValueError(f"observation missing required sentiment column {self.sentiment_col!r}")
        return float(value)

    def _crowd_side(self, observation: Observation) -> str:
        value = observation.get(self.crowd_side_col)
        if value is None:
            return ""
        return str(value).strip().upper()

    def is_consensus_active(
        self,
        observation: Observation,
        calibration_artifact: CalibrationArtifact,
    ) -> bool:
        extreme_threshold, _, _ = self._thresholds(calibration_artifact)
        sentiment = self._sentiment(observation)
        crowd_side = self._crowd_side(observation)
        return abs(sentiment) >= extreme_threshold and crowd_side in {"LONG", "SHORT"}

    def classify(
        self,
        observation: Observation,
        running_maturity: int,
        calibration_artifact: CalibrationArtifact,
    ) -> str:
        extreme_threshold, young_boundary, mature_boundary = self._thresholds(
            calibration_artifact
        )
        sentiment = self._sentiment(observation)

        if abs(sentiment) < extreme_threshold:
            return self._NON_EXTREME

        if running_maturity < young_boundary:
            return self._YOUNG
        if running_maturity < mature_boundary:
            return self._MATURING
        return self._MATURE
