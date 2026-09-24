"""Persistent Commitment Lifecycle ontology plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from bsve.state_machine.engine import _PairRuntime
from bsve.state_machine.protocol import CalibrationArtifact, Observation


@dataclass(frozen=True)
class PersistentThresholds:
    """Typed threshold view extracted from calibration artifact."""

    level_q33: float
    level_q67: float
    trajectory_q33: float
    trajectory_q67: float
    min_level_history: int
    min_trajectory_history: int


class PersistentPlugin:
    """Classifier for the Persistent Commitment Lifecycle ontology."""

    ontology_id = "persistent"
    ontology_version = "0.1.0"

    SUPPORTED_PAIRS = {
        "eur-usd",
        "gbp-usd",
        "nzd-usd",
        "eur-gbp",
        "eur-aud",
    }

    def __init__(
        self,
        *,
        sentiment_col: str = "net_sentiment",
        crowd_side_col: str = "crowd_side",
    ) -> None:
        self.sentiment_col = sentiment_col
        self.crowd_side_col = crowd_side_col

    def _thresholds(self, calibration_artifact: CalibrationArtifact) -> PersistentThresholds:
        thresholds = calibration_artifact.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("calibration artifact thresholds are missing")

        required = (
            "level_q33",
            "level_q67",
            "trajectory_q33",
            "trajectory_q67",
            "min_level_history",
            "min_trajectory_history",
        )
        missing = [k for k in required if k not in thresholds]
        if missing:
            raise ValueError(f"calibration artifact missing persistent thresholds: {missing}")

        return PersistentThresholds(
            level_q33=float(thresholds["level_q33"]),
            level_q67=float(thresholds["level_q67"]),
            trajectory_q33=float(thresholds["trajectory_q33"]),
            trajectory_q67=float(thresholds["trajectory_q67"]),
            min_level_history=int(thresholds["min_level_history"]),
            min_trajectory_history=int(thresholds["min_trajectory_history"]),
        )

    def _depth(self, observation: Observation) -> float | None:
        value = observation.get(self.sentiment_col)
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(v):
            return None
        return abs(v)

    def _crowd_side(self, observation: Observation) -> str:
        value = observation.get(self.crowd_side_col)
        if value is None:
            return ""
        try:
            numeric = float(value)
            if numeric > 0:
                return "LONG"
            if numeric < 0:
                return "SHORT"
            return ""
        except (TypeError, ValueError):
            pass
        side = str(value).strip().upper()
        if side in {"LONG", "SHORT"}:
            return side
        return ""

    def is_consensus_active(
        self,
        observation: Observation,
        calibration_artifact: CalibrationArtifact,
    ) -> bool:
        return self._crowd_side(observation) in {"LONG", "SHORT"}

    def classify(
        self,
        observation: Observation,
        running_maturity: int,
        calibration_artifact: CalibrationArtifact,
    ) -> str:
        state = observation.get("__persistent_state_id")
        return str(state) if state is not None else ""

    def process_observation(
        self,
        *,
        observation: dict[str, Any],
        pair: str,
        timestamp: pd.Timestamp,
        prior: _PairRuntime | None,
        gap_detected: bool,
        calibration_artifact: CalibrationArtifact,
        next_episode_id: Callable[[], str],
    ) -> tuple[dict[str, Any], _PairRuntime]:
        if pair not in self.SUPPORTED_PAIRS:
            raise ValueError(f"unsupported persistent pair: {pair}")

        thresholds = self._thresholds(calibration_artifact)
        crowd_side = self._crowd_side(observation)
        depth = self._depth(observation)

        prior_side = "" if prior is None else str((prior.plugin_state or {}).get("crowd_side", ""))
        history = [] if prior is None else list((prior.plugin_state or {}).get("history", []))

        side_changed = (
            prior is not None
            and crowd_side in {"LONG", "SHORT"}
            and prior_side in {"LONG", "SHORT"}
            and crowd_side != prior_side
        )
        if prior is None:
            episode_id = next_episode_id()
            maturity = 1
            history = []
        elif side_changed:
            episode_id = next_episode_id()
            maturity = 1
            history = []
        elif gap_detected:
            # A hard observational gap creates a new episode_id even when
            # crowd_side is unchanged across the gap. This differs from the
            # P0C canonical episode representation (1,421 episodes) — the
            # surface episode_id represents observed segments, not canonical
            # P0C episodes. Documented in PERSISTENT_BSVE_SURFACE_ROADMAP.md
            # Decision Log.
            episode_id = next_episode_id()
            maturity = 1
            history = []
        else:
            episode_id = prior.current_episode_id
            maturity = prior.last_maturity + 1

        prior_depths = list(history)

        level: float | None = None
        trajectory: float | None = None

        if depth is not None and crowd_side in {"LONG", "SHORT"}:
            if len(prior_depths) >= thresholds.min_level_history:
                level = float(pd.Series(prior_depths).mean())

            if len(prior_depths) >= thresholds.min_trajectory_history:
                split_idx = len(prior_depths) // 2
                early = prior_depths[:split_idx]
                late = prior_depths[split_idx:]
                if early and late:
                    trajectory = float(pd.Series(late).mean() - pd.Series(early).mean())

        state_id: str | None = None
        if level is not None and trajectory is not None:
            state_id = self._state_id(level=level, trajectory=trajectory, thresholds=thresholds)

        if prior is None:
            transition_event = "entry"
        elif side_changed:
            # exit_reversal is assigned to the first bar of the new crowd-side
            # episode (the bar that caused the reversal). This differs from
            # Reactive-JPY where exit_reversal labels the first non-extreme bar
            # after consensus ends. Both conventions are internally consistent
            # within their respective ontologies.
            transition_event = "exit_reversal"
        elif gap_detected:
            transition_event = "entry"
        elif crowd_side not in {"LONG", "SHORT"}:
            transition_event = "exit_unknown"
        elif prior.last_consensus_active and prior.last_state_id:
            if state_id and prior.last_state_id != state_id:
                transition_event = "state_transition"
            else:
                transition_event = "continuation"
        else:
            transition_event = "exit_unknown"

        if depth is not None and crowd_side in {"LONG", "SHORT"}:
            history = prior_depths + [depth]
        else:
            history = []

        runtime = _PairRuntime(
            last_timestamp=timestamp,
            last_consensus_active=crowd_side in {"LONG", "SHORT"},
            last_maturity=maturity,
            current_episode_id=episode_id,
            last_state_id=state_id,
            plugin_state={"history": history, "crowd_side": crowd_side},
        )

        row = {
            "timestamp": timestamp,
            "pair": pair,
            "surface_id": self.ontology_id,
            "surface_version": self.ontology_version,
            "state_id": state_id,
            "episode_id": episode_id,
            "maturity_bars": maturity,
            "crowd_side": crowd_side,
            "transition_event": transition_event,
        }
        return row, runtime

    def _state_id(
        self,
        *,
        level: float,
        trajectory: float,
        thresholds: PersistentThresholds,
    ) -> str:
        level_bin = self._bin3(level, thresholds.level_q33, thresholds.level_q67)
        trajectory_bin = self._bin3(
            trajectory,
            thresholds.trajectory_q33,
            thresholds.trajectory_q67,
        )
        return f"PERSISTENT_{level_bin}{trajectory_bin}"

    @staticmethod
    def _bin3(value: float, q33: float, q67: float) -> str:
        if value < q33:
            return "L"
        if value < q67:
            return "M"
        return "H"
