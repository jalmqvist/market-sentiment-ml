"""Ontology-agnostic behavioral surface generation engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from bsve.state_machine.protocol import BehavioralOntologyPlugin, CalibrationArtifact


def _normalize_crowd_side(value: Any) -> str:
    """Normalise a crowd-side value to a canonical string label.

    Handles both string labels (``"LONG"``, ``"SHORT"``) and the integer
    encoding used by the master research dataset (``1`` = LONG, ``-1`` = SHORT,
    ``0`` = neutral / no side).

    Returns one of ``"LONG"``, ``"SHORT"``, or ``""`` (neutral / unknown).

    Examples::

        _normalize_crowd_side(1)      # "LONG"
        _normalize_crowd_side(-1)     # "SHORT"
        _normalize_crowd_side(0)      # ""
        _normalize_crowd_side("LONG") # "LONG"
        _normalize_crowd_side("long") # "LONG"
        _normalize_crowd_side(None)   # ""
    """
    if value is None:
        return ""
    # Integer encoding: positive → LONG, negative → SHORT, zero → neutral.
    try:
        numeric = float(value)
        if numeric > 0:
            return "LONG"
        if numeric < 0:
            return "SHORT"
        return ""
    except (ValueError, TypeError):
        pass
    return str(value).strip().upper()

BEHAVIORAL_SURFACE_SCHEMA_VERSION = "1.0.0"


@dataclass
class _PairRuntime:
    """Runtime state tracked per pair during causal iteration."""

    last_timestamp: pd.Timestamp
    last_crowd_side: str
    last_consensus_active: bool
    last_maturity: int
    current_episode_id: str


class BehavioralSurfaceEngine:
    """Deterministic causal engine that generates a behavioral surface."""

    def __init__(
        self,
        *,
        plugin: BehavioralOntologyPlugin,
        calibration_artifact: CalibrationArtifact,
        pair_col: str = "pair",
        timestamp_col: str = "entry_time",
        crowd_side_col: str = "crowd_side",
        max_gap: str = "1h",
    ) -> None:
        self.plugin = plugin
        self.calibration_artifact = calibration_artifact
        self.pair_col = pair_col
        self.timestamp_col = timestamp_col
        self.crowd_side_col = crowd_side_col
        self.max_gap = pd.Timedelta(max_gap)
        self._pair_state: dict[str, _PairRuntime] = {}
        self._episode_counter = 0

    def _next_episode_id(self, pair: str) -> str:
        self._episode_counter += 1
        return f"{pair}:{self._episode_counter:08d}"

    def process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        pair = _pair_key(observation[self.pair_col])
        timestamp = pd.Timestamp(observation[self.timestamp_col])
        if pd.isna(timestamp):
            raise ValueError(f"invalid timestamp for pair={pair!r}: {observation[self.timestamp_col]!r}")

        crowd_side = _normalize_crowd_side(observation.get(self.crowd_side_col))

        # Expose the normalised crowd-side back into the observation dict so
        # that plugin methods receive canonical labels regardless of whether the
        # source dataset encodes crowd-side as integers or strings.
        normalised_observation = dict(observation)
        normalised_observation[self.crowd_side_col] = crowd_side

        consensus_active = self.plugin.is_consensus_active(normalised_observation, self.calibration_artifact)

        prior = self._pair_state.get(pair)
        if prior is None:
            boundary = True
        else:
            if timestamp <= prior.last_timestamp:
                raise ValueError(
                    f"non-chronological observations for pair {pair!r}: "
                    f"{timestamp} <= {prior.last_timestamp}"
                )

            gap_detected = (timestamp - prior.last_timestamp) > self.max_gap
            side_changed = crowd_side != prior.last_crowd_side
            extreme_changed = consensus_active != prior.last_consensus_active
            boundary = gap_detected or side_changed or extreme_changed

        if boundary:
            episode_id = self._next_episode_id(pair)
            maturity = 1 if consensus_active else 0
        else:
            episode_id = prior.current_episode_id
            maturity = prior.last_maturity + 1 if consensus_active else 0

        state = self.plugin.classify(normalised_observation, maturity, self.calibration_artifact)

        self._pair_state[pair] = _PairRuntime(
            last_timestamp=timestamp,
            last_crowd_side=crowd_side,
            last_consensus_active=consensus_active,
            last_maturity=maturity,
            current_episode_id=episode_id,
        )

        return {
            "timestamp": timestamp,
            "pair": pair,
            "state": state,
            "episode_id": episode_id,
            "maturity_bars": maturity,
            "crowd_side": crowd_side,
        }


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input dataset missing required columns: {missing}")


def _pair_key(value: Any) -> str:
    pair = str(value).strip()
    if not pair:
        raise ValueError("pair value must be a non-empty string")
    return pair


def generate_behavioral_surface(
    dataset: pd.DataFrame,
    *,
    plugin: BehavioralOntologyPlugin,
    calibration_artifact: CalibrationArtifact,
    dataset_version: str,
    pair_col: str = "pair",
    timestamp_col: str = "entry_time",
    crowd_side_col: str = "crowd_side",
) -> pd.DataFrame:
    """Generate one deterministic behavioral-state assignment per pair/timestamp row.

    Pair values are treated as already normalized by the dataset adapter.
    """
    _require_columns(dataset, [pair_col, timestamp_col, crowd_side_col, "net_sentiment"])

    working = dataset[[pair_col, timestamp_col, crowd_side_col, "net_sentiment"]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    if working[timestamp_col].isna().any():
        raise ValueError(f"column {timestamp_col!r} contains non-datetime values")

    working = working.sort_values([pair_col, timestamp_col], kind="mergesort").reset_index(drop=True)

    engine = BehavioralSurfaceEngine(
        plugin=plugin,
        calibration_artifact=calibration_artifact,
        pair_col=pair_col,
        timestamp_col=timestamp_col,
        crowd_side_col=crowd_side_col,
    )

    rows = [engine.process_observation(row) for row in working.to_dict(orient="records")]
    surface = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "pair",
            "state",
            "episode_id",
            "maturity_bars",
            "crowd_side",
        ],
    )

    surface.attrs["provenance"] = {
        "ontology_id": plugin.ontology_id,
        "ontology_version": plugin.ontology_version,
        "calibration_id": str(calibration_artifact.get("calibration_id", "")),
        "calibration_hash": str(calibration_artifact.get("artifact_hash", "")),
        "schema_version": BEHAVIORAL_SURFACE_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return surface


def build_behavioral_surface_manifest(surface: pd.DataFrame) -> dict[str, Any]:
    """Build reproducibility manifest for a behavioral surface DataFrame."""
    provenance = dict(surface.attrs.get("provenance", {}))

    return {
        "ontology_id": provenance.get("ontology_id", ""),
        "ontology_version": provenance.get("ontology_version", ""),
        "calibration_id": provenance.get("calibration_id", ""),
        "calibration_hash": provenance.get("calibration_hash", ""),
        "dataset_version": provenance.get("dataset_version", ""),
        "schema_version": provenance.get("schema_version", BEHAVIORAL_SURFACE_SCHEMA_VERSION),
        "generated_timestamp": provenance.get("generated_timestamp", ""),
        "row_count": int(len(surface)),
        "pair_counts": {str(k): int(v) for k, v in surface["pair"].value_counts().sort_index().items()},
        "state_counts": {str(k): int(v) for k, v in surface["state"].value_counts().sort_index().items()},
    }
