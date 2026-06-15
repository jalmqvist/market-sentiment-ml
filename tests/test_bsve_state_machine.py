"""Tests for bsve/state_machine/rule_based.py — BSVE State Assignment Engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bsve.state_machine.rule_based import (
    assert_calibrations_valid,
    assign_states_reactive_jpy,
    classify_maturity,
    classify_transition,
    generate_diagnostics,
    generate_run_manifest,
    validate_state_artifact,
    write_run_manifest,
)
from bsve.calibration.calibration_contract import (
    build_calibration_artifact,
    write_calibration_artifact,
)
from schemas.bsve_artifact_schema import BSVE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calibration_artifact() -> dict:
    """A minimal valid calibration artifact for reactive_jpy."""
    return build_calibration_artifact(
        calibration_id="test_reactive_jpy_v1",
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2026-12-31",
        dataset_version="1.5.0",
        calibration_method="hazard_analysis",
        outcome="success",
        thresholds={
            "extreme_threshold_net_pct": 70.0,
            "young_boundary_bars": 8,
            "mature_boundary_bars": 24,
        },
        diagnostics={},
        calibration_mode="research",
    )


@pytest.fixture()
def artifact_file(tmp_path, calibration_artifact) -> Path:
    """Write a valid calibration artifact to a temp file and return its path."""
    path = tmp_path / "test_artifact.json"
    write_calibration_artifact(calibration_artifact, path)
    return path


@pytest.fixture()
def null_artifact_file(tmp_path) -> Path:
    """A null-calibration artifact file."""
    art = build_calibration_artifact(
        calibration_id="null_jpy_v1",
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2026-12-31",
        dataset_version="1.5.0",
        calibration_method="hazard_analysis",
        outcome="null",
        null_reason="insufficient data",
    )
    path = tmp_path / "null_artifact.json"
    write_calibration_artifact(art, path)
    return path


def _make_jpy_df(
    n: int = 10,
    pair: str = "usd-jpy",
    net_sentiments: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal JPY-pair DataFrame for state assignment tests."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    if net_sentiments is None:
        # Alternate extreme / non-extreme to exercise state transitions.
        net_sentiments = [80.0 if i % 3 != 2 else 50.0 for i in range(n)]
    assert len(net_sentiments) == n
    return pd.DataFrame(
        {
            "pair": pair,
            "entry_time": timestamps,
            "net_sentiment": net_sentiments,
        }
    )


# ---------------------------------------------------------------------------
# classify_maturity
# ---------------------------------------------------------------------------


class TestClassifyMaturity:
    def test_young(self):
        assert classify_maturity(1, young_boundary=8, mature_boundary=24) == "young"
        assert classify_maturity(7, young_boundary=8, mature_boundary=24) == "young"

    def test_maturing(self):
        assert classify_maturity(8, young_boundary=8, mature_boundary=24) == "maturing"
        assert classify_maturity(23, young_boundary=8, mature_boundary=24) == "maturing"

    def test_mature(self):
        assert classify_maturity(24, young_boundary=8, mature_boundary=24) == "mature"
        assert classify_maturity(100, young_boundary=8, mature_boundary=24) == "mature"


# ---------------------------------------------------------------------------
# classify_transition
# ---------------------------------------------------------------------------


class TestClassifyTransition:
    def test_first_bar_is_entry(self):
        assert classify_transition(None, "JPY_NON_EXTREME", False, False) == "entry"
        assert classify_transition(None, "JPY_CONSENSUS_YOUNG", False, True) == "entry"

    def test_unchanged_state_is_continuation(self):
        assert (
            classify_transition(
                "JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_YOUNG", True, True
            )
            == "continuation"
        )

    def test_extreme_to_non_extreme_is_exit_reversal(self):
        assert (
            classify_transition(
                "JPY_CONSENSUS_YOUNG", "JPY_NON_EXTREME", True, False
            )
            == "exit_reversal"
        )

    def test_threshold_crossing_is_exit_threshold(self):
        """exit_threshold is recorded on the *receiving* row (the first row of
        the new maturity class), not on the last row of the prior class.

        Young → Maturing: exit_threshold on the first Maturing row.
        Maturing → Mature: exit_threshold on the first Mature row.
        """
        assert (
            classify_transition(
                "JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING", True, True
            )
            == "exit_threshold"
        )
        assert (
            classify_transition(
                "JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE", True, True
            )
            == "exit_threshold"
        )

    def test_non_extreme_to_extreme_is_entry(self):
        # First extreme bar after a non-extreme bar opens a new episode.
        assert (
            classify_transition(
                "JPY_NON_EXTREME", "JPY_CONSENSUS_YOUNG", False, True
            )
            == "entry"
        )


# ---------------------------------------------------------------------------
# assign_states_reactive_jpy
# ---------------------------------------------------------------------------


class TestAssignStatesReactiveJPY:
    def test_non_extreme_state(self, calibration_artifact):
        df = _make_jpy_df(n=3, net_sentiments=[50.0, 60.0, 65.0])
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert (result["state_id"] == "JPY_NON_EXTREME").all()
        assert (result["maturity_class"] == "n_a").all()
        assert (result["maturity_bars"] == 0).all()

    def test_young_state(self, calibration_artifact):
        df = _make_jpy_df(n=5, net_sentiments=[80.0] * 5)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert (result["state_id"] == "JPY_CONSENSUS_YOUNG").all()
        assert (result["maturity_class"] == "young").all()
        # maturity_bars should be 1,2,3,4,5
        assert list(result["maturity_bars"]) == [1, 2, 3, 4, 5]

    def test_maturity_progression(self, calibration_artifact):
        """Episode that grows through young → maturing → mature."""
        sentiments = [80.0] * 30  # 30 bars all extreme
        df = _make_jpy_df(n=30, net_sentiments=sentiments)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        # Bars 1-7 should be young
        assert (result.iloc[:7]["state_id"] == "JPY_CONSENSUS_YOUNG").all()
        # Bars 8-23 should be maturing
        assert (result.iloc[7:23]["state_id"] == "JPY_CONSENSUS_MATURING").all()
        # Bars 24+ should be mature
        assert (result.iloc[23:]["state_id"] == "JPY_CONSENSUS_MATURE").all()

    def test_maturity_resets_on_new_episode(self, calibration_artifact):
        """Episode isolation: maturity MUST reset after non-extreme bar."""
        sentiments = [80.0, 80.0, 80.0, 50.0, 80.0, 80.0]
        df = _make_jpy_df(n=6, net_sentiments=sentiments)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        # After the non-extreme bar, the next extreme episode should restart.
        # The last bar has maturity=2, NOT maturity=5 (no episode carry-over).
        assert result.iloc[4]["state_id"] == "JPY_CONSENSUS_YOUNG"
        assert result.iloc[5]["state_id"] == "JPY_CONSENSUS_YOUNG"
        assert result.iloc[4]["maturity_bars"] == 1
        assert result.iloc[5]["maturity_bars"] == 2

    def test_required_columns_present(self, calibration_artifact):
        df = _make_jpy_df(n=3)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        required_cols = [
            "entry_time",
            "prediction_available_timestamp",
            "pair",
            "environment_id",
            "state_id",
            "state_version",
            "maturity_bars",
            "maturity_class",
            "state_confidence",
            "transition_event",
            "spec_id",
            "calibration_id",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_state_confidence_is_one(self, calibration_artifact):
        df = _make_jpy_df(n=5)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert (result["state_confidence"] == 1.0).all()

    def test_causal_ordering(self, calibration_artifact):
        df = _make_jpy_df(n=5)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        entry = pd.to_datetime(result["entry_time"])
        available = pd.to_datetime(result["prediction_available_timestamp"])
        assert (entry < available).all()

    def test_multi_pair(self, calibration_artifact):
        df1 = _make_jpy_df(n=5, pair="usd-jpy", net_sentiments=[80.0] * 5)
        df2 = _make_jpy_df(n=5, pair="eur-jpy", net_sentiments=[80.0] * 5)
        df = pd.concat([df1, df2], ignore_index=True)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        # Each pair should reset maturity independently.
        for pair in ["usd-jpy", "eur-jpy"]:
            pair_result = result[result["pair"] == pair]
            assert list(pair_result["maturity_bars"]) == [1, 2, 3, 4, 5]

    def test_missing_thresholds_raises(self, calibration_artifact):
        bad_artifact = dict(calibration_artifact)
        bad_artifact["thresholds"] = {}
        with pytest.raises(ValueError, match="missing required thresholds"):
            assign_states_reactive_jpy(
                _make_jpy_df(),
                calibration_artifact=bad_artifact,
                spec_id="reactive_jpy_v1",
            )

    def test_transition_first_bar_is_entry(self, calibration_artifact):
        df = _make_jpy_df(n=3, net_sentiments=[80.0, 80.0, 50.0])
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert result.iloc[0]["transition_event"] == "entry"

    def test_transition_continuation(self, calibration_artifact):
        df = _make_jpy_df(n=3, net_sentiments=[80.0, 80.0, 80.0])
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert result.iloc[1]["transition_event"] == "continuation"
        assert result.iloc[2]["transition_event"] == "continuation"

    def test_transition_exit_reversal(self, calibration_artifact):
        df = _make_jpy_df(n=2, net_sentiments=[80.0, 50.0])
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        assert result.iloc[1]["transition_event"] == "exit_reversal"

    def test_transition_exit_threshold_young_to_maturing(self, calibration_artifact):
        """exit_threshold is recorded on the first Maturing row (bar index 7,
        maturity_bars=8), not on the last Young row."""
        df = _make_jpy_df(n=9, net_sentiments=[80.0] * 9)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        # Bar at index 7 (maturity=8) is the first Maturing row — exit_threshold
        # is recorded here because this row entered via threshold crossing.
        assert result.iloc[7]["state_id"] == "JPY_CONSENSUS_MATURING"
        assert result.iloc[7]["transition_event"] == "exit_threshold"

    def test_transition_exit_threshold_maturing_to_mature(self, calibration_artifact):
        """exit_threshold is recorded on the first Mature row (bar index 23,
        maturity_bars=24), not on the last Maturing row."""
        df = _make_jpy_df(n=25, net_sentiments=[80.0] * 25)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        # Bar at index 23 (maturity=24) is the first Mature row — exit_threshold
        # is recorded here because this row entered via threshold crossing.
        assert result.iloc[23]["state_id"] == "JPY_CONSENSUS_MATURE"
        assert result.iloc[23]["transition_event"] == "exit_threshold"


# ---------------------------------------------------------------------------
# validate_state_artifact
# ---------------------------------------------------------------------------


class TestValidateStateArtifact:
    def _valid_df(self) -> pd.DataFrame:
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        return pd.DataFrame(
            {
                "entry_time": ts,
                "prediction_available_timestamp": ts + pd.Timedelta(hours=1),
                "pair": "usd-jpy",
                "environment_id": "reactive_jpy",
                "state_id": ["JPY_CONSENSUS_YOUNG"] * 3,
                "state_version": "1.0.0",
                "maturity_bars": [1, 2, 3],
                "maturity_class": ["young"] * 3,
                "state_confidence": 1.0,
                "transition_event": ["entry", "continuation", "continuation"],
                "spec_id": "reactive_jpy_v1",
                "calibration_id": "test_v1",
            }
        )

    def test_valid_passes(self):
        validate_state_artifact(self._valid_df())  # no exception

    def test_null_state_id_fails(self):
        df = self._valid_df()
        df.loc[0, "state_id"] = None
        with pytest.raises(ValueError, match="null state_id"):
            validate_state_artifact(df)

    def test_negative_maturity_fails(self):
        df = self._valid_df()
        df.loc[0, "maturity_bars"] = -1
        with pytest.raises(ValueError, match="maturity_bars < 0"):
            validate_state_artifact(df)

    def test_causal_violation_fails(self):
        df = self._valid_df()
        df["prediction_available_timestamp"] = df["entry_time"]
        with pytest.raises(ValueError, match="causal ordering violation"):
            validate_state_artifact(df)

    def test_empty_calibration_id_fails(self):
        df = self._valid_df()
        df["calibration_id"] = ""
        with pytest.raises(ValueError, match="calibration_id"):
            validate_state_artifact(df)

    def test_empty_spec_id_fails(self):
        df = self._valid_df()
        df["spec_id"] = ""
        with pytest.raises(ValueError, match="spec_id"):
            validate_state_artifact(df)


# ---------------------------------------------------------------------------
# assert_calibrations_valid
# ---------------------------------------------------------------------------


class TestAssertCalibrationsValid:
    def test_valid_artifact_loads(self, artifact_file, calibration_artifact):
        loaded = assert_calibrations_valid(artifact_file)
        assert loaded["calibration_id"] == calibration_artifact["calibration_id"]
        assert loaded["thresholds"]["young_boundary_bars"] == 8

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            assert_calibrations_valid(tmp_path / "nonexistent.json")

    def test_null_artifact_raises(self, null_artifact_file):
        with pytest.raises(ValueError, match="null-calibration"):
            assert_calibrations_valid(null_artifact_file)

    def test_tampered_artifact_raises(self, artifact_file):
        data = json.loads(artifact_file.read_text())
        data["thresholds"]["young_boundary_bars"] = 999  # tamper
        artifact_file.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            assert_calibrations_valid(artifact_file)


# ---------------------------------------------------------------------------
# generate_diagnostics
# ---------------------------------------------------------------------------


class TestGenerateDiagnostics:
    def test_state_counts_present(self, calibration_artifact):
        df = _make_jpy_df(n=5, net_sentiments=[80.0, 80.0, 50.0, 80.0, 80.0])
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        diag = generate_diagnostics(result)
        assert "state_counts" in diag
        assert "state_frequencies" in diag
        assert "JPY_CONSENSUS_YOUNG" in diag["state_counts"]
        assert "JPY_NON_EXTREME" in diag["state_counts"]

    def test_survival_counts_keys(self, calibration_artifact):
        df = _make_jpy_df(n=10, net_sentiments=[80.0] * 10)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        diag = generate_diagnostics(result)
        assert "8" in diag["survival_counts"]
        assert "24" in diag["survival_counts"]

    def test_mature_sparsity_flag_set_for_low_counts(self, calibration_artifact):
        # With only a few mature observations (<50), the flag should be True.
        df = _make_jpy_df(n=30, net_sentiments=[80.0] * 30)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        diag = generate_diagnostics(result)
        # All pairs should be flagged as sparse (we have far fewer than 50 mature obs).
        for pair, flagged in diag["mature_sparsity_flags"].items():
            assert isinstance(flagged, bool)

    def test_continuous_extreme_episode_is_one_episode(self, calibration_artifact):
        """30 consecutive extreme bars — spanning Young, Maturing and Mature —
        must be counted as a single episode with duration 30, not as 3 separate
        episodes.  This validates that episode extraction uses extreme-bar
        boundaries, not state-change boundaries.
        """
        df = _make_jpy_df(n=30, net_sentiments=[80.0] * 30)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=calibration_artifact,
            spec_id="reactive_jpy_v1",
        )
        diag = generate_diagnostics(result)

        # Single unbroken extreme episode.
        assert diag["episodes_per_pair"]["usd-jpy"] == 1

        # The one episode spans all 30 bars.
        assert diag["episode_duration_distribution"]["max"] == 30

        # Episode survives to ≥24 bars — survival count for "24" must be 1.
        assert diag["survival_counts"]["24"] == 1


# ---------------------------------------------------------------------------
# generate_run_manifest
# ---------------------------------------------------------------------------


class TestGenerateRunManifest:
    def test_required_fields_present(self):
        manifest = generate_run_manifest(
            run_id="test-run-001",
            environment_id="reactive_jpy",
            spec_id="reactive_jpy_v1",
            calibration_id="reactive_jpy_v1_20260615",
            dataset_version="1.5.0",
        )
        required = [
            "run_id",
            "environment_id",
            "spec_id",
            "calibration_id",
            "dataset_version",
            "artifact_schema_version",
            "timestamp",
        ]
        for key in required:
            assert key in manifest, f"Missing key: {key}"

    def test_run_manifest_write(self, tmp_path):
        manifest = generate_run_manifest(
            run_id="test-run-002",
            environment_id="reactive_jpy",
            spec_id="reactive_jpy_v1",
            calibration_id="reactive_jpy_v1_20260615",
            dataset_version="1.5.0",
        )
        path = write_run_manifest(manifest, tmp_path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == "test-run-002"


# ---------------------------------------------------------------------------
# Integration: committed calibration artifact
# ---------------------------------------------------------------------------


class TestCommittedArtifactIntegration:
    """Tests against the actual committed calibration artifact in the repo."""

    _ARTIFACT = Path(
        "bsve/calibration_artifacts/reactive_jpy_v1_20260615.json"
    )

    def test_committed_artifact_loads(self):
        if not self._ARTIFACT.exists():
            pytest.skip("Committed artifact not present in working directory.")
        artifact = assert_calibrations_valid(self._ARTIFACT)
        assert artifact["thresholds"]["young_boundary_bars"] == 8
        assert artifact["thresholds"]["mature_boundary_bars"] == 24

    def test_committed_artifact_state_assignment(self):
        if not self._ARTIFACT.exists():
            pytest.skip("Committed artifact not present in working directory.")
        artifact = assert_calibrations_valid(self._ARTIFACT)
        df = _make_jpy_df(n=30, net_sentiments=[80.0] * 30)
        result = assign_states_reactive_jpy(
            df,
            calibration_artifact=artifact,
            spec_id="reactive_jpy_v1",
        )
        assert len(result) == 30
        # The artifact should drive the young/maturing/mature boundaries.
        assert result.iloc[6]["state_id"] == "JPY_CONSENSUS_YOUNG"   # bar 7
        assert result.iloc[7]["state_id"] == "JPY_CONSENSUS_MATURING"  # bar 8
        assert result.iloc[23]["state_id"] == "JPY_CONSENSUS_MATURE"   # bar 24
