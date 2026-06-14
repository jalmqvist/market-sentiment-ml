"""
Comprehensive tests for the BSVE PR2 Calibration Framework.

Covers:
    * Successful calibration artifacts (build, validate, load, write).
    * Null calibration artifacts (first-class, validated identically).
    * Schema validation and fail-fast behaviour.
    * CalibrationPlugin protocol and CalibrationRegistry.
    * CalibrationRunner end-to-end execution with a placeholder plugin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    import yaml as _yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

requires_yaml = pytest.mark.skipif(not _YAML_AVAILABLE, reason="pyyaml not installed")

from bsve.calibration.calibration_contract import (
    CALIBRATION_SCHEMA_VERSION,
    REQUIRED_METADATA_KEYS,
    build_calibration_artifact,
    load_calibration_artifact,
    validate_calibration_artifact,
    write_calibration_artifact,
)
from bsve.calibration.registry import (
    CalibrationPlugin,
    CalibrationRegistry,
    get_default_registry,
)
from bsve.calibration.calibration_runner import CalibrationRunner, load_state_spec


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _success_artifact(**overrides: Any):
    """Build a minimal valid success artifact."""
    base = dict(
        calibration_id="test_calib_001",
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2023-12-31",
        dataset_version="1.3.2",
        calibration_method="hazard_analysis",
        outcome="success",
        thresholds={"young_boundary_bars": 12, "mature_boundary_bars": 48},
        diagnostics={"n_episodes": 200},
    )
    base.update(overrides)
    return build_calibration_artifact(**base)


def _null_artifact(**overrides: Any):
    """Build a minimal valid null artifact."""
    base = dict(
        calibration_id="test_null_001",
        ontology_id="reactive_chf",
        ontology_version="1.0.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2023-12-31",
        dataset_version="1.3.2",
        calibration_method="percentile_jenks",
        outcome="null",
        null_reason="Insufficient episodes for reliable calibration",
    )
    base.update(overrides)
    return build_calibration_artifact(**base)


# ---------------------------------------------------------------------------
# 1. Successful calibration artifacts
# ---------------------------------------------------------------------------


class TestSuccessArtifact:
    def test_build_returns_all_required_metadata_keys(self):
        artifact = _success_artifact()
        assert REQUIRED_METADATA_KEYS.issubset(set(artifact.keys()))

    def test_build_sets_correct_schema_version(self):
        artifact = _success_artifact()
        assert artifact["schema_version"] == CALIBRATION_SCHEMA_VERSION

    def test_build_sets_outcome_success(self):
        artifact = _success_artifact()
        assert artifact["outcome"] == "success"

    def test_build_attaches_artifact_hash(self):
        artifact = _success_artifact()
        assert isinstance(artifact["artifact_hash"], str)
        assert len(artifact["artifact_hash"]) == 64  # SHA-256 hex

    def test_validate_passes_for_valid_artifact(self):
        artifact = _success_artifact()
        violations = validate_calibration_artifact(artifact, strict=False)
        assert violations == []

    def test_validate_strict_does_not_raise_for_valid_artifact(self):
        artifact = _success_artifact()
        validate_calibration_artifact(artifact, strict=True)  # must not raise

    def test_thresholds_present_for_success(self):
        artifact = _success_artifact()
        assert artifact["thresholds"]

    def test_diagnostics_stored(self):
        artifact = _success_artifact()
        assert artifact["diagnostics"]["n_episodes"] == 200


# ---------------------------------------------------------------------------
# 2. Null calibration artifacts
# ---------------------------------------------------------------------------


class TestNullArtifact:
    def test_build_null_artifact(self):
        artifact = _null_artifact()
        assert artifact["outcome"] == "null"

    def test_null_artifact_has_all_required_metadata_keys(self):
        artifact = _null_artifact()
        assert REQUIRED_METADATA_KEYS.issubset(set(artifact.keys()))

    def test_null_artifact_passes_validation(self):
        artifact = _null_artifact()
        violations = validate_calibration_artifact(artifact, strict=False)
        assert violations == []

    def test_null_artifact_carries_null_reason(self):
        artifact = _null_artifact()
        assert artifact["null_reason"]

    def test_null_artifact_hash_verifies(self):
        artifact = _null_artifact()
        # Mutation should break the hash
        artifact2 = dict(artifact)
        artifact2["null_reason"] = "tampered"
        violations = validate_calibration_artifact(artifact2, strict=False)
        assert any("hash" in v for v in violations)


# ---------------------------------------------------------------------------
# 3. Schema validation and fail-fast behaviour
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_missing_required_key_raises_in_strict_mode(self):
        artifact = _success_artifact()
        del artifact["calibration_id"]
        with pytest.raises(ValueError, match="missing required metadata keys"):
            validate_calibration_artifact(artifact, strict=True)

    def test_missing_required_key_collected_in_non_strict_mode(self):
        artifact = _success_artifact()
        del artifact["calibration_id"]
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("missing required metadata keys" in v for v in violations)

    def test_wrong_schema_version_fails(self):
        artifact = _success_artifact()
        artifact["schema_version"] = "0.0.0"
        # Recompute hash so that's not the failing point
        from bsve.calibration.calibration_contract import _compute_artifact_hash
        artifact["artifact_hash"] = _compute_artifact_hash(artifact)
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("schema_version mismatch" in v for v in violations)

    def test_invalid_outcome_fails(self):
        artifact = _success_artifact()
        artifact["outcome"] = "partial"
        from bsve.calibration.calibration_contract import _compute_artifact_hash
        artifact["artifact_hash"] = _compute_artifact_hash(artifact)
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("outcome" in v for v in violations)

    def test_tampered_hash_fails(self):
        artifact = _success_artifact()
        artifact["thresholds"]["extra"] = 999  # mutate after hash set
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("hash" in v for v in violations)

    def test_tampered_hash_raises_in_strict_mode(self):
        artifact = _success_artifact()
        artifact["thresholds"]["extra"] = 999
        with pytest.raises(ValueError, match="artifact_hash"):
            validate_calibration_artifact(artifact, strict=True)

    def test_window_start_after_end_fails(self):
        artifact = _success_artifact(
            calibration_window_start="2023-12-31",
            calibration_window_end="2019-01-01",
        )
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("calibration_window_start must be before" in v for v in violations)

    def test_overlapping_calibration_evaluation_windows_fails(self):
        artifact = _success_artifact(
            calibration_window_start="2019-01-01",
            calibration_window_end="2022-12-31",
        )
        violations = validate_calibration_artifact(
            artifact,
            strict=False,
            evaluation_window_start="2022-06-01",
            evaluation_window_end="2024-01-01",
        )
        assert any("overlap" in v for v in violations)

    def test_non_overlapping_evaluation_window_passes(self):
        artifact = _success_artifact(
            calibration_window_start="2019-01-01",
            calibration_window_end="2022-12-31",
        )
        violations = validate_calibration_artifact(
            artifact,
            strict=False,
            evaluation_window_start="2023-01-01",
            evaluation_window_end="2025-01-01",
        )
        assert violations == []

    def test_success_artifact_with_empty_thresholds_fails(self):
        artifact = build_calibration_artifact(
            calibration_id="bad_001",
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            calibration_window_start="2019-01-01",
            calibration_window_end="2023-12-31",
            dataset_version="1.3.2",
            calibration_method="hazard_analysis",
            outcome="success",
            thresholds={},
        )
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("thresholds" in v for v in violations)

    def test_null_artifact_missing_null_reason_fails(self):
        # Build a valid null artifact, then strip null_reason and recompute hash
        # so we test the validator directly (the builder sanitizes empty strings).
        from bsve.calibration.calibration_contract import _compute_artifact_hash

        artifact = _null_artifact()
        artifact["null_reason"] = ""
        artifact["artifact_hash"] = _compute_artifact_hash(artifact)
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("null_reason" in v for v in violations)

    def test_malformed_window_date_fails(self):
        artifact = _success_artifact(calibration_window_start="not-a-date")
        violations = validate_calibration_artifact(artifact, strict=False)
        assert any("calibration_window_start" in v for v in violations)


# ---------------------------------------------------------------------------
# 4. Load and write helpers
# ---------------------------------------------------------------------------


class TestLoadWriteHelpers:
    def test_write_then_load_roundtrip(self, tmp_path):
        artifact = _success_artifact()
        out = write_calibration_artifact(artifact, tmp_path / "calib.json")
        assert out.exists()
        loaded = load_calibration_artifact(out)
        assert loaded["calibration_id"] == artifact["calibration_id"]

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_calibration_artifact(tmp_path / "does_not_exist.json")

    def test_load_malformed_json_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed JSON"):
            load_calibration_artifact(p)

    def test_write_creates_parent_dirs(self, tmp_path):
        artifact = _success_artifact()
        nested = tmp_path / "a" / "b" / "c" / "calib.json"
        out = write_calibration_artifact(artifact, nested)
        assert out.exists()

    def test_write_invalid_artifact_raises(self, tmp_path):
        bad = {"schema_version": "0.0.0"}  # missing most keys
        with pytest.raises(ValueError):
            write_calibration_artifact(bad, tmp_path / "bad.json")

    def test_null_artifact_write_and_load(self, tmp_path):
        artifact = _null_artifact()
        out = write_calibration_artifact(artifact, tmp_path / "null_calib.json")
        loaded = load_calibration_artifact(out)
        assert loaded["outcome"] == "null"
        assert loaded["null_reason"]


# ---------------------------------------------------------------------------
# 5. CalibrationPlugin protocol
# ---------------------------------------------------------------------------


class _ConcretePlugin:
    """Minimal concrete implementation of CalibrationPlugin (placeholder)."""

    def calibrate(
        self,
        dataset_adapter: Any,
        state_spec: dict[str, Any],
        calibration_params: dict[str, Any],
    ):
        return build_calibration_artifact(
            calibration_id=calibration_params["calibration_id"],
            ontology_id=calibration_params["ontology_id"],
            ontology_version=calibration_params["ontology_version"],
            calibration_window_start=calibration_params["calibration_window_start"],
            calibration_window_end=calibration_params["calibration_window_end"],
            dataset_version=calibration_params["dataset_version"],
            calibration_method=calibration_params.get("calibration_method", "placeholder"),
            outcome="success",
            thresholds={"placeholder_threshold": 42.0},
        )


class _NullPlugin:
    """Placeholder plugin that always returns a null artifact."""

    def calibrate(
        self,
        dataset_adapter: Any,
        state_spec: dict[str, Any],
        calibration_params: dict[str, Any],
    ):
        return build_calibration_artifact(
            calibration_id=calibration_params["calibration_id"],
            ontology_id=calibration_params["ontology_id"],
            ontology_version=calibration_params["ontology_version"],
            calibration_window_start=calibration_params["calibration_window_start"],
            calibration_window_end=calibration_params["calibration_window_end"],
            dataset_version=calibration_params["dataset_version"],
            calibration_method=calibration_params.get("calibration_method", "placeholder"),
            outcome="null",
            null_reason="No data available in calibration window",
        )


class TestCalibrationPluginProtocol:
    def test_concrete_plugin_satisfies_protocol(self):
        plugin = _ConcretePlugin()
        assert isinstance(plugin, CalibrationPlugin)

    def test_null_plugin_satisfies_protocol(self):
        plugin = _NullPlugin()
        assert isinstance(plugin, CalibrationPlugin)

    def test_object_without_calibrate_does_not_satisfy_protocol(self):
        class NotAPlugin:
            pass

        assert not isinstance(NotAPlugin(), CalibrationPlugin)


# ---------------------------------------------------------------------------
# 6. CalibrationRegistry
# ---------------------------------------------------------------------------


class TestCalibrationRegistry:
    def test_register_and_lookup(self):
        registry = CalibrationRegistry()
        plugin = _ConcretePlugin()
        registry.register("reactive_jpy", "1.0.0", plugin)
        found = registry.lookup("reactive_jpy", "1.0.0")
        assert found is plugin

    def test_lookup_missing_key_raises_key_error(self):
        registry = CalibrationRegistry()
        with pytest.raises(KeyError, match="no calibration plugin registered"):
            registry.lookup("reactive_jpy", "1.0.0")

    def test_register_non_plugin_raises_type_error(self):
        registry = CalibrationRegistry()

        class NotPlugin:
            pass

        with pytest.raises(TypeError, match="CalibrationPlugin protocol"):
            registry.register("reactive_jpy", "1.0.0", NotPlugin())  # type: ignore[arg-type]

    def test_registered_keys(self):
        registry = CalibrationRegistry()
        registry.register("reactive_jpy", "1.0.0", _ConcretePlugin())
        registry.register("reactive_chf", "1.0.0", _NullPlugin())
        keys = registry.registered_keys()
        assert ("reactive_chf", "1.0.0") in keys
        assert ("reactive_jpy", "1.0.0") in keys

    def test_is_registered(self):
        registry = CalibrationRegistry()
        registry.register("reactive_jpy", "1.0.0", _ConcretePlugin())
        assert registry.is_registered("reactive_jpy", "1.0.0")
        assert not registry.is_registered("reactive_chf", "1.0.0")

    def test_versions_for(self):
        registry = CalibrationRegistry()
        registry.register("reactive_jpy", "1.0.0", _ConcretePlugin())
        registry.register("reactive_jpy", "2.0.0", _ConcretePlugin())
        versions = registry.versions_for("reactive_jpy")
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    def test_len(self):
        registry = CalibrationRegistry()
        assert len(registry) == 0
        registry.register("reactive_jpy", "1.0.0", _ConcretePlugin())
        assert len(registry) == 1

    def test_overwrite_registration(self):
        registry = CalibrationRegistry()
        p1 = _ConcretePlugin()
        p2 = _ConcretePlugin()
        registry.register("reactive_jpy", "1.0.0", p1)
        registry.register("reactive_jpy", "1.0.0", p2)
        assert registry.lookup("reactive_jpy", "1.0.0") is p2

    def test_get_default_registry_returns_singleton(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# 7. CalibrationRunner end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture()
def state_spec_file(tmp_path) -> Path:
    """Write a minimal state-spec YAML file for runner tests."""
    if not _YAML_AVAILABLE:
        pytest.skip("pyyaml not installed")
    spec = {
        "environment": {
            "id": "reactive_jpy",
            "version": "1.0.0",
            "family": "reactive",
        },
        "threshold_placeholders": {
            "young_boundary_bars": None,
            "mature_boundary_bars": None,
        },
    }
    p = tmp_path / "reactive_jpy_v1.yaml"
    p.write_text(_yaml.dump(spec), encoding="utf-8")
    return p


@pytest.fixture()
def runner_registry():
    registry = CalibrationRegistry()
    registry.register("reactive_jpy", "1.0.0", _ConcretePlugin())
    registry.register("reactive_chf", "1.0.0", _NullPlugin())
    return registry


@pytest.fixture()
def dummy_adapter():
    """A trivial object standing in for a MasterResearchDatasetAdapter."""

    class _DummyAdapter:
        pass

    return _DummyAdapter()


class TestCalibrationRunner:
    def test_runner_writes_artifact_file(
        self, tmp_path, state_spec_file, runner_registry, dummy_adapter
    ):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        params = {
            "calibration_id": "reactive_jpy_v1_test",
            "ontology_id": "reactive_jpy",
            "ontology_version": "1.0.0",
            "calibration_window_start": "2019-01-01",
            "calibration_window_end": "2023-12-31",
            "dataset_version": "1.3.2",
            "calibration_method": "placeholder",
        }
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=dummy_adapter,
            calibration_params=params,
        )
        assert out.exists()

    def test_runner_artifact_is_valid_json(
        self, tmp_path, state_spec_file, runner_registry, dummy_adapter
    ):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        params = {
            "calibration_id": "reactive_jpy_valid_json",
            "ontology_id": "reactive_jpy",
            "ontology_version": "1.0.0",
            "calibration_window_start": "2019-01-01",
            "calibration_window_end": "2023-12-31",
            "dataset_version": "1.3.2",
            "calibration_method": "placeholder",
        }
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=dummy_adapter,
            calibration_params=params,
        )
        loaded = json.loads(out.read_text())
        assert loaded["calibration_id"] == "reactive_jpy_valid_json"

    @requires_yaml
    def test_runner_null_plugin_writes_null_artifact(
        self, tmp_path, state_spec_file, runner_registry, dummy_adapter
    ):
        chf_spec = {
            "environment": {"id": "reactive_chf", "version": "1.0.0"},
        }
        chf_spec_path = tmp_path / "reactive_chf_v1.yaml"
        chf_spec_path.write_text(_yaml.dump(chf_spec), encoding="utf-8")

        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        params = {
            "calibration_id": "reactive_chf_null_test",
            "ontology_id": "reactive_chf",
            "ontology_version": "1.0.0",
            "calibration_window_start": "2019-01-01",
            "calibration_window_end": "2023-12-31",
            "dataset_version": "1.3.2",
            "calibration_method": "percentile_jenks",
        }
        out = runner.run(
            ontology_id="reactive_chf",
            ontology_version="1.0.0",
            state_spec_path=chf_spec_path,
            dataset_adapter=dummy_adapter,
            calibration_params=params,
        )
        loaded = json.loads(out.read_text())
        assert loaded["outcome"] == "null"
        assert loaded["null_reason"]

    def test_runner_raises_for_missing_calibration_id(
        self, tmp_path, state_spec_file, runner_registry, dummy_adapter
    ):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        with pytest.raises(ValueError, match="calibration_id"):
            runner.run(
                ontology_id="reactive_jpy",
                ontology_version="1.0.0",
                state_spec_path=state_spec_file,
                dataset_adapter=dummy_adapter,
                calibration_params={},
            )

    def test_runner_raises_for_unregistered_plugin(
        self, tmp_path, state_spec_file, dummy_adapter
    ):
        empty_registry = CalibrationRegistry()
        runner = CalibrationRunner(
            registry=empty_registry, output_dir=tmp_path / "artifacts"
        )
        with pytest.raises(KeyError, match="no calibration plugin registered"):
            runner.run(
                ontology_id="reactive_jpy",
                ontology_version="1.0.0",
                state_spec_path=state_spec_file,
                dataset_adapter=dummy_adapter,
                calibration_params={
                    "calibration_id": "reactive_jpy_missing",
                    "ontology_id": "reactive_jpy",
                    "ontology_version": "1.0.0",
                    "calibration_window_start": "2019-01-01",
                    "calibration_window_end": "2023-12-31",
                    "dataset_version": "1.3.2",
                },
            )

    def test_runner_raises_for_missing_state_spec(
        self, tmp_path, runner_registry, dummy_adapter
    ):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        with pytest.raises(FileNotFoundError):
            runner.run(
                ontology_id="reactive_jpy",
                ontology_version="1.0.0",
                state_spec_path=tmp_path / "nonexistent.yaml",
                dataset_adapter=dummy_adapter,
                calibration_params={
                    "calibration_id": "whatever",
                    "ontology_id": "reactive_jpy",
                    "ontology_version": "1.0.0",
                    "calibration_window_start": "2019-01-01",
                    "calibration_window_end": "2023-12-31",
                    "dataset_version": "1.3.2",
                },
            )

    def test_runner_rejects_overlapping_evaluation_window(
        self, tmp_path, state_spec_file, runner_registry, dummy_adapter
    ):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "artifacts"
        )
        params = {
            "calibration_id": "overlap_test",
            "ontology_id": "reactive_jpy",
            "ontology_version": "1.0.0",
            "calibration_window_start": "2019-01-01",
            "calibration_window_end": "2023-12-31",
            "dataset_version": "1.3.2",
            "calibration_method": "placeholder",
        }
        with pytest.raises(ValueError, match="overlap"):
            runner.run(
                ontology_id="reactive_jpy",
                ontology_version="1.0.0",
                state_spec_path=state_spec_file,
                dataset_adapter=dummy_adapter,
                calibration_params=params,
                evaluation_window_start="2023-06-01",
                evaluation_window_end="2025-01-01",
            )

    def test_runner_properties(self, tmp_path, runner_registry):
        runner = CalibrationRunner(
            registry=runner_registry, output_dir=tmp_path / "out"
        )
        assert runner.registry is runner_registry
        assert runner.output_dir == tmp_path / "out"


# ---------------------------------------------------------------------------
# 8. load_state_spec helper
# ---------------------------------------------------------------------------


class TestLoadStateSpec:
    @requires_yaml
    def test_loads_valid_yaml(self, tmp_path):
        spec = {"environment": {"id": "reactive_jpy"}}
        p = tmp_path / "spec.yaml"
        p.write_text(_yaml.dump(spec), encoding="utf-8")
        loaded = load_state_spec(p)
        assert loaded["environment"]["id"] == "reactive_jpy"

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_state_spec(tmp_path / "missing.yaml")
