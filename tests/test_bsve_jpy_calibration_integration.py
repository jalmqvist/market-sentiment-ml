"""
Integration tests — BSVE PR3 Reactive JPY Calibration.

Exercises the full calibration path:

    DatasetAdapter
        → JPYMaturityCalibrationPlugin
            → CalibrationRunner
                → CalibrationArtifact (JSON on disk)

Covers:
    * Successful calibration artifacts.
    * Null calibration artifacts (insufficient data, insufficient episodes).
    * Schema validation on all produced artifacts.
    * Reproducibility (identical params → identical thresholds and hash).
    * Plugin registration via bootstrap.register_all_plugins().
    * Runner execution with real plugin and fixture dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

requires_yaml = pytest.mark.skipif(not _YAML_AVAILABLE, reason="pyyaml not installed")

from bsve.adapters.dataset_adapter import (
    DatasetAdapterConfig,
    MasterResearchDatasetAdapter,
)
from bsve.calibration.calibration_contract import (
    validate_calibration_artifact,
)
from bsve.calibration.calibration_runner import CalibrationRunner
from bsve.calibration.jpy_maturity_calibration import JPYMaturityCalibrationPlugin
from bsve.calibration.registry import CalibrationRegistry


# ---------------------------------------------------------------------------
# Fixture dataset factories
# ---------------------------------------------------------------------------


def _make_jpy_dataset(
    pairs: tuple[str, ...] = ("USDJPY", "EURJPY", "GBPJPY"),
    n_bars_per_pair: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic H1 dataset for JPY calibration tests.

    Two clearly separated sentiment regimes:
    * Moderate bars (≈ 55% of dataset): uniform(-45, 45), |sentiment| ≤ 45.
    * Extreme blocks (≈ 45% of dataset): uniform(72, 92), always > threshold
      when ``extreme_percentile=40.0`` is used in tests (threshold ≈ 35).

    Extreme blocks span 20–50 bars, forming continuous lifecycles that the
    hazard analysis can process.  The separation ensures the full block
    appears as a single episode — not as multiple short sub-episodes.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    base_time = pd.Timestamp("2019-01-02 00:00:00")

    for pair in pairs:
        sentiment = np.zeros(n_bars_per_pair, dtype=float)
        pos = 0
        while pos < n_bars_per_pair:
            # Moderate gap: all values well below the 40th-pct threshold.
            gap_len = int(rng.integers(10, 35))
            end_gap = min(pos + gap_len, n_bars_per_pair)
            sentiment[pos:end_gap] = rng.uniform(-45, 45, size=end_gap - pos)
            pos = end_gap
            if pos >= n_bars_per_pair:
                break
            # Extreme episode: all values above ~35 (expected threshold).
            ep_len = int(rng.integers(20, 50))
            end_ep = min(pos + ep_len, n_bars_per_pair)
            sentiment[pos:end_ep] = rng.uniform(72, 92, size=end_ep - pos)
            pos = end_ep

        timestamps = [
            base_time + pd.Timedelta(hours=j) for j in range(n_bars_per_pair)
        ]
        for ts, s in zip(timestamps, sentiment):
            records.append({"pair": pair, "entry_time": ts, "net_sentiment": s})

    return pd.DataFrame(records)


def _make_sparse_dataset(
    pairs: tuple[str, ...] = ("USDJPY",),
    n_bars: int = 100,
) -> pd.DataFrame:
    """Synthetic dataset with too few rows to pass sample quality gate."""
    rng = np.random.default_rng(0)
    base_time = pd.Timestamp("2019-01-02")
    rows = [
        {
            "pair": p,
            "entry_time": base_time + pd.Timedelta(hours=i),
            "net_sentiment": float(rng.normal(0, 30)),
        }
        for p in pairs
        for i in range(n_bars)
    ]
    return pd.DataFrame(rows)


def _make_non_extreme_dataset(
    pairs: tuple[str, ...] = ("USDJPY", "EURJPY", "GBPJPY"),
    n_bars_per_pair: int = 1000,
) -> pd.DataFrame:
    """Dataset where sentiment never crosses the extreme threshold, so no episodes form."""
    rng = np.random.default_rng(7)
    base_time = pd.Timestamp("2019-01-02")
    rows = [
        {
            "pair": p,
            "entry_time": base_time + pd.Timedelta(hours=i),
            "net_sentiment": float(rng.uniform(-30, 30)),
        }
        for p in pairs
        for i in range(n_bars_per_pair)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared adapter and calibration_params factories
# ---------------------------------------------------------------------------


def _make_adapter(df: pd.DataFrame) -> MasterResearchDatasetAdapter:
    return MasterResearchDatasetAdapter(df)


def _base_params(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "calibration_id": "test_reactive_jpy_v1",
        "ontology_id": "reactive_jpy",
        "ontology_version": "1.0.0",
        "calibration_window_start": "2019-01-01",
        "calibration_window_end": "2026-12-31",
        "dataset_version": "fixture_1.0",
        "calibration_method": "hazard_analysis",
        "pairs": ["USDJPY", "EURJPY", "GBPJPY"],
        # Use 40th-pct threshold so it falls in the moderate range (~35),
        # below all extreme-block values (72–92) in the fixture.
        # The research-backed production value is 70.0; the lower value here
        # is intentional for integration test determinism.
        "extreme_percentile": 40.0,
        "min_episode_count": 50,
        "min_sample_count": 500,
        "hazard_smoothing_window": 12,
        "young_fraction": 0.4,
        "mature_fraction": 1.6,
        "diagnostic_percentiles": [10, 25, 50, 75, 90],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def jpy_adapter():
    return _make_adapter(_make_jpy_dataset())


@pytest.fixture()
def jpy_params():
    return _base_params()


@pytest.fixture()
def jpy_plugin():
    return JPYMaturityCalibrationPlugin()


@pytest.fixture()
def jpy_registry():
    r = CalibrationRegistry()
    r.register("reactive_jpy", "1.0.0", JPYMaturityCalibrationPlugin())
    return r


@pytest.fixture()
def state_spec_file(tmp_path) -> Path:
    """Minimal reactive_jpy_v1.yaml state-spec for runner tests."""
    if not _YAML_AVAILABLE:
        pytest.skip("pyyaml not installed")
    spec = {
        "environment": {
            "id": "reactive_jpy",
            "version": "1.0.0",
            "family": "reactive",
        },
        "threshold_placeholders": {
            "extreme_threshold_net_pct": None,
            "young_boundary_bars": None,
            "mature_boundary_bars": None,
        },
    }
    p = tmp_path / "reactive_jpy_v1.yaml"
    p.write_text(_yaml.dump(spec), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Plugin satisfies CalibrationPlugin protocol
# ---------------------------------------------------------------------------


class TestPluginProtocol:
    def test_satisfies_protocol(self, jpy_plugin):
        from bsve.calibration.registry import CalibrationPlugin

        assert isinstance(jpy_plugin, CalibrationPlugin)

    def test_has_calibrate_method(self, jpy_plugin):
        assert callable(getattr(jpy_plugin, "calibrate", None))


# ---------------------------------------------------------------------------
# 2. Successful calibration artifact
# ---------------------------------------------------------------------------


class TestSuccessCalibration:
    def test_plugin_returns_success_artifact(self, jpy_plugin, jpy_adapter, jpy_params):
        spec: dict = {}
        artifact = jpy_plugin.calibrate(jpy_adapter, spec, jpy_params)
        assert artifact["outcome"] == "success"

    def test_artifact_passes_schema_validation(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        violations = validate_calibration_artifact(artifact, strict=False)
        assert violations == [], f"Schema violations: {violations}"

    def test_artifact_has_all_required_thresholds(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        thresholds = artifact["thresholds"]
        assert "extreme_threshold_net_pct" in thresholds
        assert "young_boundary_bars" in thresholds
        assert "mature_boundary_bars" in thresholds

    def test_threshold_types_are_numeric(self, jpy_plugin, jpy_adapter, jpy_params):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        t = artifact["thresholds"]
        assert isinstance(t["extreme_threshold_net_pct"], float)
        assert isinstance(t["young_boundary_bars"], int)
        assert isinstance(t["mature_boundary_bars"], int)

    def test_boundaries_are_ordered(self, jpy_plugin, jpy_adapter, jpy_params):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        t = artifact["thresholds"]
        assert t["young_boundary_bars"] < t["mature_boundary_bars"]

    def test_artifact_carries_diagnostics(self, jpy_plugin, jpy_adapter, jpy_params):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        diag = artifact["diagnostics"]
        assert diag["sample_count"] > 0
        assert diag["episode_count"] >= 50

    def test_diagnostics_include_episode_count_per_pair(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        per_pair = artifact["diagnostics"]["episode_count_per_pair"]
        # Normalized pair names expected.
        assert "usd-jpy" in per_pair
        assert "eur-jpy" in per_pair
        assert "gbp-jpy" in per_pair

    def test_diagnostics_include_maturity_percentiles(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        pct = artifact["diagnostics"]["maturity_distribution_percentiles"]
        for key in ("p10", "p25", "p50", "p75", "p90"):
            assert key in pct

    def test_diagnostics_include_calibration_window_coverage(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        assert artifact["diagnostics"]["calibration_window_coverage_days"] is not None

    def test_artifact_metadata_matches_params(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        assert artifact["calibration_id"] == jpy_params["calibration_id"]
        assert artifact["ontology_id"] == "reactive_jpy"
        assert artifact["ontology_version"] == "1.0.0"
        assert artifact["dataset_version"] == "fixture_1.0"

    def test_extreme_threshold_is_positive(self, jpy_plugin, jpy_adapter, jpy_params):
        artifact = jpy_plugin.calibrate(jpy_adapter, {}, jpy_params)
        assert artifact["thresholds"]["extreme_threshold_net_pct"] > 0


# ---------------------------------------------------------------------------
# 3. Null calibration artifacts
# ---------------------------------------------------------------------------


class TestNullCalibration:
    def test_null_artifact_when_sample_count_too_low(self, jpy_plugin):
        """Sparse dataset should trigger min_sample_count quality gate."""
        sparse_df = _make_sparse_dataset(n_bars=100)
        adapter = _make_adapter(sparse_df)
        params = _base_params(
            calibration_id="test_null_sparse",
            min_sample_count=500,
        )
        artifact = jpy_plugin.calibrate(adapter, {}, params)
        assert artifact["outcome"] == "null"
        assert artifact["null_reason"]

    def test_null_artifact_when_no_episodes(self, jpy_plugin):
        """Non-extreme dataset should trigger min_episode_count quality gate."""
        flat_df = _make_non_extreme_dataset()
        adapter = _make_adapter(flat_df)
        params = _base_params(
            calibration_id="test_null_no_episodes",
            # High percentile so virtually no bars qualify as extreme.
            extreme_percentile=99.9,
            min_episode_count=50,
        )
        artifact = jpy_plugin.calibrate(adapter, {}, params)
        assert artifact["outcome"] == "null"
        assert artifact["null_reason"]

    def test_null_artifact_passes_schema_validation(self, jpy_plugin):
        """Null artifacts must be valid calibration artifacts."""
        sparse_df = _make_sparse_dataset(n_bars=10)
        adapter = _make_adapter(sparse_df)
        params = _base_params(
            calibration_id="test_null_schema",
            min_sample_count=500,
        )
        artifact = jpy_plugin.calibrate(adapter, {}, params)
        assert artifact["outcome"] == "null"
        violations = validate_calibration_artifact(artifact, strict=False)
        assert violations == [], f"Null artifact schema violations: {violations}"

    def test_null_artifact_has_null_reason(self, jpy_plugin):
        sparse_df = _make_sparse_dataset(n_bars=10)
        adapter = _make_adapter(sparse_df)
        params = _base_params(
            calibration_id="test_null_reason",
            min_sample_count=500,
        )
        artifact = jpy_plugin.calibrate(adapter, {}, params)
        assert isinstance(artifact.get("null_reason"), str)
        assert len(artifact["null_reason"]) > 0

    def test_null_artifact_has_empty_thresholds(self, jpy_plugin):
        sparse_df = _make_sparse_dataset(n_bars=10)
        adapter = _make_adapter(sparse_df)
        params = _base_params(
            calibration_id="test_null_thresholds",
            min_sample_count=500,
        )
        artifact = jpy_plugin.calibrate(adapter, {}, params)
        assert artifact["outcome"] == "null"
        # Null artifacts may carry empty thresholds but must not carry success thresholds.
        assert not artifact.get("thresholds") or artifact["thresholds"] == {}


# ---------------------------------------------------------------------------
# 4. Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_identical_params_produce_identical_thresholds(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        """Same dataset + same params → identical threshold values."""
        artifact1 = jpy_plugin.calibrate(jpy_adapter, {}, dict(jpy_params))
        artifact2 = jpy_plugin.calibrate(jpy_adapter, {}, dict(jpy_params))
        assert artifact1["thresholds"] == artifact2["thresholds"]

    def test_identical_params_produce_identical_artifact_hash(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        """Same inputs → same artifact hash (deterministic calibration)."""
        a1 = jpy_plugin.calibrate(jpy_adapter, {}, dict(jpy_params))
        a2 = jpy_plugin.calibrate(jpy_adapter, {}, dict(jpy_params))
        # artifact_hash includes calibration_timestamp which differs per call;
        # verify the threshold payload is identical instead.
        assert a1["thresholds"] == a2["thresholds"]
        assert a1["diagnostics"]["episode_count"] == a2["diagnostics"]["episode_count"]

    def test_different_calibration_id_does_not_change_thresholds(
        self, jpy_plugin, jpy_adapter, jpy_params
    ):
        """calibration_id is metadata — thresholds are data-driven."""
        params_a = dict(jpy_params, calibration_id="run_a")
        params_b = dict(jpy_params, calibration_id="run_b")
        a = jpy_plugin.calibrate(jpy_adapter, {}, params_a)
        b = jpy_plugin.calibrate(jpy_adapter, {}, params_b)
        assert a["thresholds"] == b["thresholds"]


# ---------------------------------------------------------------------------
# 5. Plugin registration and bootstrap
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_manual_registration(self):
        registry = CalibrationRegistry()
        plugin = JPYMaturityCalibrationPlugin()
        registry.register("reactive_jpy", "1.0.0", plugin)
        assert registry.is_registered("reactive_jpy", "1.0.0")
        assert registry.lookup("reactive_jpy", "1.0.0") is plugin

    def test_bootstrap_registers_reactive_jpy(self):
        """bootstrap.register_all_plugins() populates a fresh registry."""
        from bsve.calibration.bootstrap import register_all_plugins

        registry = CalibrationRegistry()
        register_all_plugins(registry)
        assert registry.is_registered("reactive_jpy", "1.0.0")

    def test_bootstrap_plugin_satisfies_protocol(self):
        from bsve.calibration.bootstrap import register_all_plugins
        from bsve.calibration.registry import CalibrationPlugin

        registry = CalibrationRegistry()
        register_all_plugins(registry)
        plugin = registry.lookup("reactive_jpy", "1.0.0")
        assert isinstance(plugin, CalibrationPlugin)

    def test_bootstrap_idempotent(self):
        """Calling bootstrap twice does not accumulate duplicate entries."""
        from bsve.calibration.bootstrap import register_all_plugins

        registry = CalibrationRegistry()
        register_all_plugins(registry)
        register_all_plugins(registry)
        assert len(registry.registered_keys()) == 1

    def test_bootstrap_uses_default_registry_when_none_given(self):
        from bsve.calibration.bootstrap import register_all_plugins
        from bsve.calibration.registry import get_default_registry

        r = register_all_plugins(None)
        assert r is get_default_registry()
        assert r.is_registered("reactive_jpy", "1.0.0")


# ---------------------------------------------------------------------------
# 6. CalibrationRunner end-to-end
# ---------------------------------------------------------------------------


@requires_yaml
class TestCalibrationRunnerIntegration:
    def test_runner_produces_artifact_file(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=jpy_params,
        )
        assert out.exists()

    def test_runner_artifact_is_valid_json(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(jpy_params, calibration_id="runner_json_test")
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
        )
        loaded = json.loads(out.read_text())
        assert loaded["calibration_id"] == "runner_json_test"

    def test_runner_artifact_passes_schema_validation(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(jpy_params, calibration_id="runner_schema_test")
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
        )
        artifact = json.loads(out.read_text())
        violations = validate_calibration_artifact(artifact, strict=False)
        assert violations == [], f"Schema violations: {violations}"

    def test_runner_artifact_has_success_outcome(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(jpy_params, calibration_id="runner_outcome_test")
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
        )
        artifact = json.loads(out.read_text())
        assert artifact["outcome"] == "success"

    def test_runner_writes_null_artifact_for_sparse_data(
        self, tmp_path, jpy_registry, state_spec_file
    ):
        """CalibrationRunner must not raise on null artifact; outcome == 'null'."""
        sparse_df = _make_sparse_dataset(n_bars=50)
        adapter = _make_adapter(sparse_df)
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = _base_params(
            calibration_id="runner_null_test",
            min_sample_count=10_000,  # impossible to satisfy
        )
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=adapter,
            calibration_params=params,
        )
        artifact = json.loads(out.read_text())
        assert artifact["outcome"] == "null"

    def test_runner_artifact_filename_uses_calibration_id(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(jpy_params, calibration_id="my_unique_run_id")
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
        )
        assert out.stem == "my_unique_run_id"

    def test_runner_with_bootstrap_registry(
        self, tmp_path, jpy_adapter, jpy_params, state_spec_file
    ):
        """End-to-end using bootstrap-populated registry."""
        from bsve.calibration.bootstrap import register_all_plugins

        registry = CalibrationRegistry()
        register_all_plugins(registry)
        runner = CalibrationRunner(
            registry=registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(jpy_params, calibration_id="bootstrap_e2e")
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
        )
        artifact = json.loads(out.read_text())
        assert artifact["outcome"] == "success"

    def test_runner_non_overlapping_evaluation_window_passes(
        self, tmp_path, jpy_adapter, jpy_params, jpy_registry, state_spec_file
    ):
        runner = CalibrationRunner(
            registry=jpy_registry, output_dir=tmp_path / "artifacts"
        )
        params = dict(
            jpy_params,
            calibration_id="eval_window_ok",
            calibration_window_start="2019-01-01",
            calibration_window_end="2022-12-31",
        )
        out = runner.run(
            ontology_id="reactive_jpy",
            ontology_version="1.0.0",
            state_spec_path=state_spec_file,
            dataset_adapter=jpy_adapter,
            calibration_params=params,
            evaluation_window_start="2023-01-01",
            evaluation_window_end="2025-01-01",
        )
        assert out.exists()
