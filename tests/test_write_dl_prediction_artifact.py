"""
tests/test_write_dl_prediction_artifact.py
===========================================
Unit tests for scripts/write_dl_prediction_artifact.py.

Tests cover:
1. write_dl_prediction_artifact: writes parquet + manifest.
2. Per-run parquet schema (time-series payload only).
3. Per-run manifest schema (identity + provenance + required blocks).
4. pred_direction tri-state semantics.
5. prediction_timestamp optional column.
6. Identity validation (required fields, target_horizon type).
7. Error paths: missing required columns, invalid pred_prob_up.
8. run_id auto-generation.

Run with::

    python -m pytest tests/test_write_dl_prediction_artifact.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from write_dl_prediction_artifact import (
    REQUIRED_PARQUET_COLS,
    RUN_PARQUET_COLS,
    write_dl_prediction_artifact,
)
from schemas.dl_artifact_schema import DL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_df(n: int = 4) -> pd.DataFrame:
    """Minimal valid prediction DataFrame (no identity columns)."""
    times = pd.date_range("2023-01-02", periods=n, freq="h")
    return pd.DataFrame(
        {
            "entry_time": times,
            "pair": ["eur-usd"] * n,
            "pred_prob_up": np.linspace(0.4, 0.7, n),
        }
    )


def _valid_identity() -> dict:
    return {
        "model": "MLP",
        "surface_id": "trend_vol",
        "surface_version": "unknown",
        "state_id": "LVTF",
        "dl_regime": "LVTF",
        "target_horizon": 24,
        "feature_set": "price_vol_sentiment",
    }


def _valid_provenance() -> dict:
    return {
        "dataset_version": "1.1.0",
        "model_version": "v1.0",
        "training_run_id": "run_20240115_abc",
    }


# ---------------------------------------------------------------------------
# write_dl_prediction_artifact
# ---------------------------------------------------------------------------


class TestWriteDlPredictionArtifact:
    def test_writes_parquet_and_manifest(self, tmp_path):
        pq, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=tmp_path, run_id="test_run"
        )
        assert pq.exists()
        assert mf.exists()
        assert pq.name == "test_run.parquet"
        assert mf.name == "test_run.manifest.json"

    def test_parquet_contains_required_contract_cols(self, tmp_path):
        pq, _ = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=tmp_path, run_id="r1"
        )
        loaded = pd.read_parquet(pq)
        # Must contain all required contract columns
        for col in REQUIRED_PARQUET_COLS:
            assert col in loaded.columns, f"Missing column: {col}"
        # Legacy payload columns should still be present.
        for col in RUN_PARQUET_COLS:
            assert col in loaded.columns, f"Missing legacy payload column: {col}"

    def test_parquet_row_count_matches_input(self, tmp_path):
        df = _minimal_df(n=6)
        pq, _ = write_dl_prediction_artifact(
            df, _valid_identity(), output_dir=tmp_path, run_id="r2"
        )
        loaded = pd.read_parquet(pq)
        assert len(loaded) == 6

    def test_manifest_schema(self, tmp_path):
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), _valid_provenance(),
            output_dir=tmp_path, run_id="r3"
        )
        manifest = json.loads(mf.read_text())
        assert manifest["schema_version"] == DL_SCHEMA_VERSION
        assert manifest["export_frequency"] == "H1"
        assert "generated_at_utc" in manifest
        assert "run_id" in manifest
        assert "signal_definition" in manifest
        assert "identity" in manifest
        assert "provenance" in manifest
        assert "calibration" in manifest
        assert manifest["calibration"]["method"] == "none"
        assert "train_period" in manifest
        assert "warnings" in manifest
        assert "missing_provenance_counts" in manifest
        assert "git_commit" in manifest
        assert "row_count" in manifest
        assert "pairs" in manifest
        assert "entry_time_min" in manifest
        assert "entry_time_max" in manifest
        assert "artifact_metadata" in manifest
        assert "missingness_config" in manifest
        assert "export_config" in manifest

    def test_manifest_identity_block(self, tmp_path):
        identity = _valid_identity()
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), identity, output_dir=tmp_path, run_id="r4"
        )
        manifest = json.loads(mf.read_text())
        assert manifest["identity"]["model"] == "MLP"
        assert manifest["identity"]["surface_id"] == "trend_vol"
        assert manifest["identity"]["surface_version"] == "unknown"
        assert manifest["identity"]["state_id"] == "LVTF"
        assert manifest["identity"]["dl_regime"] == "LVTF"
        assert manifest["identity"]["target_horizon"] == 24
        assert manifest["identity"]["feature_set"] == "price_vol_sentiment"

    def test_manifest_provenance_block(self, tmp_path):
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), _valid_provenance(),
            output_dir=tmp_path, run_id="r5"
        )
        manifest = json.loads(mf.read_text())
        assert manifest["provenance"]["dataset_version"] == "1.1.0"
        assert manifest["provenance"]["model_version"] == "v1.0"
        assert manifest["provenance"]["training_run_id"] == "run_20240115_abc"

    def test_manifest_train_period_from_provenance(self, tmp_path):
        provenance = {
            **_valid_provenance(),
            "train_period_start": "2018-01-01",
            "train_period_end": "2022-12-31",
        }
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), provenance,
            output_dir=tmp_path, run_id="r6"
        )
        manifest = json.loads(mf.read_text())
        assert manifest["train_period"]["start"] == "2018-01-01"
        assert manifest["train_period"]["end"] == "2022-12-31"

    def test_run_id_autogenerated_when_none(self, tmp_path):
        pq, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=tmp_path
        )
        # Both files must exist with matching stems
        assert pq.exists()
        assert mf.exists()
        assert pq.stem == mf.stem.replace(".manifest", "").replace("manifest", "")
        # run_id stem encodes identity fragments
        assert "MLP" in pq.stem
        assert "trend_vol" in pq.stem
        assert "LVTF" in pq.stem

    def test_legacy_identity_is_backward_compatible(self, tmp_path):
        legacy_identity = {
            "model": "MLP",
            "dl_regime": "reactive_jpy:JPY_CONSENSUS_MATURE",
            "target_horizon": 24,
            "feature_set": "price_vol_sentiment",
        }
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), legacy_identity, output_dir=tmp_path, run_id="legacy_identity"
        )
        manifest = json.loads(mf.read_text())
        assert manifest["identity"]["surface_id"] == "reactive_jpy"
        assert manifest["identity"]["state_id"] == "JPY_CONSENSUS_MATURE"

    def test_pred_direction_tristate(self, tmp_path):
        df = _minimal_df(4)
        df["pred_prob_up"] = [0.6, 0.5, 0.4, 0.5]
        pq, _ = write_dl_prediction_artifact(
            df, _valid_identity(), output_dir=tmp_path, run_id="tristate"
        )
        loaded = pd.read_parquet(pq)
        expected = pd.array([1, 0, -1, 0], dtype="Int64")
        pd.testing.assert_extension_array_equal(
            loaded["pred_direction"].values, expected
        )

    def test_prediction_timestamp_null_when_absent(self, tmp_path):
        pq, _ = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=tmp_path, run_id="no_ts"
        )
        loaded = pd.read_parquet(pq)
        assert "prediction_timestamp" in loaded.columns
        assert loaded["prediction_timestamp"].isna().all()
        assert "dl_feature_available" in loaded.columns
        assert (loaded["dl_feature_available"] == 1).all()

    def test_prediction_timestamp_preserved(self, tmp_path):
        df = _minimal_df(4)
        df["prediction_timestamp"] = pd.date_range("2023-01-02 00:30", periods=4, freq="h")
        pq, _ = write_dl_prediction_artifact(
            df, _valid_identity(), output_dir=tmp_path, run_id="with_ts"
        )
        loaded = pd.read_parquet(pq)
        assert not loaded["prediction_timestamp"].isna().all()

    def test_missing_required_columns_raises(self, tmp_path):
        df = pd.DataFrame({"entry_time": ["2023-01-01"], "pair": ["eur-usd"]})
        with pytest.raises(ValueError, match="missing required columns"):
            write_dl_prediction_artifact(df, _valid_identity(), output_dir=tmp_path)

    def test_invalid_pred_prob_up_raises(self, tmp_path):
        df = _minimal_df()
        df.loc[0, "pred_prob_up"] = 1.5
        with pytest.raises(ValueError, match="invalid pred_prob_up"):
            write_dl_prediction_artifact(df, _valid_identity(), output_dir=tmp_path)

    def test_missing_identity_model_raises(self, tmp_path):
        identity = {k: v for k, v in _valid_identity().items() if k != "model"}
        with pytest.raises(ValueError, match="missing required keys"):
            write_dl_prediction_artifact(_minimal_df(), identity, output_dir=tmp_path)

    def test_missing_identity_target_horizon_raises(self, tmp_path):
        identity = {k: v for k, v in _valid_identity().items() if k != "target_horizon"}
        with pytest.raises(ValueError, match="missing required keys"):
            write_dl_prediction_artifact(_minimal_df(), identity, output_dir=tmp_path)

    def test_invalid_target_horizon_type_raises(self, tmp_path):
        identity = {**_valid_identity(), "target_horizon": "twenty-four"}
        with pytest.raises(ValueError, match="target_horizon"):
            write_dl_prediction_artifact(_minimal_df(), identity, output_dir=tmp_path)

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "dir"
        write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=new_dir, run_id="r7"
        )
        assert new_dir.exists()

    def test_returns_tuple_of_paths(self, tmp_path):
        result = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), output_dir=tmp_path, run_id="r8"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        pq, mf = result
        assert str(pq).endswith(".parquet")
        assert str(mf).endswith(".manifest.json")

    def test_warnings_for_missing_provenance(self, tmp_path):
        """Manifest warnings list should mention missing provenance."""
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), provenance=None,
            output_dir=tmp_path, run_id="r9"
        )
        manifest = json.loads(mf.read_text())
        assert len(manifest["warnings"]) > 0

    def test_no_warnings_with_full_provenance(self, tmp_path):
        """With full provenance, only the prediction_timestamp warning may appear."""
        _, mf = write_dl_prediction_artifact(
            _minimal_df(), _valid_identity(), _valid_provenance(),
            output_dir=tmp_path, run_id="r10"
        )
        manifest = json.loads(mf.read_text())
        # prediction_timestamp warning may still be present (no ts in input)
        # but no provenance-missing warnings
        for w in manifest["warnings"]:
            assert "provenance" not in w.lower() or "prediction_timestamp" in w.lower()

    def test_multiple_pairs_in_one_run(self, tmp_path):
        df1 = _minimal_df(3)
        df2 = _minimal_df(3)
        df2["pair"] = "usd-jpy"
        df2["entry_time"] = pd.date_range("2023-01-05", periods=3, freq="h")
        df = pd.concat([df1, df2], ignore_index=True)
        pq, mf = write_dl_prediction_artifact(
            df, _valid_identity(), output_dir=tmp_path, run_id="multi_pair"
        )
        loaded = pd.read_parquet(pq)
        assert loaded["pair"].nunique() == 2
        manifest = json.loads(mf.read_text())
        assert len(manifest["pairs"]) == 2

    def test_constant_presence_mode_expands_hourly_grid(self, tmp_path):
        df = _minimal_df(3)
        # Create a deterministic 1-hour gap in availability.
        df["entry_time"] = pd.to_datetime(
            ["2023-01-02 00:00:00", "2023-01-02 01:00:00", "2023-01-02 03:00:00"]
        )
        pq, mf = write_dl_prediction_artifact(
            df,
            _valid_identity(),
            provenance={
                "control_mode": "constant_presence",
                "dl_add_missing_indicators": True,
                "dl_impute_optional_features": True,
                "dl_imputation_value": 0.5,
            },
            output_dir=tmp_path,
            run_id="const_presence",
        )
        loaded = pd.read_parquet(pq)
        manifest = json.loads(mf.read_text())
        assert len(loaded) == 4  # 00,01,02,03
        assert "pred_prob_up_missing" in loaded.columns
        assert int((loaded["dl_feature_available"] == 0).sum()) == 1
        assert manifest["export_config"]["control_mode"] == "constant_presence"

    def test_availability_shuffle_mode_is_deterministic(self, tmp_path):
        df = _minimal_df(6)
        pq1, _ = write_dl_prediction_artifact(
            df,
            _valid_identity(),
            provenance={
                "control_mode": "availability_shuffle",
                "availability_shuffle_seed": 7,
            },
            output_dir=tmp_path,
            run_id="shuffle_a",
        )
        pq2, _ = write_dl_prediction_artifact(
            df,
            _valid_identity(),
            provenance={
                "control_mode": "availability_shuffle",
                "availability_shuffle_seed": 7,
            },
            output_dir=tmp_path,
            run_id="shuffle_b",
        )
        a = pd.read_parquet(pq1)[["pair", "entry_time"]]
        b = pd.read_parquet(pq2)[["pair", "entry_time"]]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
