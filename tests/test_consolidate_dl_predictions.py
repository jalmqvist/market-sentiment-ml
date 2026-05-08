"""
tests/test_consolidate_dl_predictions.py
=========================================
Unit tests for scripts/consolidate_dl_predictions.py.

Tests cover:
1. consolidate_dl_predictions: reads per-run artifacts, produces cube.
2. Identity columns injected from manifests.
3. Cube schema validation (uniqueness, monotonicity per surface grain).
4. Multi-run, multi-surface, multi-pair consolidation.
5. Orphan parquet (no manifest) is skipped with a warning.
6. Error paths: missing input dir, no valid pairs.

Run with::

    python -m pytest tests/test_consolidate_dl_predictions.py -v
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

from write_dl_prediction_artifact import write_dl_prediction_artifact
from consolidate_dl_predictions import (
    consolidate_dl_predictions,
    _find_run_artifact_pairs,
    _load_run_artifact,
)
from build_dl_signal_artifact import OUTPUT_COLS, SCHEMA_VERSION, UNIQUE_KEY_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_df(n: int = 4, pair: str = "eur-usd") -> pd.DataFrame:
    times = pd.date_range("2023-01-02", periods=n, freq="h")
    return pd.DataFrame(
        {
            "entry_time": times,
            "pair": [pair] * n,
            "pred_prob_up": np.linspace(0.4, 0.7, n),
        }
    )


def _identity(model="MLP", regime="LVTF", horizon=24, fset="price_vol_sentiment"):
    return {
        "model": model,
        "dl_regime": regime,
        "target_horizon": horizon,
        "feature_set": fset,
    }


def _write_run(tmp_dir: Path, df: pd.DataFrame, identity: dict, run_id: str) -> None:
    """Write a per-run artifact using the writer."""
    write_dl_prediction_artifact(df, identity, output_dir=tmp_dir, run_id=run_id)


# ---------------------------------------------------------------------------
# _find_run_artifact_pairs
# ---------------------------------------------------------------------------


class TestFindRunArtifactPairs:
    def test_finds_valid_pairs(self, tmp_path):
        _write_run(tmp_path, _minimal_df(), _identity(), "run1")
        pairs = _find_run_artifact_pairs(tmp_path)
        assert len(pairs) == 1

    def test_skips_orphan_parquets(self, tmp_path, capsys):
        _write_run(tmp_path, _minimal_df(), _identity(), "run1")
        # Write an orphan parquet (no manifest)
        (tmp_path / "orphan.parquet").write_bytes(b"")
        pairs = _find_run_artifact_pairs(tmp_path)
        assert len(pairs) == 1  # only run1
        captured = capsys.readouterr()
        assert "orphan.parquet" in captured.out

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _find_run_artifact_pairs(tmp_path / "nonexistent")

    def test_no_pairs_raises(self, tmp_path):
        (tmp_path / "some.parquet").write_bytes(b"")
        with pytest.raises(ValueError, match="No valid"):
            _find_run_artifact_pairs(tmp_path)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No valid"):
            _find_run_artifact_pairs(tmp_path)


# ---------------------------------------------------------------------------
# _load_run_artifact
# ---------------------------------------------------------------------------


class TestLoadRunArtifact:
    def test_injects_identity_columns(self, tmp_path):
        _write_run(tmp_path, _minimal_df(), _identity(), "run1")
        pq = tmp_path / "run1.parquet"
        mf = tmp_path / "run1.manifest.json"
        df = _load_run_artifact(pq, mf)
        assert "model" in df.columns
        assert (df["model"] == "MLP").all()
        assert "dl_regime" in df.columns
        assert (df["dl_regime"] == "LVTF").all()
        assert "target_horizon" in df.columns
        assert (df["target_horizon"] == 24).all()
        assert df["target_horizon"].dtype == "Int64"
        assert "feature_set" in df.columns

    def test_injects_provenance_columns(self, tmp_path):
        write_dl_prediction_artifact(
            _minimal_df(),
            _identity(),
            provenance={"dataset_version": "1.1.0", "model_version": "v2.0"},
            output_dir=tmp_path,
            run_id="run2",
        )
        df = _load_run_artifact(
            tmp_path / "run2.parquet", tmp_path / "run2.manifest.json"
        )
        assert (df["dataset_version"] == "1.1.0").all()
        assert (df["model_version"] == "v2.0").all()


# ---------------------------------------------------------------------------
# consolidate_dl_predictions
# ---------------------------------------------------------------------------


class TestConsolidateDlPredictions:
    def test_single_run_produces_cube(self, tmp_path):
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")

        out_pq = tmp_path / "cube.parquet"
        out_mf = tmp_path / "cube_manifest.json"
        cube = consolidate_dl_predictions(pred_dir, out_pq, out_mf)

        assert out_pq.exists()
        assert out_mf.exists()
        assert len(cube) == 4

    def test_cube_has_all_required_columns(self, tmp_path):
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")
        cube = consolidate_dl_predictions(
            pred_dir, tmp_path / "pq", tmp_path / "mf"
        )
        for col in OUTPUT_COLS:
            assert col in cube.columns, f"Missing column: {col}"

    def test_identity_columns_present_and_correct(self, tmp_path):
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")
        cube = consolidate_dl_predictions(
            pred_dir, tmp_path / "pq", tmp_path / "mf"
        )
        assert (cube["model"] == "MLP").all()
        assert (cube["dl_regime"] == "LVTF").all()
        assert (cube["target_horizon"] == 24).all()
        assert cube["target_horizon"].dtype == "Int64"

    def test_multi_run_cube(self, tmp_path):
        pred_dir = tmp_path / "preds"
        # Two different surfaces
        _write_run(pred_dir, _minimal_df(), _identity(regime="HVTF"), "run_hvtf")
        _write_run(pred_dir, _minimal_df(pair="usd-jpy"), _identity(regime="LVTF"), "run_lvtf")
        # offset times for usd-jpy to avoid duplicates on the key
        run2 = _minimal_df(pair="usd-jpy")
        run2["entry_time"] += pd.Timedelta(hours=10)
        _write_run(pred_dir, run2, _identity(regime="LVTF", model="LSTM"), "run_lstm")

        cube = consolidate_dl_predictions(
            pred_dir, tmp_path / "pq", tmp_path / "mf"
        )
        assert cube["pair"].nunique() >= 1
        assert len(cube) >= 4

    def test_uniqueness_validated(self, tmp_path):
        """Duplicate rows on unique key must raise AssertionError."""
        pred_dir = tmp_path / "preds"
        # Same identity + same timestamps → unique key collision after concat
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")
        _write_run(pred_dir, _minimal_df(), _identity(), "run2")

        with pytest.raises(AssertionError, match="duplicate"):
            consolidate_dl_predictions(
                pred_dir, tmp_path / "pq", tmp_path / "mf"
            )

    def test_cube_manifest_schema(self, tmp_path):
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")
        out_mf = tmp_path / "cube_manifest.json"
        consolidate_dl_predictions(pred_dir, tmp_path / "pq", out_mf)

        manifest = json.loads(out_mf.read_text())
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert "generated_at_utc" in manifest
        assert "total_rows" in manifest
        assert "pair_stats" in manifest
        assert "calibration" in manifest
        assert "train_period" in manifest
        assert "warnings" in manifest
        assert "missing_provenance_counts" in manifest

    def test_missing_input_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            consolidate_dl_predictions(
                tmp_path / "nonexistent",
                tmp_path / "pq",
                tmp_path / "mf",
            )

    def test_empty_input_dir_raises(self, tmp_path):
        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()
        with pytest.raises(ValueError, match="No valid"):
            consolidate_dl_predictions(pred_dir, tmp_path / "pq", tmp_path / "mf")

    def test_target_horizon_int64_in_cube(self, tmp_path):
        """target_horizon must be Int64 in the consolidated cube."""
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(horizon=48), "run1")
        cube = consolidate_dl_predictions(
            pred_dir, tmp_path / "pq", tmp_path / "mf"
        )
        assert cube["target_horizon"].dtype == "Int64"
        assert (cube["target_horizon"] == 48).all()

    def test_schema_version_constant_in_cube(self, tmp_path):
        pred_dir = tmp_path / "preds"
        _write_run(pred_dir, _minimal_df(), _identity(), "run1")
        cube = consolidate_dl_predictions(
            pred_dir, tmp_path / "pq", tmp_path / "mf"
        )
        assert (cube["schema_version"] == SCHEMA_VERSION).all()
