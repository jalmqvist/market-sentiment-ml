"""
tests/test_dl_artifact_contract.py
====================================
Tests for the DL artifact contract layer (v2).

Covers:
1. validate_dl_artifact: valid artifact passes all checks.
2. Missing required field raises ValueError.
3. Causal ordering violation raises ValueError.
4. Duplicate (pair, entry_time) rows raise ValueError.
5. Non-monotonic entry_time within a pair raises ValueError.
6. Invalid schema version raises ValueError.
7. Non-strict mode collects all violations without raising.
8. Pair normalization violation detected.
9. Timezone-aware column raises ValueError.
10. DL_SCHEMA_VERSION constant is correct.
11. Full artifact written by write_dl_prediction_artifact passes validation.
12. New v2 columns present in written artifact.
13. prediction_available_timestamp defaults to entry_time.
14. artifact_created_timestamp is present and non-null.

Run with::

    python -m pytest tests/test_dl_artifact_contract.py -v
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

from schemas.dl_artifact_schema import (
    DL_ARTIFACT_CREATED_COL,
    DL_AVAILABLE_TS_COL,
    DL_GENERATED_TS_COL,
    DL_SCHEMA_VERSION,
    validate_dl_artifact,
)
from write_dl_prediction_artifact import write_dl_prediction_artifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_valid_df(n: int = 4) -> pd.DataFrame:
    """Minimal valid artifact DataFrame (all required columns, no violations)."""
    times = pd.date_range("2023-01-02", periods=n, freq="h")
    return pd.DataFrame(
        {
            "pair": ["eur-usd"] * n,
            "entry_time": times,
            DL_AVAILABLE_TS_COL: times,  # causal equality: exactly entry_time
        }
    )


def _valid_identity() -> dict:
    return {
        "model": "MLP",
        "dl_regime": "LVTF",
        "target_horizon": 24,
        "feature_set": "price_trend",
    }


def _minimal_write_df(n: int = 4) -> pd.DataFrame:
    """Minimal DataFrame for write_dl_prediction_artifact (no identity cols)."""
    times = pd.date_range("2023-01-02", periods=n, freq="h")
    return pd.DataFrame(
        {
            "entry_time": times,
            "pair": ["eur-usd"] * n,
            "pred_prob_up": np.linspace(0.4, 0.7, n),
        }
    )


# ---------------------------------------------------------------------------
# validate_dl_artifact — unit tests
# ---------------------------------------------------------------------------


class TestValidateDlArtifact:
    def test_valid_artifact_passes(self):
        df = _minimal_valid_df()
        violations = validate_dl_artifact(df, strict=False)
        assert violations == [], f"Unexpected violations: {violations}"

    def test_valid_artifact_with_metadata_passes(self):
        df = _minimal_valid_df()
        metadata = {"schema_version": DL_SCHEMA_VERSION}
        violations = validate_dl_artifact(df, metadata=metadata, strict=False)
        assert violations == [], f"Unexpected violations: {violations}"

    def test_missing_required_field_entry_time_raises(self):
        df = _minimal_valid_df().drop(columns=["entry_time"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dl_artifact(df, strict=True)

    def test_missing_required_field_pair_raises(self):
        df = _minimal_valid_df().drop(columns=["pair"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dl_artifact(df, strict=True)

    def test_missing_prediction_available_timestamp_raises(self):
        df = _minimal_valid_df().drop(columns=[DL_AVAILABLE_TS_COL])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dl_artifact(df, strict=True)

    def test_causal_ordering_violation_raises(self):
        """prediction_available_timestamp > entry_time must be caught."""
        df = _minimal_valid_df()
        # Shift prediction_available_timestamp forward — violation
        df[DL_AVAILABLE_TS_COL] = df["entry_time"] + pd.Timedelta(hours=1)
        with pytest.raises(ValueError, match="causal ordering violated"):
            validate_dl_artifact(df, strict=True)

    def test_causal_ordering_equality_passes(self):
        """prediction_available_timestamp == entry_time is valid."""
        df = _minimal_valid_df()
        df[DL_AVAILABLE_TS_COL] = df["entry_time"].copy()
        violations = validate_dl_artifact(df, strict=False)
        assert not any("causal" in v for v in violations), violations

    def test_causal_ordering_less_than_passes(self):
        """prediction_available_timestamp < entry_time is valid."""
        df = _minimal_valid_df()
        df[DL_AVAILABLE_TS_COL] = df["entry_time"] - pd.Timedelta(hours=1)
        violations = validate_dl_artifact(df, strict=False)
        assert not any("causal" in v for v in violations), violations

    def test_duplicate_pair_entry_time_raises(self):
        df = _minimal_valid_df(2)
        # Create exact duplicate rows
        df = pd.concat([df, df], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            validate_dl_artifact(df, strict=True)

    def test_non_monotonic_entry_time_raises(self):
        df = _minimal_valid_df(4)
        # Reverse the time order within the pair
        df = df.iloc[::-1].reset_index(drop=True)
        df[DL_AVAILABLE_TS_COL] = df["entry_time"].copy()
        with pytest.raises(ValueError, match="monotonically"):
            validate_dl_artifact(df, strict=True)

    def test_invalid_schema_version_raises(self):
        df = _minimal_valid_df()
        metadata = {"schema_version": "dl_signals_h1_v1"}  # old v1 version
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_dl_artifact(df, metadata=metadata, strict=True)

    def test_missing_schema_version_in_metadata_raises(self):
        df = _minimal_valid_df()
        metadata = {"dataset_version": "1.0.0"}  # no schema_version key
        with pytest.raises(ValueError, match="metadata missing 'schema_version'"):
            validate_dl_artifact(df, metadata=metadata, strict=True)

    def test_non_strict_collects_all_violations(self):
        """Non-strict mode should collect multiple violations."""
        df = pd.DataFrame(
            {
                "pair": ["EUR-USD"],  # bad normalization (uppercase)
                "entry_time": [pd.Timestamp("2023-01-02")],
                DL_AVAILABLE_TS_COL: [pd.Timestamp("2023-01-03")],  # causal violation
            }
        )
        violations = validate_dl_artifact(
            df, metadata={"schema_version": "old"}, strict=False
        )
        assert len(violations) >= 2, f"Expected multiple violations, got: {violations}"

    def test_pair_normalization_violation_detected(self):
        df = _minimal_valid_df()
        df["pair"] = "EUR-USD"  # uppercase, should be eur-usd
        with pytest.raises(ValueError, match="non-normalized"):
            validate_dl_artifact(df, strict=True)

    def test_tz_aware_entry_time_raises(self):
        df = _minimal_valid_df()
        df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
        with pytest.raises(ValueError, match="tz-naive"):
            validate_dl_artifact(df, strict=True)

    def test_null_in_pair_column_raises(self):
        df = _minimal_valid_df()
        df.loc[0, "pair"] = None
        with pytest.raises(ValueError, match="null"):
            validate_dl_artifact(df, strict=True)

    def test_null_in_entry_time_raises(self):
        df = _minimal_valid_df()
        df.loc[0, "entry_time"] = pd.NaT
        # NaT in entry_time also triggers null check
        with pytest.raises(ValueError, match="null"):
            validate_dl_artifact(df, strict=True)

    def test_null_in_prediction_available_timestamp_raises(self):
        df = _minimal_valid_df()
        df.loc[0, DL_AVAILABLE_TS_COL] = pd.NaT
        with pytest.raises(ValueError, match="null"):
            validate_dl_artifact(df, strict=True)

    def test_multi_pair_valid(self):
        times_a = pd.date_range("2023-01-02", periods=3, freq="h")
        times_b = pd.date_range("2023-01-02", periods=3, freq="h")
        df = pd.DataFrame(
            {
                "pair": ["eur-usd"] * 3 + ["usd-jpy"] * 3,
                "entry_time": list(times_a) + list(times_b),
                DL_AVAILABLE_TS_COL: list(times_a) + list(times_b),
            }
        )
        violations = validate_dl_artifact(df, strict=False)
        assert violations == [], violations


# ---------------------------------------------------------------------------
# v2 schema — integration tests via write_dl_prediction_artifact
# ---------------------------------------------------------------------------


class TestV2ArtifactSchema:
    def test_written_artifact_passes_validation(self, tmp_path):
        pq, mf = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="v2_test",
        )
        loaded = pd.read_parquet(pq)
        # Should not raise
        validate_dl_artifact(
            loaded,
            metadata={"schema_version": DL_SCHEMA_VERSION},
            strict=True,
        )

    def test_v2_columns_present_in_parquet(self, tmp_path):
        pq, _ = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="v2_cols",
        )
        loaded = pd.read_parquet(pq)
        for col in [DL_AVAILABLE_TS_COL, DL_GENERATED_TS_COL, DL_ARTIFACT_CREATED_COL]:
            assert col in loaded.columns, f"Missing v2 column: {col}"

    def test_prediction_available_timestamp_defaults_to_entry_time(self, tmp_path):
        pq, _ = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="avail_ts",
        )
        loaded = pd.read_parquet(pq)
        # prediction_available_timestamp should equal entry_time (causal equality)
        pd.testing.assert_series_equal(
            loaded[DL_AVAILABLE_TS_COL].reset_index(drop=True),
            loaded["entry_time"].reset_index(drop=True),
            check_names=False,
        )

    def test_artifact_created_timestamp_non_null(self, tmp_path):
        pq, _ = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="created_ts",
        )
        loaded = pd.read_parquet(pq)
        assert DL_ARTIFACT_CREATED_COL in loaded.columns
        assert not loaded[DL_ARTIFACT_CREATED_COL].isna().any(), (
            "artifact_created_timestamp must be non-null"
        )
        # All rows share the same artifact_created_timestamp
        assert loaded[DL_ARTIFACT_CREATED_COL].nunique() == 1

    def test_manifest_schema_version_is_v2(self, tmp_path):
        _, mf = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="schema_v2",
        )
        manifest = json.loads(mf.read_text())
        assert manifest["schema_version"] == DL_SCHEMA_VERSION

    def test_manifest_contains_artifact_created_timestamp(self, tmp_path):
        _, mf = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="manifest_ts",
        )
        manifest = json.loads(mf.read_text())
        assert DL_ARTIFACT_CREATED_COL in manifest
        assert manifest[DL_ARTIFACT_CREATED_COL] is not None

    def test_manifest_timestamp_semantics_documented(self, tmp_path):
        _, mf = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="ts_semantics",
        )
        manifest = json.loads(mf.read_text())
        ts_sem = manifest["artifact_metadata"]["timestamp_semantics"]
        assert DL_AVAILABLE_TS_COL in ts_sem
        assert DL_GENERATED_TS_COL in ts_sem
        assert DL_ARTIFACT_CREATED_COL in ts_sem

    def test_causal_invariant_holds_in_artifact(self, tmp_path):
        """prediction_available_timestamp <= entry_time for all rows."""
        pq, _ = write_dl_prediction_artifact(
            _minimal_write_df(),
            _valid_identity(),
            output_dir=tmp_path,
            run_id="causal_check",
        )
        loaded = pd.read_parquet(pq)
        violations = (loaded[DL_AVAILABLE_TS_COL] > loaded["entry_time"]).sum()
        assert violations == 0, f"{violations} causal ordering violations in artifact"

    def test_schema_version_constant_value(self):
        assert DL_SCHEMA_VERSION == "2.0.0"
