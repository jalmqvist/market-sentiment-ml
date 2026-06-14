from __future__ import annotations

import pandas as pd
import pytest

from bsve.artifacts.io import validate_bsve_artifact, write_bsve_artifact
from schemas.bsve_artifact_schema import BSVE_SCHEMA_VERSION


def _valid_df() -> pd.DataFrame:
    entry = pd.date_range("2024-01-01", periods=2, freq="h")
    return pd.DataFrame(
        {
            "entry_time": entry,
            "prediction_available_timestamp": entry + pd.Timedelta(hours=1),
            "pair": ["USDJPY", "USDJPY"],
            "environment_id": ["reactive_jpy", "reactive_jpy"],
            "state_id": ["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_YOUNG"],
            "state_version": ["1.0.0", "1.0.0"],
            "maturity_bars": [0, 1],
            "maturity_class": ["young", "maturing"],
            "state_confidence": [1.0, 1.0],
            "transition_event": ["entry", "continuation"],
            "spec_id": ["reactive_jpy_v1", "reactive_jpy_v1"],
            "calibration_id": ["pending", "pending"],
        }
    )


def test_validate_wrapper_defaults_schema_version() -> None:
    violations = validate_bsve_artifact(_valid_df(), strict=False)
    assert violations == []


def test_write_wrapper_writes_parquet(tmp_path) -> None:
    path = tmp_path / "out.parquet"
    out = write_bsve_artifact(
        _valid_df(),
        path,
        metadata={"schema_version": BSVE_SCHEMA_VERSION},
        artifact_metadata={"source": "unit-test"},
    )
    assert out.exists()


def test_fail_fast_behavior_in_write_wrapper(tmp_path) -> None:
    bad = _valid_df().drop(columns=["state_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        write_bsve_artifact(bad, tmp_path / "bad.parquet")
