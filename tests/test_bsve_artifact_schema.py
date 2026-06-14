from __future__ import annotations

import pandas as pd
import pytest

from schemas.bsve_artifact_schema import (
    BSVE_AVAILABLE_TS_COL,
    BSVE_CALIBRATION_ID_COL,
    BSVE_ENTRY_TIME_COL,
    BSVE_ENVIRONMENT_COL,
    BSVE_MATURITY_BARS_COL,
    BSVE_MATURITY_CLASS_COL,
    BSVE_PAIR_COL,
    BSVE_REQUIRED_ARTIFACT_COLS,
    BSVE_SCHEMA_VERSION,
    BSVE_SPEC_ID_COL,
    BSVE_STATE_CONFIDENCE_COL,
    BSVE_STATE_ID_COL,
    BSVE_STATE_VERSION_COL,
    BSVE_TRANSITION_EVENT_COL,
    validate_bsve_artifact,
    write_bsve_artifact,
)


def _valid_df() -> pd.DataFrame:
    entry = pd.date_range("2024-01-01", periods=3, freq="h")
    return pd.DataFrame(
        {
            BSVE_ENTRY_TIME_COL: entry,
            BSVE_AVAILABLE_TS_COL: entry + pd.Timedelta(hours=1),
            BSVE_PAIR_COL: ["USDJPY", "USDJPY", "EURCHF"],
            BSVE_ENVIRONMENT_COL: ["reactive_jpy", "reactive_jpy", "reactive_chf"],
            BSVE_STATE_ID_COL: ["A", "A", "B"],
            BSVE_STATE_VERSION_COL: ["1.0.0", "1.0.0", "1.0.0"],
            BSVE_MATURITY_BARS_COL: [0, 1, 0],
            BSVE_MATURITY_CLASS_COL: ["young", "maturing", "n_a"],
            BSVE_STATE_CONFIDENCE_COL: [1.0, 1.0, 1.0],
            BSVE_TRANSITION_EVENT_COL: ["entry", "continuation", "entry"],
            BSVE_SPEC_ID_COL: ["reactive_jpy_v1", "reactive_jpy_v1", "reactive_chf_v1"],
            BSVE_CALIBRATION_ID_COL: ["pending", "pending", "pending"],
        }
    )


def test_valid_bsve_artifact_passes() -> None:
    violations = validate_bsve_artifact(
        _valid_df(), metadata={"schema_version": BSVE_SCHEMA_VERSION}, strict=False
    )
    assert violations == []


def test_missing_column_fails_fast() -> None:
    df = _valid_df().drop(columns=[BSVE_STATE_ID_COL])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_bsve_artifact(df, strict=True)


def test_causal_ordering_violation_fails_fast() -> None:
    df = _valid_df()
    df[BSVE_AVAILABLE_TS_COL] = df[BSVE_ENTRY_TIME_COL]
    with pytest.raises(ValueError, match="causal ordering violated"):
        validate_bsve_artifact(df, strict=True)


def test_negative_maturity_fails_fast() -> None:
    df = _valid_df()
    df.loc[0, BSVE_MATURITY_BARS_COL] = -1
    with pytest.raises(ValueError, match="maturity_bars"):
        validate_bsve_artifact(df, strict=True)


def test_non_strict_collects_violations() -> None:
    df = _valid_df()
    df[BSVE_MATURITY_CLASS_COL] = "invalid"
    df[BSVE_STATE_CONFIDENCE_COL] = 2.0
    violations = validate_bsve_artifact(df, strict=False)
    assert len(violations) >= 2


def test_spec_and_calibration_resolvers_enforced() -> None:
    df = _valid_df()

    with pytest.raises(ValueError, match="spec_id not resolvable"):
        validate_bsve_artifact(df, strict=True, spec_resolver=lambda _v: False)

    with pytest.raises(ValueError, match="calibration_id not resolvable"):
        validate_bsve_artifact(df, strict=True, calibration_resolver=lambda _v: False)


def test_spec_and_calibration_ids_are_required_contract_fields() -> None:
    assert BSVE_SPEC_ID_COL in BSVE_REQUIRED_ARTIFACT_COLS
    assert BSVE_CALIBRATION_ID_COL in BSVE_REQUIRED_ARTIFACT_COLS


def test_write_bsve_artifact_writes_parquet(tmp_path) -> None:
    path = tmp_path / "bsve_states.parquet"
    out = write_bsve_artifact(
        _valid_df(),
        path,
        metadata={"schema_version": BSVE_SCHEMA_VERSION},
    )
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 3
