from __future__ import annotations

import json

import pandas as pd
import pytest

from bsve.validation.outcome_labeling import (
    assign_independent_outcome_labels,
    build_outcome_payload,
    main,
    reconstruct_consensus_episodes,
)


def _state_surface() -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp("2024-01-01")
    # Episode 1: YOUNG, ends at t=1.
    rows.append({"pair": "usd-jpy", "entry_time": ts, "state_id": "JPY_CONSENSUS_YOUNG"})
    rows.append({"pair": "usd-jpy", "entry_time": ts + pd.Timedelta(hours=1), "state_id": "JPY_CONSENSUS_YOUNG"})
    # Gap state.
    rows.append({"pair": "usd-jpy", "entry_time": ts + pd.Timedelta(hours=2), "state_id": "JPY_NON_EXTREME"})
    # Episode 2: MATURE, ends at t=4.
    rows.append({"pair": "usd-jpy", "entry_time": ts + pd.Timedelta(hours=3), "state_id": "JPY_CONSENSUS_MATURE"})
    rows.append({"pair": "usd-jpy", "entry_time": ts + pd.Timedelta(hours=4), "state_id": "JPY_CONSENSUS_MATURE"})
    return pd.DataFrame(rows)


def _market_dataset() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01")
    closes = [100.0, 101.0, 101.0, 101.0, 101.0, 101.5, 103.5]
    # vol_48b is expressed in return units (e.g. 0.01 == 1% volatility).
    vol_48b = [0.01] * len(closes)
    return pd.DataFrame(
        {
            "pair": ["usd-jpy"] * len(closes),
            "entry_time": [ts + pd.Timedelta(hours=i) for i in range(len(closes))],
            "entry_close": closes,
            "vol_48b": vol_48b,
        }
    )


def test_reconstruct_consensus_episodes_filters_non_consensus() -> None:
    episodes = reconstruct_consensus_episodes(_state_surface())
    assert set(episodes["state_id"]) == {"JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"}
    assert len(episodes) == 2


def test_assign_independent_outcome_labels_success_and_failure() -> None:
    outcomes = assign_independent_outcome_labels(
        _state_surface(),
        _market_dataset(),
        outcome_window_bars=2,
    )
    assert len(outcomes) == 2
    young = outcomes[outcomes["state_id"] == "JPY_CONSENSUS_YOUNG"].iloc[0]
    mature = outcomes[outcomes["state_id"] == "JPY_CONSENSUS_MATURE"].iloc[0]
    assert young["success_threshold"] == 0.01
    assert mature["success_threshold"] == 0.01
    assert young["outcome_label"] == "FAILURE"
    assert mature["outcome_label"] == "SUCCESS"


def test_outcome_labeling_cli_execution_and_payload(tmp_path) -> None:
    artifact = tmp_path / "bsve_states_reactive_jpy_1.0.0.parquet"
    dataset = tmp_path / "dataset.csv"
    _state_surface().assign(maturity_bars=1, transition_event="continuation").to_parquet(
        artifact,
        index=False,
    )
    _market_dataset().to_csv(dataset, index=False)

    exit_code = main(
        [
            "--artifact",
            str(artifact),
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(tmp_path),
            "--outcome-window-bars",
            "2",
        ]
    )
    assert exit_code == 0

    output = tmp_path / "independent_outcomes.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["outcome_window_bars"] == 2
    assert payload["summary"]["total_consensus_episodes"] == 2
    assert payload["summary"]["evaluable_episodes"] == 2
    assert len(payload["independent_outcomes"]) == 2


def test_build_outcome_payload_counts() -> None:
    payload = build_outcome_payload(
        _state_surface(),
        _market_dataset(),
        outcome_window_bars=2,
    )
    assert payload["summary"]["success_count"] == 1
    assert payload["summary"]["failure_count"] == 1


# ---------------------------------------------------------------------------
# Dataset-schema regression tests
# ---------------------------------------------------------------------------

def _bsve_schema_dataset() -> pd.DataFrame:
    """Realistic BSVE dataset schema: vol_12b + vol_48b, no atr_pct."""
    ts = pd.Timestamp("2024-01-01")
    closes = [100.0, 101.0, 101.0, 101.0, 101.0, 101.5, 103.5]
    return pd.DataFrame(
        {
            "pair": ["usd-jpy"] * len(closes),
            "entry_time": [ts + pd.Timedelta(hours=i) for i in range(len(closes))],
            "entry_close": closes,
            "vol_12b": [0.005] * len(closes),
            "vol_48b": [0.01] * len(closes),
        }
    )


def test_schema_regression_no_atr_pct_column() -> None:
    """Outcome labeling must succeed with the BSVE dataset schema (vol_48b, no atr_pct)."""
    market_df = _bsve_schema_dataset()
    assert "atr_pct" not in market_df.columns, "test dataset must not contain atr_pct"
    assert "vol_48b" in market_df.columns

    outcomes = assign_independent_outcome_labels(
        _state_surface(),
        market_df,
        outcome_window_bars=2,
    )
    assert len(outcomes) == 2
    assert set(outcomes["outcome_label"].dropna()) <= {"SUCCESS", "FAILURE"}


def test_schema_regression_payload_threshold_column() -> None:
    """Payload metadata must reflect vol_48b as the threshold column."""
    payload = build_outcome_payload(
        _state_surface(),
        _bsve_schema_dataset(),
        outcome_window_bars=2,
    )
    assert payload["metadata"]["threshold_column"] == "vol_48b"
    assert payload["summary"]["success_rate"] is not None


def test_schema_regression_atr_pct_dependency_absent() -> None:
    """If atr_pct is the only volatility column, outcome labeling must raise ValueError."""
    ts = pd.Timestamp("2024-01-01")
    closes = [100.0, 101.0, 101.0, 101.0, 101.0, 101.5, 103.5]
    atr_only_dataset = pd.DataFrame(
        {
            "pair": ["usd-jpy"] * len(closes),
            "entry_time": [ts + pd.Timedelta(hours=i) for i in range(len(closes))],
            "entry_close": closes,
            "atr_pct": [1.0] * len(closes),
        }
    )
    # vol_48b is required; a dataset with only atr_pct must fail.
    with pytest.raises((ValueError, KeyError)):
        assign_independent_outcome_labels(
            _state_surface(),
            atr_only_dataset,
            outcome_window_bars=2,
        )


def test_cli_threshold_column_option_metadata_surfacing(tmp_path) -> None:
    """--threshold-column selects the correct dataset column and surfaces it in metadata."""
    artifact = tmp_path / "bsve_states_reactive_jpy_1.0.0.parquet"
    dataset_csv = tmp_path / "dataset.csv"
    _state_surface().assign(maturity_bars=1, transition_event="continuation").to_parquet(
        artifact, index=False
    )
    _bsve_schema_dataset().to_csv(dataset_csv, index=False)

    exit_code = main(
        [
            "--artifact", str(artifact),
            "--dataset-path", str(dataset_csv),
            "--output-dir", str(tmp_path),
            "--outcome-window-bars", "2",
            "--threshold-column", "vol_12b",
        ]
    )
    assert exit_code == 0

    payload = json.loads((tmp_path / "independent_outcomes.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["threshold_column"] == "vol_12b"


def test_cli_nonexistent_threshold_column_exits_nonzero(tmp_path) -> None:
    """--threshold-column referencing a column absent from the dataset must cause a non-zero exit."""
    artifact = tmp_path / "bsve_states_reactive_jpy_1.0.0.parquet"
    dataset_csv = tmp_path / "dataset.csv"
    _state_surface().assign(maturity_bars=1, transition_event="continuation").to_parquet(
        artifact, index=False
    )
    _bsve_schema_dataset().to_csv(dataset_csv, index=False)

    exit_code = main(
        [
            "--artifact", str(artifact),
            "--dataset-path", str(dataset_csv),
            "--output-dir", str(tmp_path),
            "--outcome-window-bars", "2",
            "--threshold-column", "atr_pct",
        ]
    )
    assert exit_code != 0
