from __future__ import annotations

import json

import pandas as pd

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
    # atr_pct is percent-scaled (ATR / close * 100), so 1.0 means 1%.
    atr = [1.0] * len(closes)
    return pd.DataFrame(
        {
            "pair": ["usd-jpy"] * len(closes),
            "entry_time": [ts + pd.Timedelta(hours=i) for i in range(len(closes))],
            "entry_close": closes,
            "atr_pct": atr,
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
