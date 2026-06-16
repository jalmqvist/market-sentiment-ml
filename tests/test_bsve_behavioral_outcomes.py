from __future__ import annotations

import json

import pandas as pd

from bsve.validation.behavioral_outcomes import (
    analyze_behavioral_outcomes,
    compute_persistence_probabilities,
    compute_progression_analysis,
    compute_reversal_probabilities,
    compute_transition_matrix,
    main,
)


def _surface_from_states(
    states: list[str],
    *,
    pair: str = "USDJPY",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp(start)
    prev_state: str | None = None
    maturity_bars = 0

    for state in states:
        if state == prev_state:
            maturity_bars += 1
            transition_event = "continuation"
        else:
            maturity_bars = 1
            transition_event = "entry"
        rows.append(
            {
                "pair": pair,
                "entry_time": ts,
                "state_id": state,
                "maturity_bars": maturity_bars,
                "transition_event": transition_event,
            }
        )
        prev_state = state
        ts += pd.Timedelta(hours=1)

    return pd.DataFrame(rows)


def _behaviorally_distinct_surface() -> pd.DataFrame:
    states: list[str] = []
    for _ in range(24):
        states.extend(["JPY_CONSENSUS_YOUNG"] * 2)
        states.extend(["JPY_NON_EXTREME"] * 12)
    for _ in range(24):
        states.extend(["JPY_CONSENSUS_MATURING"] * 12)
        states.extend(["JPY_CONSENSUS_MATURE"] * 12)
        states.extend(["JPY_NON_EXTREME"] * 12)
    for _ in range(24):
        states.extend(["JPY_CONSENSUS_MATURE"] * 20)
        states.extend(["JPY_NON_EXTREME"] * 12)
    return _surface_from_states(states)


def _progression_surface() -> pd.DataFrame:
    states: list[str] = []
    for _ in range(20):
        states.extend(["JPY_CONSENSUS_YOUNG"] * 4)
        states.extend(["JPY_CONSENSUS_MATURING"] * 4)
        states.extend(["JPY_CONSENSUS_MATURE"] * 4)
        states.extend(["JPY_NON_EXTREME"] * 4)
    return _surface_from_states(states)


def test_transition_matrix_generation() -> None:
    df = _surface_from_states(
        [
            "JPY_NON_EXTREME",
            "JPY_NON_EXTREME",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURING",
            "JPY_NON_EXTREME",
            "JPY_CONSENSUS_MATURE",
            "JPY_CONSENSUS_MATURE",
        ]
    )

    matrix = compute_transition_matrix(df)
    assert matrix["JPY_NON_EXTREME"]["JPY_NON_EXTREME"]["count"] == 1
    assert matrix["JPY_NON_EXTREME"]["JPY_CONSENSUS_YOUNG"]["count"] == 1
    assert matrix["JPY_NON_EXTREME"]["JPY_CONSENSUS_MATURE"]["count"] == 1
    assert matrix["JPY_NON_EXTREME"]["JPY_NON_EXTREME"]["probability"] == 1 / 3
    assert matrix["JPY_CONSENSUS_YOUNG"]["JPY_CONSENSUS_YOUNG"]["probability"] == 0.5
    assert (
        matrix["JPY_CONSENSUS_YOUNG"]["JPY_CONSENSUS_MATURING"]["probability"] == 0.5
    )


def test_reversal_probability_computation() -> None:
    report = compute_reversal_probabilities(_behaviorally_distinct_surface())
    assert report["JPY_CONSENSUS_YOUNG"]["within_4"]["count"] >= 20
    assert report["JPY_CONSENSUS_YOUNG"]["within_4"]["probability"] == 1.0
    assert report["JPY_CONSENSUS_MATURING"]["within_4"]["probability"] == 0.0
    assert report["JPY_CONSENSUS_MATURE"]["within_12"]["probability"] == 0.5


def test_persistence_probability_computation() -> None:
    report = compute_persistence_probabilities(_behaviorally_distinct_surface())
    assert report["JPY_CONSENSUS_YOUNG"]["at_4"]["probability"] == 0.0
    assert report["JPY_CONSENSUS_MATURING"]["at_12"]["probability"] == 1.0
    assert report["JPY_CONSENSUS_MATURE"]["at_12"]["probability"] == 0.5


def test_progression_analysis() -> None:
    report = compute_progression_analysis(_progression_surface())
    young_to_maturing = report["JPY_CONSENSUS_YOUNG->JPY_CONSENSUS_MATURING"]
    maturing_to_mature = report["JPY_CONSENSUS_MATURING->JPY_CONSENSUS_MATURE"]
    assert young_to_maturing["sample_count"] == 20
    assert young_to_maturing["transition_probability"] == 1.0
    assert young_to_maturing["median_time_to_transition"] == 4.0
    assert maturing_to_mature["transition_probability"] == 1.0
    assert maturing_to_mature["median_time_to_transition"] == 4.0


def test_behavioral_outcomes_cli_generates_complete_report(tmp_path) -> None:
    df = _behaviorally_distinct_surface()
    artifact = tmp_path / "bsve_states_reactive_jpy_1.0.0.parquet"
    df.to_parquet(artifact, index=False)

    exit_code = main(
        [
            "--artifact",
            str(artifact),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0

    report_path = tmp_path / "behavioral_outcomes_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["behavioral_evidence_available"] is True
    assert "transition_matrix" in report
    assert "reversal_probabilities" in report
    assert "persistence_probabilities" in report
    assert "progression_analysis" in report
    assert any(test["significant"] for test in report["behavioral_tests"].values())


def test_analyze_behavioral_outcomes_returns_behavioral_test_summary() -> None:
    report = analyze_behavioral_outcomes(_behaviorally_distinct_surface())
    assert report["behavioral_evidence_available"] is True
    assert report["significant_behavioral_tests"]
    assert all(
        test["classification"] == "behavioral_evidence"
        and test["used_for_behavioral_differentiation"] is True
        for test in report["behavioral_tests"].values()
    )
