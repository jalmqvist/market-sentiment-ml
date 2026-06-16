from __future__ import annotations

import json

import pandas as pd

from bsve.validation.criterion1 import (
    MIN_OBSERVATIONS_PER_STATE,
    evaluate_criterion1,
    main,
    reconstruct_state_episodes,
    run_duration_ks_tests,
    compute_duration_statistics,
)
from bsve.validation.report import write_validation_report


def _surface_from_episodes(
    specs: list[tuple[str, int]],
    *,
    pair: str = "USDJPY",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp(start)
    prev_state: str | None = None

    for state, duration in specs:
        for i in range(duration):
            if prev_state != state and i == 0:
                event = "entry"
            else:
                event = "continuation"
            rows.append(
                {
                    "pair": pair,
                    "entry_time": ts,
                    "state_id": state,
                    "maturity_bars": i + 1,
                    "transition_event": event,
                }
            )
            ts += pd.Timedelta(hours=1)
        prev_state = state

    return pd.DataFrame(rows)


def _rich_surface() -> pd.DataFrame:
    specs: list[tuple[str, int]] = []
    specs.extend([("JPY_NON_EXTREME", 10)] * 6)
    specs.extend([("JPY_CONSENSUS_YOUNG", 2), ("JPY_NON_EXTREME", 1)] * 40)
    specs.extend([("JPY_CONSENSUS_MATURING", 10), ("JPY_NON_EXTREME", 1)] * 20)
    specs.extend([("JPY_CONSENSUS_MATURE", 20), ("JPY_NON_EXTREME", 1)] * 10)
    return _surface_from_episodes(specs)


def test_reconstruct_state_episodes() -> None:
    df = pd.DataFrame(
        {
            "pair": ["USDJPY", "USDJPY", "USDJPY", "EURJPY", "EURJPY"],
            "entry_time": pd.date_range("2024-01-01", periods=5, freq="h"),
            "state_id": [
                "JPY_NON_EXTREME",
                "JPY_NON_EXTREME",
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_MATURE",
            ],
            "maturity_bars": [1, 2, 1, 2, 24],
            "transition_event": ["entry", "continuation", "entry", "entry", "entry"],
        }
    )

    episodes = reconstruct_state_episodes(df)
    observed = sorted(zip(episodes["state_id"], episodes["duration_bars"]))
    assert observed == sorted(
        [
            ("JPY_NON_EXTREME", 2),
            ("JPY_CONSENSUS_YOUNG", 1),
            ("JPY_CONSENSUS_YOUNG", 1),
            ("JPY_CONSENSUS_MATURE", 1),
        ]
    )


def test_duration_statistics() -> None:
    episodes = pd.DataFrame(
        {
            "pair": ["USDJPY", "USDJPY", "USDJPY"],
            "state_id": [
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_MATURE",
            ],
            "start_time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "end_time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "duration_bars": [2, 6, 20],
        }
    )

    stats = compute_duration_statistics(episodes)
    young = next(row for row in stats if row["state_id"] == "JPY_CONSENSUS_YOUNG")
    assert young["episode_count"] == 2
    assert young["median_duration"] == 4.0
    assert young["max_duration"] == 6


def test_ks_test_execution() -> None:
    specs = []
    specs.extend([("JPY_CONSENSUS_YOUNG", 2), ("JPY_NON_EXTREME", 1)] * 50)
    specs.extend([("JPY_CONSENSUS_MATURING", 12), ("JPY_NON_EXTREME", 1)] * 50)
    specs.extend([("JPY_CONSENSUS_MATURE", 24), ("JPY_NON_EXTREME", 1)] * 50)
    episodes = reconstruct_state_episodes(_surface_from_episodes(specs))

    tests, warnings = run_duration_ks_tests(episodes)
    assert len(tests) == 3
    assert warnings == []
    assert any(test["significant"] for test in tests)
    assert all(
        test["classification"] == "calibration_consistency_diagnostic"
        and test["used_for_behavioral_differentiation"] is False
        for test in tests
    )


def test_minimum_observation_validation_fails() -> None:
    df = _surface_from_episodes(
        [
            ("JPY_NON_EXTREME", MIN_OBSERVATIONS_PER_STATE),
            ("JPY_CONSENSUS_YOUNG", MIN_OBSERVATIONS_PER_STATE),
            ("JPY_CONSENSUS_MATURING", MIN_OBSERVATIONS_PER_STATE),
            ("JPY_CONSENSUS_MATURE", MIN_OBSERVATIONS_PER_STATE - 1),
        ]
    )

    result, _ = evaluate_criterion1(df)
    assert result.passed is False
    assert result.status == "FAIL"
    assert any("Insufficient observations" in warning for warning in result.warnings)


def test_report_generation_inconclusive_status(tmp_path) -> None:
    df = _rich_surface()
    artifact = tmp_path / "bsve_states_reactive_jpy_1.0.0.parquet"
    df.to_parquet(artifact, index=False)

    exit_code = main([
        "--artifact",
        str(artifact),
        "--output-dir",
        str(tmp_path),
    ])
    assert exit_code == 0

    report_path = tmp_path / "bsve_validation_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation_outcome"]["passed"] is False
    assert report["validation_outcome"]["status"] == "INCONCLUSIVE"
    assert report["metadata"]["criterion"] == "criterion1_behavioral_differentiation"
    assert report["metadata"]["behavioral_evidence_available"] is False
    assert report["duration_ks_diagnostics"][0]["classification"] == (
        "calibration_consistency_diagnostic"
    )

    direct_path = write_validation_report(report, tmp_path / "nested")
    reloaded = json.loads(direct_path.read_text(encoding="utf-8"))
    assert reloaded["validation_outcome"]["status"] == "INCONCLUSIVE"


def test_status_pass_when_behavioral_evidence_available() -> None:
    df = _rich_surface()
    result, report = evaluate_criterion1(df, behavioral_evidence_available=True)
    assert result.status == "PASS"
    assert result.passed is True
    assert report["metadata"]["behavioral_evidence_available"] is True
    assert all(
        row["classification"] == "calibration_consistency_diagnostic"
        and row["used_for_behavioral_differentiation"] is False
        for row in report["duration_ks_diagnostics"]
    )
