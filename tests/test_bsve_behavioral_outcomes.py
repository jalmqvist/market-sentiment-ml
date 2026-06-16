"""Tests for bsve.validation.behavioral_outcomes."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from bsve.validation.behavioral_outcomes import (
    analyze_behavioral_outcomes,
    _compute_episode_outcomes,
    _cohens_h,
    _MIN_EPISODE_COUNT,
)


def _make_surface(
    specs: list[tuple[str, int, str]],
    *,
    pair: str = "USDJPY",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a minimal state-surface DataFrame.

    Each spec is (state_id, duration_bars, terminal_event).  The terminal event
    is assigned to the last bar of each episode; all preceding bars get
    'continuation'.
    """
    rows = []
    ts = pd.Timestamp(start)
    prev_state: str | None = None

    for state, duration, terminal_event in specs:
        for i in range(duration):
            if prev_state != state and i == 0:
                event = "entry"
            elif i == duration - 1:
                event = terminal_event
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


def _rich_surface_with_reversals() -> pd.DataFrame:
    """Surface where MATURE state has a meaningfully different reversal rate."""
    specs: list[tuple[str, int, str]] = []
    # YOUNG: 40 episodes of 3 bars each, 5 with exit_reversal (~12%)
    for i in range(40):
        terminal = "exit_reversal" if i < 5 else "exit_unknown"
        specs.append(("JPY_CONSENSUS_YOUNG", 3, terminal))
        specs.append(("JPY_NON_EXTREME", 2, "continuation"))
    # MATURING: 20 episodes, 2 with exit_reversal (~10%)
    for i in range(20):
        terminal = "exit_reversal" if i < 2 else "exit_unknown"
        specs.append(("JPY_CONSENSUS_MATURING", 8, terminal))
        specs.append(("JPY_NON_EXTREME", 2, "continuation"))
    # MATURE: 15 episodes, all exit_reversal (100%) — very different from YOUNG
    for _ in range(15):
        specs.append(("JPY_CONSENSUS_MATURE", 20, "exit_reversal"))
        specs.append(("JPY_NON_EXTREME", 2, "continuation"))
    return _make_surface(specs)


def _no_reversal_surface() -> pd.DataFrame:
    """Surface with no exit_reversal events anywhere."""
    specs: list[tuple[str, int, str]] = []
    specs.extend([("JPY_CONSENSUS_YOUNG", 3, "exit_unknown")] * 40)
    specs.extend([("JPY_NON_EXTREME", 2, "continuation")] * 40)
    specs.extend([("JPY_CONSENSUS_MATURING", 8, "exit_unknown")] * 20)
    specs.extend([("JPY_NON_EXTREME", 2, "continuation")] * 20)
    specs.extend([("JPY_CONSENSUS_MATURE", 20, "exit_unknown")] * 10)
    specs.extend([("JPY_NON_EXTREME", 2, "continuation")] * 10)
    return _make_surface(specs)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_cohens_h_identical_proportions() -> None:
    assert _cohens_h(0.5, 0.5) == pytest.approx(0.0)


def test_cohens_h_extreme_difference() -> None:
    h = _cohens_h(0.0, 1.0)
    assert h == pytest.approx(math.pi, rel=1e-6)


def test_cohens_h_symmetric() -> None:
    assert _cohens_h(0.2, 0.8) == pytest.approx(_cohens_h(0.8, 0.2))


def test_compute_episode_outcomes_counts_reversals() -> None:
    surface = _make_surface(
        [
            ("JPY_CONSENSUS_YOUNG", 3, "exit_reversal"),
            ("JPY_NON_EXTREME", 2, "continuation"),
            ("JPY_CONSENSUS_MATURE", 5, "exit_unknown"),
        ]
    )
    episodes = _compute_episode_outcomes(surface)
    young_eps = episodes[episodes["state_id"] == "JPY_CONSENSUS_YOUNG"]
    mature_eps = episodes[episodes["state_id"] == "JPY_CONSENSUS_MATURE"]
    assert int(young_eps["is_reversal"].sum()) == 1
    assert int(mature_eps["is_reversal"].sum()) == 0


def test_analyze_no_reversals_returns_not_available() -> None:
    surface = _no_reversal_surface()
    result = analyze_behavioral_outcomes(surface)
    assert result["behavioral_evidence_available"] is False
    assert result["behavioral_effect_size"] is None
    assert isinstance(result["behavioral_tests"], list)
    assert isinstance(result["behavioral_outcomes"], list)


def test_analyze_outcome_fields_present() -> None:
    surface = _no_reversal_surface()
    result = analyze_behavioral_outcomes(surface)
    for outcome in result["behavioral_outcomes"]:
        assert "state_id" in outcome
        assert "episode_count" in outcome
        assert "reversal_count" in outcome
        assert "reversal_rate" in outcome


def test_analyze_test_fields_present() -> None:
    surface = _rich_surface_with_reversals()
    result = analyze_behavioral_outcomes(surface)
    for test in result["behavioral_tests"]:
        assert "comparison" in test
        assert "state_a" in test
        assert "state_b" in test
        assert "test" in test
        assert test["test"] == "fisher_exact"
        assert "effect_size_metric" in test
        assert test["effect_size_metric"] == "cohens_h"
        assert "significant" in test
        assert "skipped" in test


def test_analyze_detects_behavioral_differentiation() -> None:
    """Strongly differentiated reversal rates should yield available evidence."""
    surface = _rich_surface_with_reversals()
    result = analyze_behavioral_outcomes(surface)
    # MATURE (100% reversal) vs YOUNG (~12%) should be highly significant
    assert result["behavioral_evidence_available"] is True
    assert result["behavioral_effect_size"] is not None
    assert result["behavioral_effect_size"] > 0.0


def test_analyze_skips_when_insufficient_episodes() -> None:
    """States with fewer than _MIN_EPISODE_COUNT episodes must produce skipped tests."""
    specs: list[tuple[str, int, str]] = []
    # Only 2 YOUNG episodes — below _MIN_EPISODE_COUNT
    specs.extend([("JPY_CONSENSUS_YOUNG", 3, "exit_reversal")] * 2)
    specs.extend([("JPY_CONSENSUS_MATURING", 8, "exit_unknown")] * 10)
    specs.extend([("JPY_CONSENSUS_MATURE", 20, "exit_reversal")] * 10)
    surface = _make_surface(specs)
    result = analyze_behavioral_outcomes(surface)
    skipped = [t for t in result["behavioral_tests"] if t["skipped"]]
    assert len(skipped) > 0
    for t in skipped:
        assert t["p_value"] is None
        assert t["effect_size"] is None


def test_analyze_effect_size_is_max_of_significant() -> None:
    """behavioral_effect_size should be the max Cohen's h among significant tests."""
    surface = _rich_surface_with_reversals()
    result = analyze_behavioral_outcomes(surface)
    if not result["behavioral_evidence_available"]:
        pytest.skip("no significant tests in this surface")
    significant_effects = [
        t["effect_size"]
        for t in result["behavioral_tests"]
        if t["significant"] and t["effect_size"] is not None
    ]
    assert result["behavioral_effect_size"] == pytest.approx(max(significant_effects))
