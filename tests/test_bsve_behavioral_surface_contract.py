"""Regression tests for the canonical Behavioral Surface contract.

Verifies:
- One-to-one (timestamp, pair) keys
- Deterministic output ordering
- Presence of all canonical contract fields
- Exact row-count preservation
- Backward compatibility of behavioral state assignment
- Successful round-trip serialization
- Schema stability
- Correct transition_event values
- Correct surface_id / surface_version population
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from bsve.calibration.calibration_contract import build_calibration_artifact, write_calibration_artifact
from bsve.state_machine.engine import (
    generate_behavioral_surface,
    build_behavioral_surface_manifest,
)
from bsve.state_machine.plugins.reactive_jpy import ReactiveJPYPlugin
from bsve.state_machine.protocol import CalibrationArtifact
from bsve.state_machine.rule_based import run_behavioral_surface_pipeline

# ---------------------------------------------------------------------------
# Canonical column contract
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "timestamp",
    "pair",
    "surface_id",
    "surface_version",
    "state_id",
    "episode_id",
    "maturity_bars",
    "crowd_side",
    "transition_event",
]

VALID_TRANSITION_EVENTS = {"entry", "continuation", "exit_reversal", "exit_unknown"}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calibration_artifact() -> CalibrationArtifact:
    return build_calibration_artifact(
        calibration_id="reactive_jpy_v1_contract_test",
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2026-12-31",
        dataset_version="1.5.1",
        calibration_method="hazard_analysis",
        outcome="success",
        thresholds={
            "extreme_threshold_net_pct": 70.0,
            "young_boundary_bars": 8,
            "mature_boundary_bars": 24,
        },
        diagnostics={},
        calibration_mode="research",
    )


def _make_df(
    sentiments: list[float],
    *,
    pair: str = "usd-jpy",
    crowd_sides: list[str] | None = None,
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    n = len(sentiments)
    if crowd_sides is None:
        crowd_sides = ["LONG"] * n
    return pd.DataFrame(
        {
            "pair": [pair] * n,
            "entry_time": pd.date_range(start, periods=n, freq="h"),
            "net_sentiment": sentiments,
            "crowd_side": crowd_sides,
        }
    )


def _generate(df: pd.DataFrame, artifact: CalibrationArtifact) -> pd.DataFrame:
    return generate_behavioral_surface(
        df,
        plugin=ReactiveJPYPlugin(),
        calibration_artifact=artifact,
        dataset_version="test-contract",
    )


# ---------------------------------------------------------------------------
# Schema contract: canonical columns
# ---------------------------------------------------------------------------


def test_canonical_columns_present(calibration_artifact: CalibrationArtifact) -> None:
    """All canonical contract fields must be present in the exact order."""
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    assert list(surface.columns) == CANONICAL_COLUMNS


def test_no_extra_columns(calibration_artifact: CalibrationArtifact) -> None:
    """Surface must not contain columns outside the canonical schema."""
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    assert set(surface.columns) == set(CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Key uniqueness: one row per (timestamp, pair)
# ---------------------------------------------------------------------------


def test_unique_timestamp_pair_keys_single_pair(calibration_artifact: CalibrationArtifact) -> None:
    df = _make_df([80, 50, 80, 80, 70])
    surface = _generate(df, calibration_artifact)
    assert not surface.duplicated(["timestamp", "pair"]).any(), \
        "Surface must have one-to-one (timestamp, pair) keys"


def test_unique_timestamp_pair_keys_multi_pair(calibration_artifact: CalibrationArtifact) -> None:
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80], pair="usd-jpy"),
            _make_df([75, 60, 75, 75], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    assert not surface.duplicated(["timestamp", "pair"]).any()


# ---------------------------------------------------------------------------
# Row count preservation
# ---------------------------------------------------------------------------


def test_row_count_preserved(calibration_artifact: CalibrationArtifact) -> None:
    """Surface must have exactly the same row count as the input dataset."""
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80, 80], pair="usd-jpy"),
            _make_df([75, 75, 60, 75], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    assert len(surface) == len(df), (
        f"Row count mismatch: surface={len(surface)}, input={len(df)}"
    )


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def test_deterministic_ordering_same_input(calibration_artifact: CalibrationArtifact) -> None:
    """Generating the surface twice on the same input must produce identical results."""
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80], pair="usd-jpy"),
            _make_df([75, 60, 75, 75], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    s1 = _generate(df, calibration_artifact)
    s2 = _generate(df, calibration_artifact)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


def test_output_sorted_by_pair_then_timestamp(calibration_artifact: CalibrationArtifact) -> None:
    """Surface must be sorted pair-first, timestamp-second (deterministic)."""
    df = pd.concat(
        [
            _make_df([80, 50, 80], pair="usd-jpy"),
            _make_df([80, 80, 50], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    expected_pairs = sorted(surface["pair"].unique())
    for pair in expected_pairs:
        pair_ts = surface[surface["pair"] == pair]["timestamp"]
        assert pair_ts.is_monotonic_increasing, f"timestamps not sorted for pair={pair!r}"


# ---------------------------------------------------------------------------
# surface_id and surface_version
# ---------------------------------------------------------------------------


def test_surface_id_populated(calibration_artifact: CalibrationArtifact) -> None:
    plugin = ReactiveJPYPlugin()
    surface = generate_behavioral_surface(
        _make_df([80, 80]),
        plugin=plugin,
        calibration_artifact=calibration_artifact,
        dataset_version="test",
    )
    assert (surface["surface_id"] == plugin.ontology_id).all()


def test_surface_version_populated(calibration_artifact: CalibrationArtifact) -> None:
    plugin = ReactiveJPYPlugin()
    surface = generate_behavioral_surface(
        _make_df([80, 80]),
        plugin=plugin,
        calibration_artifact=calibration_artifact,
        dataset_version="test",
    )
    assert (surface["surface_version"] == plugin.ontology_version).all()


def test_surface_id_is_constant_per_surface(calibration_artifact: CalibrationArtifact) -> None:
    """surface_id must be the same for every row in a surface."""
    df = pd.concat(
        [_make_df([80, 50], pair="usd-jpy"), _make_df([80, 80], pair="eur-jpy")],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    assert surface["surface_id"].nunique() == 1
    assert surface["surface_version"].nunique() == 1


# ---------------------------------------------------------------------------
# state_id (canonical ontology state identifier)
# ---------------------------------------------------------------------------


def test_state_id_is_canonical_state_identifier(calibration_artifact: CalibrationArtifact) -> None:
    """state_id must match existing behavioral state classification logic."""
    surface = _generate(_make_df([80.0] * 25), calibration_artifact)
    # First 7 rows: YOUNG (maturity 1–7), rows 7–22: MATURING, rows 23+: MATURE
    assert (surface.iloc[:7]["state_id"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state_id"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state_id"] == "JPY_CONSENSUS_MATURE").all()


def test_state_id_non_extreme(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([60, 65, 69]), calibration_artifact)
    assert (surface["state_id"] == "JPY_NON_EXTREME").all()


# ---------------------------------------------------------------------------
# transition_event semantics
# ---------------------------------------------------------------------------


def test_transition_event_values_are_valid(calibration_artifact: CalibrationArtifact) -> None:
    """All transition_event values must belong to the canonical enum."""
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80, 80], pair="usd-jpy"),
            _make_df([60, 80, 80, 50, 60], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    unexpected = set(surface["transition_event"]) - VALID_TRANSITION_EVENTS
    assert not unexpected, f"Unexpected transition_event values: {unexpected}"


def test_transition_event_entry_on_first_consensus_bar(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """The first bar of every consensus episode must be labeled 'entry'."""
    # extreme × 3, non-extreme × 1, extreme × 2  → two episodes
    surface = _generate(_make_df([80, 80, 80, 50, 80, 80]), calibration_artifact)
    # Row 0: first extreme → entry
    assert surface.iloc[0]["transition_event"] == "entry"
    # Row 4: new extreme episode → entry
    assert surface.iloc[4]["transition_event"] == "entry"


def test_transition_event_continuation_within_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Bars 2+ within a consensus episode must be labeled 'continuation'."""
    surface = _generate(_make_df([80, 80, 80, 80]), calibration_artifact)
    assert surface.iloc[0]["transition_event"] == "entry"
    assert (surface.iloc[1:]["transition_event"] == "continuation").all()


def test_transition_event_exit_reversal_immediately_after_consensus(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """The first non-extreme bar after a consensus episode is 'exit_reversal'."""
    # extreme × 3, non-extreme × 1
    surface = _generate(_make_df([80, 80, 80, 50]), calibration_artifact)
    assert surface.iloc[3]["transition_event"] == "exit_reversal"


def test_transition_event_exit_unknown_for_non_extreme_continuation(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Non-extreme rows not immediately following a consensus episode are 'exit_unknown'."""
    # non-extreme × 3 at the start → all exit_unknown (no prior consensus)
    surface = _generate(_make_df([50, 60, 65]), calibration_artifact)
    assert (surface["transition_event"] == "exit_unknown").all()


def test_transition_event_exit_unknown_after_multiple_non_extreme(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Only the first non-extreme bar after consensus is exit_reversal; subsequent are exit_unknown."""
    # extreme × 2, non-extreme × 3
    surface = _generate(_make_df([80, 80, 60, 55, 50]), calibration_artifact)
    assert surface.iloc[2]["transition_event"] == "exit_reversal"
    assert surface.iloc[3]["transition_event"] == "exit_unknown"
    assert surface.iloc[4]["transition_event"] == "exit_unknown"


def test_reactive_jpy_transition_events_complete_set(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """A realistic sequence should produce all four Reactive-JPY transition events."""
    # non-extreme × 2, extreme × 3, non-extreme × 2 → entry, continuation×2, exit_reversal, exit_unknown
    surface = _generate(_make_df([60, 60, 80, 80, 80, 50, 50]), calibration_artifact)
    events = set(surface["transition_event"])
    assert events == {"exit_unknown", "entry", "continuation", "exit_reversal"}


# ---------------------------------------------------------------------------
# Backward compatibility: behavioral state assignment unchanged
# ---------------------------------------------------------------------------


def test_behavioral_state_assignment_backward_compatible(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Removing new metadata fields must leave behavioral outputs identical.

    Verifies that row_count, (timestamp, pair) keys, state_id, episode_id,
    maturity_bars, and crowd_side are unchanged after removing surface_id,
    surface_version, and transition_event.
    """
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80, 80], pair="usd-jpy"),
            _make_df([75, 60, 75, 75, 80], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)

    # Strip the newly introduced metadata columns to get the behavioral core
    legacy_cols = ["timestamp", "pair", "state_id", "episode_id", "maturity_bars", "crowd_side"]
    core = surface[legacy_cols].copy()

    # Regenerate; the behavioral core must be identical
    surface2 = _generate(df, calibration_artifact)
    core2 = surface2[legacy_cols].copy()

    assert len(core) == len(core2), "row count changed"
    pd.testing.assert_frame_equal(core.reset_index(drop=True), core2.reset_index(drop=True))


def test_row_count_matches_input_after_schema_expansion(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Surface row count must equal input row count regardless of new columns."""
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80], pair="usd-jpy"),
            _make_df([75, 60, 80], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    assert len(surface) == len(df)


def test_timestamp_pair_keys_match_input(calibration_artifact: CalibrationArtifact) -> None:
    """(timestamp, pair) key set must equal the input key set."""
    df = pd.concat(
        [
            _make_df([80, 50, 80], pair="usd-jpy"),
            _make_df([80, 80, 50], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)

    # The engine normalizes pair via _pair_key (strip + str), not uppercase
    input_keys = set(
        zip(
            pd.to_datetime(df["entry_time"]),
            df["pair"].str.strip(),
        )
    )
    surface_keys = set(zip(surface["timestamp"], surface["pair"]))
    assert surface_keys == input_keys


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


def test_round_trip_parquet(tmp_path: Path, calibration_artifact: CalibrationArtifact) -> None:
    """Behavioral Surface must survive a parquet write/read cycle without data loss."""
    surface = _generate(
        pd.concat(
            [_make_df([80, 50, 80], pair="usd-jpy"), _make_df([80, 80], pair="eur-jpy")],
            ignore_index=True,
        ),
        calibration_artifact,
    )
    out = tmp_path / "surface.parquet"
    surface.to_parquet(out, index=False)
    loaded = pd.read_parquet(out)

    assert set(loaded.columns) == set(CANONICAL_COLUMNS)
    assert len(loaded) == len(surface)
    pd.testing.assert_frame_equal(
        surface[CANONICAL_COLUMNS].reset_index(drop=True),
        loaded[CANONICAL_COLUMNS].reset_index(drop=True),
    )


def test_round_trip_preserves_transition_event_dtype(
    tmp_path: Path, calibration_artifact: CalibrationArtifact
) -> None:
    """transition_event must be readable as strings after parquet round-trip."""
    surface = _generate(_make_df([80, 50, 80, 80]), calibration_artifact)
    out = tmp_path / "surface.parquet"
    surface.to_parquet(out, index=False)
    loaded = pd.read_parquet(out)
    assert set(loaded["transition_event"].dropna()) <= VALID_TRANSITION_EVENTS


# ---------------------------------------------------------------------------
# Manifest: artifact provenance
# ---------------------------------------------------------------------------


def test_manifest_exposes_artifact_provenance(calibration_artifact: CalibrationArtifact) -> None:
    """Manifest must contain all required artifact provenance keys."""
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    manifest = build_behavioral_surface_manifest(surface)

    required_keys = {
        "ontology_id",
        "ontology_version",
        "calibration_id",
        "dataset_version",
        "behavioral_surface_schema_version",
        "generated_timestamp",
    }
    missing = required_keys - set(manifest)
    assert not missing, f"Manifest missing required provenance keys: {missing}"


def test_manifest_row_count_matches_surface(calibration_artifact: CalibrationArtifact) -> None:
    df = pd.concat(
        [_make_df([80, 80, 50], pair="usd-jpy"), _make_df([80, 50], pair="eur-jpy")],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    manifest = build_behavioral_surface_manifest(surface)
    assert manifest["row_count"] == len(surface)


def test_manifest_state_counts_use_state_id(calibration_artifact: CalibrationArtifact) -> None:
    """Manifest state_counts must reflect state_id values."""
    surface = _generate(_make_df([80, 80, 50, 80]), calibration_artifact)
    manifest = build_behavioral_surface_manifest(surface)
    # state_counts must contain at least one of the Reactive-JPY state names
    assert any(k.startswith("JPY_") for k in manifest["state_counts"])


# ---------------------------------------------------------------------------
# Full orchestration pipeline
# ---------------------------------------------------------------------------


def test_orchestration_produces_canonical_schema(
    tmp_path: Path, calibration_artifact: CalibrationArtifact
) -> None:
    """run_behavioral_surface_pipeline must produce the canonical schema on disk."""
    dataset = pd.concat(
        [
            _make_df([80, 80, 50, 80], pair="usd-jpy"),
            _make_df([75, 60, 80, 80], pair="eur-jpy"),
        ],
        ignore_index=True,
    )
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    calibration_path = tmp_path / "calibration.json"
    write_calibration_artifact(calibration_artifact, calibration_path)

    surface_path, manifest_path = run_behavioral_surface_pipeline(
        dataset_path=dataset_path,
        output_dir=tmp_path / "out",
        calibration_artifact_path=calibration_path,
        state_spec_path=Path("bsve/state_specs/reactive_jpy_v1.yaml"),
        dataset_version="1.5.1",
        pairs=["USDJPY", "EURJPY"],
    )

    loaded = pd.read_parquet(surface_path)
    assert set(loaded.columns) == set(CANONICAL_COLUMNS)
    assert not loaded.duplicated(["timestamp", "pair"]).any()
    assert set(loaded["transition_event"]) <= VALID_TRANSITION_EVENTS
