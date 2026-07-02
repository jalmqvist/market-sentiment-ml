"""Tests for PR4 Behavioral Surface Generator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bsve.calibration.calibration_contract import (
    build_calibration_artifact,
    write_calibration_artifact,
)
from bsve.state_machine.engine import (
    BehavioralSurfaceEngine,
    build_behavioral_surface_manifest,
    generate_behavioral_surface,
)
from bsve.state_machine.protocol import CalibrationArtifact
from bsve.state_machine.plugins.reactive_jpy import ReactiveJPYPlugin
from bsve.state_machine.rule_based import run_behavioral_surface_pipeline


@pytest.fixture()
def calibration_artifact() -> CalibrationArtifact:
    return build_calibration_artifact(
        calibration_id="reactive_jpy_v1_20260615",
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
    assert len(crowd_sides) == n

    return pd.DataFrame(
        {
            "pair": [pair] * n,
            "entry_time": pd.date_range(start, periods=n, freq="h"),
            "net_sentiment": sentiments,
            "crowd_side": crowd_sides,
        }
    )


def _generate(df: pd.DataFrame, calibration_artifact: CalibrationArtifact) -> pd.DataFrame:
    return generate_behavioral_surface(
        df,
        plugin=ReactiveJPYPlugin(),
        calibration_artifact=calibration_artifact,
        dataset_version="test-1",
    )


def test_streaming_equivalence(calibration_artifact: CalibrationArtifact) -> None:
    df = pd.concat(
        [
            _make_df([80, 80, 50, 80, 80], pair="usd-jpy"),
            _make_df([75, 75, 75, 60, 75], pair="eur-jpy"),
        ],
        ignore_index=True,
    ).sort_values(["pair", "entry_time"], kind="mergesort")

    batch = _generate(df, calibration_artifact)

    engine = BehavioralSurfaceEngine(
        plugin=ReactiveJPYPlugin(),
        calibration_artifact=calibration_artifact,
    )
    rows = [engine.process_observation(row) for row in df.to_dict(orient="records")]
    stream = pd.DataFrame(rows)[batch.columns]

    pd.testing.assert_frame_equal(batch.reset_index(drop=True), stream.reset_index(drop=True))


def test_causal_alignment_when_appending_future_rows(
    calibration_artifact: CalibrationArtifact,
) -> None:
    base = _make_df([80, 80, 50, 80, 80])
    future = _make_df([80, 80], start="2024-01-01 05:00:00")

    surface_base = _generate(base, calibration_artifact)
    surface_extended = _generate(pd.concat([base, future], ignore_index=True), calibration_artifact)

    cols = ["state_id", "episode_id", "maturity_bars", "crowd_side"]
    pd.testing.assert_frame_equal(
        surface_base[cols].reset_index(drop=True),
        surface_extended.iloc[: len(surface_base)][cols].reset_index(drop=True),
    )


def test_running_maturity_not_final_duration(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80, 80, 80, 80]), calibration_artifact)
    assert list(surface["maturity_bars"]) == [1, 2, 3, 4]


def test_gap_does_not_break_episode_by_default(calibration_artifact: CalibrationArtifact) -> None:
    """Large timestamp gaps do NOT break episodes by default (max_gap=None).

    The Reactive-JPY calibration defines episodes purely by whether
    abs(net_sentiment) >= extreme_threshold.  It does not use wall-clock time
    to terminate episodes.  The engine must match that semantics by default.
    """
    df = _make_df([80, 80, 80])
    # Introduce a 3-hour gap — historically this would have fragmented the episode.
    df.loc[2, "entry_time"] = pd.Timestamp("2024-01-01 04:00:00")

    surface = _generate(df, calibration_artifact)
    # All observations have extreme sentiment → one continuous episode.
    assert list(surface["maturity_bars"]) == [1, 2, 3]
    assert surface["episode_id"].nunique() == 1


def test_gap_breaks_episode_when_max_gap_is_set(calibration_artifact: CalibrationArtifact) -> None:
    """An explicit max_gap still fragments episodes when the gap exceeds the threshold."""
    df = _make_df([80, 80, 80])
    df.loc[2, "entry_time"] = pd.Timestamp("2024-01-01 04:00:00")  # 3h gap

    engine = BehavioralSurfaceEngine(
        plugin=ReactiveJPYPlugin(),
        calibration_artifact=calibration_artifact,
        max_gap="1h",
    )
    rows = [engine.process_observation(r) for r in df.to_dict(orient="records")]
    surface = pd.DataFrame(rows)

    assert list(surface["maturity_bars"]) == [1, 2, 1]
    assert surface.iloc[0]["episode_id"] == surface.iloc[1]["episode_id"]
    assert surface.iloc[2]["episode_id"] != surface.iloc[1]["episode_id"]


def test_episode_identity_unbroken_across_crowd_side_change(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Crowd-side change no longer breaks an episode — only sentiment threshold matters.

    Calibration ground truth uses only abs(net_sentiment) >= threshold to define
    episode boundaries.  A LONG→SHORT transition within an extreme sentiment run
    does not end the episode.
    """
    surface = _generate(
        _make_df([80, 80, 80], crowd_sides=["LONG", "LONG", "SHORT"]),
        calibration_artifact,
    )
    assert list(surface["maturity_bars"]) == [1, 2, 3]
    assert surface.iloc[2]["episode_id"] == surface.iloc[1]["episode_id"]


def test_reactive_jpy_state_boundaries(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80.0] * 25), calibration_artifact)

    assert (surface.iloc[:7]["state_id"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state_id"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state_id"] == "JPY_CONSENSUS_MATURE").all()


def test_extreme_threshold_boundaries(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([69.999, 70.000, 70.001]), calibration_artifact)
    assert list(surface["state_id"]) == [
        "JPY_NON_EXTREME",
        "JPY_CONSENSUS_YOUNG",
        "JPY_CONSENSUS_YOUNG",
    ]
    assert list(surface["maturity_bars"]) == [0, 1, 2]


def test_maturity_boundaries_exact_transitions(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80.0] * 24), calibration_artifact)
    assert surface.iloc[6]["maturity_bars"] == 7
    assert surface.iloc[6]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert surface.iloc[7]["maturity_bars"] == 8
    assert surface.iloc[7]["state_id"] == "JPY_CONSENSUS_MATURING"
    assert surface.iloc[22]["maturity_bars"] == 23
    assert surface.iloc[22]["state_id"] == "JPY_CONSENSUS_MATURING"
    assert surface.iloc[23]["maturity_bars"] == 24
    assert surface.iloc[23]["state_id"] == "JPY_CONSENSUS_MATURE"


def test_consensus_interruption_creates_new_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    assert list(surface["maturity_bars"]) == [1, 0, 1]
    assert surface.iloc[0]["episode_id"] != surface.iloc[2]["episode_id"]


def test_non_extreme_rows_are_non_extreme_state(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([60, 65, 69]), calibration_artifact)
    assert (surface["state_id"] == "JPY_NON_EXTREME").all()
    assert (surface["maturity_bars"] == 0).all()


def test_surface_contains_required_columns(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80, 80]), calibration_artifact)
    assert list(surface.columns) == [
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


def test_provenance_and_manifest(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    provenance = surface.attrs["provenance"]

    for key in [
        "ontology_id",
        "ontology_version",
        "calibration_id",
        "calibration_hash",
        "schema_version",
        "dataset_version",
        "generated_timestamp",
    ]:
        assert key in provenance

    manifest = build_behavioral_surface_manifest(surface)
    for key in [
        "ontology_id",
        "ontology_version",
        "calibration_id",
        "calibration_hash",
        "dataset_version",
        "row_count",
        "pair_counts",
        "state_counts",
        "schema_version",
        "behavioral_surface_schema_version",
        "generated_timestamp",
    ]:
        assert key in manifest
    assert manifest["row_count"] == len(surface)


def test_orchestration_pipeline_exports_surface_and_manifest(
    tmp_path: Path,
    calibration_artifact: CalibrationArtifact,
) -> None:
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

    assert surface_path.exists()
    assert manifest_path.exists()

    surface = pd.read_parquet(surface_path)
    assert set(surface.columns) == {
        "timestamp",
        "pair",
        "surface_id",
        "surface_version",
        "state_id",
        "episode_id",
        "maturity_bars",
        "crowd_side",
        "transition_event",
    }

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_version"] == "1.5.1"


# ---------------------------------------------------------------------------
# Integer crowd-side encoding (master research dataset uses 1/-1/0)
# ---------------------------------------------------------------------------


def _make_df_int_sides(
    sentiments: list[float],
    sides: list[int],
    *,
    pair: str = "usd-jpy",
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Build a test dataset using integer crowd_side encoding (1=LONG, -1=SHORT, 0=neutral)."""
    n = len(sentiments)
    assert len(sides) == n
    return pd.DataFrame(
        {
            "pair": [pair] * n,
            "entry_time": pd.date_range(start, periods=n, freq="h"),
            "net_sentiment": sentiments,
            "crowd_side": sides,
        }
    )


def test_integer_crowd_side_long_episode_continues(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side=1 (LONG) with extreme sentiment must form a continuous episode."""
    df = _make_df_int_sides([80, 80, 80, 80], [1, 1, 1, 1])
    surface = _generate(df, calibration_artifact)

    assert list(surface["maturity_bars"]) == [1, 2, 3, 4]
    assert surface["episode_id"].nunique() == 1
    assert (surface["crowd_side"] == "LONG").all()


def test_integer_crowd_side_short_episode_continues(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side=-1 (SHORT) with extreme sentiment must form a continuous episode."""
    df = _make_df_int_sides([80, 80, 80], [-1, -1, -1])
    surface = _generate(df, calibration_artifact)

    assert list(surface["maturity_bars"]) == [1, 2, 3]
    assert surface["episode_id"].nunique() == 1
    assert (surface["crowd_side"] == "SHORT").all()


def test_integer_crowd_side_zero_non_extreme_sentiment(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side=0 (neutral) with non-extreme sentiment stays NON_EXTREME.

    Crowd-side is no longer part of the consensus-active predicate; only
    abs(net_sentiment) >= threshold matters.  Non-extreme sentiment therefore
    produces NON_EXTREME state regardless of crowd_side.
    """
    # Non-extreme sentiment → NON_EXTREME state, zero maturity.
    df = _make_df_int_sides([60, 65], [0, 0])
    surface = _generate(df, calibration_artifact)
    assert (surface["state_id"] == "JPY_NON_EXTREME").all()
    assert (surface["maturity_bars"] == 0).all()

    # Extreme sentiment with neutral crowd_side: maturity accumulates because
    # is_consensus_active depends only on abs(net_sentiment) >= threshold.
    df2 = _make_df_int_sides([80, 80, 80], [0, 0, 0])
    surface2 = _generate(df2, calibration_artifact)
    assert list(surface2["maturity_bars"]) == [1, 2, 3]
    assert surface2["episode_id"].nunique() == 1


def test_integer_crowd_side_change_does_not_break_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Flipping from integer 1 (LONG) to -1 (SHORT) must NOT start a new episode.

    The calibration ground truth defines episode boundaries solely by
    abs(net_sentiment) crossing the extreme threshold.  A crowd-side reversal
    within an extreme sentiment run is not an episode boundary.
    """
    df = _make_df_int_sides([80, 80, 80], [1, 1, -1])
    surface = _generate(df, calibration_artifact)

    assert list(surface["maturity_bars"]) == [1, 2, 3]
    assert surface["episode_id"].nunique() == 1
    assert surface.iloc[0]["crowd_side"] == "LONG"
    assert surface.iloc[2]["crowd_side"] == "SHORT"


def test_integer_crowd_side_maturity_progression(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side must produce the full YOUNG → MATURING → MATURE progression."""
    df = _make_df_int_sides([80.0] * 25, [1] * 25)
    surface = _generate(df, calibration_artifact)

    assert (surface.iloc[:7]["state_id"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state_id"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state_id"] == "JPY_CONSENSUS_MATURE").all()


# ---------------------------------------------------------------------------
# Episode continuity regression tests — calibration semantics alignment
# ---------------------------------------------------------------------------


def _make_df_with_gaps(
    sentiments: list[float],
    *,
    pair: str = "usd-jpy",
    gap_hours: float = 8.0,
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Build a dataset with large gaps between observations (simulates ~3x/day sampling)."""
    n = len(sentiments)
    start_ts = pd.Timestamp(start)
    timestamps = [start_ts + pd.Timedelta(hours=gap_hours * i) for i in range(n)]
    return pd.DataFrame(
        {
            "pair": [pair] * n,
            "entry_time": timestamps,
            "net_sentiment": sentiments,
            "crowd_side": ["LONG"] * n,
        }
    )


def test_large_gap_extreme_sentiment_forms_single_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Consecutive extreme-sentiment observations spaced 8h apart form one episode.

    The historical dataset is sampled ~3 times per day (~8h gaps).  The
    Reactive-JPY calibration never terminates an episode because of the
    inter-observation gap; it only terminates when abs(net_sentiment) drops
    below the threshold.  The engine must reproduce this behaviour.
    """
    df = _make_df_with_gaps([80.0] * 10, gap_hours=8.0)
    surface = _generate(df, calibration_artifact)

    assert surface["episode_id"].nunique() == 1, "all observations must share one episode"
    assert list(surface["maturity_bars"]) == list(range(1, 11))


def test_episode_interrupted_by_non_extreme_then_resumes(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """A non-extreme observation ends the episode; re-entering extreme starts a new one.

    This matches calibration semantics exactly: the boundary is the
    threshold crossing, not the clock.
    """
    # extreme × 5 → non-extreme × 1 → extreme × 5
    sentiments = [80.0] * 5 + [60.0] + [80.0] * 5
    df = _make_df_with_gaps(sentiments, gap_hours=8.0)
    surface = _generate(df, calibration_artifact)

    first_ep = surface.iloc[0]["episode_id"]
    # Rows 0-4: extreme → episode 1
    assert all(surface.iloc[i]["episode_id"] == first_ep for i in range(5))
    # Row 5: non-extreme → maturity resets to 0
    assert surface.iloc[5]["maturity_bars"] == 0
    # Rows 6-10: new extreme episode
    second_ep = surface.iloc[6]["episode_id"]
    assert second_ep != first_ep
    assert list(surface.iloc[6:]["maturity_bars"]) == list(range(1, 6))


def test_maturing_and_mature_states_reachable_with_large_gaps(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """MATURING and MATURE states must be reachable even when observations are far apart.

    With the previous max_gap="1h" default, datasets sampled ~3x/day (~8h
    gaps) could never reach MATURING (requires maturity >= 8) or MATURE
    (requires maturity >= 24) because every observation triggered a new
    episode.  After the fix, a long enough run of extreme observations should
    advance through all three maturity states.
    """
    # 25 observations with extreme sentiment, 8h spacing
    df = _make_df_with_gaps([80.0] * 25, gap_hours=8.0)
    surface = _generate(df, calibration_artifact)

    states = list(surface["state_id"])
    assert "JPY_CONSENSUS_MATURING" in states, "MATURING state must be reachable"
    assert "JPY_CONSENSUS_MATURE" in states, "MATURE state must be reachable"
    # Verify exact boundaries match calibration thresholds (young<8, maturing<24, mature>=24)
    assert (surface.iloc[:7]["state_id"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state_id"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state_id"] == "JPY_CONSENSUS_MATURE").all()


def test_episode_survival_counts_match_calibration_semantics(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Verify episode survival statistics match calibration replay on the same mini-dataset.

    Constructs a synthetic dataset that mirrors the calibration's episode
    extraction logic and asserts that the engine produces the same episode
    count and maximum episode length.
    """
    from bsve.calibration.jpy_maturity_calibration import extract_consensus_lifecycles

    # Build a dataset: two long extreme runs separated by a non-extreme gap
    # Run A: 30 extreme observations (8h gaps)
    # Gap: 3 non-extreme observations
    # Run B: 15 extreme observations (8h gaps)
    sentiments = [80.0] * 30 + [60.0] * 3 + [80.0] * 15
    df = _make_df_with_gaps(sentiments, gap_hours=8.0)

    # --- Calibration replay (ground truth) ---
    cal_df = df.rename(columns={"net_sentiment": "sentiment_net"})
    extreme_threshold = float(
        calibration_artifact["thresholds"]["extreme_threshold_net_pct"]
    )
    lifecycles = extract_consensus_lifecycles(cal_df, "usd-jpy", extreme_threshold, min_episode_bars=1)
    cal_episode_count = len(lifecycles)
    cal_max_length = max(lc.duration_bars for lc in lifecycles)

    # --- Engine replay ---
    surface = _generate(df, calibration_artifact)
    engine_episode_count = surface[surface["maturity_bars"] == 1]["episode_id"].nunique()
    engine_max_maturity = int(surface["maturity_bars"].max())

    # Engine episodes == calibration episodes (2 runs → 2 episodes)
    assert engine_episode_count == cal_episode_count, (
        f"episode count mismatch: engine={engine_episode_count}, "
        f"calibration={cal_episode_count}"
    )
    # Engine max maturity should match calibration max episode length (run A = 30)
    assert engine_max_maturity == cal_max_length, (
        f"max length mismatch: engine={engine_max_maturity}, "
        f"calibration={cal_max_length}"
    )
