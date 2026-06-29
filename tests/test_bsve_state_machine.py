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

    cols = ["state", "episode_id", "maturity_bars", "crowd_side"]
    pd.testing.assert_frame_equal(
        surface_base[cols].reset_index(drop=True),
        surface_extended.iloc[: len(surface_base)][cols].reset_index(drop=True),
    )


def test_running_maturity_not_final_duration(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80, 80, 80, 80]), calibration_artifact)
    assert list(surface["maturity_bars"]) == [1, 2, 3, 4]


def test_gap_handling_breaks_episode(calibration_artifact: CalibrationArtifact) -> None:
    df = _make_df([80, 80, 80])
    df.loc[2, "entry_time"] = pd.Timestamp("2024-01-01 04:00:00")  # 3h gap

    surface = _generate(df, calibration_artifact)
    assert list(surface["maturity_bars"]) == [1, 2, 1]
    assert surface.iloc[0]["episode_id"] == surface.iloc[1]["episode_id"]
    assert surface.iloc[2]["episode_id"] != surface.iloc[1]["episode_id"]


def test_episode_identity_breaks_on_crowd_side_change(
    calibration_artifact: CalibrationArtifact,
) -> None:
    surface = _generate(
        _make_df([80, 80, 80], crowd_sides=["LONG", "LONG", "SHORT"]),
        calibration_artifact,
    )
    assert list(surface["maturity_bars"]) == [1, 2, 1]
    assert surface.iloc[2]["episode_id"] != surface.iloc[1]["episode_id"]


def test_reactive_jpy_state_boundaries(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80.0] * 25), calibration_artifact)

    assert (surface.iloc[:7]["state"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state"] == "JPY_CONSENSUS_MATURE").all()


def test_extreme_threshold_boundaries(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([69.999, 70.000, 70.001]), calibration_artifact)
    assert list(surface["state"]) == [
        "JPY_NON_EXTREME",
        "JPY_CONSENSUS_YOUNG",
        "JPY_CONSENSUS_YOUNG",
    ]
    assert list(surface["maturity_bars"]) == [0, 1, 2]


def test_maturity_boundaries_exact_transitions(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80.0] * 24), calibration_artifact)
    assert surface.iloc[6]["maturity_bars"] == 7
    assert surface.iloc[6]["state"] == "JPY_CONSENSUS_YOUNG"
    assert surface.iloc[7]["maturity_bars"] == 8
    assert surface.iloc[7]["state"] == "JPY_CONSENSUS_MATURING"
    assert surface.iloc[22]["maturity_bars"] == 23
    assert surface.iloc[22]["state"] == "JPY_CONSENSUS_MATURING"
    assert surface.iloc[23]["maturity_bars"] == 24
    assert surface.iloc[23]["state"] == "JPY_CONSENSUS_MATURE"


def test_consensus_interruption_creates_new_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    surface = _generate(_make_df([80, 50, 80]), calibration_artifact)
    assert list(surface["maturity_bars"]) == [1, 0, 1]
    assert surface.iloc[0]["episode_id"] != surface.iloc[2]["episode_id"]


def test_non_extreme_rows_are_non_extreme_state(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([60, 65, 69]), calibration_artifact)
    assert (surface["state"] == "JPY_NON_EXTREME").all()
    assert (surface["maturity_bars"] == 0).all()


def test_surface_contains_required_columns(calibration_artifact: CalibrationArtifact) -> None:
    surface = _generate(_make_df([80, 80]), calibration_artifact)
    assert list(surface.columns) == [
        "timestamp",
        "pair",
        "state",
        "episode_id",
        "maturity_bars",
        "crowd_side",
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
        "state",
        "episode_id",
        "maturity_bars",
        "crowd_side",
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


def test_integer_crowd_side_zero_is_non_extreme(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side=0 (neutral) must not form an active consensus episode.

    With a non-extreme sentiment, the state is JPY_NON_EXTREME regardless of
    crowd_side. With extreme sentiment and crowd_side=0, maturity cannot
    accumulate (is_consensus_active returns False), so maturity stays at 0.
    """
    # Non-extreme sentiment → NON_EXTREME state, zero maturity.
    df = _make_df_int_sides([60, 65], [0, 0])
    surface = _generate(df, calibration_artifact)
    assert (surface["state"] == "JPY_NON_EXTREME").all()
    assert (surface["maturity_bars"] == 0).all()

    # Extreme sentiment but neutral crowd_side: maturity cannot accumulate.
    df2 = _make_df_int_sides([80, 80, 80], [0, 0, 0])
    surface2 = _generate(df2, calibration_artifact)
    assert (surface2["maturity_bars"] == 0).all()
    assert surface2["episode_id"].nunique() == 1  # same episode (no boundary from side or extreme change)


def test_integer_crowd_side_side_change_breaks_episode(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Flipping from integer 1 (LONG) to -1 (SHORT) must start a new episode."""
    df = _make_df_int_sides([80, 80, 80], [1, 1, -1])
    surface = _generate(df, calibration_artifact)

    assert list(surface["maturity_bars"]) == [1, 2, 1]
    assert surface.iloc[2]["episode_id"] != surface.iloc[1]["episode_id"]
    assert surface.iloc[0]["crowd_side"] == "LONG"
    assert surface.iloc[2]["crowd_side"] == "SHORT"


def test_integer_crowd_side_maturity_progression(
    calibration_artifact: CalibrationArtifact,
) -> None:
    """Integer crowd_side must produce the full YOUNG → MATURING → MATURE progression."""
    df = _make_df_int_sides([80.0] * 25, [1] * 25)
    surface = _generate(df, calibration_artifact)

    assert (surface.iloc[:7]["state"] == "JPY_CONSENSUS_YOUNG").all()
    assert (surface.iloc[7:23]["state"] == "JPY_CONSENSUS_MATURING").all()
    assert (surface.iloc[23:]["state"] == "JPY_CONSENSUS_MATURE").all()
