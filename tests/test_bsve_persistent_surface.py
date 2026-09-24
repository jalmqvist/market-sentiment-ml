from __future__ import annotations

import pandas as pd
import pytest

from bsve.calibration.calibration_contract import build_calibration_artifact
from bsve.dataset_augmentation import augment_with_behavioral_surface
from bsve.state_machine.engine import generate_behavioral_surface
from bsve.state_machine.plugins.persistent import PersistentPlugin


PERSISTENT_PAIRS = ["eur-usd", "gbp-usd", "nzd-usd", "eur-gbp", "eur-aud"]


@pytest.fixture()
def calibration_artifact() -> dict:
    return build_calibration_artifact(
        calibration_id="persistent_v0_1_0_test",
        ontology_id="persistent",
        ontology_version="0.1.0",
        calibration_window_start="2019-01-01",
        calibration_window_end="2023-12-31",
        dataset_version="1.6.1",
        calibration_method="persistent_tertile_training_fold",
        outcome="success",
        thresholds={
            "level_q33": 20.0,
            "level_q67": 35.0,
            "trajectory_q33": 4.0,
            "trajectory_q67": 10.0,
            "min_level_history": 3,
            "min_trajectory_history": 4,
        },
        diagnostics={},
    )


def _frame(
    sentiments: list[float],
    *,
    pair: str = "eur-usd",
    start: str = "2024-01-01 00:00:00",
    freq: str = "h",
) -> pd.DataFrame:
    n = len(sentiments)
    sides = ["LONG" if s > 0 else "SHORT" if s < 0 else "" for s in sentiments]
    return pd.DataFrame(
        {
            "pair": [pair] * n,
            "entry_time": pd.date_range(start, periods=n, freq=freq),
            "net_sentiment": sentiments,
            "crowd_side": sides,
        }
    )


def _generate(df: pd.DataFrame, artifact: dict, *, max_gap: str = "30D") -> pd.DataFrame:
    return generate_behavioral_surface(
        df,
        plugin=PersistentPlugin(),
        calibration_artifact=artifact,
        dataset_version="1.6.1",
        max_gap=max_gap,
    )


def test_persistent_pair_scope_constant() -> None:
    assert PersistentPlugin.SUPPORTED_PAIRS == set(PERSISTENT_PAIRS)


def test_unsupported_pair_rejected(calibration_artifact: dict) -> None:
    df = _frame([10, 20, 30, 40, 50], pair="usd-jpy")
    with pytest.raises(ValueError, match="unsupported persistent pair"):
        _generate(df, calibration_artifact)


def test_insufficient_history_remains_unassigned(calibration_artifact: dict) -> None:
    surface = _generate(_frame([10, 20, 30, 40]), calibration_artifact)
    assert surface["state_id"].isna().all()


def test_transition_to_valid_state_after_minimum_history(calibration_artifact: dict) -> None:
    surface = _generate(_frame([10, 20, 30, 40, 50]), calibration_artifact)
    assert surface.iloc[:4]["state_id"].isna().all()
    assert surface.iloc[4]["state_id"] == "PERSISTENT_MH"


def test_state_change_does_not_create_new_episode(calibration_artifact: dict) -> None:
    surface = _generate(_frame([10, 20, 30, 40, 50, 60, 10]), calibration_artifact)
    assert surface["episode_id"].nunique() == 1
    assert "state_transition" in set(surface["transition_event"])


def test_reversal_creates_new_episode(calibration_artifact: dict) -> None:
    df = pd.concat(
        [_frame([10, 20, 30, 40, 50]), _frame([-60, -55], start="2024-01-01 05:00:00")],
        ignore_index=True,
    )
    surface = _generate(df, calibration_artifact)
    assert surface.iloc[5]["episode_id"] != surface.iloc[4]["episode_id"]
    assert surface.iloc[5]["transition_event"] == "exit_reversal"


def test_gap_resets_observed_segment_history(calibration_artifact: dict) -> None:
    df = _frame([10, 20, 30, 40, 50])
    df.loc[4, "entry_time"] = pd.Timestamp("2024-01-03 00:00:00")
    surface = _generate(df, calibration_artifact, max_gap="12h")

    assert surface.iloc[4]["maturity_bars"] == 1
    assert surface.iloc[4]["episode_id"] != surface.iloc[3]["episode_id"]
    assert pd.isna(surface.iloc[4]["state_id"])
    assert surface.iloc[4]["transition_event"] == "entry"


def test_exit_unknown_for_missing_crowd_side(calibration_artifact: dict) -> None:
    df = _frame([10, 20, 30, 40, 50])
    df.loc[4, "crowd_side"] = ""
    surface = _generate(df, calibration_artifact)
    assert surface.iloc[4]["transition_event"] == "exit_unknown"


def test_deterministic_generation(calibration_artifact: dict) -> None:
    df = _frame([10, 20, 30, 40, 50, 60])
    s1 = _generate(df, calibration_artifact)
    s2 = _generate(df, calibration_artifact)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


def test_augmentation_path_accepts_persistent_surface(calibration_artifact: dict) -> None:
    base = _frame([10, 20, 30, 40, 50])
    surface = _generate(base, calibration_artifact)
    augmented, stats = augment_with_behavioral_surface(base, surface)
    assert len(augmented) == len(base)
    assert stats["rows_matched"] == len(base)
