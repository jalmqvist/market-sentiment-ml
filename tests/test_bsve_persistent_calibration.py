from __future__ import annotations

import pandas as pd
import pytest

from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter
from bsve.calibration.persistent_calibration import PersistentCalibrationPlugin


def _dataset() -> pd.DataFrame:
    rows = []
    pairs = ["EURUSD", "GBPUSD", "NZDUSD", "EURGBP", "EURAUD"]
    for pair_idx, pair in enumerate(pairs):
        times = pd.date_range("2021-01-01", periods=16, freq="h")
        base = 10 + pair_idx
        sentiments = [
            base,
            base + 5,
            base + 10,
            base + 20,
            base + 30,
            base + 40,
            -(base + 15),
            -(base + 25),
            -(base + 35),
            -(base + 45),
            -(base + 50),
            base + 12,
            base + 22,
            base + 32,
            base + 42,
            base + 52,
        ]
        for ts, s in zip(times, sentiments):
            rows.append(
                {
                    "pair": pair,
                    "entry_time": ts,
                    "net_sentiment": float(s),
                    "crowd_side": 1 if s > 0 else -1,
                }
            )
    return pd.DataFrame(rows)


def _spec() -> dict:
    return {
        "environment": {
            "id": "persistent",
            "version": "0.1.0",
            "pairs": ["EURUSD", "GBPUSD", "NZDUSD", "EURGBP", "EURAUD"],
        },
        "minimum_history": {"level": 3, "trajectory": 4},
        "observed_segments": {"max_gap": "30D"},
    }


def test_persistent_calibration_builds_success_artifact() -> None:
    adapter = MasterResearchDatasetAdapter(_dataset())
    plugin = PersistentCalibrationPlugin()

    artifact = plugin.calibrate(
        adapter,
        _spec(),
        {
            "calibration_id": "persistent_fold_001",
            "dataset_version": "1.6.1",
            "calibration_window_start": "2021-01-01",
            "calibration_window_end": "2021-12-31",
            "calibration_method": "persistent_tertile_training_fold",
            "calibration_mode": "walkforward",
        },
    )

    assert artifact["ontology_id"] == "persistent"
    assert artifact["ontology_version"] == "0.1.0"
    assert artifact["dataset_version"] == "1.6.1"
    assert artifact["outcome"] == "success"

    th = artifact["thresholds"]
    assert th["min_level_history"] == 3
    assert th["min_trajectory_history"] == 4
    assert th["level_q33"] < th["level_q67"]
    assert th["trajectory_q33"] < th["trajectory_q67"]
    assert sorted(th["pairs"]) == ["eur-aud", "eur-gbp", "eur-usd", "gbp-usd", "nzd-usd"]


def test_persistent_calibration_rejects_degenerate_input() -> None:
    df = pd.DataFrame(
        {
            "pair": ["EURUSD"] * 12,
            "entry_time": pd.date_range("2021-01-01", periods=12, freq="h"),
            "net_sentiment": [50.0] * 12,
            "crowd_side": [1] * 12,
        }
    )
    adapter = MasterResearchDatasetAdapter(df)
    plugin = PersistentCalibrationPlugin()

    with pytest.raises(ValueError, match="distinct level values|degenerate level tertiles|empty level bin"):
        plugin.calibrate(
            adapter,
            _spec(),
            {
                "calibration_id": "persistent_fold_bad",
                "dataset_version": "1.6.1",
                "calibration_window_start": "2021-01-01",
                "calibration_window_end": "2021-12-31",
                "calibration_method": "persistent_tertile_training_fold",
            },
        )
