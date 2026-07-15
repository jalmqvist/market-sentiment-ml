from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.deep_learning import train as train_module
WRITER_MODULE = importlib.import_module("scripts.write_dl_prediction_artifact")


def _synthetic_dataset(n_per_pair: int = 120) -> pd.DataFrame:
    rows = []
    for pair in ["eur-usd", "gbp-usd", "usd-jpy"]:
        times = pd.date_range("2020-01-01", periods=n_per_pair, freq="h")
        x = np.linspace(-1.0, 1.0, n_per_pair)
        rows.append(
            pd.DataFrame(
                {
                    "pair": pair,
                    "entry_time": times,
                    "regime": "LVTF",
                    "ret_24b": x,
                    "trend_12b": x,
                    "trend_24b": x * 0.8,
                    "trend_48b": x * 0.6,
                    "vol_12b": np.abs(x),
                    "vol_48b": np.abs(x) * 0.9,
                    "net_sentiment": x * 0.3,
                    "abs_sentiment": np.abs(x) * 0.3,
                    "sentiment_change": np.gradient(x),
                    "sentiment_z": x * 0.5,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _run_train_main(monkeypatch, tmp_path: Path, argv: list[str]) -> dict:
    captured: dict = {}

    def _fake_writer(df, identity, provenance, output_dir, run_id=None):
        captured["df"] = df.copy()
        captured["identity"] = dict(identity)
        captured["provenance"] = dict(provenance or {})
        pq = tmp_path / "artifact.parquet"
        mf = tmp_path / "artifact.manifest.json"
        pq.write_text("stub")
        mf.write_text("{}")
        return pq, mf

    monkeypatch.setattr(
        train_module,
        "load_dataset",
        lambda *args, **kwargs: _synthetic_dataset(),
    )
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])

    train_module.main()
    return captured


def _synthetic_behavioral_dataset(n_per_state: int = 120) -> pd.DataFrame:
    rows = []
    states = ["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]
    for state_idx, state in enumerate(states):
        times = pd.date_range("2020-01-01", periods=n_per_state, freq="h")
        x = np.linspace(-1.0, 1.0, n_per_state) + (0.01 * state_idx)
        rows.append(
            pd.DataFrame(
                {
                    "pair": "usd-jpy",
                    "entry_time": times,
                    "regime": "LVTF",
                    "surface_id": "reactive_jpy",
                    "state_id": state,
                    "episode_id": [f"ep-{state_idx}-{i}" for i in range(n_per_state)],
                    "maturity_bars": np.arange(1, n_per_state + 1),
                    "crowd_side": "LONG",
                    "transition_event": "continuation",
                    "ret_24b": x,
                    "trend_12b": x,
                    "trend_24b": x * 0.8,
                    "trend_48b": x * 0.6,
                    "vol_12b": np.abs(x),
                    "vol_48b": np.abs(x) * 0.9,
                    "net_sentiment": x * 0.3,
                    "abs_sentiment": np.abs(x) * 0.3,
                    "sentiment_change": np.gradient(x),
                    "sentiment_z": x * 0.5,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_pairs_flag_backward_compatibility(monkeypatch, tmp_path):
    captured = _run_train_main(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version",
            "test",
            "--pairs",
            "EURUSD",
            "--regime",
            "LVTF",
            "--epochs",
            "1",
        ],
    )

    assert set(captured["df"]["pair"].unique()) == {"eur-usd"}
    assert captured["provenance"]["training_pairs"] == ["eur-usd"]
    assert captured["provenance"]["inference_pairs"] == ["eur-usd"]


def test_cross_family_train_and_inference_pairs(monkeypatch, tmp_path):
    captured = _run_train_main(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version",
            "test",
            "--train-pairs",
            "EURUSD,GBPUSD",
            "--predict-pairs",
            "USDJPY",
            "--regime",
            "LVTF",
            "--epochs",
            "1",
            "--export-split",
            "test",
        ],
    )

    assert set(captured["df"]["pair"].unique()) == {"usd-jpy"}
    assert captured["provenance"]["training_pairs"] == ["eur-usd", "gbp-usd"]
    assert captured["provenance"]["inference_pairs"] == ["usd-jpy"]


def test_partition_arg_conflict_rejected_before_dataset_load(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_module,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_dataset should not be called")),
    )
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--dataset-version",
        "test",
        "--regime",
        "LVTF",
        "--surface",
        "reactive_jpy",
        "--state",
        "JPY_CONSENSUS_YOUNG",
    ])
    with np.testing.assert_raises_regex(ValueError, r"--regime, --surface, --state"):
        train_module.main()


def test_behavioral_filter_and_provenance(monkeypatch, tmp_path):
    dataset = _synthetic_behavioral_dataset()
    captured = {}

    def _fake_writer(df, identity, provenance, output_dir, run_id=None):
        captured["df"] = df.copy()
        captured["identity"] = dict(identity)
        captured["provenance"] = dict(provenance or {})
        pq = tmp_path / "artifact.parquet"
        mf = tmp_path / "artifact.manifest.json"
        pq.write_text("stub")
        mf.write_text("{}")
        return pq, mf

    monkeypatch.setattr(train_module, "load_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--dataset-version",
        "test",
        "--surface",
        "reactive_jpy",
        "--state",
        "JPY_CONSENSUS_YOUNG",
        "--epochs",
        "1",
        "--export-split",
        "all",
    ])

    train_module.main()

    assert len(captured["df"]) == 120
    assert captured["identity"]["dl_regime"] == "reactive_jpy:JPY_CONSENSUS_YOUNG"
    assert captured["identity"]["surface_id"] == "reactive_jpy"
    assert captured["identity"]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert captured["identity"]["surface_version"] == "unknown"
    assert captured["provenance"]["surface_id"] == "reactive_jpy"
    assert captured["provenance"]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert captured["provenance"]["surface_version"] == "unknown"
    assert captured["provenance"]["dataset_variant"] == "core"
    assert captured["provenance"]["dataset_version"] == "test"
    assert captured["provenance"]["ontology_version"] is None


def test_behavioral_states_create_distinct_artifacts(monkeypatch, tmp_path):
    dataset = _synthetic_behavioral_dataset()
    monkeypatch.setattr(train_module, "load_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)

    for state in ["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]:
        monkeypatch.setattr(sys, "argv", [
            "train.py",
            "--dataset-version",
            "test",
            "--surface",
            "reactive_jpy",
            "--state",
            state,
            "--epochs",
            "1",
            "--export-split",
            "all",
        ])
        train_module.main()

    parquet_files = sorted(tmp_path.glob("*.parquet"))
    manifest_files = sorted(tmp_path.glob("*.manifest.json"))
    assert len(parquet_files) == 2
    assert len(manifest_files) == 2

    manifest_dl_regimes = {
        json.loads(path.read_text(encoding="utf-8"))["identity"]["dl_regime"]
        for path in manifest_files
    }
    assert manifest_dl_regimes == {
        "reactive_jpy:JPY_CONSENSUS_YOUNG",
        "reactive_jpy:JPY_CONSENSUS_MATURE",
    }
    manifest_surface_state = set()
    for path in manifest_files:
        identity = json.loads(path.read_text(encoding="utf-8"))["identity"]
        manifest_surface_state.add((identity["surface_id"], identity["state_id"]))
    assert manifest_surface_state == {
        ("reactive_jpy", "JPY_CONSENSUS_YOUNG"),
        ("reactive_jpy", "JPY_CONSENSUS_MATURE"),
    }
