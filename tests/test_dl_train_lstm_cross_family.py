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

from research.deep_learning import train_lstm as lstm_module

WRITER_MODULE = importlib.import_module("scripts.write_dl_prediction_artifact")


def _synthetic_lstm_dataset(n_per_pair: int = 72) -> pd.DataFrame:
    rows = []
    pairs = ["eur-usd", "gbp-usd", "usd-jpy"]
    for pair_idx, pair in enumerate(pairs):
        snapshot_time = pd.date_range("2020-01-01", periods=n_per_pair, freq="30min")
        entry_time = snapshot_time.floor("h")
        x = np.linspace(-1.0, 1.0, n_per_pair) + (0.01 * pair_idx)
        rows.append(
            pd.DataFrame(
                {
                    "pair": pair,
                    "snapshot_time": snapshot_time,
                    "entry_time": entry_time,
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


def _run_lstm_main(monkeypatch, tmp_path: Path, argv: list[str]) -> dict:
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
        lstm_module,
        "load_dataset",
        lambda *args, **kwargs: _synthetic_lstm_dataset(),
    )
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_lstm.py", *argv])

    lstm_module.main()
    return captured


def _synthetic_behavioral_lstm_dataset(n_per_state: int = 96) -> pd.DataFrame:
    rows = []
    states = ["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]
    for state_idx, state in enumerate(states):
        snapshot_time = pd.date_range("2020-01-01", periods=n_per_state, freq="30min")
        entry_time = snapshot_time.floor("h")
        x = np.linspace(-1.0, 1.0, n_per_state) + (0.01 * state_idx)
        rows.append(
            pd.DataFrame(
                {
                    "pair": "usd-jpy",
                    "snapshot_time": snapshot_time,
                    "entry_time": entry_time,
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


def test_build_sequences_pair_safe_and_metadata_aligned():
    df = _synthetic_lstm_dataset(n_per_pair=12).copy()
    df["target_direction"] = (df["ret_24b"] > 0).astype(int)
    features = [
        "trend_12b",
        "trend_24b",
        "trend_48b",
        "vol_12b",
        "vol_48b",
        "net_sentiment",
        "abs_sentiment",
        "sentiment_change",
        "sentiment_z",
    ]

    seq_len = 4
    X, y, meta_df = lstm_module.build_sequences(df, features, "target_direction", seq_len)

    assert len(X) == len(y) == len(meta_df)
    assert set(meta_df["pair"].unique()) == {"eur-usd", "gbp-usd", "usd-jpy"}
    assert meta_df.groupby("pair").size().to_dict() == {
        "eur-usd": 8,
        "gbp-usd": 8,
        "usd-jpy": 8,
    }
    for _, grp in meta_df.groupby("pair"):
        assert grp["snapshot_time"].is_monotonic_increasing


def test_lstm_cross_family_predict_universe_export(monkeypatch, tmp_path):
    captured = _run_lstm_main(
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
            "--seq-len",
            "4",
            "--export-split",
            "all",
        ],
    )

    assert set(captured["df"]["pair"].unique()) == {"usd-jpy"}
    assert captured["identity"]["model"] == "lstm"
    assert captured["provenance"]["training_pairs"] == ["eur-usd", "gbp-usd"]
    assert captured["provenance"]["inference_pairs"] == ["usd-jpy"]
    assert captured["df"][["pair", "entry_time"]].duplicated().sum() == 0


def test_partition_arg_conflict_rejected_before_dataset_load(monkeypatch, tmp_path):
    monkeypatch.setattr(
        lstm_module,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_dataset should not be called")),
    )
    monkeypatch.setattr(sys, "argv", [
        "train_lstm.py",
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
        lstm_module.main()


def test_behavioral_filter_and_provenance(monkeypatch, tmp_path):
    dataset = _synthetic_behavioral_lstm_dataset()
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

    monkeypatch.setattr(lstm_module, "load_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_lstm.py",
        "--dataset-version",
        "test",
        "--surface",
        "reactive_jpy",
        "--state",
        "JPY_CONSENSUS_YOUNG",
        "--epochs",
        "1",
        "--seq-len",
        "4",
        "--export-split",
        "all",
    ])

    lstm_module.main()

    filtered = dataset[
        (dataset["surface_id"] == "reactive_jpy") &
        (dataset["state_id"] == "JPY_CONSENSUS_YOUNG")
    ].copy()
    filtered["target_direction"] = 0
    _, _, infer_meta_df = lstm_module.build_sequences(
        filtered,
        [
            "trend_12b",
            "trend_24b",
            "trend_48b",
            "vol_12b",
            "vol_48b",
            "net_sentiment",
            "abs_sentiment",
            "sentiment_change",
            "sentiment_z",
        ],
        "target_direction",
        4,
    )
    expected_rows = len(
        infer_meta_df.assign(entry_time=pd.to_datetime(infer_meta_df["entry_time"]).dt.tz_localize(None))
        [["pair", "entry_time"]]
        .drop_duplicates()
    )

    assert len(captured["df"]) == expected_rows
    assert captured["identity"]["dl_regime"] == "reactive_jpy:JPY_CONSENSUS_YOUNG"
    assert captured["provenance"]["surface_id"] == "reactive_jpy"
    assert captured["provenance"]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert captured["provenance"]["dataset_variant"] == "core"
    assert captured["provenance"]["dataset_version"] == "test"
    assert captured["provenance"]["ontology_version"] is None


def test_behavioral_states_create_distinct_artifacts(monkeypatch, tmp_path):
    dataset = _synthetic_behavioral_lstm_dataset()
    monkeypatch.setattr(lstm_module, "load_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)

    for state in ["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]:
        monkeypatch.setattr(sys, "argv", [
            "train_lstm.py",
            "--dataset-version",
            "test",
            "--surface",
            "reactive_jpy",
            "--state",
            state,
            "--epochs",
            "1",
            "--seq-len",
            "4",
            "--export-split",
            "all",
        ])
        lstm_module.main()

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
