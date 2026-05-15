from __future__ import annotations

import importlib
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
