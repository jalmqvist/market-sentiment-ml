"""
tests/test_dl_dataset_variant.py
==================================
Regression tests for PR3.1 — Dataset Variant Support.

Covers:
- Default variant ("core") for MLP and LSTM
- Behavioral dataset variant for MLP and LSTM
- Artifact provenance records the actual variant used
- Manifest provenance records the actual variant used
- MLP/LSTM parity: both pipelines accept and record variants identically
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.deep_learning import train as train_module
from research.deep_learning import train_lstm as lstm_module

WRITER_MODULE = importlib.import_module("scripts.write_dl_prediction_artifact")


# ---------------------------------------------------------------------------
# Shared synthetic datasets
# ---------------------------------------------------------------------------

def _synthetic_mlp_dataset(n_per_pair: int = 120) -> pd.DataFrame:
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


def _synthetic_behavioral_mlp_dataset(n_per_state: int = 120) -> pd.DataFrame:
    rows = []
    for state_idx, state in enumerate(["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]):
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


def _synthetic_lstm_dataset(n_per_pair: int = 72) -> pd.DataFrame:
    rows = []
    for pair_idx, pair in enumerate(["eur-usd", "gbp-usd", "usd-jpy"]):
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


def _synthetic_behavioral_lstm_dataset(n_per_state: int = 96) -> pd.DataFrame:
    rows = []
    for state_idx, state in enumerate(["JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"]):
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


# ---------------------------------------------------------------------------
# Helpers to run train.py / train_lstm.py main() and capture artifact calls
# ---------------------------------------------------------------------------

def _run_mlp(monkeypatch, tmp_path: Path, argv: list[str], dataset=None) -> dict:
    captured: dict = {}

    def _fake_writer(df, identity, provenance, output_dir, run_id=None):
        captured["df"] = df.copy()
        captured["identity"] = dict(identity)
        captured["provenance"] = dict(provenance or {})
        pq = tmp_path / f"artifact_{len(list(tmp_path.glob('*.parquet')))}.parquet"
        mf = tmp_path / f"artifact_{len(list(tmp_path.glob('*.manifest.json')))}.manifest.json"
        pq.write_text("stub")
        mf.write_text(json.dumps({"identity": captured["identity"], "provenance": captured["provenance"]}))
        return pq, mf

    if dataset is None:
        dataset = _synthetic_mlp_dataset()
    monkeypatch.setattr(train_module, "load_dataset", lambda *a, **kw: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])

    train_module.main()
    return captured


def _run_lstm(monkeypatch, tmp_path: Path, argv: list[str], dataset=None) -> dict:
    captured: dict = {}

    def _fake_writer(df, identity, provenance, output_dir, run_id=None):
        captured["df"] = df.copy()
        captured["identity"] = dict(identity)
        captured["provenance"] = dict(provenance or {})
        pq = tmp_path / f"artifact_{len(list(tmp_path.glob('*.parquet')))}.parquet"
        mf = tmp_path / f"artifact_{len(list(tmp_path.glob('*.manifest.json')))}.manifest.json"
        pq.write_text("stub")
        mf.write_text(json.dumps({"identity": captured["identity"], "provenance": captured["provenance"]}))
        return pq, mf

    if dataset is None:
        dataset = _synthetic_lstm_dataset()
    monkeypatch.setattr(lstm_module, "load_dataset", lambda *a, **kw: dataset.copy())
    monkeypatch.setattr(WRITER_MODULE, "write_dl_prediction_artifact", _fake_writer)
    monkeypatch.setattr(WRITER_MODULE, "PREDICTIONS_DIR_DEFAULT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_lstm.py", *argv])

    lstm_module.main()
    return captured


# ---------------------------------------------------------------------------
# Tests: default variant ("core") — backwards-compatibility regression
# ---------------------------------------------------------------------------

def test_mlp_default_variant_is_core(monkeypatch, tmp_path):
    """Omitting --dataset-variant records 'core' in provenance (MLP)."""
    captured = _run_mlp(
        monkeypatch,
        tmp_path,
        ["--dataset-version", "1.5.1", "--regime", "LVTF", "--epochs", "1"],
    )
    assert captured["provenance"]["dataset_variant"] == "core"
    assert captured["provenance"]["dataset_version"] == "1.5.1"


def test_lstm_default_variant_is_core(monkeypatch, tmp_path):
    """Omitting --dataset-variant records 'core' in provenance (LSTM)."""
    captured = _run_lstm(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--regime", "LVTF",
            "--epochs", "1",
            "--seq-len", "4",
        ],
    )
    assert captured["provenance"]["dataset_variant"] == "core"
    assert captured["provenance"]["dataset_version"] == "1.5.1"


# ---------------------------------------------------------------------------
# Tests: Behavioral dataset variant
# ---------------------------------------------------------------------------

def test_mlp_behavioral_variant_recorded_in_provenance(monkeypatch, tmp_path):
    """--dataset-variant reactive_jpy_v1_core is recorded in artifact provenance (MLP)."""
    dataset = _synthetic_behavioral_mlp_dataset()
    captured = _run_mlp(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    assert captured["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert captured["provenance"]["dataset_version"] == "1.5.1"


def test_lstm_behavioral_variant_recorded_in_provenance(monkeypatch, tmp_path):
    """--dataset-variant reactive_jpy_v1_core is recorded in artifact provenance (LSTM)."""
    dataset = _synthetic_behavioral_lstm_dataset()
    captured = _run_lstm(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--seq-len", "4",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    assert captured["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert captured["provenance"]["dataset_version"] == "1.5.1"


# ---------------------------------------------------------------------------
# Tests: Artifact provenance — variant propagates to behavioral provenance block
# ---------------------------------------------------------------------------

def test_mlp_behavioral_provenance_uses_selected_variant(monkeypatch, tmp_path):
    """Behavioral provenance block records the selected variant, not a hard-coded value (MLP)."""
    dataset = _synthetic_behavioral_mlp_dataset()
    captured = _run_mlp(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    # Behavioral provenance block (merged via resolve_behavioral_provenance)
    # must reflect the selected variant rather than a hard-coded "core"
    assert captured["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert captured["provenance"]["surface_id"] == "reactive_jpy"
    assert captured["provenance"]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert captured["provenance"]["ontology_version"] is None


def test_lstm_behavioral_provenance_uses_selected_variant(monkeypatch, tmp_path):
    """Behavioral provenance block records the selected variant, not a hard-coded value (LSTM)."""
    dataset = _synthetic_behavioral_lstm_dataset()
    captured = _run_lstm(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--seq-len", "4",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    assert captured["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert captured["provenance"]["surface_id"] == "reactive_jpy"
    assert captured["provenance"]["state_id"] == "JPY_CONSENSUS_YOUNG"
    assert captured["provenance"]["ontology_version"] is None


# ---------------------------------------------------------------------------
# Tests: Manifest provenance — written JSON contains the actual variant
# ---------------------------------------------------------------------------

def test_mlp_manifest_records_actual_variant(monkeypatch, tmp_path):
    """Manifest JSON file records the actual dataset variant (MLP)."""
    dataset = _synthetic_behavioral_mlp_dataset()
    _run_mlp(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    manifest_files = list(tmp_path.glob("*.manifest.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"


def test_lstm_manifest_records_actual_variant(monkeypatch, tmp_path):
    """Manifest JSON file records the actual dataset variant (LSTM)."""
    dataset = _synthetic_behavioral_lstm_dataset()
    _run_lstm(
        monkeypatch,
        tmp_path,
        [
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--surface", "reactive_jpy",
            "--state", "JPY_CONSENSUS_YOUNG",
            "--epochs", "1",
            "--seq-len", "4",
            "--export-split", "all",
        ],
        dataset=dataset,
    )
    manifest_files = list(tmp_path.glob("*.manifest.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"


# ---------------------------------------------------------------------------
# Tests: MLP/LSTM parity — both pipelines produce identical provenance fields
# ---------------------------------------------------------------------------

def test_mlp_lstm_parity_default_variant(monkeypatch, tmp_path):
    """MLP and LSTM record identical dataset_variant provenance for default 'core' variant."""
    mlp_cap = _run_mlp(
        monkeypatch,
        tmp_path / "mlp",
        ["--dataset-version", "1.5.1", "--regime", "LVTF", "--epochs", "1"],
    )
    (tmp_path / "mlp").mkdir(exist_ok=True)
    lstm_cap = _run_lstm(
        monkeypatch,
        tmp_path / "lstm",
        [
            "--dataset-version", "1.5.1",
            "--regime", "LVTF",
            "--epochs", "1",
            "--seq-len", "4",
        ],
    )
    assert mlp_cap["provenance"]["dataset_variant"] == lstm_cap["provenance"]["dataset_variant"] == "core"
    assert mlp_cap["provenance"]["dataset_version"] == lstm_cap["provenance"]["dataset_version"] == "1.5.1"


def test_mlp_lstm_parity_behavioral_variant(monkeypatch, tmp_path):
    """MLP and LSTM record identical dataset_variant provenance for a Behavioral variant."""
    mlp_dataset = _synthetic_behavioral_mlp_dataset()
    lstm_dataset = _synthetic_behavioral_lstm_dataset()

    common_argv_suffix = [
        "--dataset-version", "1.5.1",
        "--dataset-variant", "reactive_jpy_v1_core",
        "--surface", "reactive_jpy",
        "--state", "JPY_CONSENSUS_YOUNG",
        "--epochs", "1",
        "--export-split", "all",
    ]

    (tmp_path / "mlp").mkdir()
    mlp_cap = _run_mlp(
        monkeypatch,
        tmp_path / "mlp",
        common_argv_suffix,
        dataset=mlp_dataset,
    )

    (tmp_path / "lstm").mkdir()
    lstm_cap = _run_lstm(
        monkeypatch,
        tmp_path / "lstm",
        common_argv_suffix + ["--seq-len", "4"],
        dataset=lstm_dataset,
    )

    assert mlp_cap["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert lstm_cap["provenance"]["dataset_variant"] == "reactive_jpy_v1_core"
    assert mlp_cap["provenance"]["dataset_version"] == lstm_cap["provenance"]["dataset_version"]
    assert mlp_cap["provenance"]["dataset_variant"] == lstm_cap["provenance"]["dataset_variant"]


# ---------------------------------------------------------------------------
# Tests: dataset_loader accepts arbitrary variant without raising ValueError
# ---------------------------------------------------------------------------

def test_dataset_loader_accepts_behavioral_variant(tmp_path):
    """load_dataset no longer restricts variants to a fixed allowlist."""
    from research.deep_learning.dataset_loader import load_dataset
    import config as cfg

    version = "test_variant_v"
    variant = "reactive_jpy_v1_core"
    out_dir = Path(cfg.OUTPUT_DIR) / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal CSV that satisfies loader requirements
    df = pd.DataFrame({
        "snapshot_time": pd.date_range("2020-01-01", periods=5, freq="h"),
        "pair": "usd-jpy",
        "ret_48b": [0.1, -0.1, 0.2, -0.2, 0.0],
    })
    csv_path = out_dir / f"master_research_dataset_{variant}.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset(version, variant=variant)
    assert len(loaded) == 5
