"""Integration tests for PR7.3 — epoch_sweep mode in run_behavioral_suite.

Covers:
- Named epoch grids (EPOCH_GRIDS constant)
- _resolve_epoch_list with named grid and explicit --epoch-list
- --mode epoch_sweep CLI argument parsing
- _build_experiment_id prefix for epoch_sweep
- run_epoch_sweep end-to-end: generates sweep_summary.csv, per-epoch
  walk-forward outputs, and auto-invokes convergence analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.behavioral import run_behavioral_suite as suite
from analysis.behavioral.utils import RunResult


DATASET_VERSION = "1.5.1"
DATASET_VARIANT = "reactive_jpy_v1_core"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset(repo_root: Path) -> None:
    out_dir = repo_root / "data" / "output" / DATASET_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.date_range("2020-01-01", periods=36, freq="MS")
    rows = []
    for idx, t in enumerate(ts):
        rows.append(
            {
                "pair": "usd-jpy",
                "entry_time": t,
                "snapshot_time": t,
                "surface_id": "reactive_jpy",
                "state_id": "STATE_A" if idx % 2 == 0 else "STATE_B",
                "regime": "HV" if idx % 3 == 0 else "LV",
                "ret_24b": 0.05 if idx % 4 in {0, 1} else -0.02,
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / f"master_research_dataset_{DATASET_VARIANT}.csv",
        index=False,
    )


def _parse_flag(command: list[str], flag: str) -> str:
    i = command.index(flag)
    return command[i + 1]


def _build_fake_runner(repo_root: Path):
    calls: list[list[str]] = []

    def _runner(*, command: list[str], repo_root: Path, log_path: Path) -> RunResult:
        model = "mlp" if command[1].endswith("train.py") else "lstm"
        surface_id = _parse_flag(command, "--surface")
        state_id = _parse_flag(command, "--state")
        test_start = pd.Timestamp(_parse_flag(command, "--wf-test-start"))

        run_id = f"{model}_{surface_id}_{state_id}_{len(calls)}"
        pred_dir = repo_root / "data" / "output" / "dl_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        pq_path = pred_dir / f"{run_id}.parquet"
        mf_path = pred_dir / f"{run_id}.manifest.json"

        pred = pd.DataFrame(
            {
                "pair": ["usd-jpy", "usd-jpy"],
                "entry_time": [test_start, test_start + pd.Timedelta(days=1)],
                "pred_prob_up": [0.8, 0.2],
                "signal_strength": [0.6, -0.6],
            }
        )
        pred.to_parquet(pq_path, index=False)
        mf_path.write_text(
            json.dumps({"identity": {"model": model}, "provenance": {}}),
            encoding="utf-8",
        )

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("epoch sweep wf run", encoding="utf-8")
        calls.append(command)
        return RunResult(
            command=command,
            returncode=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            log_path=log_path,
            reported_parquet_path=pq_path,
            reported_manifest_path=mf_path,
        )

    return _runner, calls


# ---------------------------------------------------------------------------
# Named epoch grids
# ---------------------------------------------------------------------------

class TestEpochGrids:
    def test_default_grid_exists(self):
        assert "default" in suite.EPOCH_GRIDS
        assert suite.EPOCH_GRIDS["default"] == [5, 10, 25, 50, 75, 100, 125, 150, 200]

    def test_dense_grid_exists(self):
        assert "dense" in suite.EPOCH_GRIDS
        assert len(suite.EPOCH_GRIDS["dense"]) >= 5

    def test_publication_grid_exists(self):
        assert "publication" in suite.EPOCH_GRIDS
        assert suite.EPOCH_GRIDS["publication"] == [5, 10, 25, 50, 75, 100, 125, 150, 200, 300, 400, 500]

    def test_all_grids_are_sorted(self):
        for name, epochs in suite.EPOCH_GRIDS.items():
            assert epochs == sorted(epochs), f"Grid '{name}' is not sorted"

    def test_all_grid_epochs_positive(self):
        for name, epochs in suite.EPOCH_GRIDS.items():
            assert all(e >= 1 for e in epochs), f"Grid '{name}' has non-positive epochs"


# ---------------------------------------------------------------------------
# _resolve_epoch_list
# ---------------------------------------------------------------------------

class TestResolveEpochList:
    def _make_args(self, epoch_grid=None, epoch_list=None):
        import argparse
        args = argparse.Namespace(
            epoch_grid=epoch_grid or suite._DEFAULT_EPOCH_GRID,
            epoch_list=epoch_list,
        )
        return args

    def test_default_grid(self):
        args = self._make_args()
        result = suite._resolve_epoch_list(args)
        assert result == suite.EPOCH_GRIDS[suite._DEFAULT_EPOCH_GRID]

    def test_named_grid_dense(self):
        args = self._make_args(epoch_grid="dense")
        result = suite._resolve_epoch_list(args)
        assert result == suite.EPOCH_GRIDS["dense"]

    def test_epoch_list_overrides_grid(self):
        args = self._make_args(epoch_grid="default", epoch_list="5,10,20")
        result = suite._resolve_epoch_list(args)
        assert result == [5, 10, 20]

    def test_epoch_list_deduplicates_and_sorts(self):
        args = self._make_args(epoch_list="50,10,50,5")
        result = suite._resolve_epoch_list(args)
        assert result == [5, 10, 50]

    def test_epoch_list_single_value(self):
        args = self._make_args(epoch_list="42")
        assert suite._resolve_epoch_list(args) == [42]

    def test_epoch_list_empty_raises(self):
        args = self._make_args(epoch_list="   ")
        with pytest.raises(ValueError, match="--epoch-list is empty"):
            suite._resolve_epoch_list(args)

    def test_epoch_list_non_integer_raises(self):
        args = self._make_args(epoch_list="5,ten,20")
        with pytest.raises(ValueError, match="--epoch-list must be comma-separated integers"):
            suite._resolve_epoch_list(args)

    def test_epoch_list_zero_raises(self):
        args = self._make_args(epoch_list="0,10,20")
        with pytest.raises(ValueError, match="All epoch counts must be >= 1"):
            suite._resolve_epoch_list(args)

    def test_unknown_grid_raises(self):
        args = self._make_args(epoch_grid="nonexistent")
        args.epoch_list = None
        with pytest.raises(ValueError, match="Unknown epoch grid"):
            suite._resolve_epoch_list(args)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgsEpochSweep:
    def test_mode_epoch_sweep_accepted(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
            ]
        )
        assert args.mode == "epoch_sweep"

    def test_default_epoch_grid(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
            ]
        )
        assert args.epoch_grid == suite._DEFAULT_EPOCH_GRID

    def test_custom_epoch_grid(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
                "--epoch-grid", "dense",
            ]
        )
        assert args.epoch_grid == "dense"

    def test_epoch_list_argument(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
                "--epoch-list", "5,10,50",
            ]
        )
        assert args.epoch_list == "5,10,50"

    def test_plateau_threshold_default(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
            ]
        )
        assert args.plateau_threshold == 0.005

    def test_plateau_threshold_custom(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
                "--plateau-threshold", "0.01",
            ]
        )
        assert args.plateau_threshold == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# _build_experiment_id prefix
# ---------------------------------------------------------------------------

class TestBuildExperimentIdEpochSweep:
    def test_epoch_sweep_prefix(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
            ]
        )
        exp_id = suite._build_experiment_id(args)
        assert exp_id.startswith("behavioral_epoch_sweep_")

    def test_explicit_experiment_id_respected(self):
        args = suite._parse_args(
            [
                "--dataset-version", DATASET_VERSION,
                "--dataset-variant", DATASET_VARIANT,
                "--mode", "epoch_sweep",
                "--experiment-id", "my_sweep",
            ]
        )
        assert suite._build_experiment_id(args) == "my_sweep"


# ---------------------------------------------------------------------------
# run_epoch_sweep end-to-end
# ---------------------------------------------------------------------------

def test_run_epoch_sweep_writes_outputs(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = suite._parse_args(
        [
            "--dataset-version", DATASET_VERSION,
            "--dataset-variant", DATASET_VARIANT,
            "--mode", "epoch_sweep",
            "--surface", "reactive_jpy",
            "--models", "both",
            "--repo-root", str(repo_root),
            "--output-root", str(tmp_path / "out"),
            "--experiment-id", "sweep_test",
            "--epoch-list", "5,10",
            "--wf-train-years", "1",
            "--wf-test-months", "3",
            "--wf-step-months", "3",
        ]
    )
    summary = suite.run_epoch_sweep(args)

    # Top-level summary keys
    assert summary["sweep_id"] == "sweep_test"
    assert summary["epochs"] == [5, 10]
    assert summary["total_failures"] == 0

    sweep_dir = Path(summary["sweep_dir"])
    assert sweep_dir.exists()

    # sweep_summary.csv
    manifest_path = Path(summary["sweep_manifest"])
    assert manifest_path.exists()
    manifest_df = pd.read_csv(manifest_path)
    assert set(manifest_df.columns) >= {"epoch", "experiment_dir"}
    assert set(manifest_df["epoch"].tolist()) == {5, 10}

    # Per-epoch walk-forward outputs
    runs_dir = sweep_dir / "runs"
    assert runs_dir.exists()
    for epoch in [5, 10]:
        ep_dir = runs_dir / f"sweep_test_ep{epoch}"
        assert (ep_dir / "metrics.csv").exists(), f"metrics.csv missing for epoch {epoch}"
        assert (ep_dir / "summary.csv").exists()
        assert (ep_dir / "experiment_manifest.json").exists()

    # Convergence analysis outputs
    assert summary["convergence_report"] is not None
    assert Path(summary["convergence_report"]).exists()
    assert summary["epoch_summary"] is not None
    assert Path(summary["epoch_summary"]).exists()
    assert summary["plots_dir"] is not None
    plots_dir = Path(summary["plots_dir"])
    assert (plots_dir / "pr_auc_vs_epoch.png").exists()

    # walk-forward runs were dispatched with correct epoch counts
    epoch_flags_seen = set()
    for cmd in calls:
        if "--epochs" in cmd:
            epoch_flags_seen.add(int(_parse_flag(cmd, "--epochs")))
    assert epoch_flags_seen == {5, 10}


def test_run_epoch_sweep_named_grid(tmp_path, monkeypatch):
    """epoch_sweep with --epoch-grid uses the correct epoch list."""
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    # Use a 2-epoch "dense" grid by patching EPOCH_GRIDS for test isolation
    monkeypatch.setattr(suite, "EPOCH_GRIDS", {**suite.EPOCH_GRIDS, "dense": [5, 10]})

    args = suite._parse_args(
        [
            "--dataset-version", DATASET_VERSION,
            "--dataset-variant", DATASET_VARIANT,
            "--mode", "epoch_sweep",
            "--surface", "reactive_jpy",
            "--models", "mlp",
            "--repo-root", str(repo_root),
            "--output-root", str(tmp_path / "out"),
            "--experiment-id", "grid_sweep",
            "--epoch-grid", "dense",
            "--wf-train-years", "1",
            "--wf-test-months", "3",
            "--wf-step-months", "3",
        ]
    )
    summary = suite.run_epoch_sweep(args)

    assert summary["epochs"] == [5, 10]
    manifest_df = pd.read_csv(summary["sweep_manifest"])
    assert set(manifest_df["epoch"].tolist()) == {5, 10}


def test_run_epoch_sweep_respects_selected_state(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = suite._parse_args(
        [
            "--dataset-version", DATASET_VERSION,
            "--dataset-variant", DATASET_VARIANT,
            "--mode", "epoch_sweep",
            "--surface", "reactive_jpy",
            "--state", "STATE_A",
            "--models", "mlp",
            "--repo-root", str(repo_root),
            "--output-root", str(tmp_path / "out"),
            "--experiment-id", "state_sweep",
            "--epoch-list", "5,10",
            "--wf-train-years", "1",
            "--wf-test-months", "3",
            "--wf-step-months", "3",
        ]
    )
    summary = suite.run_epoch_sweep(args)

    assert summary["epochs"] == [5, 10]
    assert all(_parse_flag(cmd, "--state") == "STATE_A" for cmd in calls)
