from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.behavioral import run_behavioral_suite as suite
from analysis.behavioral.analyze_manifests import summarize_manifests
from analysis.behavioral.utils import RunResult, discover_behavioral_states


DATASET_VERSION = "1.5.1"
DATASET_VARIANT = "reactive_jpy_v1_core"


def _write_dataset(repo_root: Path) -> None:
    out_dir = repo_root / "data" / "output" / DATASET_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "pair": ["usd-jpy"] * 6,
            "entry_time": pd.date_range("2024-01-01", periods=6, freq="h"),
            "snapshot_time": pd.date_range("2024-01-01", periods=6, freq="h"),
            "surface_id": [
                "reactive_jpy",
                "reactive_jpy",
                "reactive_jpy",
                "reactive_jpy",
                "reactive_jpy",
                "reactive_jpy",
            ],
            "state_id": [
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_YOUNG",
                "JPY_CONSENSUS_MATURE",
                "JPY_CONSENSUS_MATURE",
                "JPY_CONSENSUS_MATURE",
                "JPY_CONSENSUS_YOUNG",
            ],
            "ret_48b": [0.1, 0.2, -0.1, 0.0, 0.05, -0.03],
        }
    )
    df.to_csv(out_dir / f"master_research_dataset_{DATASET_VARIANT}.csv", index=False)


def _parse_flag(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _build_fake_runner(repo_root: Path, fail_for: set[tuple[str, str, str]] | None = None):
    calls: list[dict] = []
    fail_for = fail_for or set()

    def _runner(*, command: list[str], repo_root: Path, log_path: Path) -> RunResult:
        model = "mlp" if command[1].endswith("train.py") else "lstm"
        surface_id = _parse_flag(command, "--surface")
        state_id = _parse_flag(command, "--state")
        run_id = f"{model}_{surface_id}_{state_id}_{len(calls)}"

        predictions_dir = repo_root / "data" / "output" / "dl_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        pred = pd.DataFrame(
            {
                "pair": ["usd-jpy", "usd-jpy"],
                "entry_time": pd.date_range("2024-01-01", periods=2, freq="h"),
                "pred_prob_up": [0.6, 0.4] if model == "mlp" else [0.55, 0.45],
                "signal_strength": [0.2, -0.2] if model == "mlp" else [0.1, -0.1],
            }
        )
        pred.to_parquet(predictions_dir / f"{run_id}.parquet", index=False)

        manifest = {
            "identity": {
                "model": model,
                "dl_regime": f"{surface_id}:{state_id}",
                "target_horizon": 24,
                "feature_set": "price_trend",
            },
            "provenance": {
                "dataset_version": DATASET_VERSION,
                "dataset_variant": DATASET_VARIANT,
                "surface_id": surface_id,
                "state_id": state_id,
            },
            "warnings": ["synthetic warning"],
            "missing_provenance_counts": {"dataset_version": 0, "model_version": 1},
        }
        (predictions_dir / f"{run_id}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        repo_logs = repo_root / "logs"
        repo_logs.mkdir(parents=True, exist_ok=True)
        (repo_logs / f"{run_id}.log").write_text("synthetic trainer log", encoding="utf-8")

        returncode = 1 if (model, surface_id, state_id) in fail_for else 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("synthetic run log", encoding="utf-8")

        calls.append(
            {
                "command": command,
                "model": model,
                "surface_id": surface_id,
                "state_id": state_id,
                "returncode": returncode,
            }
        )

        return RunResult(
            command=command,
            returncode=returncode,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            log_path=log_path,
        )

    return _runner, calls


def _build_args(repo_root: Path, output_root: Path) -> suite.argparse.Namespace:
    return suite._parse_args(
        [
            "--dataset-version",
            DATASET_VERSION,
            "--dataset-variant",
            DATASET_VARIANT,
            "--models",
            "both",
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(output_root),
            "--experiment-id",
            "exp_test",
        ]
    )


def test_automatic_state_discovery():
    df = pd.DataFrame(
        {
            "surface_id": ["reactive_jpy", "reactive_jpy", "reactive_jpy", None],
            "state_id": ["A", "A", "B", "B"],
        }
    )
    states = discover_behavioral_states(df)
    assert states == [
        {"surface_id": "reactive_jpy", "state_id": "A"},
        {"surface_id": "reactive_jpy", "state_id": "B"},
    ]


def test_state_discovery_supports_state_filter():
    df = pd.DataFrame(
        {
            "surface_id": ["reactive_jpy", "reactive_jpy", "reactive_chf"],
            "state_id": ["A", "B", "A"],
        }
    )
    states = discover_behavioral_states(
        df,
        selected_surface_id="reactive_jpy",
        selected_state_id="B",
    )
    assert states == [{"surface_id": "reactive_jpy", "state_id": "B"}]


def test_orchestrates_both_trainers_for_each_state(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = _build_args(repo_root, tmp_path / "analysis_output")
    summary = suite.run_suite(args)

    assert summary["runs"] == 4
    assert len(calls) == 4
    assert {call["model"] for call in calls} == {"mlp", "lstm"}


def test_run_suite_respects_selected_state(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = suite._parse_args(
        [
            "--dataset-version",
            DATASET_VERSION,
            "--dataset-variant",
            DATASET_VARIANT,
            "--models",
            "both",
            "--surface",
            "reactive_jpy",
            "--state",
            "JPY_CONSENSUS_MATURE",
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(tmp_path / "analysis_output"),
            "--experiment-id",
            "exp_state_filter",
        ]
    )
    summary = suite.run_suite(args)

    assert summary["runs"] == 2
    assert len(calls) == 2
    assert {call["state_id"] for call in calls} == {"JPY_CONSENSUS_MATURE"}


def test_artifact_collection_and_layout(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, _ = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = _build_args(repo_root, tmp_path / "analysis_output")
    suite.run_suite(args)

    exp_dir = tmp_path / "analysis_output" / "exp_test"
    assert (exp_dir / "summary.csv").exists()
    assert (exp_dir / "metrics.csv").exists()
    assert (exp_dir / "report.md").exists()
    assert len(list((exp_dir / "manifests").glob("*.json"))) == 4
    assert len(list((exp_dir / "prediction_artifacts").glob("*.parquet"))) == 4


def test_manifest_parsing_and_report_generation(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, _ = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = _build_args(repo_root, tmp_path / "analysis_output")
    suite.run_suite(args)

    exp_dir = tmp_path / "analysis_output" / "exp_test"
    manifests = sorted((exp_dir / "manifests").glob("*.manifest.json"))
    manifest_df = summarize_manifests(manifests)
    assert (manifest_df["warning_count"] > 0).all()

    report_text = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "synthetic warning" in report_text
    assert "missing_provenance" in report_text


def test_failed_subprocesses_reported(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, _ = _build_fake_runner(
        repo_root,
        fail_for={("lstm", "reactive_jpy", "JPY_CONSENSUS_MATURE")},
    )
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = _build_args(repo_root, tmp_path / "analysis_output")
    summary = suite.run_suite(args)

    assert summary["failures"] == 1
    exp_dir = tmp_path / "analysis_output" / "exp_test"
    run_df = pd.read_csv(exp_dir / "summary.csv")
    assert (run_df["status"] == "failed").sum() == 1
    report_text = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "## Failed Runs" in report_text


def test_reproducible_output_directory_layout(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, _ = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = _build_args(repo_root, tmp_path / "analysis_output")
    suite.run_suite(args)

    exp_dir = tmp_path / "analysis_output" / "exp_test"
    assert (exp_dir / "experiment_manifest.json").exists()
    assert (exp_dir / "report.md").exists()
    assert (exp_dir / "summary.csv").exists()
    assert (exp_dir / "metrics.csv").exists()
    for rel in ["manifests", "prediction_artifacts", "plots", "logs"]:
        assert (exp_dir / rel).is_dir()

    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["version"] == DATASET_VERSION
    assert manifest["dataset"]["variant"] == DATASET_VARIANT
    assert len(manifest["discovered_states"]) == 2
