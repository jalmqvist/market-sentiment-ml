from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.behavioral import run_behavioral_suite as suite
from analysis.behavioral.utils import RunResult


DATASET_VERSION = "1.5.1"
DATASET_VARIANT = "reactive_jpy_v1_core"


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
        mf_path.write_text(json.dumps({"identity": {"model": model}, "provenance": {}}), encoding="utf-8")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("wf run", encoding="utf-8")
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


def test_parse_args_walkforward_mode_supports_surface_alias():
    args = suite._parse_args(
        [
            "--dataset-version", DATASET_VERSION,
            "--dataset-variant", DATASET_VARIANT,
            "--mode", "walkforward",
            "--surface", "reactive_jpy",
        ]
    )
    assert args.mode == "walkforward"
    assert args.surface_id == "reactive_jpy"


def test_walkforward_suite_writes_outputs(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_dataset(repo_root)

    runner, calls = _build_fake_runner(repo_root)
    monkeypatch.setattr(suite, "run_training_command", runner)

    args = suite._parse_args(
        [
            "--dataset-version", DATASET_VERSION,
            "--dataset-variant", DATASET_VARIANT,
            "--mode", "walkforward",
            "--surface", "reactive_jpy",
            "--models", "both",
            "--repo-root", str(repo_root),
            "--output-root", str(tmp_path / "out"),
            "--experiment-id", "wf_test",
            "--wf-train-years", "1",
            "--wf-test-months", "3",
            "--wf-step-months", "3",
        ]
    )
    summary = suite.run_walkforward_suite(args)

    assert summary["runs"] > 0
    assert len(calls) == summary["runs"]
    assert any("--wf-test-start" in cmd for cmd in calls)

    exp_dir = tmp_path / "out" / "wf_test"
    assert (exp_dir / "summary.csv").exists()
    assert (exp_dir / "metrics.csv").exists()
    assert (exp_dir / "report.md").exists()
    assert (exp_dir / "experiment_manifest.json").exists()

    metrics_df = pd.read_csv(exp_dir / "metrics.csv")
    assert "pr_auc" in metrics_df.columns
    assert set(metrics_df["baseline"].dropna().unique()) >= {
        "behavioral_surface",
        "permutation",
        "base_rate",
        "random_matched_partition",
    }

    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "walkforward"
    assert manifest["walkforward_protocol"]["protocol"] == "mpml_reference_v1"

    report_text = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "does not evaluate trading suitability" in report_text
