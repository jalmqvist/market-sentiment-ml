from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.behavioral.analyze_epoch_sweep import analyze_epoch_sweep


def _write_experiment(exp_dir: Path, epoch: int, state_values: dict[str, float]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fold in [1, 2, 3]:
        for state_id, pr_auc_base in state_values.items():
            for model, model_delta in {"mlp": 0.0, "lstm": -0.005}.items():
                behavioral_pr = pr_auc_base + model_delta
                rows.append(
                    {
                        "metric_group": "walkforward_fold",
                        "baseline": "behavioral_surface",
                        "fold": fold,
                        "surface_id": "reactive_jpy",
                        "state_id": state_id,
                        "model": model,
                        "pr_auc": behavioral_pr,
                        "mcc": 0.20 + behavioral_pr / 10.0,
                        "balanced_accuracy": 0.55 + behavioral_pr / 10.0,
                        "brier_score": 0.24 - behavioral_pr / 10.0,
                        "calibration_ece": 0.12 - behavioral_pr / 20.0,
                    }
                )
                for baseline, pr_gap in {
                    "permutation": 0.10,
                    "random_matched_partition": 0.08,
                    "trend_volatility": 0.05,
                    "base_rate": 0.03,
                }.items():
                    control_pr = behavioral_pr - pr_gap
                    rows.append(
                        {
                            "metric_group": "walkforward_fold",
                            "baseline": baseline,
                            "fold": fold,
                            "surface_id": "reactive_jpy",
                            "state_id": state_id,
                            "model": model,
                            "pr_auc": control_pr,
                            "mcc": 0.18 + control_pr / 10.0,
                            "balanced_accuracy": 0.52 + control_pr / 10.0,
                            "brier_score": 0.26 - control_pr / 10.0,
                            "calibration_ece": 0.14 - control_pr / 20.0,
                        }
                    )

    pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "mlp", "epoch": epoch, "status": "success"},
            {"model": "lstm", "epoch": epoch, "status": "success"},
        ]
    ).to_csv(exp_dir / "summary.csv", index=False)
    (exp_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "models_executed": ["mlp", "lstm"],
                "discovered_states": [
                    {"surface_id": "reactive_jpy", "state_id": "STATE_A"},
                    {"surface_id": "reactive_jpy", "state_id": "STATE_B"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_analyze_epoch_sweep_generates_outputs(tmp_path, capsys):
    sweep_dir = tmp_path / "sweep"
    epochs = {
        5: {"STATE_A": 0.60, "STATE_B": 0.55},
        10: {"STATE_A": 0.648, "STATE_B": 0.60},
        15: {"STATE_A": 0.650, "STATE_B": 0.63},
        20: {"STATE_A": 0.651, "STATE_B": 0.67},
    }
    manifest_rows = []
    for epoch, state_values in epochs.items():
        exp_dir = sweep_dir / f"epoch_{epoch}"
        _write_experiment(exp_dir, epoch, state_values)
        manifest_rows.append({"epoch": epoch, "experiment_dir": exp_dir.name})

    sweep_manifest = sweep_dir / "sweep_summary.csv"
    pd.DataFrame(manifest_rows).to_csv(sweep_manifest, index=False)

    output_dir = sweep_dir / "analysis"
    summary = analyze_epoch_sweep(
        sweep_manifest=sweep_manifest,
        output_dir=output_dir,
        plateau_threshold=0.005,
    )

    assert Path(summary["epoch_summary"]).exists()
    assert Path(summary["convergence_report"]).exists()
    assert (output_dir / "plots" / "pr_auc_vs_epoch.png").exists()
    assert (output_dir / "plots" / "relative_improvement_vs_controls.png").exists()
    assert (output_dir / "plots" / "calibration_vs_epoch.png").exists()

    epoch_summary_df = pd.read_csv(output_dir / "epoch_summary.csv")
    assert "pr_auc_relative_pct_vs_permutation" in epoch_summary_df.columns
    assert "calibration_ece_relative_pct_vs_base_rate" in epoch_summary_df.columns
    behavioral_df = epoch_summary_df[epoch_summary_df["baseline"] == "behavioral_surface"]
    assert set(behavioral_df["epoch"].tolist()) == {5, 10, 15, 20}

    report_text = (output_dir / "convergence_report.md").read_text(encoding="utf-8")
    assert "## Research Recommendation" in report_text
    assert "STATE_A" in report_text
    assert "extend the sweep to 40 epochs" in report_text

    stdout = capsys.readouterr().out
    assert "Epoch Sweep Analysis" in stdout
    assert "Best epoch by Behavioral State" in stdout
