from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from analysis.behavioral.analyze_manifests import summarize_manifests
from analysis.behavioral.compare_predictions import (
    compare_mlp_lstm_predictions,
    summarize_prediction_artifact,
)
from analysis.behavioral.controls import generate_controls
from analysis.behavioral.coverage import build_coverage_table
from analysis.behavioral.metrics import compute_prediction_metrics_from_path
from analysis.behavioral.reporting import (
    write_markdown_report,
    write_metrics_csv,
    write_summary_csv,
)
from analysis.behavioral.utils import (
    build_training_command,
    copy_files,
    diff_new_files,
    discover_behavioral_states,
    get_git_commit,
    load_dataset_for_suite,
    read_json,
    resolve_dataset_csv_path,
    run_training_command,
    sanitize_fragment,
    select_models,
    snapshot_files,
    utc_now_iso,
)


OUTPUT_LAYOUT_DIRS = [
    "manifests",
    "prediction_artifacts",
    "plots",
    "logs",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the behavioral experiment suite across discovered states.",
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-variant", required=True)
    parser.add_argument("--surface-id", default=None)
    parser.add_argument("--models", default="both", help="mlp, lstm, or both")
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-quantile", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--train-pairs", default=None)
    parser.add_argument("--predict-pairs", default=None)
    parser.add_argument("--export-split", default="all", choices=["all", "test"])
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/output"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/output/dl_predictions"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args(argv)


def _build_experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return args.experiment_id
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    variant = sanitize_fragment(args.dataset_variant)
    return f"behavioral_suite_{args.dataset_version}_{variant}_{timestamp}"


def _prepare_experiment_dir(root: Path, experiment_id: str) -> Path:
    experiment_dir = root / experiment_id
    for rel in OUTPUT_LAYOUT_DIRS:
        (experiment_dir / rel).mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    models = select_models(args.models)
    experiment_id = _build_experiment_id(args)
    experiment_dir = _prepare_experiment_dir(args.output_root, experiment_id)

    dataset_output_dir = args.repo_root / "data" / "output"
    dataset_path = resolve_dataset_csv_path(
        args.dataset_version,
        args.dataset_variant,
        output_dir=dataset_output_dir,
    )
    dataset_df = load_dataset_for_suite(
        args.dataset_version,
        args.dataset_variant,
        output_dir=dataset_output_dir,
    )
    states = discover_behavioral_states(dataset_df, selected_surface_id=args.surface_id)

    started_at = utc_now_iso()
    run_rows: list[dict[str, object]] = []
    manifest_paths: list[Path] = []
    prediction_paths: list[Path] = []

    predictions_dir = args.repo_root / args.predictions_dir
    logs_dir = args.repo_root / args.logs_dir

    for state in states:
        for model in models:
            # Filesystem snapshots serve as fallback when trainer does not
            # report artifact paths directly via log lines.
            before_parquet = snapshot_files(predictions_dir, "*.parquet")
            before_manifest = snapshot_files(predictions_dir, "*.manifest.json")
            before_logs = snapshot_files(logs_dir, "*.log")

            command = build_training_command(
                trainer=model,
                dataset_version=args.dataset_version,
                dataset_variant=args.dataset_variant,
                surface_id=state["surface_id"],
                state_id=state["state_id"],
                feature_set=args.feature_set,
                target_horizon=args.target_horizon,
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                label_quantile=args.label_quantile,
                seq_len=args.seq_len,
                train_pairs=args.train_pairs,
                predict_pairs=args.predict_pairs,
                export_split=args.export_split,
            )

            run_tag = f"{model}_{sanitize_fragment(state['surface_id'])}_{sanitize_fragment(state['state_id'])}"
            run_log_path = experiment_dir / "logs" / f"run_{run_tag}.log"
            result = run_training_command(
                command=command,
                repo_root=args.repo_root,
                log_path=run_log_path,
            )

            # Prefer trainer-reported artifact paths; fall back to filesystem diff.
            if result.reported_parquet_path is not None:
                new_parquet = [result.reported_parquet_path]
            else:
                after_parquet = snapshot_files(predictions_dir, "*.parquet")
                new_parquet = diff_new_files(before_parquet, after_parquet)

            if result.reported_manifest_path is not None:
                new_manifest = [result.reported_manifest_path]
            else:
                after_manifest = snapshot_files(predictions_dir, "*.manifest.json")
                new_manifest = diff_new_files(before_manifest, after_manifest)

            after_logs = snapshot_files(logs_dir, "*.log")
            new_logs = diff_new_files(before_logs, after_logs)

            copied_parquet = copy_files(new_parquet, experiment_dir / "prediction_artifacts")
            copied_manifest = copy_files(new_manifest, experiment_dir / "manifests")
            copy_files(new_logs, experiment_dir / "logs")

            if copied_manifest:
                manifest_paths.extend(copied_manifest)
            if copied_parquet:
                prediction_paths.extend(copied_parquet)

            run_rows.append(
                {
                    "surface_id": state["surface_id"],
                    "state_id": state["state_id"],
                    "model": model,
                    "status": "success" if result.returncode == 0 else "failed",
                    "returncode": int(result.returncode),
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "duration_seconds": result.duration_seconds,
                    "command": " ".join(shlex.quote(part) for part in command),
                    "manifest_file": copied_manifest[0].name if copied_manifest else None,
                    "artifact_file": copied_parquet[0].name if copied_parquet else None,
                    "log_file": _relative_to(run_log_path, args.repo_root),
                    "artifact_discovery": (
                        "trainer_reported"
                        if result.reported_parquet_path is not None
                        else "filesystem_diff"
                    ),
                }
            )

    run_df = write_summary_csv(run_rows, experiment_dir / "summary.csv")
    coverage_df = build_coverage_table(dataset_df, states)
    manifest_df = summarize_manifests(manifest_paths) if manifest_paths else pd.DataFrame()

    # Scientific metrics per prediction artifact
    prediction_metric_rows: list[dict[str, object]] = []
    for run_row in run_rows:
        if run_row.get("artifact_file") is None:
            continue
        artifact_path = experiment_dir / "prediction_artifacts" / str(run_row["artifact_file"])
        if not artifact_path.exists():
            continue
        try:
            pm = compute_prediction_metrics_from_path(
                artifact_path,
                surface_id=str(run_row["surface_id"]),
                state_id=str(run_row["state_id"]),
            )
            pm["model"] = run_row["model"]
            prediction_metric_rows.append({"metric_group": "prediction_metrics", **pm})
        except Exception:
            pass

    # Legacy artifact summary (kept for backwards compatibility)
    legacy_summary_rows = [
        {"metric_group": "prediction_artifact", **summarize_prediction_artifact(path)}
        for path in prediction_paths
    ]

    compare_rows: list[dict[str, object]] = []
    for state in states:
        state_run_df = run_df[
            (run_df["surface_id"] == state["surface_id"]) &
            (run_df["state_id"] == state["state_id"]) &
            (run_df["status"] == "success")
        ]
        mlp_row = state_run_df[state_run_df["model"] == "mlp"]
        lstm_row = state_run_df[state_run_df["model"] == "lstm"]
        mlp_path = (
            (experiment_dir / "prediction_artifacts" / mlp_row.iloc[0]["artifact_file"]).resolve()
            if not mlp_row.empty and pd.notna(mlp_row.iloc[0]["artifact_file"])
            else None
        )
        lstm_path = (
            (experiment_dir / "prediction_artifacts" / lstm_row.iloc[0]["artifact_file"]).resolve()
            if not lstm_row.empty and pd.notna(lstm_row.iloc[0]["artifact_file"])
            else None
        )
        compare_rows.append(
            compare_mlp_lstm_predictions(
                mlp_path=mlp_path,
                lstm_path=lstm_path,
                surface_id=state["surface_id"],
                state_id=state["state_id"],
            )
        )
    compare_df = pd.DataFrame(compare_rows)

    # Baseline controls
    controls_df = generate_controls(dataset_df, states)

    metric_rows: list[dict[str, object]] = []
    metric_rows.extend(
        {"metric_group": "coverage", **row}
        for row in coverage_df.to_dict(orient="records")
    )
    metric_rows.extend(
        {"metric_group": "mlp_lstm_compare", **row}
        for row in compare_df.to_dict(orient="records")
    )
    metric_rows.extend(prediction_metric_rows)
    metric_rows.extend(legacy_summary_rows)
    metrics_df = write_metrics_csv(metric_rows, experiment_dir / "metrics.csv")

    # Build the scientific metrics DataFrame for report display
    sci_metrics_df = pd.DataFrame(
        [r for r in prediction_metric_rows if r.get("metric_group") == "prediction_metrics"]
    )
    if not sci_metrics_df.empty and "metric_group" in sci_metrics_df.columns:
        sci_metrics_df = sci_metrics_df.drop(columns=["metric_group"])

    finished_at = utc_now_iso()
    config_payload = {
        "experiment_id": experiment_id,
        "dataset_version": args.dataset_version,
        "dataset_variant": args.dataset_variant,
        "selected_surface_id": args.surface_id,
        "models": models,
        "feature_set": args.feature_set,
        "target_horizon": args.target_horizon,
        "train_pairs": args.train_pairs,
        "predict_pairs": args.predict_pairs,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": get_git_commit(args.repo_root),
    }

    write_markdown_report(
        output_path=experiment_dir / "report.md",
        experiment_id=experiment_id,
        config=config_payload,
        run_df=run_df,
        coverage_df=coverage_df,
        manifest_df=manifest_df,
        compare_df=compare_df,
        discovered_states=states,
        metrics_df=sci_metrics_df if not sci_metrics_df.empty else None,
        controls_df=controls_df,
    )

    experiment_manifest = {
        "experiment_id": experiment_id,
        "created_at": started_at,
        "completed_at": finished_at,
        "success": bool((run_df["status"] == "success").all()) if not run_df.empty else False,
        "cli": {
            "argv": sys.argv,
            "parsed": vars(args),
        },
        "dataset": {
            "version": args.dataset_version,
            "variant": args.dataset_variant,
            "path": str(dataset_path),
        },
        "discovered_states": states,
        "models_executed": models,
        "git_commit": config_payload["git_commit"],
        "runs": run_df.to_dict(orient="records"),
        "manifest_files": [path.name for path in manifest_paths],
        "prediction_artifacts": [path.name for path in prediction_paths],
    }
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, indent=2, default=str),
        encoding="utf-8",
    )

    summary = {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "runs": len(run_df),
        "failures": int((run_df["status"] != "success").sum()) if not run_df.empty else 0,
        "metrics_rows": len(metrics_df),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_suite(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
