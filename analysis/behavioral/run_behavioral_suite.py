from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

from analysis.walkforward.calibration import compute_calibration
from analysis.walkforward.controls import build_control_rows
from analysis.walkforward.evaluate import aggregate_metric_table, compute_predictive_metrics
from analysis.walkforward.utils import (
    build_binary_labels,
    filter_window,
    match_predictions_with_labels,
    resolve_target_column,
    resolve_time_column,
    train_threshold,
)
from research.utils.mpml_walkforward_reference import (
    generate_walkforward_folds_by_pos,
    walkforward_signature,
)

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


# ---------------------------------------------------------------------------
# Named training profiles
# ---------------------------------------------------------------------------

#: Named experiment profiles.  Each profile defines default hyperparameters
#: that are appropriate for the stated research intent.  Profiles can be
#: selected with ``--profile <name>``; individual hyperparameters can still
#: be overridden via explicit CLI flags.
PROFILES: dict[str, dict[str, object]] = {
    "smoke": {
        "epochs": 2,
        "hidden_dim": 16,
        "description": (
            "Minimal smoke-test profile (2 epochs, small model). "
            "Verifies the pipeline runs end-to-end; results are not scientifically meaningful."
        ),
    },
    "standard": {
        "epochs": 10,
        "hidden_dim": 32,
        "description": (
            "Standard characterization profile (10 epochs). "
            "Produces scientifically meaningful initial results for a Behavioral Surface."
        ),
    },
    "publication": {
        "epochs": 50,
        "hidden_dim": 64,
        "description": (
            "Publication-quality profile (50 epochs, larger model). "
            "Use when preparing findings for external reporting or comparison."
        ),
    },
}

_DEFAULT_PROFILE = "standard"


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Apply profile defaults to *args*, respecting explicit CLI overrides.

    The profile sets default values for ``epochs`` and ``hidden_dim`` only
    when those flags were not explicitly supplied (i.e. they are still ``None``
    from argparse, which uses ``None`` as the sentinel default for both flags).
    """
    profile_name = getattr(args, "profile", _DEFAULT_PROFILE) or _DEFAULT_PROFILE
    profile = PROFILES.get(profile_name)
    if profile is None:
        known = ", ".join(PROFILES)
        raise ValueError(f"Unknown profile '{profile_name}'. Known profiles: {known}.")

    # Only apply profile defaults when the user did not explicitly supply the
    # flag.  Both --epochs and --hidden-dim default to None in argparse, so a
    # None value here reliably means "not provided on the command line".
    if args.epochs is None:
        args.epochs = profile["epochs"]
    if args.hidden_dim is None:
        args.hidden_dim = profile["hidden_dim"]

    args.profile_name = profile_name
    args.profile_description = profile["description"]
    return args


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Behavioral Characterization Suite across discovered Behavioral Surface states.\n\n"
            "Named profiles control the default training intensity:\n"
            "  smoke       — 2 epochs, pipeline smoke-test only\n"
            "  standard    — 10 epochs, initial scientific characterization (default)\n"
            "  publication — 50 epochs, publication-quality results"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-variant", required=True)
    parser.add_argument("--surface-id", "--surface", dest="surface_id", default=None)
    parser.add_argument(
        "--mode",
        default="characterization",
        choices=["characterization", "walkforward"],
    )
    parser.add_argument("--models", default="both", help="mlp, lstm, or both")
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument(
        "--profile",
        default=_DEFAULT_PROFILE,
        choices=list(PROFILES),
        help=(
            "Named training profile (default: standard). "
            "Sets default epoch count and model size. "
            "Override individual hyperparameters with --epochs / --hidden-dim."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (overrides --profile default).")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Hidden layer dimension (overrides --profile default).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-quantile", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--wf-train-years", type=int, default=7)
    parser.add_argument("--wf-test-months", type=int, default=6)
    parser.add_argument("--wf-step-months", type=int, default=6)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--train-pairs", default=None)
    parser.add_argument("--predict-pairs", default=None)
    parser.add_argument("--export-split", default="all", choices=["all", "test"])
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/output"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/output/dl_predictions"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    return _apply_profile(args)



def _build_experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return args.experiment_id
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    variant = sanitize_fragment(args.dataset_variant)
    prefix = "behavioral_walkforward" if args.mode == "walkforward" else "behavioral_suite"
    return f"{prefix}_{args.dataset_version}_{variant}_{timestamp}"


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


def _write_walkforward_report(
    *,
    output_path: Path,
    experiment_id: str,
    config_payload: dict[str, object],
    run_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
) -> None:
    lines = [
        f"# Behavioral Walk-forward Predictive Validation Report: {experiment_id}",
        "",
        "This report evaluates predictive performance only.",
        "It does not evaluate trading suitability.",
        "",
        "## Executive Summary",
        f"- total_runs: {len(run_df)}",
        f"- successful_runs: {(run_df['status'] == 'success').sum() if not run_df.empty else 0}",
        f"- failed_runs: {(run_df['status'] != 'success').sum() if not run_df.empty else 0}",
        f"- folds: {run_df['fold'].nunique() if 'fold' in run_df.columns and not run_df.empty else 0}",
    ]

    if not aggregated_df.empty:
        lines.extend(["", "## Aggregate Predictive Comparison"])
        disp = [
            c
            for c in [
                "model",
                "surface_id",
                "state_id",
                "baseline",
                "folds",
                "pr_auc_mean",
                "brier_score_mean",
                "mcc_mean",
                "balanced_accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
            ]
            if c in aggregated_df.columns
        ]
        lines.append(aggregated_df[disp].to_markdown(index=False))

    if not fold_metrics_df.empty:
        lines.extend(["", "## Per-fold Metrics"])
        disp = [
            c
            for c in [
                "fold",
                "model",
                "surface_id",
                "state_id",
                "baseline",
                "n",
                "positive_rate",
                "pr_auc",
                "brier_score",
                "calibration_ece",
                "mcc",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
            ]
            if c in fold_metrics_df.columns
        ]
        lines.append(fold_metrics_df[disp].to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Reproducibility",
            f"- dataset_version: `{config_payload.get('dataset_version')}`",
            f"- dataset_variant: `{config_payload.get('dataset_variant')}`",
            f"- selected_surface_id: `{config_payload.get('selected_surface_id')}`",
            f"- models: `{', '.join(config_payload.get('models', []))}`",
            f"- walkforward_protocol: `{config_payload.get('walkforward_protocol')}`",
            f"- git_commit: `{config_payload.get('git_commit')}`",
            f"- started_at: `{config_payload.get('started_at')}`",
            f"- finished_at: `{config_payload.get('finished_at')}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "profile": getattr(args, "profile_name", "custom"),
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


def run_walkforward_suite(args: argparse.Namespace) -> dict[str, object]:
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
    time_col = resolve_time_column(dataset_df)
    target_col = resolve_target_column(dataset_df, args.target_horizon)
    state_partitions: dict[tuple[str, str], pd.DataFrame] = {}
    for state in states:
        key = (state["surface_id"], state["state_id"])
        state_partitions[key] = dataset_df[
            (dataset_df["surface_id"] == state["surface_id"])
            & (dataset_df["state_id"] == state["state_id"])
        ].copy()

    dates = pd.DatetimeIndex(pd.to_datetime(dataset_df[time_col], errors="coerce").dropna().sort_values().unique())
    folds = generate_walkforward_folds_by_pos(
        dates=dates,
        train_years=args.wf_train_years,
        test_months=args.wf_test_months,
        step_months=args.wf_step_months,
    )
    protocol = walkforward_signature(
        train_years=args.wf_train_years,
        test_months=args.wf_test_months,
        step_months=args.wf_step_months,
    )

    started_at = utc_now_iso()
    run_rows: list[dict[str, object]] = []
    manifest_paths: list[Path] = []
    prediction_paths: list[Path] = []
    metric_rows: list[dict[str, object]] = []
    calibration_curve_rows: list[dict[str, object]] = []

    predictions_dir = args.repo_root / args.predictions_dir
    logs_dir = args.repo_root / args.logs_dir

    for fold in folds:
        train_start = pd.Timestamp(fold["train_start_dt"])
        train_end = pd.Timestamp(fold["train_end_dt"])
        test_start = pd.Timestamp(fold["test_start_dt"])
        test_end = pd.Timestamp(fold["test_end_dt"])

        fold_test_df = filter_window(dataset_df, time_col=time_col, start=test_start, end=test_end)

        for state in states:
            state_df = state_partitions[(state["surface_id"], state["state_id"])]
            state_train_df = filter_window(state_df, time_col=time_col, start=train_start, end=train_end)
            state_test_df = filter_window(state_df, time_col=time_col, start=test_start, end=test_end)
            threshold = train_threshold(
                state_train_df,
                target_col=target_col,
                label_quantile=args.label_quantile,
            )
            if threshold is None:
                continue

            label_train = build_binary_labels(
                state_train_df,
                target_col=target_col,
                threshold=threshold,
            )
            label_test = build_binary_labels(
                state_test_df,
                target_col=target_col,
                threshold=threshold,
            )
            if label_test.empty:
                continue

            # Regime-conditioned controls (Trend/Volatility baseline)
            regime_train_df = None
            regime_test_df = None
            if "regime" in state_train_df.columns and "regime" in state_test_df.columns:
                regime_train_df = label_train.merge(
                    state_train_df[["pair", "entry_time", "regime"]].copy(),
                    on=["pair", "entry_time"],
                    how="left",
                )
                regime_test_df = label_test.merge(
                    state_test_df[["pair", "entry_time", "regime"]].copy(),
                    on=["pair", "entry_time"],
                    how="left",
                )

            train_positive_rate = float(label_train["y_true"].mean()) if not label_train.empty else None

            for model in models:
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
                    export_split="test",
                    walkforward_window={
                        "train_start": fold["train_start_dt"],
                        "train_end": fold["train_end_dt"],
                        "test_start": fold["test_start_dt"],
                        "test_end": fold["test_end_dt"],
                    },
                )

                run_tag = (
                    f"wf_fold{fold['fold']}_{model}_"
                    f"{sanitize_fragment(state['surface_id'])}_{sanitize_fragment(state['state_id'])}"
                )
                run_log_path = experiment_dir / "logs" / f"run_{run_tag}.log"
                result = run_training_command(
                    command=command,
                    repo_root=args.repo_root,
                    log_path=run_log_path,
                )

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

                run_row = {
                    "fold": int(fold["fold"]),
                    "train_start_dt": fold["train_start_dt"],
                    "train_end_dt": fold["train_end_dt"],
                    "test_start_dt": fold["test_start_dt"],
                    "test_end_dt": fold["test_end_dt"],
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
                    "threshold": float(threshold),
                }
                run_rows.append(run_row)

                if result.returncode != 0 or not copied_parquet:
                    continue

                artifact_path = experiment_dir / "prediction_artifacts" / copied_parquet[0].name
                try:
                    pred_df = pd.read_parquet(artifact_path)
                except Exception:
                    continue
                y_true, y_prob = match_predictions_with_labels(pred_df, label_test)
                behavioral_metrics = compute_predictive_metrics(y_true, y_prob)
                calibration_summary, curve_rows = compute_calibration(
                    y_true,
                    y_prob,
                    n_bins=args.calibration_bins,
                )
                metric_rows.append(
                    {
                        "metric_group": "walkforward_fold",
                        "baseline": "behavioral_surface",
                        "fold": int(fold["fold"]),
                        "surface_id": state["surface_id"],
                        "state_id": state["state_id"],
                        "model": model,
                        **behavioral_metrics,
                        **calibration_summary,
                    }
                )
                for curve in curve_rows:
                    calibration_curve_rows.append(
                        {
                            "fold": int(fold["fold"]),
                            "surface_id": state["surface_id"],
                            "state_id": state["state_id"],
                            "model": model,
                            **curve,
                        }
                    )

                control_rows = build_control_rows(
                    y_true=y_true,
                    y_prob_behavioral=y_prob,
                    fold_id=int(fold["fold"]),
                    model=model,
                    surface_id=state["surface_id"],
                    state_id=state["state_id"],
                    train_positive_rate=train_positive_rate,
                    regime_train_df=regime_train_df,
                    regime_test_df=regime_test_df,
                )
                for row in control_rows:
                    metric_rows.append({"metric_group": "walkforward_fold", **row})

    run_df = write_summary_csv(run_rows, experiment_dir / "summary.csv")
    fold_metrics_df = pd.DataFrame(metric_rows)
    aggregate_rows = aggregate_metric_table(metric_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)

    metrics_payload = metric_rows + [{"metric_group": "walkforward_aggregate", **row} for row in aggregate_rows]
    metrics_df = write_metrics_csv(metrics_payload, experiment_dir / "metrics.csv")
    if calibration_curve_rows:
        pd.DataFrame(calibration_curve_rows).to_csv(experiment_dir / "calibration_curve.csv", index=False)

    finished_at = utc_now_iso()
    config_payload = {
        "experiment_id": experiment_id,
        "dataset_version": args.dataset_version,
        "dataset_variant": args.dataset_variant,
        "selected_surface_id": args.surface_id,
        "models": models,
        "feature_set": args.feature_set,
        "target_horizon": args.target_horizon,
        "walkforward_protocol": protocol,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": get_git_commit(args.repo_root),
    }

    _write_walkforward_report(
        output_path=experiment_dir / "report.md",
        experiment_id=experiment_id,
        config_payload=config_payload,
        run_df=run_df,
        fold_metrics_df=fold_metrics_df,
        aggregated_df=aggregate_df,
    )

    experiment_manifest = {
        "experiment_id": experiment_id,
        "created_at": started_at,
        "completed_at": finished_at,
        "mode": "walkforward",
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
        "walkforward_protocol": protocol,
        "walkforward_folds": folds,
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
        "folds": len(folds),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_walkforward_suite(args) if args.mode == "walkforward" else run_suite(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
