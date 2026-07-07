from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from analysis.behavioral.interpretation import (
    Finding,
    Observation,
    derive_research_recommendation,
    format_executive_summary,
    format_findings,
    format_key_observations,
    generate_findings,
    generate_key_observations,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_summary_csv(summary_rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_path, index=False)
    return df


def write_metrics_csv(metrics_rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    df.to_csv(output_path, index=False)
    return df


def _states_table(states: list[dict[str, str]]) -> str:
    if not states:
        return "No behavioral states discovered."
    rows = [{"surface_id": s["surface_id"], "state_id": s["state_id"]} for s in states]
    return pd.DataFrame(rows).to_markdown(index=False)


def write_markdown_report(
    *,
    output_path: Path,
    experiment_id: str,
    config: dict,
    run_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    discovered_states: list[dict[str, str]] | None = None,
    metrics_df: pd.DataFrame | None = None,
    controls_df: pd.DataFrame | None = None,
    key_observations: list[Observation] | None = None,
) -> None:
    failures = run_df[run_df["status"] != "success"] if not run_df.empty else run_df

    # Manifest rows with any issue
    warning_rows = pd.DataFrame()
    if not manifest_df.empty:
        has_issue = pd.Series(False, index=manifest_df.index)
        for col in ["error_count", "warning_count"]:
            if col in manifest_df.columns:
                has_issue = has_issue | (manifest_df[col].fillna(0).astype(int) > 0)
        warning_rows = manifest_df[has_issue]

    # ------------------------------------------------------------------
    # Synthesize findings and recommendation (PR5.1)
    # ------------------------------------------------------------------
    auto_metrics = metrics_df if metrics_df is not None else pd.DataFrame()
    findings: list[Finding] = generate_findings(coverage_df, compare_df, auto_metrics)
    recommendation = derive_research_recommendation(
        run_df=run_df,
        findings=findings,
        coverage_df=coverage_df,
    )

    # ------------------------------------------------------------------
    # Primary report: Executive Summary → Scientific Findings → Recommendation
    # ------------------------------------------------------------------
    lines: list[str] = [
        f"# Behavioral Characterization Report: {experiment_id}",
        "",
        format_executive_summary(
            experiment_id=experiment_id,
            run_df=run_df,
            coverage_df=coverage_df,
            discovered_states=discovered_states or [],
            findings=findings,
            recommendation=recommendation,
        ),
        "",
        format_findings(findings),
    ]

    # ------------------------------------------------------------------
    # Appendix: engineering diagnostics and raw metrics
    # ------------------------------------------------------------------
    lines.extend([
        "",
        "---",
        "",
        "## Appendix",
        "",
        "### Discovered Behavioral Surface States",
    ])
    lines.append(_states_table(discovered_states or []))

    lines.extend([
        "",
        "### Execution Summary",
        f"- total_runs: {len(run_df)}",
        f"- successful_runs: {(run_df['status'] == 'success').sum() if not run_df.empty else 0}",
        f"- failed_runs: {len(failures) if failures is not None else 0}",
    ])

    lines.extend(["", "### Coverage"])
    if coverage_df.empty:
        lines.append("No coverage rows generated.")
    else:
        display_cols = [c for c in coverage_df.columns if c in [
            "scope", "row_count", "pair_count",
            "coverage_fraction", "state_fraction_of_behavioral",
            "timestamp_unique", "timestamp_min", "timestamp_max",
        ]]
        lines.append(coverage_df[display_cols].to_markdown(index=False))

    lines.extend(["", "### Prediction Comparison (MLP vs LSTM)"])
    if compare_df.empty:
        lines.append("No comparable MLP/LSTM prediction overlap found.")
    else:
        lines.append(compare_df.to_markdown(index=False))

    # Scientific metrics
    if metrics_df is not None and not metrics_df.empty:
        lines.extend(["", "### Scientific Prediction Metrics"])
        metric_display_cols = [c for c in [
            "artifact_file", "state_id", "n_predictions",
            "prediction_entropy_mean", "prediction_confidence_mean",
            "effective_prediction_coverage", "sharpness",
            "pair_balance", "coverage_days",
        ] if c in metrics_df.columns]
        if metric_display_cols:
            lines.append(metrics_df[metric_display_cols].to_markdown(index=False))

    # Controls
    if controls_df is not None and not controls_df.empty:
        lines.extend(["", "### Baseline Controls"])
        control_display_cols = [c for c in [
            "scope", "control_type", "row_count",
            "coverage_fraction", "pair_count",
            "timestamp_unique",
        ] if c in controls_df.columns]
        if control_display_cols:
            lines.append(controls_df[control_display_cols].to_markdown(index=False))

    # Key observations (legacy section, kept for traceability)
    if key_observations is not None:
        lines.extend(["", format_key_observations(key_observations)])
    else:
        obs = generate_key_observations(coverage_df, compare_df, auto_metrics, controls_df)
        lines.extend(["", format_key_observations(obs)])

    # Manifest issues (errors first, then warnings)
    lines.extend(["", "### Manifest Issues"])
    if warning_rows.empty:
        lines.append("No manifest errors or warnings detected.")
    else:
        # Show error rows
        if "error_count" in manifest_df.columns:
            error_rows = manifest_df[manifest_df["error_count"].fillna(0).astype(int) > 0]
            if not error_rows.empty:
                lines.extend(["", "#### Errors"])
                lines.append(error_rows[["manifest_file", "errors"]].to_markdown(index=False))
        # Show warning rows
        if "warning_count" in manifest_df.columns:
            warn_rows = manifest_df[manifest_df["warning_count"].fillna(0).astype(int) > 0]
            if not warn_rows.empty:
                lines.extend(["", "#### Warnings"])
                disp_cols = ["manifest_file", "warnings"]
                if "notes" in manifest_df.columns:
                    disp_cols = ["manifest_file", "warnings", "notes"]
                lines.append(warn_rows[[c for c in disp_cols if c in warn_rows.columns]].to_markdown(index=False))

    # Legacy compatibility: also show 'all_messages' column if no separate columns available
    if not manifest_df.empty and "warning_count" not in manifest_df.columns and "warnings" in manifest_df.columns:
        legacy_warn = manifest_df[manifest_df.get("warning_count", 0) > 0]
        if not legacy_warn.empty:
            lines.extend(["", "#### Legacy Manifest Warnings"])
            lines.append(legacy_warn[["manifest_file", "warnings"]].to_markdown(index=False))

    if not failures.empty:
        lines.extend(["", "## Failed Runs"])
        lines.append(failures[["surface_id", "state_id", "model", "returncode", "log_file"]].to_markdown(index=False))

    lines.extend([
        "",
        "### Reproducibility",
        f"- dataset_version: `{config.get('dataset_version')}`",
        f"- dataset_variant: `{config.get('dataset_variant')}`",
        f"- selected_models: `{','.join(config.get('models', []))}`",
        f"- feature_set: `{config.get('feature_set')}`",
        f"- target_horizon: `{config.get('target_horizon')}`",
        f"- git_commit: `{config.get('git_commit')}`",
        f"- started_at: `{config.get('started_at')}`",
        f"- finished_at: `{config.get('finished_at')}`",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _build_control_comparison_bullets(aggregated_df: pd.DataFrame) -> list[str]:
    if aggregated_df.empty:
        return []

    behavior_df = aggregated_df[aggregated_df["baseline"] == "behavioral_surface"].copy()
    if behavior_df.empty:
        return []

    rows: list[str] = []
    for baseline in [
        "permutation",
        "base_rate",
        "random_matched_partition",
        "trend_volatility",
    ]:
        baseline_df = aggregated_df[aggregated_df["baseline"] == baseline].copy()
        if baseline_df.empty:
            continue

        merged = behavior_df.merge(
            baseline_df,
            on=["model", "surface_id", "state_id"],
            how="inner",
            suffixes=("_behavioral", "_control"),
        )
        if merged.empty:
            continue

        pr_wins = (
            merged["pr_auc_mean_behavioral"].astype(float)
            > merged["pr_auc_mean_control"].astype(float)
        ).sum()
        brier_wins = (
            merged["brier_score_mean_behavioral"].astype(float)
            < merged["brier_score_mean_control"].astype(float)
        ).sum()
        total = len(merged)
        rows.append(
            f"- Versus `{baseline}`: PR-AUC is higher in {int(pr_wins)}/{total} model-state comparisons; "
            f"Brier score is lower in {int(brier_wins)}/{total} comparisons."
        )
    return rows


def _write_fold_performance_plot(
    *,
    fold_metrics_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    wf_df = fold_metrics_df[fold_metrics_df["baseline"] == "behavioral_surface"].copy()
    if wf_df.empty or "fold" not in wf_df.columns or "pr_auc" not in wf_df.columns:
        return False

    wf_df["line_key"] = (
        wf_df["model"].astype(str)
        + " | "
        + wf_df["surface_id"].astype(str)
        + " | "
        + wf_df["state_id"].astype(str)
    )
    plt.figure(figsize=(10, 5))
    for line_key, group in wf_df.groupby("line_key"):
        sorted_group = group.sort_values("fold")
        plt.plot(
            sorted_group["fold"].astype(int),
            sorted_group["pr_auc"].astype(float),
            marker="o",
            linewidth=1.5,
            label=line_key,
        )
    plt.xlabel("Walk-forward fold")
    plt.ylabel("PR-AUC")
    plt.title("Predictive Performance by Walk-forward Fold")
    plt.grid(alpha=0.3)
    if wf_df["line_key"].nunique() <= 8:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _write_calibration_curve_plot(
    *,
    calibration_curve_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    if calibration_curve_df.empty:
        return False
    required = {"mean_pred", "observed_freq"}
    if not required.issubset(calibration_curve_df.columns):
        return False

    curve_df = calibration_curve_df.dropna(subset=["mean_pred", "observed_freq"]).copy()
    if curve_df.empty:
        return False

    grouped = (
        curve_df.groupby("bin", dropna=False)[["mean_pred", "observed_freq"]]
        .mean()
        .reset_index(drop=True)
        .sort_values("mean_pred")
    )
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Perfect calibration")
    plt.plot(
        grouped["mean_pred"].astype(float),
        grouped["observed_freq"].astype(float),
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="Behavioral Surface",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def write_walkforward_report(
    *,
    output_path: Path,
    experiment_id: str,
    config_payload: dict[str, object],
    run_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    calibration_curve_df: pd.DataFrame | None = None,
) -> None:
    plots_dir = output_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fold_plot_rel = "plots/fold_pr_auc.png"
    calibration_plot_rel = "plots/calibration_curve.png"
    has_fold_plot = _write_fold_performance_plot(
        fold_metrics_df=fold_metrics_df,
        output_path=output_path.parent / fold_plot_rel,
    )
    has_calibration_plot = _write_calibration_curve_plot(
        calibration_curve_df=calibration_curve_df if calibration_curve_df is not None else pd.DataFrame(),
        output_path=output_path.parent / calibration_plot_rel,
    )

    summary_lines = [
        "## Executive Summary",
        f"- total_runs: {len(run_df)}",
        f"- successful_runs: {(run_df['status'] == 'success').sum() if not run_df.empty else 0}",
        f"- failed_runs: {(run_df['status'] != 'success').sum() if not run_df.empty else 0}",
        f"- folds: {run_df['fold'].nunique() if 'fold' in run_df.columns and not run_df.empty else 0}",
    ]
    summary_lines.extend(_build_control_comparison_bullets(aggregated_df))

    behavior_agg = aggregated_df[aggregated_df.get("baseline") == "behavioral_surface"] if not aggregated_df.empty else pd.DataFrame()
    if not behavior_agg.empty and "pr_auc_mean" in behavior_agg.columns:
        pr_mean = _safe_float(behavior_agg["pr_auc_mean"].mean())
        if pr_mean is not None:
            summary_lines.append(f"- Behavioral Surface mean PR-AUC across model-state groups: {pr_mean:.3f}.")

    behavior_fold = fold_metrics_df[fold_metrics_df.get("baseline") == "behavioral_surface"] if not fold_metrics_df.empty else pd.DataFrame()
    if not behavior_fold.empty and "calibration_ece" in behavior_fold.columns:
        ece_mean = _safe_float(behavior_fold["calibration_ece"].mean())
        if ece_mean is not None:
            summary_lines.append(
                f"- Mean Expected Calibration Error (ECE) for Behavioral Surface predictions: {ece_mean:.3f}."
            )

    lines = [
        f"# Behavioral Walk-forward Predictive Validation Report: {experiment_id}",
        "",
        "This report evaluates predictive discrimination, calibration, and robustness only.",
        "It does not evaluate trading suitability, returns, or strategy profitability.",
        "",
        *summary_lines,
        "",
        "## Scientific Findings",
        "",
        "Predictive performance is interpreted relative to baseline controls rather than as isolated metrics.",
        "A Behavioral Surface is considered stronger only when discrimination and error metrics improve versus controls.",
        "Calibration is reported separately so discrimination gains can be interpreted together with probability reliability.",
    ]

    if has_fold_plot or has_calibration_plot:
        lines.extend(["", "## Plots"])
        if has_fold_plot:
            lines.extend(
                [
                    "",
                    "### Predictive Performance by Fold (PR-AUC)",
                    f"![Predictive performance by walk-forward fold]({fold_plot_rel})",
                ]
            )
        if has_calibration_plot:
            lines.extend(
                [
                    "",
                    "### Calibration Curve",
                    f"![Calibration curve reliability diagram]({calibration_plot_rel})",
                ]
            )

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
        if disp:
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
        if disp:
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
