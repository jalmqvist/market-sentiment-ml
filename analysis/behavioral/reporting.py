from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.behavioral.interpretation import format_key_observations, generate_key_observations, Observation


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

    lines = [
        f"# Behavioral Experiment Report: {experiment_id}",
        "",
        "## Reproducibility",
        f"- dataset_version: `{config.get('dataset_version')}`",
        f"- dataset_variant: `{config.get('dataset_variant')}`",
        f"- selected_models: `{','.join(config.get('models', []))}`",
        f"- feature_set: `{config.get('feature_set')}`",
        f"- target_horizon: `{config.get('target_horizon')}`",
        f"- git_commit: `{config.get('git_commit')}`",
        f"- started_at: `{config.get('started_at')}`",
        f"- finished_at: `{config.get('finished_at')}`",
        "",
        "## Discovered Behavioral Surface States",
    ]
    lines.append(_states_table(discovered_states or []))

    lines.extend([
        "",
        "## Execution Summary",
        f"- total_runs: {len(run_df)}",
        f"- successful_runs: {(run_df['status'] == 'success').sum() if not run_df.empty else 0}",
        f"- failed_runs: {len(failures) if failures is not None else 0}",
        "",
        "## Coverage",
    ])

    if coverage_df.empty:
        lines.append("No coverage rows generated.")
    else:
        display_cols = [c for c in coverage_df.columns if c in [
            "scope", "row_count", "pair_count",
            "coverage_fraction", "state_fraction_of_behavioral",
            "timestamp_unique", "timestamp_min", "timestamp_max",
        ]]
        lines.append(coverage_df[display_cols].to_markdown(index=False))

    lines.extend(["", "## Prediction Comparison (MLP vs LSTM)"])
    if compare_df.empty:
        lines.append("No comparable MLP/LSTM prediction overlap found.")
    else:
        lines.append(compare_df.to_markdown(index=False))

    # Scientific metrics
    if metrics_df is not None and not metrics_df.empty:
        lines.extend(["", "## Scientific Prediction Metrics"])
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
        lines.extend(["", "## Baseline Controls"])
        control_display_cols = [c for c in [
            "scope", "control_type", "row_count",
            "coverage_fraction", "pair_count",
            "timestamp_unique",
        ] if c in controls_df.columns]
        if control_display_cols:
            lines.append(controls_df[control_display_cols].to_markdown(index=False))

    # Key observations
    if key_observations is not None:
        lines.extend(["", format_key_observations(key_observations)])
    else:
        # Auto-generate if all required dataframes are present
        if not coverage_df.empty:
            auto_metrics = metrics_df if metrics_df is not None else pd.DataFrame()
            auto_compare = compare_df
            obs = generate_key_observations(coverage_df, auto_compare, auto_metrics, controls_df)
            lines.extend(["", format_key_observations(obs)])

    # Manifest issues (errors first, then warnings)
    lines.extend(["", "## Manifest Issues"])
    if warning_rows.empty:
        lines.append("No manifest errors or warnings detected.")
    else:
        # Show error rows
        if "error_count" in manifest_df.columns:
            error_rows = manifest_df[manifest_df["error_count"].fillna(0).astype(int) > 0]
            if not error_rows.empty:
                lines.extend(["", "### Errors"])
                lines.append(error_rows[["manifest_file", "errors"]].to_markdown(index=False))
        # Show warning rows
        if "warning_count" in manifest_df.columns:
            warn_rows = manifest_df[manifest_df["warning_count"].fillna(0).astype(int) > 0]
            if not warn_rows.empty:
                lines.extend(["", "### Warnings"])
                disp_cols = ["manifest_file", "warnings"]
                if "notes" in manifest_df.columns:
                    disp_cols = ["manifest_file", "warnings", "notes"]
                lines.append(warn_rows[[c for c in disp_cols if c in warn_rows.columns]].to_markdown(index=False))

    # Legacy compatibility: also show 'all_messages' column if no separate columns available
    if not manifest_df.empty and "warning_count" not in manifest_df.columns and "warnings" in manifest_df.columns:
        legacy_warn = manifest_df[manifest_df.get("warning_count", 0) > 0]
        if not legacy_warn.empty:
            lines.extend(["", "### Legacy Manifest Warnings"])
            lines.append(legacy_warn[["manifest_file", "warnings"]].to_markdown(index=False))

    if not failures.empty:
        lines.extend(["", "## Failed Runs"])
        lines.append(failures[["surface_id", "state_id", "model", "returncode", "log_file"]].to_markdown(index=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
