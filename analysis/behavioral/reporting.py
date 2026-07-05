from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary_csv(summary_rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_path, index=False)
    return df


def write_metrics_csv(metrics_rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    df.to_csv(output_path, index=False)
    return df


def write_markdown_report(
    *,
    output_path: Path,
    experiment_id: str,
    config: dict,
    run_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> None:
    failures = run_df[run_df["status"] != "success"] if not run_df.empty else run_df
    warning_rows = manifest_df[manifest_df.get("warning_count", 0) > 0] if not manifest_df.empty else manifest_df

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
        "## Execution Summary",
        f"- total_runs: {len(run_df)}",
        f"- successful_runs: {(run_df['status'] == 'success').sum() if not run_df.empty else 0}",
        f"- failed_runs: {len(failures) if failures is not None else 0}",
        "",
        "## Coverage",
    ]

    if coverage_df.empty:
        lines.append("No coverage rows generated.")
    else:
        lines.append(coverage_df.to_markdown(index=False))

    lines.extend(["", "## Prediction Comparison (MLP vs LSTM)"])
    if compare_df.empty:
        lines.append("No comparable MLP/LSTM prediction overlap found.")
    else:
        lines.append(compare_df.to_markdown(index=False))

    lines.extend(["", "## Manifest Warnings"])
    if warning_rows.empty:
        lines.append("No manifest warnings detected.")
    else:
        lines.append(warning_rows[["manifest_file", "warnings"]].to_markdown(index=False))

    if not failures.empty:
        lines.extend(["", "## Failed Runs"])
        lines.append(failures[["surface_id", "state_id", "model", "returncode", "log_file"]].to_markdown(index=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
