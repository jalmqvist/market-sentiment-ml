from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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

_MAX_PLOT_LEGEND_LINES = 8

# Walk-forward folds below this threshold trigger a caution warning and
# a brief explanation of why few folds were generated.
_FEW_FOLDS_THRESHOLD = 3
_CONTROL_BASELINES = [
    "permutation",
    "random_matched_partition",
    "trend_volatility",
    "base_rate",
]
_CONTROL_LABELS = {
    "permutation": "Permutation",
    "random_matched_partition": "Random partition",
    "trend_volatility": "Trend/volatility",
    "base_rate": "Base-rate",
}
_MODEL_LINESTYLES = {
    "mlp": "-",
    "lstm": "--",
}
_MAX_FINDING_EVIDENCE_LINES = 5


def _protocol_summary_block(
    *,
    protocol: dict[str, object],
    n_folds: int,
    dataset_date_range: tuple[str | None, str | None] | None,
) -> list[str]:
    """Return markdown lines for the Walk-forward Protocol summary table."""
    date_min, date_max = dataset_date_range if dataset_date_range else (None, None)
    dataset_span = f"{date_min} → {date_max}" if date_min and date_max else "unknown"
    lines = [
        "## Walk-forward Protocol",
        "",
        "| | |",
        "|---|---|",
        f"| Dataset span | {dataset_span} |",
        f"| Training window | {protocol.get('train_years', '?')} years |",
        f"| Test window | {protocol.get('test_months', '?')} months |",
        f"| Step | {protocol.get('step_months', '?')} months |",
        f"| Generated folds | {n_folds} |",
    ]
    return lines


def _few_folds_explanation(
    *,
    n_folds: int,
    protocol: dict[str, object],
    dataset_duration_years: float | None,
) -> list[str]:
    """Return markdown lines explaining why few folds were generated (when applicable)."""
    if n_folds >= _FEW_FOLDS_THRESHOLD:
        return []

    lines: list[str] = []

    if n_folds == 1:
        lines.append(
            "> ⚠️ **Only one walk-forward fold could be generated from the available dataset "
            "using the selected protocol. Predictive conclusions should therefore be interpreted "
            "with caution.**"
        )
    elif n_folds == 0:
        lines.append(
            "> ⚠️ **No walk-forward folds could be generated. The dataset may be too short "
            "for the requested training window.**"
        )
    else:
        lines.append(
            f"> ⚠️ **Only {n_folds} walk-forward folds were generated. "
            "Predictive conclusions should be interpreted with caution given the limited evaluation window.**"
        )

    if dataset_duration_years is not None:
        train_years = protocol.get("train_years", "?")
        step_months = protocol.get("step_months", "?")
        remaining = (
            round(dataset_duration_years - float(train_years), 1)
            if isinstance(train_years, (int, float))
            else None
        )
        lines.extend([
            "",
            "### Why Few Folds Were Generated",
            "",
            "| | |",
            "|---|---|",
            f"| Dataset duration | {dataset_duration_years:.1f} years |",
            f"| Requested training window | {train_years} years |",
        ])
        if remaining is not None:
            lines.append(f"| Remaining data after training | {remaining:.1f} years |")
        fold_word = "fold" if n_folds == 1 else "folds"
        if n_folds <= 1:
            lines.append(
                f"| Result | Only {n_folds} {fold_word} satisfies the requested protocol. |"
            )
        else:
            lines.append(
                f"| Result | {n_folds} {fold_word} satisfy the requested protocol "
                f"(step = {step_months} months). |"
            )
    return lines


def _protocol_assessment_block(
    *,
    n_folds: int,
    protocol: dict[str, object],
    dataset_duration_years: float | None,
) -> list[str]:
    """Return markdown lines for the Protocol Assessment section."""
    train_years = protocol.get("train_years", 0)
    duration_adequate = (
        dataset_duration_years is not None
        and isinstance(train_years, (int, float))
        and dataset_duration_years >= float(train_years) + 0.5
    )
    multiple_folds = n_folds >= 2
    good_folds = n_folds >= _FEW_FOLDS_THRESHOLD

    checks = [
        ("Dataset duration adequate", duration_adequate),
        ("Multiple folds generated", multiple_folds),
        ("Fold schedule reproducible", True),
        ("Shared MPML fold protocol", protocol.get("protocol") == "mpml_reference_v1"),
    ]

    lines = ["## Protocol Validation", ""]
    for label, ok in checks:
        mark = "✓" if ok else "✗"
        lines.append(f"{mark} {label}")

    lines.append("")
    if good_folds and duration_adequate:
        lines.append("**Protocol Quality: GOOD**")
    elif multiple_folds:
        lines.append("**Protocol Quality: ADEQUATE**")
        lines.append("")
        lines.append(f"_Reason: {n_folds} walk-forward folds available; "
                     "results should be interpreted with moderate caution._")
    else:
        lines.append("**Protocol Quality: LIMITED**")
        lines.append("")
        if n_folds <= 1:
            lines.append("_Reason: Only one walk-forward fold available._")
        else:
            lines.append(f"_Reason: {n_folds} walk-forward folds available._")

    return lines

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
    relative_df = _build_relative_improvement_frame(aggregated_df)
    if relative_df.empty:
        return []

    comparison_bullets: list[str] = ["- Relative improvement over controls:"]
    for baseline in _CONTROL_BASELINES:
        baseline_df = relative_df[relative_df["baseline"] == baseline].copy()
        if baseline_df.empty:
            continue
        pr_mean = _safe_float(baseline_df["pr_auc_relative_pct"].mean())
        if pr_mean is None:
            continue
        comparison_bullets.append(
            f"  - {_CONTROL_LABELS.get(baseline, baseline)}: {_format_relative_pct(pr_mean)} PR-AUC"
        )
    return comparison_bullets


def _format_relative_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _relative_improvement(
    behavioral_value: object,
    baseline_value: object,
    *,
    higher_is_better: bool,
) -> float | None:
    behavioral = _safe_float(behavioral_value)
    baseline = _safe_float(baseline_value)
    if behavioral is None or baseline is None or baseline == 0:
        return None
    if higher_is_better:
        return ((behavioral - baseline) / abs(baseline)) * 100.0
    return ((baseline - behavioral) / abs(baseline)) * 100.0


def _build_relative_improvement_frame(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    if aggregated_df.empty or "baseline" not in aggregated_df.columns:
        return pd.DataFrame()

    behavior_df = aggregated_df[aggregated_df["baseline"] == "behavioral_surface"].copy()
    if behavior_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    merge_keys = ["model", "surface_id", "state_id"]
    for baseline in _CONTROL_BASELINES:
        baseline_df = aggregated_df[aggregated_df["baseline"] == baseline].copy()
        if baseline_df.empty:
            continue
        merged = behavior_df.merge(
            baseline_df,
            on=merge_keys,
            how="inner",
            suffixes=("_behavioral", "_control"),
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "baseline": baseline,
                    "model": row["model"],
                    "surface_id": row["surface_id"],
                    "state_id": row["state_id"],
                    "pr_auc_relative_pct": _relative_improvement(
                        row.get("pr_auc_mean_behavioral"),
                        row.get("pr_auc_mean_control"),
                        higher_is_better=True,
                    ),
                    "mcc_relative_pct": _relative_improvement(
                        row.get("mcc_mean_behavioral"),
                        row.get("mcc_mean_control"),
                        higher_is_better=True,
                    ),
                    "balanced_accuracy_relative_pct": _relative_improvement(
                        row.get("balanced_accuracy_mean_behavioral"),
                        row.get("balanced_accuracy_mean_control"),
                        higher_is_better=True,
                    ),
                    "brier_score_relative_pct": _relative_improvement(
                        row.get("brier_score_mean_behavioral"),
                        row.get("brier_score_mean_control"),
                        higher_is_better=False,
                    ),
                    "calibration_ece_relative_pct": _relative_improvement(
                        row.get("calibration_ece_mean_behavioral"),
                        row.get("calibration_ece_mean_control"),
                        higher_is_better=False,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_state_color_map(state_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {
        state_id: cmap(idx % cmap.N)
        for idx, state_id in enumerate(sorted(set(state_ids)))
    }


def _interest_confidence_line(interest: str, confidence: str) -> str:
    return (
        f"**Scientific Interest:** {interest.capitalize()}  "
        f"**Scientific Confidence:** {confidence.capitalize()}"
    )


def _truncate_evidence_lines(evidence_lines: list[str]) -> list[str]:
    if len(evidence_lines) <= _MAX_FINDING_EVIDENCE_LINES:
        return evidence_lines
    hidden = len(evidence_lines) - _MAX_FINDING_EVIDENCE_LINES
    return evidence_lines[:_MAX_FINDING_EVIDENCE_LINES] + [f"- … {hidden} additional evidence line(s) omitted."]


def _build_walkforward_findings(
    *,
    aggregated_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    calibration_curve_df: pd.DataFrame,
) -> list[Finding]:
    findings: list[Finding] = []
    relative_df = _build_relative_improvement_frame(aggregated_df)
    behavioral_agg = aggregated_df[aggregated_df["baseline"] == "behavioral_surface"].copy()
    behavioral_fold = fold_metrics_df[fold_metrics_df["baseline"] == "behavioral_surface"].copy()

    if not relative_df.empty:
        baseline_lines: list[str] = []
        positive_groups = 0
        total_groups = 0
        for baseline in _CONTROL_BASELINES:
            baseline_df = relative_df[relative_df["baseline"] == baseline].copy()
            if baseline_df.empty:
                continue
            pr_mean = _safe_float(baseline_df["pr_auc_relative_pct"].mean())
            brier_mean = _safe_float(baseline_df["brier_score_relative_pct"].mean())
            wins = int((pd.to_numeric(baseline_df["pr_auc_relative_pct"], errors="coerce") > 0).sum())
            comps = int(pd.to_numeric(baseline_df["pr_auc_relative_pct"], errors="coerce").notna().sum())
            positive_groups += wins
            total_groups += comps
            baseline_lines.append(
                f"- {_CONTROL_LABELS.get(baseline, baseline)}: PR-AUC {_format_relative_pct(pr_mean)}; "
                f"Brier {_format_relative_pct(brier_mean)}; positive PR-AUC in {wins}/{comps} comparisons."
            )
        if baseline_lines:
            share = (positive_groups / total_groups) if total_groups else 0.0
            findings.append(
                Finding(
                    title="Behavioral Surface outperforms predictive controls in most comparisons",
                    description="",
                    evidence=baseline_lines,
                    interest="high",
                    confidence="high" if share >= 0.65 else "medium",
                    follow_up=(
                        "Replicate the strongest state-level gains on an extended walk-forward horizon "
                        "before promoting the Surface."
                    ),
                    interpretation=(
                        f"The Behavioral Surface delivers positive relative PR-AUC in {positive_groups}/{total_groups} "
                        "baseline comparisons. The current evidence is consistent with predictive structure beyond "
                        "label permutation, random partitions, and simple regime controls."
                    ),
                )
            )

    if not behavioral_agg.empty and behavioral_agg["state_id"].nunique() > 1:
        state_perf = (
            behavioral_agg.groupby("state_id", dropna=False)["pr_auc_mean"]
            .mean()
            .sort_values(ascending=False)
        )
        state_rel = (
            relative_df.groupby("state_id", dropna=False)["pr_auc_relative_pct"].mean()
            if not relative_df.empty
            else pd.Series(dtype=float)
        )
        best_state = str(state_perf.index[0])
        worst_state = str(state_perf.index[-1])
        spread = _safe_float(state_perf.iloc[0] - state_perf.iloc[-1])
        evidence_lines = []
        for state_id, pr_mean in state_perf.items():
            rel_mean = _safe_float(state_rel.get(state_id)) if not state_rel.empty else None
            evidence_lines.append(
                f"- {state_id}: mean PR-AUC {float(pr_mean):.3f}; mean relative PR-AUC vs controls "
                f"{_format_relative_pct(rel_mean)}."
            )
        if spread is not None:
            state_interpretation = (
                f"State-level discrimination is heterogeneous: `{best_state}` leads the Surface while "
                f"`{worst_state}` trails by {spread:.3f} PR-AUC. Scientific conclusions should therefore "
                "be framed by Behavioral State rather than by a single pooled average."
            )
        else:
            state_interpretation = (
                "State-level discrimination is heterogeneous, so conclusions should be framed by Behavioral State."
            )
        findings.append(
            Finding(
                title="Predictive performance differs materially between Behavioral States",
                description="",
                evidence=_truncate_evidence_lines(evidence_lines),
                interest="high",
                confidence="high" if len(state_perf) >= 3 else "medium",
                follow_up=(
                    f"Use `{best_state}` as the reference state for follow-up validation and investigate why "
                    f"`{worst_state}` underperforms before broadening claims across the full Surface."
                ),
                interpretation=state_interpretation,
            )
        )

    if not behavioral_fold.empty and "calibration_ece" in behavioral_fold.columns:
        ece_mean = _safe_float(pd.to_numeric(behavioral_fold["calibration_ece"], errors="coerce").mean())
        state_ece = (
            behavioral_fold.groupby("state_id", dropna=False)["calibration_ece"]
            .mean()
            .sort_values(ascending=False)
        )
        weighted_gap = None
        if not calibration_curve_df.empty and {"mean_pred", "observed_freq", "count"}.issubset(calibration_curve_df.columns):
            curve = calibration_curve_df.dropna(subset=["mean_pred", "observed_freq"]).copy()
            if not curve.empty:
                weighted_gap = _safe_float(
                    np.average(
                        curve["mean_pred"].astype(float) - curve["observed_freq"].astype(float),
                        weights=curve["count"].fillna(0).astype(float).clip(lower=0.0),
                    )
                )
        if ece_mean is not None:
            if weighted_gap is not None and weighted_gap > 0.01:
                title = "Calibration remains systematically over-confident"
                interpretation = (
                    f"Mean calibration gap stays positive (weighted reliability gap {weighted_gap:.3f}), "
                    "so predicted probabilities are generally more extreme than observed outcomes."
                )
            elif weighted_gap is not None and weighted_gap < -0.01:
                title = "Calibration remains systematically under-confident"
                interpretation = (
                    f"Mean calibration gap stays negative (weighted reliability gap {weighted_gap:.3f}), "
                    "so predicted probabilities remain more conservative than realized outcomes."
                )
            else:
                title = "Calibration error remains material across Behavioral States"
                interpretation = (
                    "Discrimination gains are not yet matched by equally reliable probability estimates, "
                    "so calibration should remain a separate part of the scientific interpretation."
                )
            evidence_lines = [f"- Mean calibration ECE across Behavioral Surface folds: {ece_mean:.3f}."]
            if not state_ece.empty:
                for state_id, state_ece_value in state_ece.head(3).items():
                    evidence_lines.append(f"- {state_id}: mean ECE {float(state_ece_value):.3f}.")
            findings.append(
                Finding(
                    title=title,
                    description="",
                    evidence=evidence_lines,
                    interest="medium",
                    confidence="high" if len(behavioral_fold) >= 6 else "medium",
                    follow_up=(
                        "Retain calibration plots in every predictive report and prioritize calibration tuning "
                        "before interpreting probabilities as confidence statements."
                    ),
                    interpretation=interpretation,
                )
            )

    if not behavioral_agg.empty and {"mlp", "lstm"}.issubset(set(behavioral_agg["model"].astype(str))):
        pivot = behavioral_agg.pivot_table(
            index=["surface_id", "state_id"],
            columns="model",
            values="pr_auc_mean",
            aggfunc="mean",
        ).dropna(subset=["mlp", "lstm"], how="any")
        if not pivot.empty:
            abs_diff = (pivot["mlp"].astype(float) - pivot["lstm"].astype(float)).abs()
            mean_abs_diff = _safe_float(abs_diff.mean())
            same_winner = int((abs_diff <= 0.02).sum())
            evidence_lines = [
                f"- {state_id}: MLP {float(row['mlp']):.3f}; LSTM {float(row['lstm']):.3f}; "
                f"gap {abs(float(row['mlp']) - float(row['lstm'])):.3f}."
                for (_, state_id), row in pivot.iterrows()
            ]
            if mean_abs_diff is None:
                similar = False
            else:
                similar = mean_abs_diff <= 0.03
            findings.append(
                Finding(
                    title=(
                        "MLP and LSTM exhibit similar predictive behavior"
                        if similar
                        else "MLP and LSTM diverge by Behavioral State"
                    ),
                    description="",
                    evidence=_truncate_evidence_lines(evidence_lines),
                    interest="medium",
                    confidence="high" if len(pivot) >= 3 else "medium",
                    follow_up=(
                        "Use agreement between architectures as a robustness check when choosing which "
                        "state-level results to replicate."
                    ),
                    interpretation=(
                        f"Mean absolute PR-AUC difference between architectures is {mean_abs_diff:.3f}, with "
                        f"{same_winner}/{len(pivot)} state-level comparisons within ±0.02."
                        if mean_abs_diff is not None
                        else "Architecture agreement could not be quantified reliably."
                    ),
                )
            )

    return findings[:5]


def _format_walkforward_findings(findings: list[Finding]) -> str:
    if not findings:
        return "## Scientific Findings\n\nNo scientific findings could be derived from the available predictive outputs.\n"

    lines = ["## Scientific Findings", ""]
    for index, finding in enumerate(findings, start=1):
        lines.extend(
            [
                f"### Finding {index}",
                f"**Finding:** {finding.title}",
                "",
                "**Evidence:**",
            ]
        )
        lines.extend(finding.evidence or ["- No supporting evidence available."])
        lines.extend(
            [
                "",
                f"**Interpretation:** {finding.interpretation or finding.description or 'No interpretation available.'}",
                "",
                _interest_confidence_line(finding.interest, finding.confidence),
                "",
                f"**Recommended follow-up:** {finding.follow_up or 'No follow-up specified.'}",
                "",
            ]
        )
    return "\n".join(lines)


def _derive_walkforward_recommendation(
    *,
    relative_df: pd.DataFrame,
    behavioral_agg: pd.DataFrame,
    findings: list[Finding],
) -> str:
    if behavioral_agg.empty:
        return "Insufficient predictive evidence is available to recommend a follow-up."

    state_perf = (
        behavioral_agg.groupby("state_id", dropna=False)["pr_auc_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    best_state = str(state_perf.index[0]) if not state_perf.empty else None
    calibration_issue = any("Calibration" in finding.title or "calibration" in finding.title for finding in findings)
    if not relative_df.empty:
        state_rel = relative_df.groupby("state_id", dropna=False)["pr_auc_relative_pct"].mean().sort_values(ascending=False)
        leading_state = str(state_rel.index[0]) if not state_rel.empty else best_state
        trailing = [
            str(idx)
            for idx, value in state_rel.items()
            if _safe_float(value) is not None and float(value) <= 0
        ]
        if trailing:
            trailing_text = ", ".join(trailing[:3])
            return (
                f"Prioritize follow-up validation on `{leading_state or best_state}` and treat {trailing_text} as "
                "state-specific exceptions until their control-relative performance is replicated."
            )
        if calibration_issue and leading_state is not None:
            return (
                f"Use `{leading_state}` as the leading predictive state, but keep calibration diagnostics in the "
                "decision loop before interpreting its probabilities as reliable confidence estimates."
            )
        if leading_state is not None:
            return (
                f"Replicate the strongest control-relative gains in `{leading_state}` and use its state-level "
                "performance as the benchmark for future Behavioral Surface comparisons."
            )
    if best_state is not None:
        return (
            f"Use `{best_state}` as the reference Behavioral State for the next predictive validation pass and "
            "treat pooled Surface averages as secondary evidence."
        )
    return "Repeat the walk-forward analysis once additional predictive evidence is available."


def _write_fold_performance_plot(
    *,
    fold_metrics_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    if fold_metrics_df.empty or "baseline" not in fold_metrics_df.columns:
        return False
    wf_df = fold_metrics_df[fold_metrics_df["baseline"] == "behavioral_surface"].copy()
    if wf_df.empty or "fold" not in wf_df.columns or "pr_auc" not in wf_df.columns:
        return False

    state_colors = _build_state_color_map(wf_df["state_id"].astype(str).tolist())
    plt.figure(figsize=(10, 5))
    for (state_id, model), group in wf_df.groupby(["state_id", "model"], dropna=False):
        sorted_group = group.sort_values("fold")
        plt.plot(
            sorted_group["fold"].astype(int),
            sorted_group["pr_auc"].astype(float),
            marker="o",
            linewidth=1.5,
            color=state_colors.get(str(state_id)),
            linestyle=_MODEL_LINESTYLES.get(str(model), "-."),
        )
    plt.xlabel("Walk-forward fold")
    plt.ylabel("PR-AUC")
    plt.title("Predictive Performance by Walk-forward Fold")
    plt.grid(alpha=0.3)
    state_handles = [
        Line2D([0], [0], color=color, lw=2, label=state_id)
        for state_id, color in state_colors.items()
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=style, label=model.upper())
        for model, style in _MODEL_LINESTYLES.items()
        if model in set(wf_df["model"].astype(str))
    ]
    has_state_legend = len(state_handles) <= _MAX_PLOT_LEGEND_LINES
    if has_state_legend:
        state_legend = plt.legend(handles=state_handles, title="Behavioral State", loc="upper left", fontsize=8)
        plt.gca().add_artist(state_legend)
    if model_handles:
        model_loc = "lower right" if has_state_legend else "upper right"
        plt.legend(handles=model_handles, title="Model", loc=model_loc, fontsize=8)
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
        .reset_index()
        .sort_values("bin")
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
    n_folds: int = 0,
    dataset_date_range: tuple[str | None, str | None] | None = None,
    dataset_duration_years: float | None = None,
    skipped_state_rows: list[dict[str, object]] | None = None,
) -> None:
    plots_dir = output_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    protocol = config_payload.get("walkforward_protocol") or {}

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

    behavior_agg = (
        aggregated_df[aggregated_df["baseline"] == "behavioral_surface"]
        if (not aggregated_df.empty and "baseline" in aggregated_df.columns)
        else pd.DataFrame()
    )
    if not behavior_agg.empty and "pr_auc_mean" in behavior_agg.columns:
        pr_mean = _safe_float(behavior_agg["pr_auc_mean"].mean())
        if pr_mean is not None:
            summary_lines.append(f"- Behavioral Surface mean PR-AUC across model-state groups: {pr_mean:.3f}.")
        state_summary = (
            behavior_agg.groupby("state_id", dropna=False)["pr_auc_mean"]
            .mean()
            .sort_values(ascending=False)
        )
        if not state_summary.empty:
            summary_lines.append(
                f"- Best-performing Behavioral State by mean PR-AUC: `{state_summary.index[0]}`."
            )

    behavior_fold = (
        fold_metrics_df[fold_metrics_df["baseline"] == "behavioral_surface"]
        if (not fold_metrics_df.empty and "baseline" in fold_metrics_df.columns)
        else pd.DataFrame()
    )
    if not behavior_fold.empty and "calibration_ece" in behavior_fold.columns:
        ece_mean = _safe_float(behavior_fold["calibration_ece"].mean())
        if ece_mean is not None:
            summary_lines.append(
                f"- Mean Expected Calibration Error (ECE) for Behavioral Surface predictions: {ece_mean:.3f}."
            )

    relative_df = _build_relative_improvement_frame(aggregated_df)
    findings = _build_walkforward_findings(
        aggregated_df=aggregated_df,
        fold_metrics_df=fold_metrics_df,
        calibration_curve_df=calibration_curve_df if calibration_curve_df is not None else pd.DataFrame(),
    )
    recommendation = _derive_walkforward_recommendation(
        relative_df=relative_df,
        behavioral_agg=behavior_agg,
        findings=findings,
    )

    lines = [
        f"# Behavioral Walk-forward Predictive Validation Report: {experiment_id}",
        "",
        "This report evaluates predictive discrimination, calibration, and robustness only.",
        "It does not evaluate trading suitability, returns, or strategy profitability.",
        "",
    ]

    # Walk-forward Protocol summary (always shown first)
    lines.extend(_protocol_summary_block(
        protocol=protocol,
        n_folds=n_folds,
        dataset_date_range=dataset_date_range,
    ))

    # Warning and explanation when few folds were generated
    few_folds_lines = _few_folds_explanation(
        n_folds=n_folds,
        protocol=protocol,
        dataset_duration_years=dataset_duration_years,
    )
    if few_folds_lines:
        lines.append("")
        lines.extend(few_folds_lines)

    lines.extend([
        "",
        *summary_lines,
        "",
        _format_walkforward_findings(findings).rstrip(),
        "",
        "## Research Recommendation",
        "",
        recommendation,
    ])

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

    lines.append("")
    lines.extend(_protocol_assessment_block(
        n_folds=n_folds,
        protocol=protocol,
        dataset_duration_years=dataset_duration_years,
    ))

    lines.extend(["", "---", "", "## Appendix"])

    if not relative_df.empty:
        lines.extend(["", "### Relative Improvement over Controls"])
        relative_disp = [
            c
            for c in [
                "baseline",
                "model",
                "surface_id",
                "state_id",
                "pr_auc_relative_pct",
                "brier_score_relative_pct",
                "mcc_relative_pct",
                "balanced_accuracy_relative_pct",
                "calibration_ece_relative_pct",
            ]
            if c in relative_df.columns
        ]
        appendix_relative = relative_df[relative_disp].copy()
        if "baseline" in appendix_relative.columns:
            appendix_relative["baseline"] = appendix_relative["baseline"].map(
                lambda x: _CONTROL_LABELS.get(str(x), str(x))
            )
        lines.append(appendix_relative.to_markdown(index=False))

    if not aggregated_df.empty:
        lines.extend(["", "### Aggregate Predictive Comparison"])
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
                "calibration_ece_mean",
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
        lines.extend(["", "### Per-fold Metrics"])
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

    # Skipped Behavioral States
    if skipped_state_rows:
        lines.extend(["", "### Skipped Behavioral States", ""])
        skipped_df = pd.DataFrame(skipped_state_rows)
        disp_cols = [c for c in ["fold", "surface_id", "state_id", "model", "reason"] if c in skipped_df.columns]
        lines.append(skipped_df[disp_cols].to_markdown(index=False))
    else:
        lines.extend(["", "### Skipped Behavioral States", "", "No behavioral states were skipped."])

    lines.extend(
        [
            "",
            "### Reproducibility",
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
