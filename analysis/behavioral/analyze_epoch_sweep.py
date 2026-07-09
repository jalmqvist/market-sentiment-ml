from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


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
_METRICS = [
    "pr_auc",
    "mcc",
    "balanced_accuracy",
    "brier_score",
    "calibration_ece",
]


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


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


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _build_state_color_map(state_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {
        state_id: cmap(idx % cmap.N)
        for idx, state_id in enumerate(sorted(set(state_ids)))
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze repeated walk-forward experiments across epoch counts.",
    )
    parser.add_argument("sweep_manifest", type=Path, help="CSV with columns: epoch,experiment_dir")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for epoch summary, report, and plots (default: <manifest_dir>/epoch_sweep_analysis).",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.005,
        help="PR-AUC change threshold for plateau detection across the final two epoch intervals.",
    )
    return parser.parse_args(argv)


def _resolve_experiment_dir(base_dir: Path, raw_value: object) -> Path:
    """
    Resolve an experiment directory from a sweep manifest.

    Supported path styles:

      1. Absolute
         /home/user/.../behavioral_walkforward_xxx

      2. Repository-relative
         analysis/output/behavioral_walkforward_xxx

      3. Manifest-relative
         ../behavioral_walkforward_xxx
    """

    path = Path(str(raw_value))

    #
    # Absolute path
    #
    if path.is_absolute():
        return path

    #
    # Repository-relative path
    #
    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = (repo_root / path).resolve()

    if repo_candidate.exists():
        return repo_candidate

    #
    # Manifest-relative path
    #
    return (base_dir / path).resolve()


def _load_sweep_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epoch", "experiment_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sweep manifest missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Sweep manifest is empty.")
    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    return df.sort_values("epoch").reset_index(drop=True)


def _load_experiment_outputs(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics_path = exp_dir / "metrics.csv"
    summary_path = exp_dir / "summary.csv"
    manifest_path = exp_dir / "experiment_manifest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {exp_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in {exp_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"experiment_manifest.json not found in {exp_dir}")
    metrics_df = pd.read_csv(metrics_path)
    summary_df = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return metrics_df, summary_df, manifest


def _extract_fold_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    if "metric_group" in metrics_df.columns:
        fold_df = metrics_df[metrics_df["metric_group"] == "walkforward_fold"].copy()
        if not fold_df.empty:
            return fold_df
    required = {"baseline", "model", "state_id"}
    return metrics_df.copy() if required.issubset(metrics_df.columns) else pd.DataFrame()


def _aggregate_epoch_metrics(sweep_df: pd.DataFrame, manifest_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    aggregated_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for row in sweep_df.to_dict(orient="records"):
        epoch = int(row["epoch"])
        exp_dir = _resolve_experiment_dir(manifest_dir, row["experiment_dir"])
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory not found:\n"
                f"  requested: {row['experiment_dir']}\n"
                f"  resolved : {exp_dir}"
            )
        metrics_df, summary_df, manifest = _load_experiment_outputs(exp_dir)
        fold_df = _extract_fold_metrics(metrics_df)
        if fold_df.empty:
            continue
        available_metrics = [metric for metric in _METRICS if metric in fold_df.columns]
        group_cols = ["surface_id", "state_id", "model", "baseline"]
        grouped = (
            fold_df.groupby(group_cols, dropna=False)[available_metrics]
            .mean()
            .reset_index()
        )
        grouped.insert(0, "epoch", epoch)
        for metric in available_metrics:
            grouped.rename(columns={metric: f"{metric}_mean"}, inplace=True)
        aggregated_frames.append(grouped)
        metadata_rows.append(
            {
                "epoch": epoch,
                "experiment_dir": str(exp_dir),
                "models": ", ".join(sorted(set(summary_df.get("model", pd.Series(dtype=str)).dropna().astype(str)))),
                "surface_ids": ", ".join(sorted(set(grouped["surface_id"].dropna().astype(str)))),
                "states": int(grouped["state_id"].nunique()),
                "fold_rows": int(len(fold_df)),
                "manifest_models": ", ".join(manifest.get("models_executed", [])),
            }
        )
    if not aggregated_frames:
        return pd.DataFrame(), metadata_rows
    return pd.concat(aggregated_frames, ignore_index=True), metadata_rows


def _build_relative_improvement_summary(epoch_df: pd.DataFrame) -> pd.DataFrame:
    if epoch_df.empty:
        return pd.DataFrame()
    behavior_df = epoch_df[epoch_df["baseline"] == "behavioral_surface"].copy()
    if behavior_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    merge_keys = ["epoch", "surface_id", "state_id", "model"]
    for baseline in _CONTROL_BASELINES:
        baseline_df = epoch_df[epoch_df["baseline"] == baseline].copy()
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
                    "epoch": int(row["epoch"]),
                    "surface_id": row["surface_id"],
                    "state_id": row["state_id"],
                    "model": row["model"],
                    "baseline": baseline,
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


def _build_epoch_summary(epoch_df: pd.DataFrame, relative_df: pd.DataFrame) -> pd.DataFrame:
    if epoch_df.empty:
        return pd.DataFrame()
    summary_df = epoch_df.copy()
    if relative_df.empty:
        return summary_df.sort_values(["epoch", "surface_id", "state_id", "model", "baseline"]).reset_index(drop=True)

    behavior_df = summary_df[summary_df["baseline"] == "behavioral_surface"].copy()
    if behavior_df.empty:
        return summary_df.sort_values(["epoch", "surface_id", "state_id", "model", "baseline"]).reset_index(drop=True)

    wide = (
        relative_df.pivot_table(
            index=["epoch", "surface_id", "state_id", "model"],
            columns="baseline",
            values=[
                "pr_auc_relative_pct",
                "mcc_relative_pct",
                "balanced_accuracy_relative_pct",
                "brier_score_relative_pct",
                "calibration_ece_relative_pct",
            ],
            aggfunc="mean",
        )
        .sort_index(axis=1)
    )
    wide.columns = [f"{metric}_vs_{baseline}" for metric, baseline in wide.columns]
    wide = wide.reset_index()
    merged = behavior_df.merge(
        wide,
        on=["epoch", "surface_id", "state_id", "model"],
        how="left",
    )
    baseline_df = summary_df[summary_df["baseline"] != "behavioral_surface"].copy()
    full = pd.concat([merged, baseline_df], ignore_index=True, sort=False)
    return full.sort_values(["epoch", "surface_id", "state_id", "model", "baseline"]).reset_index(drop=True)


def _detect_convergence(behavior_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if behavior_df.empty:
        return pd.DataFrame()

    state_epoch_df = (
        behavior_df.groupby(["surface_id", "state_id", "epoch"], dropna=False)["pr_auc_mean"]
        .mean()
        .reset_index()
        .sort_values(["surface_id", "state_id", "epoch"])
    )
    for (surface_id, state_id), group in state_epoch_df.groupby(["surface_id", "state_id"], dropna=False):
        epochs = group["epoch"].astype(int).tolist()
        pr_values = group["pr_auc_mean"].astype(float).to_numpy()
        valid_mask = np.isfinite(pr_values)
        if int(valid_mask.sum()) < 2:
            continue
        epochs = [epoch for epoch, is_valid in zip(epochs, valid_mask) if is_valid]
        pr_values = pr_values[valid_mask]
        best_idx = int(np.argmax(pr_values))
        best_epoch = int(epochs[best_idx])
        best_pr = float(pr_values[best_idx])
        deltas = np.diff(pr_values)
        plateau = len(deltas) >= 2 and bool(np.all(np.abs(deltas[-2:]) < threshold))
        near_best_epochs: list[int] = []
        if plateau:
            near_best_epochs = [
                int(epoch)
                for epoch, pr_value in zip(epochs, pr_values)
                if abs(best_pr - float(pr_value)) <= threshold
            ]
        recommended_epoch = min(near_best_epochs) if near_best_epochs else best_epoch
        rows.append(
            {
                "surface_id": str(surface_id),
                "state_id": str(state_id),
                "best_epoch": best_epoch,
                "best_pr_auc": best_pr,
                "plateau_detected": plateau,
                "recommended_epoch": int(recommended_epoch),
                "last_delta": float(deltas[-1]) if len(deltas) >= 1 else float("nan"),
                "penultimate_delta": float(deltas[-2]) if len(deltas) >= 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["surface_id", "state_id"]).reset_index(drop=True)


def _write_metric_plot(
    *,
    behavior_df: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if behavior_df.empty or metric_column not in behavior_df.columns:
        return False
    plot_df = behavior_df.dropna(subset=[metric_column]).copy()
    if plot_df.empty:
        return False
    state_colors = _build_state_color_map(plot_df["state_id"].astype(str).tolist())
    plt.figure(figsize=(10, 5))
    for (state_id, model), group in plot_df.groupby(["state_id", "model"], dropna=False):
        ordered = group.sort_values("epoch")
        plt.plot(
            ordered["epoch"].astype(int),
            ordered[metric_column].astype(float),
            marker="o",
            linewidth=1.6,
            color=state_colors.get(str(state_id)),
            linestyle=_MODEL_LINESTYLES.get(str(model), "-."),
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    state_handles = [
        Line2D([0], [0], color=color, lw=2, label=state_id)
        for state_id, color in state_colors.items()
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=style, label=model.upper())
        for model, style in _MODEL_LINESTYLES.items()
        if model in set(plot_df["model"].astype(str))
    ]
    state_legend = plt.legend(handles=state_handles, title="Behavioral State", loc="upper left", fontsize=8)
    plt.gca().add_artist(state_legend)
    if model_handles:
        plt.legend(handles=model_handles, title="Model", loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _write_relative_improvement_plot(relative_df: pd.DataFrame, output_path: Path) -> bool:
    if relative_df.empty or "pr_auc_relative_pct" not in relative_df.columns:
        return False
    plot_df = (
        relative_df.groupby(["epoch", "baseline"], dropna=False)["pr_auc_relative_pct"]
        .mean()
        .reset_index()
    )
    plot_df = plot_df.dropna(subset=["pr_auc_relative_pct"])
    if plot_df.empty:
        return False
    plt.figure(figsize=(10, 5))
    for baseline, group in plot_df.groupby("baseline", dropna=False):
        ordered = group.sort_values("epoch")
        plt.plot(
            ordered["epoch"].astype(int),
            ordered["pr_auc_relative_pct"].astype(float),
            marker="o",
            linewidth=1.8,
            label=_CONTROL_LABELS.get(str(baseline), str(baseline)),
        )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Relative PR-AUC improvement (%)")
    plt.title("Relative Improvement over Controls")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _build_recommendation(convergence_df: pd.DataFrame, epochs: list[int]) -> str:
    if convergence_df.empty:
        return "No convergence recommendation could be derived from the available sweep outputs."
    plateau_states = convergence_df[convergence_df["plateau_detected"]].copy()
    continuing_states = convergence_df[~convergence_df["plateau_detected"]].copy()

    lines: list[str] = []
    if not plateau_states.empty:
        states = ", ".join(plateau_states["state_id"].astype(str).tolist())
        default_epoch = int(plateau_states["recommended_epoch"].max())
        lines.append(
            f"{len(plateau_states)} Behavioral State(s) appear to have reached predictive convergence: {states}."
        )
        lines.append(f"Use approximately {default_epoch} epochs as the default starting point for this Surface.")
    if not continuing_states.empty:
        states = ", ".join(continuing_states["state_id"].astype(str).tolist())
        next_epoch = int(max(epochs) * 2) if epochs else None
        if next_epoch is not None:
            lines.append(
                f"{states} continue improving through the highest evaluated epoch; extend the sweep to {next_epoch} epochs."
            )
        else:
            lines.append(f"{states} continue improving through the highest evaluated epoch.")
    return "\n\n".join(lines)


def _render_report(
    *,
    epoch_summary_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    metadata_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    behavior_df = epoch_summary_df[epoch_summary_df["baseline"] == "behavioral_surface"].copy()
    epochs = sorted(set(epoch_summary_df["epoch"].astype(int))) if not epoch_summary_df.empty else []
    surface_names = ", ".join(sorted(set(behavior_df.get("surface_id", pd.Series(dtype=str)).dropna().astype(str))))
    models = ", ".join(sorted(set(behavior_df.get("model", pd.Series(dtype=str)).dropna().astype(str).str.upper())))

    lines = [
        "# Epoch Sweep Convergence Report",
        "",
        "## Executive Summary",
        f"- Surface: `{surface_names or 'unknown'}`",
        f"- Models: `{models or 'unknown'}`",
        f"- Epoch range: `{min(epochs) if epochs else 'N/A'} → {max(epochs) if epochs else 'N/A'}`",
        f"- Sweep points: {len(epochs)}",
    ]

    if not relative_df.empty:
        lines.extend(["", "### Relative Improvement over Controls"])
        for baseline in _CONTROL_BASELINES:
            baseline_df = relative_df[relative_df["baseline"] == baseline].copy()
            if baseline_df.empty:
                continue
            pr_mean = _safe_float(baseline_df["pr_auc_relative_pct"].mean())
            lines.append(f"- {_CONTROL_LABELS.get(baseline, baseline)}: {_format_pct(pr_mean)} PR-AUC")

    if not convergence_df.empty:
        lines.extend(["", "## Best Epoch by Behavioral State", ""])
        lines.append(
            convergence_df[["state_id", "best_epoch", "recommended_epoch", "plateau_detected"]].to_markdown(index=False)
        )

    if metadata_rows:
        lines.extend(["", "## Sweep Provenance", ""])
        lines.append(pd.DataFrame(metadata_rows).sort_values("epoch").to_markdown(index=False))

    lines.extend([
        "",
        "## Research Recommendation",
        "",
        _build_recommendation(convergence_df, epochs),
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _console_summary(
    *,
    epoch_summary_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> str:
    behavior_df = epoch_summary_df[epoch_summary_df["baseline"] == "behavioral_surface"].copy()
    surface_names = ", ".join(sorted(set(behavior_df.get("surface_id", pd.Series(dtype=str)).dropna().astype(str))))
    models = ", ".join(sorted(set(behavior_df.get("model", pd.Series(dtype=str)).dropna().astype(str).str.upper())))
    epochs = sorted(set(epoch_summary_df["epoch"].astype(int))) if not epoch_summary_df.empty else []

    lines = [
        "=" * 56,
        "Epoch Sweep Analysis",
        "=" * 56,
        "",
        f"Surface:\n{surface_names or 'unknown'}",
        "",
        f"Model:\n{models or 'unknown'}",
        "",
        f"Epoch range:\n{min(epochs) if epochs else 'N/A'} → {max(epochs) if epochs else 'N/A'}",
        "",
        "=" * 56,
        "Best epoch by Behavioral State",
        "=" * 56,
        "",
    ]
    if convergence_df.empty:
        lines.append("No convergence rows generated.")
    else:
        lines.append(convergence_df[["state_id", "best_epoch"]].to_string(index=False))
    return "\n".join(lines)


def analyze_epoch_sweep(
    *,
    sweep_manifest: Path,
    output_dir: Path | None = None,
    plateau_threshold: float = 0.005,
) -> dict[str, object]:
    sweep_manifest = sweep_manifest.resolve()
    sweep_df = _load_sweep_manifest(sweep_manifest)
    output_dir = (output_dir or (sweep_manifest.parent / "epoch_sweep_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epoch_df, metadata_rows = _aggregate_epoch_metrics(sweep_df, sweep_manifest.parent)
    if epoch_df.empty:
        raise ValueError("No walk-forward metrics could be loaded from the sweep manifest.")

    relative_df = _build_relative_improvement_summary(epoch_df)
    epoch_summary_df = _build_epoch_summary(epoch_df, relative_df)
    behavior_df = epoch_summary_df[epoch_summary_df["baseline"] == "behavioral_surface"].copy()
    convergence_df = _detect_convergence(behavior_df, plateau_threshold)

    epoch_summary_path = output_dir / "epoch_summary.csv"
    report_path = output_dir / "convergence_report.md"
    epoch_summary_df.to_csv(epoch_summary_path, index=False)
    _render_report(
        epoch_summary_df=epoch_summary_df,
        relative_df=relative_df,
        convergence_df=convergence_df,
        metadata_rows=metadata_rows,
        output_path=report_path,
    )

    _write_metric_plot(
        behavior_df=behavior_df,
        metric_column="pr_auc_mean",
        ylabel="PR-AUC",
        title="PR-AUC vs Epoch",
        output_path=plots_dir / "pr_auc_vs_epoch.png",
    )
    _write_relative_improvement_plot(
        relative_df=relative_df,
        output_path=plots_dir / "relative_improvement_vs_controls.png",
    )
    _write_metric_plot(
        behavior_df=behavior_df,
        metric_column="calibration_ece_mean",
        ylabel="Calibration ECE",
        title="Calibration vs Epoch",
        output_path=plots_dir / "calibration_vs_epoch.png",
    )

    summary_text = _console_summary(
        epoch_summary_df=epoch_summary_df,
        convergence_df=convergence_df,
    )
    print(summary_text)
    return {
        "epoch_summary": str(epoch_summary_path),
        "convergence_report": str(report_path),
        "plots_dir": str(plots_dir),
        "states": int(convergence_df["state_id"].nunique()) if not convergence_df.empty else 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analyze_epoch_sweep(
        sweep_manifest=args.sweep_manifest,
        output_dir=args.output_dir,
        plateau_threshold=args.plateau_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
