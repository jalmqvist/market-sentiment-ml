from __future__ import annotations

import argparse
import json
import re
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


def _format_decimal(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _build_state_color_map(state_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {
        state_id: cmap(idx % cmap.N)
        for idx, state_id in enumerate(sorted(set(state_ids)))
    }


def _sanitize_state_fragment(state_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(state_id)).strip("_") or "STATE"


def _format_epoch_range(start: int | None, end: int | None) -> str:
    if start is None or end is None or start == end:
        return "—"
    return f"{start}–{end}"


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

    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = (repo_root / path).resolve()

    if repo_candidate.exists():
        return repo_candidate

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
                    "model": str(row["model"]).lower(),
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


def _classify_convergence(
    *,
    epochs: list[int],
    pr_values: np.ndarray,
    threshold: float,
    stable_epochs: list[int],
    best_idx: int,
) -> tuple[str, str]:
    if len(pr_values) < 2:
        return "Unstable", "Insufficient epoch points to establish a reproducible trend."

    span = max(epochs) - min(epochs)
    spread = float(np.max(pr_values) - np.min(pr_values))
    deltas = np.diff(pr_values)
    sign_changes = int(np.sum(np.sign(deltas[1:]) != np.sign(deltas[:-1]))) if len(deltas) >= 2 else 0
    total_gain = float(pr_values[-1] - pr_values[0])

    if spread <= threshold:
        return (
            "Epoch insensitive",
            f"PR-AUC spread ({spread:.4f}) stays within threshold ({threshold:.4f}) across tested epochs.",
        )

    if best_idx == len(epochs) - 1 and total_gain > threshold:
        return (
            "Still improving",
            f"Best performance occurs at the highest tested epoch with cumulative gain {total_gain:.4f}.",
        )

    early_cutoff = max(1, int(round((len(epochs) - 1) * 0.25)))
    if best_idx <= early_cutoff and abs(float(pr_values[-1]) - float(np.max(pr_values))) <= threshold:
        return (
            "Rapid convergence",
            f"Peak appears early (epoch {epochs[best_idx]}) and remains within threshold of the peak afterward.",
        )

    if len(stable_epochs) >= 2:
        stable_width = stable_epochs[-1] - stable_epochs[0]
        if stable_width >= max(epochs[1] - epochs[0], int(span * 0.30)):
            return (
                "Gradual convergence",
                f"A broad near-peak region ({stable_epochs[0]}–{stable_epochs[-1]}) indicates gradual stabilization.",
            )

    if sign_changes >= max(2, len(deltas) // 2):
        return (
            "Unstable",
            f"Frequent direction changes in epoch-to-epoch deltas ({sign_changes}) indicate unstable learning dynamics.",
        )

    return (
        "Unstable",
        "Performance does not sustain a broad near-peak range and does not show consistent convergence.",
    )


def _detect_convergence(behavior_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if behavior_df.empty:
        return pd.DataFrame()

    state_epoch_df = (
        behavior_df.groupby(["surface_id", "state_id", "model", "epoch"], dropna=False)[["pr_auc_mean", "calibration_ece_mean"]]
        .mean()
        .reset_index()
        .sort_values(["surface_id", "state_id", "model", "epoch"])
    )
    for (surface_id, state_id, model), group in state_epoch_df.groupby(["surface_id", "state_id", "model"], dropna=False):
        ordered = group.sort_values("epoch")
        epochs = ordered["epoch"].astype(int).tolist()
        pr_values = ordered["pr_auc_mean"].astype(float).to_numpy()
        cal_values = ordered["calibration_ece_mean"].astype(float).to_numpy() if "calibration_ece_mean" in ordered.columns else np.array([])

        valid_mask = np.isfinite(pr_values)
        if int(valid_mask.sum()) < 1:
            continue
        epochs = [epoch for epoch, is_valid in zip(epochs, valid_mask) if is_valid]
        pr_values = pr_values[valid_mask]
        if len(cal_values) == len(valid_mask):
            cal_values = cal_values[valid_mask]

        best_idx = int(np.argmax(pr_values))
        best_epoch = int(epochs[best_idx])
        best_pr = float(pr_values[best_idx])
        stable_epochs = [int(epoch) for epoch, pr in zip(epochs, pr_values) if abs(best_pr - float(pr)) <= threshold]
        stable_start = stable_epochs[0] if len(stable_epochs) >= 2 else None
        stable_end = stable_epochs[-1] if len(stable_epochs) >= 2 else None
        plateau_width = int(stable_end - stable_start) if stable_start is not None and stable_end is not None else 0

        convergence_class, rationale = _classify_convergence(
            epochs=epochs,
            pr_values=pr_values,
            threshold=threshold,
            stable_epochs=stable_epochs,
            best_idx=best_idx,
        )

        calibration_trend = "N/A"
        calibration_delta = None
        if len(cal_values) >= 2 and np.all(np.isfinite(cal_values)):
            calibration_delta = float(cal_values[-1] - cal_values[0])
            if calibration_delta < -threshold:
                calibration_trend = "Improving"
            elif calibration_delta > threshold:
                calibration_trend = "Degrading"
            else:
                calibration_trend = "Stable"

        rows.append(
            {
                "surface_id": str(surface_id),
                "state_id": str(state_id),
                "model": str(model).lower(),
                "best_epoch": best_epoch,
                "recommended_epoch": stable_start if stable_start is not None else best_epoch,
                "best_pr_auc": best_pr,
                "peak_pr_auc": best_pr,
                "stable_epoch_start": stable_start,
                "stable_epoch_end": stable_end,
                "stable_epoch_range": _format_epoch_range(stable_start, stable_end),
                "plateau_width": plateau_width,
                "convergence_class": convergence_class,
                "classification_rationale": rationale,
                "calibration_trend": calibration_trend,
                "calibration_delta": calibration_delta,
            }
        )
    return pd.DataFrame(rows).sort_values(["surface_id", "state_id", "model"]).reset_index(drop=True)


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
            linewidth=1.8,
            color=state_colors.get(str(state_id)),
            linestyle=_MODEL_LINESTYLES.get(str(model).lower(), "-."),
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
        if model in set(plot_df["model"].astype(str).str.lower())
    ]
    state_legend = plt.legend(handles=state_handles, title="Behavioral State", loc="upper left", fontsize=8)
    plt.gca().add_artist(state_legend)
    if model_handles:
        plt.legend(handles=model_handles, title="Architecture", loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _write_relative_improvement_plot(relative_df: pd.DataFrame, output_path: Path) -> bool:
    if relative_df.empty or "pr_auc_relative_pct" not in relative_df.columns:
        return False
    plot_df = (
        relative_df.groupby(["epoch", "state_id", "model"], dropna=False)["pr_auc_relative_pct"]
        .mean()
        .reset_index()
    )
    plot_df = plot_df.dropna(subset=["pr_auc_relative_pct"])
    if plot_df.empty:
        return False

    state_colors = _build_state_color_map(plot_df["state_id"].astype(str).tolist())
    plt.figure(figsize=(10, 5))
    for (state_id, model), group in plot_df.groupby(["state_id", "model"], dropna=False):
        ordered = group.sort_values("epoch")
        plt.plot(
            ordered["epoch"].astype(int),
            ordered["pr_auc_relative_pct"].astype(float),
            marker="o",
            linewidth=1.8,
            color=state_colors.get(str(state_id)),
            linestyle=_MODEL_LINESTYLES.get(str(model).lower(), "-."),
        )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Relative PR-AUC improvement (%)")
    plt.title("Relative Improvement over Controls")
    plt.grid(alpha=0.3)
    state_handles = [
        Line2D([0], [0], color=color, lw=2, label=state_id)
        for state_id, color in state_colors.items()
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=style, label=model.upper())
        for model, style in _MODEL_LINESTYLES.items()
        if model in set(plot_df["model"].astype(str).str.lower())
    ]
    state_legend = plt.legend(handles=state_handles, title="Behavioral State", loc="upper left", fontsize=8)
    plt.gca().add_artist(state_legend)
    if model_handles:
        plt.legend(handles=model_handles, title="Architecture", loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _write_state_relative_improvement_plots(relative_df: pd.DataFrame, plots_dir: Path) -> list[str]:
    if relative_df.empty:
        return []

    produced: list[str] = []
    for state_id, state_df in relative_df.groupby("state_id", dropna=False):
        state_plot = state_df.dropna(subset=["pr_auc_relative_pct"]).copy()
        if state_plot.empty:
            continue

        plt.figure(figsize=(10, 5))
        for (baseline, model), group in state_plot.groupby(["baseline", "model"], dropna=False):
            ordered = group.sort_values("epoch")
            plt.plot(
                ordered["epoch"].astype(int),
                ordered["pr_auc_relative_pct"].astype(float),
                marker="o",
                linewidth=1.8,
                linestyle=_MODEL_LINESTYLES.get(str(model).lower(), "-."),
                label=f"{_CONTROL_LABELS.get(str(baseline), str(baseline))} ({str(model).upper()})",
            )
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        plt.xlabel("Epoch")
        plt.ylabel("Relative PR-AUC improvement (%)")
        plt.title(f"Relative Improvement vs Controls — {state_id}")
        plt.grid(alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        filename = f"relative_improvement_vs_controls__{_sanitize_state_fragment(str(state_id))}.png"
        path = plots_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        produced.append(filename)
    return produced


def _write_peak_pr_auc_plot(behavior_df: pd.DataFrame, output_path: Path) -> bool:
    if behavior_df.empty:
        return False

    peak_df = (
        behavior_df.groupby(["state_id", "model"], dropna=False)["pr_auc_mean"]
        .max()
        .reset_index()
        .dropna(subset=["pr_auc_mean"])
    )
    if peak_df.empty:
        return False

    pivot = peak_df.pivot_table(index="state_id", columns="model", values="pr_auc_mean", aggfunc="max")
    if pivot.empty:
        return False
    pivot = pivot.sort_index()

    models = [m for m in ["mlp", "lstm"] if m in pivot.columns]
    if not models:
        models = list(pivot.columns)

    x = np.arange(len(pivot.index))
    width = 0.36 if len(models) > 1 else 0.55
    plt.figure(figsize=(10, 5))
    for idx, model in enumerate(models):
        shift = (idx - (len(models) - 1) / 2.0) * width
        plt.bar(x + shift, pivot[model].astype(float).to_numpy(), width=width, label=str(model).upper())

    plt.xticks(x, [str(state) for state in pivot.index], rotation=25, ha="right")
    plt.ylabel("Peak PR-AUC")
    plt.title("Peak PR-AUC by State")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _build_recommendation(convergence_df: pd.DataFrame, epochs: list[int]) -> str:
    if convergence_df.empty:
        return "No convergence recommendation could be derived from the available sweep outputs."

    class_counts = (
        convergence_df.groupby("convergence_class", dropna=False)["state_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    lines = [
        "Convergence classifications are derived from PR-AUC trajectories using the configured plateau threshold.",
        "",
        "Class distribution:",
    ]
    lines.extend([f"- {klass}: {count} state(s)" for klass, count in class_counts.items()])

    improving = convergence_df[convergence_df["convergence_class"] == "Still improving"]
    if not improving.empty and epochs:
        next_epoch = int(max(epochs) * 2)
        states = ", ".join(sorted(set(improving["state_id"].astype(str))))
        lines.extend([
            "",
            f"Continue the sweep for still-improving states ({states}) to approximately {next_epoch} epochs.",
        ])

    return "\n".join(lines)


def _build_cross_architecture_agreement(convergence_df: pd.DataFrame) -> pd.DataFrame:
    if convergence_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = convergence_df.groupby(["surface_id", "state_id"], dropna=False)
    for (surface_id, state_id), group in grouped:
        mlp = group[group["model"] == "mlp"]
        lstm = group[group["model"] == "lstm"]
        if mlp.empty or lstm.empty:
            rows.append(
                {
                    "surface_id": str(surface_id),
                    "state_id": str(state_id),
                    "class_agreement": "N/A",
                    "best_epoch_delta": np.nan,
                    "stable_range_overlap": "N/A",
                    "peak_pr_auc_delta": np.nan,
                    "calibration_agreement": "N/A",
                    "agreement_summary": "Only one architecture available; agreement not assessable.",
                }
            )
            continue

        mlp_row = mlp.iloc[0]
        lstm_row = lstm.iloc[0]
        class_agree = str(mlp_row["convergence_class"]) == str(lstm_row["convergence_class"])
        best_epoch_delta = abs(int(mlp_row["best_epoch"]) - int(lstm_row["best_epoch"]))
        peak_delta = abs(float(mlp_row["peak_pr_auc"]) - float(lstm_row["peak_pr_auc"]))
        cal_agree = str(mlp_row["calibration_trend"]) == str(lstm_row["calibration_trend"])

        overlap = False
        if pd.notna(mlp_row["stable_epoch_start"]) and pd.notna(lstm_row["stable_epoch_start"]):
            overlap = (
                int(mlp_row["stable_epoch_start"]) <= int(lstm_row["stable_epoch_end"])
                and int(lstm_row["stable_epoch_start"]) <= int(mlp_row["stable_epoch_end"])
            )

        summary_parts = []
        summary_parts.append("Convergence class aligned" if class_agree else "Convergence class differs")
        summary_parts.append(f"best-epoch gap={best_epoch_delta}")
        summary_parts.append(f"peak PR-AUC gap={peak_delta:.4f}")
        summary_parts.append("calibration trend aligned" if cal_agree else "calibration trend differs")

        rows.append(
            {
                "surface_id": str(surface_id),
                "state_id": str(state_id),
                "class_agreement": "Yes" if class_agree else "No",
                "best_epoch_delta": int(best_epoch_delta),
                "stable_range_overlap": "Yes" if overlap else "No",
                "peak_pr_auc_delta": peak_delta,
                "calibration_agreement": "Yes" if cal_agree else "No",
                "agreement_summary": "; ".join(summary_parts) + ".",
            }
        )
    return pd.DataFrame(rows).sort_values(["surface_id", "state_id"]).reset_index(drop=True)


def _state_recommendation(convergence_df: pd.DataFrame, state_id: str) -> tuple[str, str]:
    state_rows = convergence_df[convergence_df["state_id"] == state_id].copy()
    if state_rows.empty:
        return "N/A", "Collect additional sweep evidence."

    rec_epoch = int(state_rows["recommended_epoch"].max())
    classes = sorted(set(state_rows["convergence_class"].astype(str)))

    if "Still improving" in classes:
        future = "Extend the epoch range for this state and monitor whether gains persist." 
    elif "Unstable" in classes:
        future = "Increase fold coverage and inspect optimization variance before finalizing training epochs."
    elif "Epoch insensitive" in classes:
        future = "Prefer lower training cost; evaluate robustness under alternate seeds and controls."
    else:
        future = "Validate this epoch choice with additional out-of-sample periods and adjacent surfaces."
    return str(rec_epoch), future


def _render_report(
    *,
    epoch_summary_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
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
        lines.extend(["", "## State-level Convergence Summary", ""])
        display = convergence_df[[
            "state_id",
            "model",
            "best_epoch",
            "stable_epoch_range",
            "peak_pr_auc",
            "plateau_width",
            "convergence_class",
        ]].copy()
        display = display.rename(
            columns={
                "state_id": "State",
                "model": "Architecture",
                "best_epoch": "Best epoch",
                "stable_epoch_range": "Stable epoch range",
                "peak_pr_auc": "Peak PR-AUC",
                "plateau_width": "Plateau width",
                "convergence_class": "Convergence class",
            }
        )
        lines.append(display.to_markdown(index=False))

        lines.extend(["", "### Classification rationale", ""])
        rationale_df = convergence_df[["state_id", "model", "convergence_class", "classification_rationale"]].copy()
        rationale_df.columns = ["State", "Architecture", "Convergence class", "Why this label"]
        lines.append(rationale_df.to_markdown(index=False))

    lines.extend(["", "## Cross-architecture agreement", ""])
    if agreement_df.empty:
        lines.append("No paired MLP/LSTM state rows were available for agreement analysis.")
    else:
        display = agreement_df[[
            "state_id",
            "class_agreement",
            "best_epoch_delta",
            "stable_range_overlap",
            "peak_pr_auc_delta",
            "calibration_agreement",
            "agreement_summary",
        ]].copy()
        display.columns = [
            "State",
            "Class agreement",
            "Best epoch Δ",
            "Stable range overlap",
            "Peak PR-AUC Δ",
            "Calibration agreement",
            "Evidence",
        ]
        lines.append(display.to_markdown(index=False))

    unique_states = sorted(set(convergence_df.get("state_id", pd.Series(dtype=str)).dropna().astype(str)))
    if unique_states:
        lines.extend(["", "## Scientific interpretation", ""])
        for state_id in unique_states:
            state_rows = convergence_df[convergence_df["state_id"] == state_id].copy()
            if state_rows.empty:
                continue
            recommended_epoch, future_work = _state_recommendation(convergence_df, state_id)
            confidence = "Moderate"
            state_agreement = agreement_df[agreement_df["state_id"] == state_id]
            if not state_agreement.empty:
                row = state_agreement.iloc[0]
                if row.get("class_agreement") == "Yes" and row.get("calibration_agreement") == "Yes":
                    confidence = "High"
                elif row.get("class_agreement") == "No":
                    confidence = "Low"

            peak = _safe_float(state_rows["peak_pr_auc"].max())
            classes = ", ".join(sorted(set(state_rows["convergence_class"].astype(str))))
            stable_ranges = ", ".join(sorted(set(state_rows["stable_epoch_range"].astype(str))))
            cal_behavior = ", ".join(sorted(set(state_rows["calibration_trend"].astype(str))))

            lines.extend(
                [
                    f"### {state_id}",
                    "",
                    "#### Scientific interpretation",
                    f"{state_id} exhibits {classes.lower()} behavior with stable range(s): {stable_ranges}.",
                    "",
                    "#### Evidence",
                    f"Peak PR-AUC: {_format_decimal(peak)}; calibration behavior: {cal_behavior}.",
                    "",
                    "#### Confidence",
                    f"{confidence} confidence, driven by cross-architecture agreement and stability diagnostics.",
                    "",
                    "#### Recommended epoch",
                    f"{recommended_epoch}",
                    "",
                    "#### Future work",
                    future_work,
                    "",
                ]
            )

    if metadata_rows:
        lines.extend(["## Sweep Provenance", ""])
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
        lines.append(
            convergence_df[["state_id", "model", "best_epoch", "stable_epoch_range", "convergence_class"]].to_string(index=False)
        )
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
    agreement_df = _build_cross_architecture_agreement(convergence_df)

    epoch_summary_path = output_dir / "epoch_summary.csv"
    report_path = output_dir / "convergence_report.md"
    epoch_summary_df.to_csv(epoch_summary_path, index=False)
    _render_report(
        epoch_summary_df=epoch_summary_df,
        relative_df=relative_df,
        convergence_df=convergence_df,
        agreement_df=agreement_df,
        metadata_rows=metadata_rows,
        output_path=report_path,
    )

    produced_plots: list[str] = []
    if _write_metric_plot(
        behavior_df=behavior_df,
        metric_column="pr_auc_mean",
        ylabel="PR-AUC",
        title="PR-AUC vs Epoch",
        output_path=plots_dir / "pr_auc_vs_epoch.png",
    ):
        produced_plots.append("pr_auc_vs_epoch.png")

    if _write_relative_improvement_plot(
        relative_df=relative_df,
        output_path=plots_dir / "relative_improvement_vs_controls.png",
    ):
        produced_plots.append("relative_improvement_vs_controls.png")

    if _write_metric_plot(
        behavior_df=behavior_df,
        metric_column="calibration_ece_mean",
        ylabel="Calibration ECE",
        title="Calibration vs Epoch",
        output_path=plots_dir / "calibration_vs_epoch.png",
    ):
        produced_plots.append("calibration_vs_epoch.png")

    produced_plots.extend(_write_state_relative_improvement_plots(relative_df, plots_dir))
    if _write_peak_pr_auc_plot(behavior_df, plots_dir / "peak_pr_auc_by_state.png"):
        produced_plots.append("peak_pr_auc_by_state.png")

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
        "plot_files": sorted(set(produced_plots)),
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
