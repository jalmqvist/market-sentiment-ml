"""Compare two or more completed behavioral experiment directories.

Usage::

    python analysis/behavioral/compare_experiments.py <exp_dir_1> <exp_dir_2> [...]

The comparison operates entirely on existing experiment outputs — no retraining
or prediction regeneration is required.

The tool tolerates experiments with:
- different Behavioral Surfaces
- different state ontologies
- different model families

Output is written to stdout as a markdown report.  Use ``--output`` to
redirect to a file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_experiment_manifest(exp_dir: Path) -> dict[str, Any]:
    """Load the top-level experiment_manifest.json from *exp_dir*."""
    manifest_path = exp_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"experiment_manifest.json not found in {exp_dir!s}. "
            "Is this a valid experiment directory?"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load metrics.csv from *exp_dir*."""
    path = exp_dir / "metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_summary(exp_dir: Path) -> pd.DataFrame:
    """Load summary.csv from *exp_dir*."""
    path = exp_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _extract_coverage(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with metric_group == 'coverage'."""
    if metrics_df.empty or "metric_group" not in metrics_df.columns:
        return pd.DataFrame()
    return metrics_df[metrics_df["metric_group"] == "coverage"].copy()


def _extract_prediction_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with metric_group == 'prediction_metrics'."""
    if metrics_df.empty or "metric_group" not in metrics_df.columns:
        return pd.DataFrame()
    return metrics_df[metrics_df["metric_group"] == "prediction_metrics"].copy()


def _extract_compare(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with metric_group == 'mlp_lstm_compare'."""
    if metrics_df.empty or "metric_group" not in metrics_df.columns:
        return pd.DataFrame()
    return metrics_df[metrics_df["metric_group"] == "mlp_lstm_compare"].copy()


def _state_set(manifest: dict) -> set[str]:
    """Return the set of 'surface_id:state_id' strings from the manifest."""
    states = manifest.get("discovered_states", [])
    return {f"{s.get('surface_id', '?')}:{s.get('state_id', '?')}" for s in states}


def _provenance_row(exp_id: str, manifest: dict) -> dict[str, Any]:
    """Extract a flat provenance dict from a manifest."""
    dataset = manifest.get("dataset", {})
    cli_parsed = manifest.get("cli", {}).get("parsed", {})
    return {
        "experiment_id": exp_id,
        "dataset_version": dataset.get("version"),
        "dataset_variant": dataset.get("variant"),
        "git_commit": manifest.get("git_commit"),
        "created_at": manifest.get("created_at"),
        "completed_at": manifest.get("completed_at"),
        "models": ", ".join(manifest.get("models_executed", [])),
        "feature_set": cli_parsed.get("feature_set"),
        "target_horizon": cli_parsed.get("target_horizon"),
        "n_states": len(manifest.get("discovered_states", [])),
        "success": manifest.get("success"),
    }


# ---------------------------------------------------------------------------
# Coverage comparison
# ---------------------------------------------------------------------------

def compare_coverage(
    experiments: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build a side-by-side coverage comparison table.

    Parameters
    ----------
    experiments:
        List of ``(experiment_label, coverage_df)`` pairs.  The
        *coverage_df* should be the rows extracted from metrics.csv with
        ``metric_group == 'coverage'``.

    Returns
    -------
    pd.DataFrame
        One row per coverage scope, one column pair per experiment.
    """
    if not experiments:
        return pd.DataFrame()

    # Collect all unique scopes across experiments
    all_scopes: list[str] = []
    seen: set[str] = set()
    for _, cov_df in experiments:
        if cov_df.empty or "scope" not in cov_df.columns:
            continue
        for scope in cov_df["scope"].tolist():
            if scope not in seen:
                all_scopes.append(scope)
                seen.add(scope)

    rows: list[dict[str, Any]] = []
    for scope in all_scopes:
        row: dict[str, Any] = {"scope": scope}
        for label, cov_df in experiments:
            if cov_df.empty or "scope" not in cov_df.columns:
                row[f"{label}:row_count"] = None
                row[f"{label}:coverage_fraction"] = None
                continue
            match = cov_df[cov_df["scope"] == scope]
            if match.empty:
                row[f"{label}:row_count"] = None
                row[f"{label}:coverage_fraction"] = None
            else:
                row[f"{label}:row_count"] = int(match.iloc[0].get("row_count", 0))
                frac = match.iloc[0].get("coverage_fraction")
                row[f"{label}:coverage_fraction"] = (
                    round(float(frac), 4) if frac is not None and not (isinstance(frac, float) and np.isnan(frac)) else None
                )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction distribution comparison
# ---------------------------------------------------------------------------

def compare_prediction_distributions(
    experiments: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compare prediction metric distributions across experiments.

    Parameters
    ----------
    experiments:
        List of ``(label, prediction_metrics_df)`` pairs.
    """
    rows: list[dict[str, Any]] = []
    METRICS = [
        "prediction_entropy_mean",
        "prediction_confidence_mean",
        "effective_prediction_coverage",
        "sharpness",
        "pair_balance",
        "pred_prob_mean",
        "signal_strength_mean",
    ]
    for label, pm_df in experiments:
        if pm_df.empty:
            row: dict[str, Any] = {"experiment": label}
            for m in METRICS:
                row[m] = None
            rows.append(row)
            continue
        row = {"experiment": label, "n_artifacts": len(pm_df)}
        for m in METRICS:
            if m in pm_df.columns:
                vals = pd.to_numeric(pm_df[m], errors="coerce").dropna()
                row[m] = round(float(vals.mean()), 6) if len(vals) > 0 else None
            else:
                row[m] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement comparison
# ---------------------------------------------------------------------------

def compare_prediction_agreement(
    experiments: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compare MLP/LSTM agreement across experiments.

    Handles non-identical state sets by using a union of all state identifiers.
    """
    # Collect all unique (surface_id, state_id) pairs across experiments
    all_states: list[tuple[str, str]] = []
    seen_states: set[tuple[str, str]] = set()
    for _, cmp_df in experiments:
        if cmp_df.empty:
            continue
        for _, row in cmp_df.iterrows():
            key = (str(row.get("surface_id", "")), str(row.get("state_id", "")))
            if key not in seen_states:
                all_states.append(key)
                seen_states.add(key)

    if not all_states:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for surface_id, state_id in all_states:
        row: dict[str, Any] = {"surface_id": surface_id, "state_id": state_id}
        for label, cmp_df in experiments:
            if cmp_df.empty:
                row[f"{label}:agreement_rate"] = None
                row[f"{label}:overlap_pct_of_mlp"] = None
                row[f"{label}:pred_prob_correlation"] = None
                continue
            match = cmp_df[
                (cmp_df["surface_id"].astype(str) == surface_id)
                & (cmp_df["state_id"].astype(str) == state_id)
            ]
            if match.empty:
                row[f"{label}:agreement_rate"] = None
                row[f"{label}:overlap_pct_of_mlp"] = None
                row[f"{label}:pred_prob_correlation"] = None
            else:
                for col in ["agreement_rate", "overlap_pct_of_mlp", "pred_prob_correlation"]:
                    val = match.iloc[0].get(col)
                    row[f"{label}:{col}"] = (
                        round(float(val), 4)
                        if val is not None and not (isinstance(val, float) and np.isnan(val))
                        else None
                    )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Occupancy comparison
# ---------------------------------------------------------------------------

def compare_occupancy(
    experiments: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compare per-state occupancy (row counts and coverage fractions) across experiments."""
    # Extract state-level coverage rows
    all_states: list[str] = []
    seen_states: set[str] = set()
    for _, cov_df in experiments:
        if cov_df.empty or "scope" not in cov_df.columns:
            continue
        for scope in cov_df["scope"].tolist():
            if scope.startswith("state:") and scope not in seen_states:
                all_states.append(scope)
                seen_states.add(scope)

    if not all_states:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scope in all_states:
        row: dict[str, Any] = {"scope": scope}
        for label, cov_df in experiments:
            if cov_df.empty or "scope" not in cov_df.columns:
                row[f"{label}:row_count"] = None
                row[f"{label}:coverage_fraction"] = None
                continue
            match = cov_df[cov_df["scope"] == scope]
            if match.empty:
                row[f"{label}:row_count"] = None
                row[f"{label}:coverage_fraction"] = None
            else:
                row[f"{label}:row_count"] = int(match.iloc[0].get("row_count", 0))
                frac = match.iloc[0].get("coverage_fraction")
                row[f"{label}:coverage_fraction"] = (
                    round(float(frac), 4)
                    if frac is not None and not (isinstance(frac, float) and np.isnan(frac))
                    else None
                )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------

def compare_experiments(
    exp_dirs: list[Path],
) -> dict[str, Any]:
    """Compare two or more completed experiment directories.

    Parameters
    ----------
    exp_dirs:
        Paths to completed experiment directories, each containing
        ``experiment_manifest.json``, ``metrics.csv``, and ``summary.csv``.

    Returns
    -------
    dict containing:
        - ``manifests``: list of raw manifest dicts
        - ``provenance_df``: pandas DataFrame
        - ``coverage_df``: side-by-side coverage comparison
        - ``occupancy_df``: per-state occupancy comparison
        - ``distribution_df``: prediction distribution comparison
        - ``agreement_df``: MLP/LSTM agreement comparison
        - ``state_sets``: dict mapping experiment_id → set of state keys
        - ``shared_states``: states present in all experiments
        - ``unique_states``: states appearing in only a subset of experiments
    """
    if len(exp_dirs) < 1:
        raise ValueError("At least one experiment directory is required.")

    manifests: list[dict] = []
    labels: list[str] = []
    coverage_pairs: list[tuple[str, pd.DataFrame]] = []
    pm_pairs: list[tuple[str, pd.DataFrame]] = []
    cmp_pairs: list[tuple[str, pd.DataFrame]] = []

    for exp_dir in exp_dirs:
        manifest = _load_experiment_manifest(exp_dir)
        exp_id = manifest.get("experiment_id", exp_dir.name)
        manifests.append(manifest)
        labels.append(exp_id)

        metrics_df = _load_metrics(exp_dir)
        coverage_pairs.append((exp_id, _extract_coverage(metrics_df)))
        pm_pairs.append((exp_id, _extract_prediction_metrics(metrics_df)))
        cmp_pairs.append((exp_id, _extract_compare(metrics_df)))

    # Provenance table
    provenance_rows = [_provenance_row(lbl, mf) for lbl, mf in zip(labels, manifests)]
    provenance_df = pd.DataFrame(provenance_rows)

    # State-set analysis
    state_sets: dict[str, set[str]] = {lbl: _state_set(mf) for lbl, mf in zip(labels, manifests)}
    if state_sets:
        all_state_keys = set.union(*state_sets.values())
        shared_states = set.intersection(*state_sets.values()) if len(state_sets) > 1 else set(next(iter(state_sets.values())))
        unique_states = all_state_keys - shared_states
    else:
        all_state_keys = set()
        shared_states = set()
        unique_states = set()

    return {
        "manifests": manifests,
        "labels": labels,
        "provenance_df": provenance_df,
        "coverage_df": compare_coverage(coverage_pairs),
        "occupancy_df": compare_occupancy(coverage_pairs),
        "distribution_df": compare_prediction_distributions(pm_pairs),
        "agreement_df": compare_prediction_agreement(cmp_pairs),
        "state_sets": state_sets,
        "shared_states": sorted(shared_states),
        "unique_states": sorted(unique_states),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def render_comparison_report(result: dict[str, Any]) -> str:
    """Render the comparison result as a markdown string."""
    lines: list[str] = [
        "# Behavioral Experiment Comparison",
        "",
        f"Comparing {len(result['labels'])} experiment(s): "
        + ", ".join(f"`{lbl}`" for lbl in result["labels"]),
        "",
        "## Provenance",
        "",
        _df_to_md(result["provenance_df"]),
        "",
        "## State-Set Analysis",
        "",
    ]

    shared = result["shared_states"]
    unique = result["unique_states"]
    state_sets: dict[str, set[str]] = result["state_sets"]
    if len(result["labels"]) > 1:
        if shared:
            lines.append(f"**States present in all experiments ({len(shared)}):**")
            for s in shared:
                lines.append(f"- `{s}`")
        if unique:
            lines.append(f"\n**States present in only a subset of experiments ({len(unique)}):**")
            for s in unique:
                owners = [lbl for lbl, ss in state_sets.items() if s in ss]
                lines.append(f"- `{s}` (found in: {', '.join(owners)})")
    else:
        states_list = sorted(next(iter(state_sets.values()), set()))
        lines.append(f"**States ({len(states_list)}):**")
        for s in states_list:
            lines.append(f"- `{s}`")

    lines.extend([
        "",
        "## Coverage Comparison",
        "",
        _df_to_md(result["coverage_df"]),
        "",
        "## State Occupancy Comparison",
        "",
        _df_to_md(result["occupancy_df"]),
        "",
        "## Prediction Distribution Comparison",
        "",
        _df_to_md(result["distribution_df"]),
        "",
        "## MLP/LSTM Agreement Comparison",
        "",
        _df_to_md(result["agreement_df"]),
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more completed behavioral experiment directories.\n\n"
            "Operates entirely on existing experiment outputs — no retraining required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "exp_dirs",
        nargs="+",
        type=Path,
        metavar="EXP_DIR",
        help="Paths to completed experiment directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write report to FILE instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compare_experiments(args.exp_dirs)
    report = render_comparison_report(result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
