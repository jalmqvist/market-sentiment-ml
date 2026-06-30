"""Behavioral Surface inspection utility.

Performs structural sanity checks on a generated Behavioral Surface before
downstream validation.  Intended to catch implementation errors — such as
broken episode construction or maturity tracking — before Criterion 1 is
executed.

Usage::

    python -m bsve.validation.inspect_surface \\
        --surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \\
        --calibration bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \\
        --output-dir bsve.test/inspection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Core inspection logic
# ---------------------------------------------------------------------------


def inspect_surface(
    surface: pd.DataFrame,
    *,
    calibration_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a structural inspection report for a Behavioral Surface DataFrame.

    Parameters
    ----------
    surface:
        DataFrame produced by :func:`bsve.state_machine.engine.generate_behavioral_surface`.
        Must contain at minimum the columns ``pair``, ``state``, ``episode_id``,
        and ``maturity_bars``.
    calibration_artifact:
        Optional calibration artifact dict.  When provided its thresholds are
        included in the report and used for the survival-count checks.

    Returns
    -------
    dict
        Structured inspection report suitable for JSON serialisation.
    """
    required = {"pair", "state", "episode_id", "maturity_bars"}
    missing = required - set(surface.columns)
    if missing:
        raise ValueError(f"surface is missing required columns: {sorted(missing)}")

    n_obs = int(len(surface))
    n_episodes = int(surface["episode_id"].nunique())

    # --- pair frequencies ---
    pair_counts = {str(k): int(v) for k, v in surface["pair"].value_counts().sort_index().items()}

    # --- state frequencies ---
    state_counts = {str(k): int(v) for k, v in surface["state"].value_counts().sort_index().items()}

    # --- episode length distribution ---
    ep_lengths = surface.groupby("episode_id").size()
    ep_length_stats = {
        "min": int(ep_lengths.min()),
        "p25": float(ep_lengths.quantile(0.25)),
        "median": float(ep_lengths.median()),
        "mean": float(ep_lengths.mean()),
        "p75": float(ep_lengths.quantile(0.75)),
        "max": int(ep_lengths.max()),
    }

    # --- longest episodes ---
    top_episodes = (
        ep_lengths.sort_values(ascending=False)
        .head(10)
        .rename("length")
        .reset_index()
        .to_dict(orient="records")
    )

    # --- maturity statistics ---
    max_maturity = int(surface["maturity_bars"].max())

    # --- episode maturity statistics ---
    ep_max_maturity = (
        surface.groupby("episode_id")["maturity_bars"]
        .max()
    )

    # --- survival counts ---
    survival_counts: dict[str, int] = {}
    for threshold in [8, 16, 24, 32, 48]:
        survival_counts[f">= {threshold} bars"] = int(
            (ep_max_maturity >= threshold).sum()
        )

    # --- calibration thresholds ---
    thresholds_section: dict[str, Any] = {}
    if calibration_artifact is not None:
        raw_thresholds = calibration_artifact.get("thresholds", {})
        thresholds_section = {
            "extreme_threshold_net_pct": raw_thresholds.get("extreme_threshold_net_pct"),
            "young_boundary_bars": raw_thresholds.get("young_boundary_bars"),
            "mature_boundary_bars": raw_thresholds.get("mature_boundary_bars"),
        }

    # --- warnings ---
    warnings: list[str] = []
    consensus_states = {"JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE"}
    has_maturing = any(
        k in state_counts and state_counts[k] > 0
        for k in ("JPY_CONSENSUS_MATURING",)
    )
    has_mature = any(
        k in state_counts and state_counts[k] > 0
        for k in ("JPY_CONSENSUS_MATURE",)
    )
    has_any_consensus = any(
        state_counts.get(s, 0) > 0 for s in consensus_states
    )

    if not has_any_consensus:
        warnings.append("No consensus-state observations found (YOUNG, MATURING, MATURE all absent).")
    if has_any_consensus and not has_maturing:
        warnings.append("No MATURING observations — episode durations may be too short.")
    if has_any_consensus and not has_mature:
        warnings.append("No MATURE observations — episode durations may be too short.")
    if max_maturity == 0:
        warnings.append(
            "Maximum maturity is zero — consensus_active is never True. "
            "Check crowd_side column encoding and extreme threshold."
        )
    if n_episodes > 0 and ep_length_stats["mean"] < 2.0:
        warnings.append(
            f"Average episode length is very short ({ep_length_stats['mean']:.2f} bars). "
            "Episode construction may be broken."
        )
    if n_obs > 0 and n_episodes / n_obs > 0.9:
        warnings.append(
            f"Episode count ({n_episodes}) is close to observation count ({n_obs}). "
            "Nearly every observation is starting a new episode."
        )

    return {
        "total_observations": n_obs,
        "total_episodes": n_episodes,
        "episode_lengths": ep_lengths.tolist(),
        "max_maturity_bars": max_maturity,
        "episode_peak_maturity": ep_max_maturity.tolist(),
        "pair_counts": pair_counts,
        "state_counts": state_counts,
        "episode_length_stats": ep_length_stats,
        "top_episodes_by_length": top_episodes,
        "survival_counts": survival_counts,
        "calibration_thresholds": thresholds_section,
        "warnings": warnings,
    }

# ---------------------------------------------------------------------------
# Public summary API
# ---------------------------------------------------------------------------

def summarize_surface(
    surface: pd.DataFrame,
    *,
    calibration_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return the canonical statistical summary for a Behavioral Surface.

    This is a thin public wrapper around :func:`inspect_surface` intended for
    reuse by downstream validation utilities (for example calibration-drift
    analysis) without invoking the CLI.
    """
    return inspect_surface(
        surface,
        calibration_artifact=calibration_artifact,
    )

# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    sep = "-" * 62
    print()
    print("BSVE Behavioral Surface Inspection")
    print(sep)
    print(f"Total observations : {report['total_observations']:,}")
    print(f"Total episodes     : {report['total_episodes']:,}")
    print(f"Max maturity (bars): {report['max_maturity_bars']}")

    print()
    print("Pair frequencies")
    print(sep)
    for pair, count in sorted(report["pair_counts"].items()):
        print(f"  {pair:<16} {count:>6,}")

    print()
    print("State frequencies")
    print(sep)
    for state, count in sorted(report["state_counts"].items()):
        print(f"  {state:<35} {count:>6,}")

    print()
    print("Episode length distribution")
    print(sep)
    ls = report["episode_length_stats"]
    print(f"  min    : {ls['min']}")
    print(f"  P25    : {ls['p25']:.1f}")
    print(f"  median : {ls['median']:.1f}")
    print(f"  mean   : {ls['mean']:.2f}")
    print(f"  P75    : {ls['p75']:.1f}")
    print(f"  max    : {ls['max']}")

    print()
    print("Longest episodes (top 10)")
    print(sep)
    for ep in report["top_episodes_by_length"]:
        print(f"  {ep['episode_id']:<40}  {ep['length']:>4} bars")

    sc = report.get("survival_counts", {})
    if sc:
        print()
        print("Survival counts")
        print(sep)
        for label, count in sc.items():
            print(f"  {label:<16} {count:>6,}")

    thresholds = report.get("calibration_thresholds", {})
    if any(v is not None for v in thresholds.values()):
        print()
        print("Calibration thresholds")
        print(sep)
        if thresholds.get("extreme_threshold_net_pct") is not None:
            print(f"  extreme_threshold_net_pct : {thresholds['extreme_threshold_net_pct']}")
        if thresholds.get("young_boundary_bars") is not None:
            print(f"  young_boundary_bars       : {thresholds['young_boundary_bars']}")
        if thresholds.get("mature_boundary_bars") is not None:
            print(f"  mature_boundary_bars      : {thresholds['mature_boundary_bars']}")

    warnings = report.get("warnings", [])
    if warnings:
        print()
        print("⚠  Warnings")
        print(sep)
        for w in warnings:
            print(f"  • {w}")
    else:
        print()
        print("✓  No structural warnings detected.")

    print()


# ---------------------------------------------------------------------------
# Optional plot generation (matplotlib is an optional dependency)
# ---------------------------------------------------------------------------


def _try_generate_plots(
    surface: pd.DataFrame,
    output_dir: Path,
    report: dict[str, Any],
) -> list[Path]:
    """Generate inspection plots if matplotlib is available.

    Failures are silently ignored so that the CLI remains useful in
    environments without matplotlib.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return []

    generated: list[Path] = []

    # 01 — episode length distribution
    try:
        ep_lengths = surface.groupby("episode_id").size()
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins = min(50, max(10, len(ep_lengths) // 10))
        ax.hist(ep_lengths, bins=n_bins, edgecolor="white")
        ax.set_xlabel("Episode length (bars)")
        ax.set_ylabel("Count")
        ax.set_title("Episode Length Distribution")
        p = output_dir / "01_episode_length_distribution.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
    except Exception:  # noqa: BLE001
        pass

    # 02 — state frequencies
    try:
        state_counts = report["state_counts"]
        labels = list(state_counts.keys())
        values = list(state_counts.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, values, edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title("State Frequencies")
        p = output_dir / "02_state_frequencies.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        generated.append(p)
    except Exception:  # noqa: BLE001
        pass

    # 03 — maturity distribution
    if "maturity_bars" in surface.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            surface["maturity_bars"].hist(ax=ax, bins=min(50, int(surface["maturity_bars"].max()) + 1))
            ax.set_xlabel("Maturity (bars)")
            ax.set_ylabel("Count")
            ax.set_title("Maturity Distribution")
            p = output_dir / "03_maturity_distribution.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            generated.append(p)
        except Exception:  # noqa: BLE001
            pass

    return generated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BSVE Behavioral Surface Inspector — structural sanity checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--surface",
        required=True,
        help="Path to the behavioral surface parquet file.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional path to the calibration artifact JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write inspection plots and JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    surface = pd.read_parquet(args.surface)

    calibration_artifact: dict[str, Any] | None = None
    if args.calibration:
        calibration_artifact = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    report = summarize_surface(
        surface,
        calibration_artifact=calibration_artifact,
    )
    _print_report(report)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "inspection_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"[BSVE] Inspection report written: {report_path}")

        plots = _try_generate_plots(surface, output_dir, report)
        for p in plots:
            print(f"[BSVE] Plot written: {p}")


if __name__ == "__main__":
    main()
