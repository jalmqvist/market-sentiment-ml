"""Independent behavioral outcome analysis for assigned BSVE states."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency

NON_EXTREME_STATE = "JPY_NON_EXTREME"
MATURITY_STATES = [
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
]
STATE_ORDER = [NON_EXTREME_STATE, *MATURITY_STATES]
HORIZONS = [4, 8, 12]
PROGRESSION_TARGETS = [
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING"),
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"),
    ("JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE"),
]
PERSISTENCE_TARGETS = {
    "JPY_CONSENSUS_YOUNG": {
        "JPY_CONSENSUS_YOUNG",
        "JPY_CONSENSUS_MATURING",
        "JPY_CONSENSUS_MATURE",
    },
    "JPY_CONSENSUS_MATURING": {
        "JPY_CONSENSUS_MATURING",
        "JPY_CONSENSUS_MATURE",
    },
    "JPY_CONSENSUS_MATURE": {
        "JPY_CONSENSUS_MATURE",
    },
}
ALPHA = 0.05
MIN_BEHAVIORAL_TEST_SAMPLES = 20


def load_state_surface(path: str | Path) -> pd.DataFrame:
    """Load a BSVE state-surface artifact."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"state-surface artifact not found: {artifact_path}")

    df = pd.read_parquet(artifact_path)
    required = {"pair", "entry_time", "state_id", "maturity_bars", "transition_event"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"artifact missing required columns: {missing}")

    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    if out["entry_time"].isna().any():
        raise ValueError("artifact contains invalid entry_time values")
    return out


def _prepare_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the artifact and add entry/next-state helper columns."""
    ordered = df.sort_values(["pair", "entry_time"]).reset_index(drop=True).copy()
    prev_state = ordered.groupby("pair")["state_id"].shift()
    ordered["_is_state_entry"] = prev_state.isna() | ordered["state_id"].ne(prev_state)
    ordered["_next_state"] = ordered.groupby("pair")["state_id"].shift(-1)
    return ordered


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    """Compute a Wilson-score interval using z as the confidence z-score."""
    if total <= 0:
        return {"lower": None, "upper": None}

    p = successes / total
    z2 = z**2
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
        / denom
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def compute_transition_matrix(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compute next-state transition counts and probabilities."""
    ordered = _prepare_surface(df)
    usable = ordered[ordered["_next_state"].notna()]
    result: dict[str, dict[str, dict[str, float | int]]] = {}

    for state_from in STATE_ORDER:
        from_df = usable[usable["state_id"] == state_from]
        total = int(len(from_df))
        row: dict[str, dict[str, float | int]] = {}
        for state_to in STATE_ORDER:
            count = int((from_df["_next_state"] == state_to).sum())
            row[state_to] = {
                "count": count,
                "probability": (count / total) if total else 0.0,
            }
        result[state_from] = row
    return result


def _iter_entry_contexts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract state-entry rows together with their future state trajectories."""
    ordered = _prepare_surface(df)
    entries: list[dict[str, Any]] = []
    for _, group in ordered.groupby("pair", sort=False):
        states = group["state_id"].tolist()
        for position, (_, row) in enumerate(group.iterrows()):
            state = str(row["state_id"])
            if not bool(row["_is_state_entry"]) or state not in MATURITY_STATES:
                continue
            future_states = states[position + 1 :]
            entries.append(
                {
                    "pair": str(row["pair"]),
                    "entry_time": row["entry_time"],
                    "state_id": state,
                    "future_states": future_states,
                }
            )
    return entries


def compute_reversal_probabilities(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute reversal-to-non-extreme probabilities within fixed horizons."""
    entries = _iter_entry_contexts(df)
    result: dict[str, dict[str, Any]] = {}

    for state in MATURITY_STATES:
        state_entries = [entry for entry in entries if entry["state_id"] == state]
        state_report: dict[str, Any] = {"entry_count": len(state_entries)}
        for horizon in HORIZONS:
            eligible = [
                entry for entry in state_entries if len(entry["future_states"]) >= horizon
            ]
            success_count = sum(
                NON_EXTREME_STATE in entry["future_states"][:horizon] for entry in eligible
            )
            total = len(eligible)
            interval = _wilson_interval(success_count, total)
            state_report[f"within_{horizon}"] = {
                "count": total,
                "success_count": int(success_count),
                "probability": (success_count / total) if total else None,
                "confidence_interval": interval,
                "censored_count": len(state_entries) - total,
            }
        result[state] = state_report
    return result


def compute_persistence_probabilities(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute persistence probabilities at fixed horizons."""
    entries = _iter_entry_contexts(df)
    result: dict[str, dict[str, Any]] = {}

    for state in MATURITY_STATES:
        state_entries = [entry for entry in entries if entry["state_id"] == state]
        state_report: dict[str, Any] = {"entry_count": len(state_entries)}
        allowed_targets = PERSISTENCE_TARGETS[state]
        for horizon in HORIZONS:
            eligible = [
                entry for entry in state_entries if len(entry["future_states"]) >= horizon
            ]
            success_count = sum(
                entry["future_states"][horizon - 1] in allowed_targets for entry in eligible
            )
            total = len(eligible)
            interval = _wilson_interval(success_count, total)
            state_report[f"at_{horizon}"] = {
                "count": total,
                "success_count": int(success_count),
                "probability": (success_count / total) if total else None,
                "confidence_interval": interval,
                "censored_count": len(state_entries) - total,
            }
        result[state] = state_report
    return result


def compute_progression_analysis(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Measure whether maturity states progress before consensus ends."""
    entries = _iter_entry_contexts(df)
    result: dict[str, dict[str, Any]] = {}

    for source_state, target_state in PROGRESSION_TARGETS:
        source_entries = [entry for entry in entries if entry["state_id"] == source_state]
        times_to_transition: list[int] = []

        for entry in source_entries:
            active_future: list[str] = []
            for future_state in entry["future_states"]:
                if future_state == NON_EXTREME_STATE:
                    break
                active_future.append(str(future_state))

            for idx, future_state in enumerate(active_future, start=1):
                if future_state == target_state:
                    times_to_transition.append(idx)
                    break

        success_count = len(times_to_transition)
        sample_count = len(source_entries)
        key = f"{source_state}->{target_state}"
        result[key] = {
            "source_state": source_state,
            "target_state": target_state,
            "sample_count": sample_count,
            "success_count": success_count,
            "transition_probability": (
                success_count / sample_count if sample_count else None
            ),
            "median_time_to_transition": (
                float(pd.Series(times_to_transition).median())
                if times_to_transition
                else None
            ),
        }

    return result


def _cramers_v(statistic: float, total: int, rows: int, cols: int) -> float | None:
    """Compute Cramér's V from a chi-squared statistic and table dimensions."""
    if total <= 0:
        return None
    scale = min(rows - 1, cols - 1)
    if scale <= 0:
        return None
    return math.sqrt(statistic / (total * scale))


def _run_behavioral_test(
    counts_by_state: dict[str, dict[str, int]],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Run a chi-squared behavioral comparison across maturity states."""
    contingency = []
    insufficient_states = []
    total = 0

    for state in MATURITY_STATES:
        success = int(counts_by_state[state]["success_count"])
        count = int(counts_by_state[state]["count"])
        failure = count - success
        total += count
        contingency.append([success, failure])
        if count < MIN_BEHAVIORAL_TEST_SAMPLES:
            insufficient_states.append(state)

    if insufficient_states:
        return {
            "metric": metric_name,
            "classification": "behavioral_evidence",
            "used_for_behavioral_differentiation": True,
            "sufficient_samples": False,
            "sample_threshold_per_state": MIN_BEHAVIORAL_TEST_SAMPLES,
            "states": MATURITY_STATES,
            "counts_by_state": counts_by_state,
            "test_statistic": None,
            "p_value": None,
            "effect_size": None,
            "significant": False,
            "warning": (
                "Insufficient behavioral test samples for states: "
                + ", ".join(insufficient_states)
                if insufficient_states
                else "No eligible observations for behavioral test."
            ),
        }

    statistic, p_value, _, _ = chi2_contingency(contingency)  # dof, expected unused
    return {
        "metric": metric_name,
        "classification": "behavioral_evidence",
        "used_for_behavioral_differentiation": True,
        "sufficient_samples": True,
        "sample_threshold_per_state": MIN_BEHAVIORAL_TEST_SAMPLES,
        "states": MATURITY_STATES,
        "counts_by_state": counts_by_state,
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": _cramers_v(float(statistic), total, len(contingency), 2),
        "significant": bool(p_value < ALPHA),
        "warning": None,
    }


def compute_behavioral_tests(
    reversal_probabilities: dict[str, dict[str, Any]],
    persistence_probabilities: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run independent contingency-based behavioral comparisons."""
    tests: dict[str, dict[str, Any]] = {}

    for horizon in HORIZONS:
        reversal_counts = {
            state: reversal_probabilities[state][f"within_{horizon}"]
            for state in MATURITY_STATES
        }
        tests[f"reversal_within_{horizon}"] = _run_behavioral_test(
            reversal_counts,
            metric_name=f"reversal_within_{horizon}",
        )

        persistence_counts = {
            state: persistence_probabilities[state][f"at_{horizon}"]
            for state in MATURITY_STATES
        }
        tests[f"persistence_at_{horizon}"] = _run_behavioral_test(
            persistence_counts,
            metric_name=f"persistence_at_{horizon}",
        )

    return tests


def analyze_behavioral_outcomes(df: pd.DataFrame) -> dict[str, Any]:
    """Run independent behavioral outcome analyses on assigned states."""
    transition_matrix = compute_transition_matrix(df)
    reversal_probabilities = compute_reversal_probabilities(df)
    persistence_probabilities = compute_persistence_probabilities(df)
    progression_analysis = compute_progression_analysis(df)
    behavioral_tests = compute_behavioral_tests(
        reversal_probabilities,
        persistence_probabilities,
    )

    valid_tests = [
        name
        for name, test in behavioral_tests.items()
        if test["sufficient_samples"] and test["effect_size"] is not None
    ]
    significant_tests = [
        name for name, test in behavioral_tests.items() if test["significant"]
    ]

    return {
        "metadata": {
            "module": "bsve.validation.behavioral_outcomes",
            "supported_environment": "reactive_jpy",
            "alpha": ALPHA,
            "horizons": HORIZONS,
            "sample_threshold_per_state": MIN_BEHAVIORAL_TEST_SAMPLES,
        },
        "behavioral_evidence_available": bool(valid_tests),
        "significant_behavioral_tests": significant_tests,
        "transition_matrix": transition_matrix,
        "reversal_probabilities": reversal_probabilities,
        "persistence_probabilities": persistence_probabilities,
        "progression_analysis": progression_analysis,
        "behavioral_tests": behavioral_tests,
    }


def write_behavioral_outcomes_report(
    report: dict[str, Any], output_dir: str | Path
) -> Path:
    """Write deterministic behavioral outcome JSON report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "behavioral_outcomes_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _print_summary(report: dict[str, Any], report_path: Path) -> None:
    print("[BSVE] Behavioral Outcome Validation (Reactive-JPY)")
    print("-" * 60)
    print(f"Behavioral evidence available: {report['behavioral_evidence_available']}")
    print("Significant behavioral tests:")
    significant = report["significant_behavioral_tests"]
    if not significant:
        print("  none")
    else:
        for name in significant:
            test = report["behavioral_tests"][name]
            print(
                f"  {name:<24} chi2={test['test_statistic']:.4f} "
                f"p={test['p_value']:.6g} effect={test['effect_size']:.4f}"
            )
    print(f"Report: {report_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent behavioral outcome analysis on a state-surface artifact.",
        prog="python -m bsve.validation.behavioral_outcomes",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to state-surface artifact parquet (bsve_states_reactive_jpy_1.0.0.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where behavioral_outcomes_report.json is written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        df = load_state_surface(args.artifact)
        report = analyze_behavioral_outcomes(df)
        report_path = write_behavioral_outcomes_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
