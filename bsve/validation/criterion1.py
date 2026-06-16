"""BSVE Criterion 1 validation for Reactive-JPY behavioral differentiation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp

from bsve.validation.behavioral_outcomes import analyze_behavioral_outcomes
from bsve.validation.report import write_validation_report

CRITERION_NAME = "criterion1_behavioral_differentiation"
MIN_OBSERVATIONS_PER_STATE = 50
MIN_OUTCOME_EPISODES_PER_STATE = 5
MIN_BEHAVIORAL_EFFECT_SIZE = 0.10
STATE_ORDER = [
    "JPY_NON_EXTREME",
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
]
CONSENSUS_STATES = [
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
]
KS_COMPARISONS = [
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING"),
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"),
    ("JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE"),
]
OUTCOME_COMPARISONS = KS_COMPARISONS
TRANSITION_EVENTS = ["entry", "continuation", "exit_reversal", "exit_unknown"]
SURVIVAL_THRESHOLDS = [8, 24, 48]


@dataclass
class ValidationResult:
    criterion_name: str
    status: str
    passed: bool
    sample_counts: dict[str, int]
    statistical_tests: list[dict[str, Any]]
    warnings: list[str]
    notes: list[str]


def _ordered_states(observed: pd.Series) -> list[str]:
    extras = sorted(set(observed.astype(str)) - set(STATE_ORDER))
    return [*STATE_ORDER, *extras]


def _cohens_h(p1: float, p2: float) -> float:
    return float(abs(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))))


def load_state_surface(path: str | Path) -> pd.DataFrame:
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


def load_independent_outcomes(path: str | Path) -> list[dict[str, Any]]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"independent outcome file not found: {payload_path}")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    outcomes = payload.get("independent_outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("independent outcome payload missing 'independent_outcomes' list")
    return outcomes


def reconstruct_state_episodes(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["pair", "entry_time"]).copy()
    shifted_pair = ordered["pair"].ne(ordered["pair"].shift())
    shifted_state = ordered["state_id"].ne(ordered["state_id"].shift())
    ordered["episode_id"] = (shifted_pair | shifted_state).cumsum()

    episodes = (
        ordered.groupby(["pair", "state_id", "episode_id"], as_index=False)
        .agg(
            start_time=("entry_time", "min"),
            end_time=("entry_time", "max"),
            duration_bars=("entry_time", "size"),
        )
        .drop(columns=["episode_id"])
    )
    episodes["duration_bars"] = episodes["duration_bars"].astype(int)
    return episodes.sort_values(["pair", "start_time", "state_id"]).reset_index(drop=True)


def compute_state_frequency_report(df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df))
    states = _ordered_states(df["state_id"])

    per_state = []
    for state in states:
        state_df = df[df["state_id"] == state]
        pair_counts = state_df.groupby("pair").size().astype(int).sort_index()
        per_state.append(
            {
                "state_id": state,
                "observations": int(len(state_df)),
                "pct_total": (float(len(state_df)) / total) if total else 0.0,
                "per_pair": {str(pair): int(count) for pair, count in pair_counts.items()},
            }
        )

    return {
        "total_observations": total,
        "per_state": per_state,
    }


def compute_duration_statistics(episodes: pd.DataFrame) -> list[dict[str, Any]]:
    states = _ordered_states(episodes["state_id"]) if not episodes.empty else STATE_ORDER
    rows = []
    for state in states:
        durations = episodes.loc[episodes["state_id"] == state, "duration_bars"]
        if durations.empty:
            rows.append(
                {
                    "state_id": state,
                    "episode_count": 0,
                    "median_duration": 0.0,
                    "mean_duration": 0.0,
                    "p25_duration": 0.0,
                    "p75_duration": 0.0,
                    "max_duration": 0,
                }
            )
            continue

        rows.append(
            {
                "state_id": state,
                "episode_count": int(len(durations)),
                "median_duration": float(durations.median()),
                "mean_duration": float(durations.mean()),
                "p25_duration": float(durations.quantile(0.25)),
                "p75_duration": float(durations.quantile(0.75)),
                "max_duration": int(durations.max()),
            }
        )
    return rows


def run_duration_ks_tests(episodes: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    tests: list[dict[str, Any]] = []
    warnings: list[str] = []

    for left, right in KS_COMPARISONS:
        d1 = episodes.loc[episodes["state_id"] == left, "duration_bars"]
        d2 = episodes.loc[episodes["state_id"] == right, "duration_bars"]

        if d1.empty or d2.empty:
            warnings.append(
                f"KS test skipped for {left} vs {right}: insufficient episode samples"
            )
            tests.append(
                {
                    "comparison": f"{left} vs {right}",
                    "state_a": left,
                    "state_b": right,
                    "ks_statistic": None,
                    "p_value": None,
                    "significant": False,
                    "classification": "calibration_consistency_diagnostic",
                    "used_for_behavioral_differentiation": False,
                }
            )
            continue

        result = ks_2samp(d1.to_numpy(), d2.to_numpy())
        tests.append(
            {
                "comparison": f"{left} vs {right}",
                "state_a": left,
                "state_b": right,
                "ks_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "significant": bool(result.pvalue < 0.05),
                "classification": "calibration_consistency_diagnostic",
                "used_for_behavioral_differentiation": False,
            }
        )

    return tests, warnings


def compute_survival_table(episodes: pd.DataFrame) -> list[dict[str, Any]]:
    states = _ordered_states(episodes["state_id"]) if not episodes.empty else STATE_ORDER
    rows: list[dict[str, Any]] = []
    for state in states:
        durations = episodes.loc[episodes["state_id"] == state, "duration_bars"]
        episode_count = int(len(durations))
        survival: dict[str, float] = {}
        for threshold in SURVIVAL_THRESHOLDS:
            key = f"survival_at_{threshold}"
            if episode_count == 0:
                survival[key] = 0.0
            else:
                survival[key] = float((durations >= threshold).mean())

        rows.append(
            {
                "state_id": state,
                "episode_count": episode_count,
                **survival,
            }
        )
    return rows


def compute_transition_frequencies(df: pd.DataFrame) -> list[dict[str, Any]]:
    states = _ordered_states(df["state_id"])
    rows: list[dict[str, Any]] = []

    for state in states:
        state_df = df[df["state_id"] == state]
        counts = state_df["transition_event"].value_counts().to_dict()
        rows.append(
            {
                "state_id": state,
                **{event: int(counts.get(event, 0)) for event in TRANSITION_EVENTS},
            }
        )
    return rows


def summarize_independent_behavioral_evidence(
    independent_outcomes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not independent_outcomes:
        empty_distribution = [
            {
                "state_id": state,
                "episode_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": None,
            }
            for state in CONSENSUS_STATES
        ]
        return {
            "labels_available": False,
            "behavioral_evidence_available": False,
            "minimum_outcome_samples_met": False,
            "has_significant_differentiation": False,
            "effect_size": None,
            "effect_size_threshold_met": False,
            "independent_outcome_distribution": empty_distribution,
            "outcome_tests": [],
            "warnings": [
                "Independent outcome labels not found. Run bsve.validation.outcome_labeling first."
            ],
        }

    frame = pd.DataFrame(independent_outcomes)
    if frame.empty:
        return {
            "labels_available": True,
            "behavioral_evidence_available": False,
            "minimum_outcome_samples_met": False,
            "has_significant_differentiation": False,
            "effect_size": None,
            "effect_size_threshold_met": False,
            "independent_outcome_distribution": [],
            "outcome_tests": [],
            "warnings": ["Independent outcome payload contains no episode rows."],
        }

    if "outcome_available" in frame.columns:
        frame = frame[frame["outcome_available"] == True]  # noqa: E712
    if "outcome_label" in frame.columns:
        frame = frame[frame["outcome_label"].isin(["SUCCESS", "FAILURE"])]

    distribution: list[dict[str, Any]] = []
    rates: dict[str, tuple[int, int]] = {}
    for state in CONSENSUS_STATES:
        state_rows = frame[frame["state_id"] == state]
        n = int(len(state_rows))
        success = int((state_rows["outcome_label"] == "SUCCESS").sum()) if n else 0
        failure = int((state_rows["outcome_label"] == "FAILURE").sum()) if n else 0
        rates[state] = (n, success)
        distribution.append(
            {
                "state_id": state,
                "episode_count": n,
                "success_count": success,
                "failure_count": failure,
                "success_rate": (float(success / n) if n else None),
            }
        )

    min_samples_met = all(
        row["episode_count"] >= MIN_OUTCOME_EPISODES_PER_STATE for row in distribution
    )

    outcome_tests: list[dict[str, Any]] = []
    significant_effect_sizes: list[float] = []
    for state_a, state_b in OUTCOME_COMPARISONS:
        n_a, s_a = rates[state_a]
        n_b, s_b = rates[state_b]

        if n_a < MIN_OUTCOME_EPISODES_PER_STATE or n_b < MIN_OUTCOME_EPISODES_PER_STATE:
            outcome_tests.append(
                {
                    "comparison": f"{state_a} vs {state_b}",
                    "state_a": state_a,
                    "state_b": state_b,
                    "test": "fisher_exact",
                    "p_value": None,
                    "significant": False,
                    "effect_size": None,
                    "effect_size_metric": "cohens_h",
                    "skipped": True,
                    "skip_reason": (
                        f"insufficient independent outcomes: {n_a} ({state_a}), "
                        f"{n_b} ({state_b}); minimum is {MIN_OUTCOME_EPISODES_PER_STATE}"
                    ),
                    "classification": "independent_behavioral_evidence",
                    "used_for_behavioral_evidence": True,
                }
            )
            continue

        p_a = s_a / n_a
        p_b = s_b / n_b
        _, p_value = fisher_exact([[s_a, n_a - s_a], [s_b, n_b - s_b]])
        effect_size = _cohens_h(p_a, p_b)
        is_significant = bool(p_value < 0.05)
        if is_significant:
            significant_effect_sizes.append(effect_size)

        outcome_tests.append(
            {
                "comparison": f"{state_a} vs {state_b}",
                "state_a": state_a,
                "state_b": state_b,
                "test": "fisher_exact",
                "p_value": float(p_value),
                "significant": is_significant,
                "effect_size": float(effect_size),
                "effect_size_metric": "cohens_h",
                "skipped": False,
                "skip_reason": None,
                "classification": "independent_behavioral_evidence",
                "used_for_behavioral_evidence": True,
            }
        )

    effect_size = max(significant_effect_sizes) if significant_effect_sizes else None
    has_significant_differentiation = any(test["significant"] for test in outcome_tests)
    effect_size_threshold_met = (
        effect_size is not None and effect_size >= MIN_BEHAVIORAL_EFFECT_SIZE
    )

    return {
        "labels_available": True,
        "behavioral_evidence_available": bool(min_samples_met),
        "minimum_outcome_samples_met": bool(min_samples_met),
        "has_significant_differentiation": has_significant_differentiation,
        "effect_size": effect_size,
        "effect_size_threshold_met": effect_size_threshold_met,
        "independent_outcome_distribution": distribution,
        "outcome_tests": outcome_tests,
        "warnings": [],
    }


def evaluate_criterion1(
    df: pd.DataFrame,
    *,
    independent_outcomes: list[dict[str, Any]] | None = None,
    descriptive_behavioral_diagnostics_available: bool = False,
    behavioral_evidence_status: str = "duration_derived_outcomes_not_independent",
    behavioral_effect_size: float | None = None,
    behavioral_tests: list[dict[str, Any]] | None = None,
    behavioral_outcomes: list[dict[str, Any]] | None = None,
) -> tuple[ValidationResult, dict[str, Any]]:
    frequency_report = compute_state_frequency_report(df)
    episodes = reconstruct_state_episodes(df)
    duration_statistics = compute_duration_statistics(episodes)
    ks_tests, ks_warnings = run_duration_ks_tests(episodes)
    survival_table = compute_survival_table(episodes)
    transitions = compute_transition_frequencies(df)
    independent_evidence = summarize_independent_behavioral_evidence(independent_outcomes)

    sample_counts = {
        row["state_id"]: int(row["observations"])
        for row in frequency_report["per_state"]
    }
    low_sample_states = [
        state for state, count in sample_counts.items() if count < MIN_OBSERVATIONS_PER_STATE
    ]

    warnings = [*ks_warnings, *independent_evidence["warnings"]]
    notes = [
        "Criterion 1 validates behavioral differentiation, not trading performance.",
        f"Minimum observations per state threshold: {MIN_OBSERVATIONS_PER_STATE}.",
        (
            "Independent evidence uses fixed-horizon post-episode forward outcomes and does "
            "not depend on maturity duration labels."
        ),
        (
            "Duration KS tests and duration-derived reversal diagnostics are reported as "
            "descriptive diagnostics only."
        ),
    ]

    if low_sample_states:
        warnings.append(
            "Insufficient observations for states: "
            + ", ".join(sorted(low_sample_states))
        )
        status = "FAIL"
    elif not independent_evidence["labels_available"]:
        warnings.append(
            "Independent outcome labels are required for Criterion 1 PASS and are currently missing."
        )
        status = "INCONCLUSIVE"
    elif not independent_evidence["minimum_outcome_samples_met"]:
        warnings.append(
            "Independent outcome labels exist but have insufficient per-state episode counts."
        )
        status = "FAIL"
    elif not independent_evidence["has_significant_differentiation"]:
        warnings.append(
            "Independent outcome labels exist but no statistically significant behavioral differentiation was observed."
        )
        status = "INCONCLUSIVE"
    elif not independent_evidence["effect_size_threshold_met"]:
        warnings.append(
            "Independent outcome differentiation is significant but effect size is below the configured threshold."
        )
        status = "INCONCLUSIVE"
    else:
        status = "PASS"

    result = ValidationResult(
        criterion_name=CRITERION_NAME,
        status=status,
        passed=(status == "PASS"),
        sample_counts=sample_counts,
        statistical_tests=ks_tests,
        warnings=warnings,
        notes=notes,
    )

    generated_at = df["entry_time"].max()
    report = {
        "metadata": {
            "criterion": CRITERION_NAME,
            "module": "bsve.validation.criterion1",
            "generated_at": generated_at.isoformat() if pd.notna(generated_at) else None,
            "min_observations_per_state": MIN_OBSERVATIONS_PER_STATE,
            "min_outcome_episodes_per_state": MIN_OUTCOME_EPISODES_PER_STATE,
            "minimum_behavioral_effect_size": MIN_BEHAVIORAL_EFFECT_SIZE,
            "ks_alpha": 0.05,
            "supported_environment": "reactive_jpy",
            "behavioral_evidence_available": independent_evidence["behavioral_evidence_available"],
            "independent_behavioral_evidence": independent_evidence["labels_available"],
            "behavioral_evidence_status": (
                "independent_outcomes_available"
                if independent_evidence["labels_available"]
                else "independent_outcomes_missing"
            ),
            "behavioral_diagnostics_classification": "descriptive_diagnostic",
            "descriptive_behavioral_diagnostics_available": (
                descriptive_behavioral_diagnostics_available
            ),
            "behavioral_effect_size": independent_evidence["effect_size"],
            "legacy_descriptive_behavioral_effect_size": behavioral_effect_size,
            "legacy_behavioral_evidence_status": behavioral_evidence_status,
        },
        "state_frequencies": frequency_report,
        "duration_statistics": duration_statistics,
        "duration_ks_diagnostics": ks_tests,
        "ks_test_results": ks_tests,
        "survival_analysis": survival_table,
        "transition_frequencies": transitions,
        "independent_outcome_distribution": independent_evidence[
            "independent_outcome_distribution"
        ],
        "outcome_tests": independent_evidence["outcome_tests"],
        "independent_behavioral_evidence": {
            "labels_available": independent_evidence["labels_available"],
            "behavioral_evidence_available": independent_evidence[
                "behavioral_evidence_available"
            ],
            "minimum_outcome_samples_met": independent_evidence[
                "minimum_outcome_samples_met"
            ],
            "has_significant_differentiation": independent_evidence[
                "has_significant_differentiation"
            ],
            "effect_size": independent_evidence["effect_size"],
            "effect_size_threshold_met": independent_evidence[
                "effect_size_threshold_met"
            ],
        },
        "descriptive_diagnostics": {
            "behavioral_outcomes": behavioral_outcomes,
            "behavioral_tests": behavioral_tests,
        },
        "validation_outcome": asdict(result),
    }
    return result, report


def _print_summary(result: ValidationResult, report: dict[str, Any], report_path: Path) -> None:
    print("[BSVE] Criterion 1 Validation (Reactive-JPY)")
    print("-" * 60)
    print(f"Status: {result.status}")
    print("Sample counts:")
    for state in STATE_ORDER:
        if state in result.sample_counts:
            print(f"  {state:<24} {result.sample_counts[state]:>8d}")
    print("Independent outcome tests:")
    for test in report.get("outcome_tests", []):
        p_value = test["p_value"]
        if p_value is None:
            print(f"  {test['comparison']:<58} skipped")
            continue
        status = "significant" if test["significant"] else "not-significant"
        print(
            f"  {test['comparison']:<44} p={p_value:.6g} "
            f"effect={test['effect_size']:.4f} ({status})"
        )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    print(f"Report: {report_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BSVE Criterion 1 validation on a state-surface artifact.",
        prog="python -m bsve.validation.criterion1",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to state-surface artifact parquet (bsve_states_reactive_jpy_1.0.0.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where bsve_validation_report.json is written.",
    )
    parser.add_argument(
        "--independent-outcomes",
        default=None,
        help=(
            "Optional path to independent_outcomes.json. Defaults to "
            "<output-dir>/independent_outcomes.json when present."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        df = load_state_surface(args.artifact)
        behavioral = analyze_behavioral_outcomes(df)

        independent_path = (
            Path(args.independent_outcomes)
            if args.independent_outcomes
            else Path(args.output_dir) / "independent_outcomes.json"
        )
        independent_outcomes = (
            load_independent_outcomes(independent_path)
            if independent_path.exists()
            else None
        )

        result, report = evaluate_criterion1(
            df,
            independent_outcomes=independent_outcomes,
            descriptive_behavioral_diagnostics_available=behavioral[
                "descriptive_behavioral_diagnostics_available"
            ],
            behavioral_evidence_status=behavioral["behavioral_evidence_status"],
            behavioral_effect_size=behavioral["behavioral_effect_size"],
            behavioral_tests=behavioral["behavioral_tests"],
            behavioral_outcomes=behavioral["behavioral_outcomes"],
        )
        report_path = write_validation_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, report, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
