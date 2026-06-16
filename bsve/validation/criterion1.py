"""BSVE Criterion 1 validation for Reactive-JPY behavioral differentiation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from bsve.validation.report import write_validation_report

CRITERION_NAME = "criterion1_behavioral_differentiation"
MIN_OBSERVATIONS_PER_STATE = 50
MIN_BEHAVIORAL_EFFECT_SIZE = 0.10
STATE_ORDER = [
    "JPY_NON_EXTREME",
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
]
KS_COMPARISONS = [
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING"),
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"),
    ("JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE"),
]
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


def evaluate_criterion1(
    df: pd.DataFrame,
    *,
    behavioral_evidence_available: bool = False,
    behavioral_effect_size: float | None = None,
) -> tuple[ValidationResult, dict[str, Any]]:
    frequency_report = compute_state_frequency_report(df)
    episodes = reconstruct_state_episodes(df)
    duration_statistics = compute_duration_statistics(episodes)
    ks_tests, ks_warnings = run_duration_ks_tests(episodes)
    survival_table = compute_survival_table(episodes)
    transitions = compute_transition_frequencies(df)

    sample_counts = {
        row["state_id"]: int(row["observations"])
        for row in frequency_report["per_state"]
    }
    low_sample_states = [
        state for state, count in sample_counts.items() if count < MIN_OBSERVATIONS_PER_STATE
    ]

    warnings = list(ks_warnings)
    notes = [
        "Criterion 1 validates behavioral differentiation, not trading performance.",
        f"Minimum observations per state threshold: {MIN_OBSERVATIONS_PER_STATE}.",
        "Duration KS tests are calibration-consistency diagnostics and are not treated as behavioral differentiation evidence.",
        (
            "Progression analysis (YOUNG → MATURING, YOUNG → MATURE, MATURING → MATURE) is "
            "currently descriptive only. It is reported for ontology interpretation and future "
            "research, but is not used in Criterion 1 PASS/FAIL determination."
        ),
    ]

    if low_sample_states:
        warnings.append(
            "Insufficient observations for states: "
            + ", ".join(sorted(low_sample_states))
        )
        status = "FAIL"
    elif behavioral_evidence_available:
        effect_size_sufficient = (
            behavioral_effect_size is not None
            and behavioral_effect_size >= MIN_BEHAVIORAL_EFFECT_SIZE
        )
        if effect_size_sufficient:
            status = "PASS"
        else:
            if behavioral_effect_size is None:
                warnings.append(
                    "Behavioral evidence is marked available but no effect size was supplied; "
                    f"a minimum effect size of {MIN_BEHAVIORAL_EFFECT_SIZE} is required for a PASS status."
                )
            else:
                warnings.append(
                    f"Behavioral effect size {behavioral_effect_size:.4f} is below the minimum "
                    f"threshold of {MIN_BEHAVIORAL_EFFECT_SIZE}; result is INCONCLUSIVE."
                )
            status = "INCONCLUSIVE"
    else:
        warnings.append(
            "Current Reactive-JPY Criterion 1 runs use duration-derived diagnostics only; independent behavioral evidence is required for a PASS status."
        )
        status = "INCONCLUSIVE"

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
            "minimum_behavioral_effect_size": MIN_BEHAVIORAL_EFFECT_SIZE,
            "ks_alpha": 0.05,
            "supported_environment": "reactive_jpy",
            "behavioral_evidence_available": behavioral_evidence_available,
            "behavioral_effect_size": behavioral_effect_size,
            "progression_analysis_role": (
                "descriptive_only — progression analysis (YOUNG → MATURING, YOUNG → MATURE, "
                "MATURING → MATURE) is reported for ontology interpretation and future research, "
                "but is not used in Criterion 1 PASS/FAIL determination."
            ),
        },
        "state_frequencies": frequency_report,
        "duration_statistics": duration_statistics,
        "duration_ks_diagnostics": ks_tests,
        "ks_test_results": ks_tests,
        "survival_analysis": survival_table,
        "transition_frequencies": transitions,
        "validation_outcome": asdict(result),
    }
    return result, report


def _print_summary(result: ValidationResult, report_path: Path) -> None:
    print("[BSVE] Criterion 1 Validation (Reactive-JPY)")
    print("-" * 60)
    print(f"Status: {result.status}")
    print("Sample counts:")
    for state in STATE_ORDER:
        if state in result.sample_counts:
            print(f"  {state:<24} {result.sample_counts[state]:>8d}")
    print("KS tests:")
    for test in result.statistical_tests:
        p_value = test["p_value"]
        if p_value is None:
            print(f"  {test['comparison']:<58} skipped")
            continue
        status = "significant" if test["significant"] else "not-significant"
        print(
            f"  {test['comparison']:<44} ks={test['ks_statistic']:.4f} "
            f"p={p_value:.6g} ({status})"
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        df = load_state_surface(args.artifact)
        result, report = evaluate_criterion1(df)
        report_path = write_validation_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
