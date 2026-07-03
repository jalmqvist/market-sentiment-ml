#!/usr/bin/env python3
"""
Behavioral validation.

Evaluate a labeled Behavioral Surface against the frozen BSVE validation
criteria.

The implementation is ontology-agnostic. Any labeled Behavioral Surface
conforming to the canonical BSVE schema may be analyzed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact, norm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "pair",
    "state_id",
    "episode_id",
    "maturity_bars",
    "crowd_side",
    "crowd_failed",
]

DEFAULT_REFERENCE_STATE = "JPY_CONSENSUS_YOUNG"
DEFAULT_TARGET_STATE = "JPY_CONSENSUS_MATURING"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_surface(path: str | Path) -> pd.DataFrame:

    surface = pd.read_parquet(path)

    surface["timestamp"] = pd.to_datetime(surface["timestamp"])

    return surface


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_surface(
    surface: pd.DataFrame,
    *,
    reference_state: str,
    target_state: str,
) -> None:

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in surface.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    duplicates = surface.duplicated(
        subset=["timestamp", "pair"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate observations detected."
        )

    if surface["crowd_failed"].isna().any():
        raise ValueError(
            "Outcome labels contain missing values."
        )

    required_states = {
        reference_state,
        target_state,
    }

    observed = set(surface["state_id"])

    missing_states = required_states - observed

    if missing_states:

        raise ValueError(
            f"Required states absent: {sorted(missing_states)}"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def contingency_table(
    df: pd.DataFrame,
    *,
    reference_state: str,
    target_state: str,
) -> tuple[np.ndarray, dict[str, float]]:

    failed = (
        df["crowd_failed"]
        .astype(bool)
    )

    table = np.array([

        [
            ((df["state_id"] == reference_state) & failed).sum(),
            ((df["state_id"] == reference_state) & ~failed).sum(),
        ],

        [
            ((df["state_id"] == target_state) & failed).sum(),
            ((df["state_id"] == target_state) & ~failed).sum(),
        ],

    ])

    young_fail_rate = (
        table[0, 0]
        / table[0].sum()
        if table[0].sum()
        else np.nan
    )

    maturing_fail_rate = (
        table[1, 0]
        / table[1].sum()
        if table[1].sum()
        else np.nan
    )

    return table, {

        "reference_failure_rate":
            young_fail_rate,

        "target_failure_rate":
            maturing_fail_rate,

        "difference":
            maturing_fail_rate
            - young_fail_rate,
    }


def relative_risk(
    table: np.ndarray,
) -> float:

    risk1 = table[1, 0] / table[1].sum()

    risk0 = table[0, 0] / table[0].sum()

    return risk1 / risk0


def odds_ratio(
    table: np.ndarray,
) -> float:

    a, b = table[1]
    c, d = table[0]

    if b == 0 or c == 0:
        return np.inf

    return (a * d) / (b * c)


def difference_confidence_interval(
    table: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wald confidence interval for the difference in crowd-failure rates
    (target − reference).

    The current BSVE validation datasets are sufficiently large that the
    normal approximation is appropriate.
    """

    n_reference = table[0].sum()
    n_target = table[1].sum()

    p_reference = table[0, 0] / n_reference
    p_target = table[1, 0] / n_target

    difference = p_target - p_reference

    standard_error = np.sqrt(
        (
            p_reference * (1.0 - p_reference) / n_reference
        )
        +
        (
            p_target * (1.0 - p_target) / n_target
        )
    )

    z = norm.ppf(
        1.0 - (1.0 - confidence) / 2.0
    )

    return (
        difference - z * standard_error,
        difference + z * standard_error,
    )

def fisher_test(
    table: np.ndarray,
) -> tuple[float, float]:
    """
    One-sided Fisher exact test.

    Table orientation:

                    Failed   Succeeded
    Reference
    Target

    The BSVE hypothesis is that the TARGET behavioral state exhibits a
    higher crowd-failure probability than the REFERENCE state.

    SciPy defines one-sided alternatives relative to the odds ratio of the
    first row compared with the second. For this table orientation the
    correct hypothesis therefore corresponds to ``alternative="less"``.

    This orientation is covered by regression tests.
    """

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="less",
    )

    return odds_ratio, p_value


# ---------------------------------------------------------------------------
# Pair decomposition
# ---------------------------------------------------------------------------


def analyze_pair(
    df: pd.DataFrame,
    *,
    reference_state: str,
    target_state: str,
) -> dict[str, Any]:
    table, rates = contingency_table(
        df,
        reference_state=reference_state,
        target_state=target_state,
    )

    odds, p = fisher_test(table)

    ci_low, ci_high = difference_confidence_interval(
        table
    )

    return {

        "table":
            table.tolist(),

        "reference_failure_rate":
            rates["reference_failure_rate"],

        "target_failure_rate":
            rates["target_failure_rate"],

        "difference":
            rates["difference"],

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "relative_risk":
            relative_risk(table),

        "odds_ratio":
            odds_ratio(table),

        "fisher_p":
            p,

        "direction":
            (
                "expected"
                if rates["difference"] > 0
                else "reversed"
            ),
    }

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def behavioral_validation(
    surface: pd.DataFrame,
    *,
    reference_state: str,
    target_state: str,
) -> dict[str, Any]:

    validate_surface(
        surface,
        reference_state=reference_state,
        target_state=target_state,
    )

    analysis_df = surface[
        surface["state_id"].isin(
            [
                reference_state,
                target_state,
            ]
        )
    ].copy()

    pooled = analyze_pair(
        analysis_df,
        reference_state=reference_state,
        target_state=target_state,
    )

    pairs: dict[str, dict[str, Any]] = {}

    for pair, pair_df in analysis_df.groupby("pair"):

        pairs[pair] = analyze_pair(
            pair_df,
            reference_state=reference_state,
            target_state=target_state,
        )

    expected_pairs = sum(
        result["direction"] == "expected"
        for result in pairs.values()
    )

    # --------------------------------------------------------------
    # Frozen validation criteria
    # --------------------------------------------------------------

    criteria = {

        "criterion_1":

            pooled["target_failure_rate"]
            >
            pooled["reference_failure_rate"],

        "criterion_2":

            pooled["fisher_p"] <= 0.05,

        "criterion_3":

            pooled["difference"] >= 0.05,

        "criterion_4":

            expected_pairs == len(pairs),

    }

    confirmed = bool(
        pooled["target_failure_rate"]
        >
        pooled["reference_failure_rate"]

        and

        pooled["fisher_p"] <= 0.05

        and

        pooled["difference"] >= 0.05

        and

        expected_pairs == len(pairs)
    )

    inconclusive = bool(

        (

                pooled["target_failure_rate"]
                >
                pooled["reference_failure_rate"]

                and

                pooled["fisher_p"] > 0.05

        )

        or

        (

                expected_pairs == len(pairs) - 1

        )

        or

        (

                pooled["target_failure_rate"]
                >
                pooled["reference_failure_rate"]

                and

                pooled["fisher_p"] <= 0.05

                and

                pooled["difference"] < 0.05

        )

    )

    if confirmed:

        verdict = "CONFIRMED"

    elif inconclusive:

        verdict = "INCONCLUSIVE"

    else:

        verdict = "NOT_CONFIRMED"

    return {

        "generated_timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "observations":
            len(analysis_df),

        "expected_pairs":
            expected_pairs,

        "pooled":
            pooled,

        "pairs":
            pairs,

        "criteria":
            criteria,

        "verdict":
            verdict,

    }

# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:

    pooled = report["pooled"]

    print()
    print("=" * 72)
    print("BSVE BEHAVIORAL VALIDATION")
    print("=" * 72)

    print()

    print(f"Observations analysed : {report['observations']:,}")

    print()

    print("Pooled contingency table")
    print("-" * 72)

    table = np.array(pooled["table"])

    print()

    print("                 Failed   Succeeded")

    print(
        f"Young        {table[0,0]:8d}{table[0,1]:12d}"
    )

    print(
        f"Maturing     {table[1,0]:8d}{table[1,1]:12d}"
    )

    print()

    print(
        f"Reference failure rate      : "
        f"{100*pooled['reference_failure_rate']:.2f}%"
    )

    print(
        f"Target failure rate         : "
        f"{100*pooled['target_failure_rate']:.2f}%"
    )

    print(
        f"Difference             : "
        f"{100*pooled['difference']:+.2f} percentage points"
    )

    print(
        f"95% CI                : "
        f"[{100 * pooled['ci_low']:.2f}%, "
        f"{100 * pooled['ci_high']:.2f}%]"
    )

    print(
        f"Relative risk          : "
        f"{pooled['relative_risk']:.3f}"
    )

    print(
        f"Odds ratio             : "
        f"{pooled['odds_ratio']:.3f}"
    )

    print(
        f"Fisher p (one-sided)   : "
        f"{pooled['fisher_p']:.6f}"
    )

    print()

    print("Pair decomposition")
    print("-" * 72)

    for pair, result in sorted(report["pairs"].items()):

        print(
            f"{pair:<10}"
            f"{100 * result['reference_failure_rate']:6.2f}% -> "
            f"{100 * result['target_failure_rate']:6.2f}%   "
            f"Δ {100 * result['difference']:+6.2f}%   "
            f"95% CI "
            f"[{100 * result['ci_low']:+5.2f}%, "
            f"{100 * result['ci_high']:+5.2f}%]   "
            f"p={result['fisher_p']:.4f}"
        )

    print()

    print("Validation criteria")
    print("-" * 72)

    labels = {

        "criterion_1":
            "Target failure rate exceeds Reference",

        "criterion_2":
            "Fisher p ≤ 0.05",

        "criterion_3":
            "Difference ≥ 5 percentage points",

        "criterion_4":
            "Directional effect in every pair",

    }

    for key in [
        "criterion_1",
        "criterion_2",
        "criterion_3",
        "criterion_4",
    ]:

        passed = report["criteria"][key]

        print(
            f"{'✓' if passed else '✗'} "
            f"{labels[key]}"
        )

    print()

    print("=" * 72)
    print(f"VALIDATION RESULT : {report['verdict']}")
    print("=" * 72)

    print()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_results(
    report: dict[str, Any],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "behavioral_validation_report.json"
    )

    def _json_default(obj):

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        raise TypeError(
            f"Object of type {type(obj).__name__} "
            "is not JSON serializable."
        )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    pooled = pd.DataFrame(
        [
            report["pooled"]
        ]
    )

    pooled.to_csv(
        output_dir
        / "behavioral_validation_pooled.csv",
        index=False,
    )

    pair_rows = []

    for pair, values in report["pairs"].items():
        row = dict(values)

        table = np.asarray(
            values["table"]
        )

        row.update({

            "reference_failed":
                int(table[0, 0]),

            "reference_succeeded":
                int(table[0, 1]),

            "target_failed":
                int(table[1, 0]),

            "target_succeeded":
                int(table[1, 1]),

        })

        row["pair"] = pair

        pair_rows.append(row)

    pd.DataFrame(pair_rows).to_csv(
        output_dir
        / "behavioral_validation_pairs.csv",
        index=False,
    )

    print(
        f"[BSVE] Analysis written: {report_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the BSVE BEHAVIORAL VALIDATION."
        )
    )

    parser.add_argument(
        "--labeled-surface",
        required=True,
        help="Labeled Behavioral Surface.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--reference-state",
        default=DEFAULT_REFERENCE_STATE,
        help="Reference behavioral state.",
    )

    parser.add_argument(
        "--target-state",
        default=DEFAULT_TARGET_STATE,
        help="Behavioral state compared against the reference state.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()

    print()

    print("=" * 72)
    print("BSVE BEHAVIORAL VALIDATION")
    print("=" * 72)

    print()

    print(
        "Loading labeled Behavioral Surface..."
    )

    surface = load_surface(
        args.labeled_surface
    )

    print(
        f"  {len(surface):,} observations"
    )

    report = behavioral_validation(
        surface,
        reference_state=args.reference_state,
        target_state=args.target_state,
    )

    print_report(
        report
    )

    export_results(
        report,
        Path(args.output_dir),
    )

    print()

    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()