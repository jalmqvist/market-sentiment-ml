#!/usr/bin/env python3
"""
Behavioral Surface calibration drift analysis.

Compares a development Behavioral Surface with an out-of-sample Behavioral
Surface using the canonical summary produced by
bsve.validation.inspect_surface.summarize_surface().

The objective is descriptive rather than inferential. Calibration drift is
reported as contextual information to aid interpretation of downstream
behavioral validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bsve.validation.inspect_surface import summarize_surface


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_surface(path: str | Path) -> pd.DataFrame:

    surface = pd.read_parquet(path)

    surface["timestamp"] = pd.to_datetime(surface["timestamp"])

    return surface


def load_calibration(path: str | Path | None):

    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentages(counts: dict[str, int]) -> dict[str, float]:

    total = sum(counts.values())

    if total == 0:
        return {}

    return {
        k: 100.0 * v / total
        for k, v in counts.items()
    }


def _compare_state_counts(
    development: dict[str, int],
    oos: dict[str, int],
) -> dict[str, Any]:

    dev_pct = _percentages(development)
    oos_pct = _percentages(oos)

    states = sorted(
        set(development)
        | set(oos)
    )

    comparison = {}

    for state in states:

        comparison[state] = {
            "development_count":
                development.get(state, 0),

            "oos_count":
                oos.get(state, 0),

            "development_pct":
                dev_pct.get(state, 0.0),

            "oos_pct":
                oos_pct.get(state, 0.0),

            "difference_pct":
                oos_pct.get(state, 0.0)
                - dev_pct.get(state, 0.0),
        }

    return comparison


def _compare_episode_statistics(
    development: dict[str, Any],
    oos: dict[str, Any],
) -> dict[str, Any]:

    metrics = [
        "mean",
        "median",
        "max",
    ]

    comparison = {}

    for metric in metrics:

        comparison[metric] = {
            "development":
                development[metric],

            "oos":
                oos[metric],

            "difference":
                oos[metric]
                - development[metric],
        }

    return comparison


def _compare_survival(
    development: dict[str, int],
    oos: dict[str, int],
) -> dict[str, Any]:

    labels = sorted(
        set(development)
        | set(oos)
    )

    comparison = {}

    for label in labels:

        comparison[label] = {
            "development":
                development.get(label, 0),

            "oos":
                oos.get(label, 0),

            "difference":
                oos.get(label, 0)
                - development.get(label, 0),
        }

    return comparison


def _compare_pair_counts(
    development: dict[str, int],
    oos: dict[str, int],
) -> dict[str, Any]:

    pairs = sorted(
        set(development)
        | set(oos)
    )

    comparison = {}

    for pair in pairs:

        comparison[pair] = {
            "development":
                development.get(pair, 0),

            "oos":
                oos.get(pair, 0),

            "difference":
                oos.get(pair, 0)
                - development.get(pair, 0),
        }

    return comparison


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def calibration_drift(
    development_surface: pd.DataFrame,
    oos_surface: pd.DataFrame,
    *,
    development_calibration: dict[str, Any] | None = None,
    oos_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:

    development = summarize_surface(
        development_surface,
        calibration_artifact=development_calibration,
    )

    oos = summarize_surface(
        oos_surface,
        calibration_artifact=oos_calibration,
    )

    comparison = {

        "state_occupancy":
            _compare_state_counts(
                development["state_counts"],
                oos["state_counts"],
            ),

        "episode_statistics":
            _compare_episode_statistics(
                development["episode_length_stats"],
                oos["episode_length_stats"],
            ),

        "survival":
            _compare_survival(
                development["survival_counts"],
                oos["survival_counts"],
            ),

        "pair_counts":
            _compare_pair_counts(
                development["pair_counts"],
                oos["pair_counts"],
            ),

    }

    assessment = []

    if development["warnings"]:
        assessment.append(
            "Development surface contains structural warnings."
        )

    if oos["warnings"]:
        assessment.append(
            "OOS surface contains structural warnings."
        )

    if not assessment:

        assessment.append(
            "No structural anomalies detected. "
            "Interpret statistical validation in the context "
            "of the observed descriptive differences."
        )

    return {

        "generated_timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "development":
            development,

        "oos":
            oos,

        "comparison":
            comparison,

        "assessment":
            assessment,
    }

# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:

    print()
    print("=" * 72)
    print("BSVE CALIBRATION DRIFT")
    print("=" * 72)

    print()

    print("State occupancy")
    print("-" * 72)

    for state, values in report["comparison"]["state_occupancy"].items():

        print(
            f"{state:<30}"
            f"{values['development_pct']:7.2f}%"
            f" -> "
            f"{values['oos_pct']:7.2f}%"
            f"   Δ {values['difference_pct']:+6.2f}%"
        )

    print()

    print("Episode statistics")
    print("-" * 72)

    for metric, values in report["comparison"]["episode_statistics"].items():

        print(
            f"{metric:<10}"
            f"{values['development']:10.2f}"
            f"{values['oos']:10.2f}"
            f"{values['difference']:10.2f}"
        )

    print()

    print("Episode survival")
    print("-" * 72)

    for label, values in report["comparison"]["survival"].items():

        print(
            f"{label:<15}"
            f"{values['development']:8d}"
            f"{values['oos']:8d}"
            f"{values['difference']:8d}"
        )

    print()

    print("Pair counts")
    print("-" * 72)

    for pair, values in report["comparison"]["pair_counts"].items():

        print(
            f"{pair:<12}"
            f"{values['development']:8d}"
            f"{values['oos']:8d}"
            f"{values['difference']:8d}"
        )

    print()

    print("Assessment")
    print("-" * 72)

    for line in report["assessment"]:
        print(f"• {line}")

    print()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_report(
    report: dict[str, Any],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "calibration_drift_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[BSVE] Drift report written: {report_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description="Compare two Behavioral Surfaces for calibration drift."
    )

    parser.add_argument(
        "--development",
        required=True,
        help="Development Behavioral Surface parquet.",
    )

    parser.add_argument(
        "--oos",
        required=True,
        help="Out-of-sample Behavioral Surface parquet.",
    )

    parser.add_argument(
        "--development-calibration",
        default=None,
        help="Optional development calibration artifact.",
    )

    parser.add_argument(
        "--oos-calibration",
        default=None,
        help="Optional OOS calibration artifact.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the JSON report.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()

    print()
    print("=" * 72)
    print("BSVE CALIBRATION DRIFT")
    print("=" * 72)

    print()

    print("Loading development Behavioral Surface...")

    development_surface = load_surface(
        args.development,
    )

    print(
        f"  {len(development_surface):,} observations"
    )

    print()

    print("Loading OOS Behavioral Surface...")

    oos_surface = load_surface(
        args.oos,
    )

    print(
        f"  {len(oos_surface):,} observations"
    )

    development_calibration = load_calibration(
        args.development_calibration,
    )

    oos_calibration = load_calibration(
        args.oos_calibration,
    )

    report = calibration_drift(
        development_surface,
        oos_surface,
        development_calibration=development_calibration,
        oos_calibration=oos_calibration,
    )

    print_report(report)

    export_report(
        report,
        Path(args.output_dir),
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()