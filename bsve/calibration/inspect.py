"""
BSVE Calibration Artifact Inspector.

Read-only CLI utility for inspecting calibration artifact JSON files.

Usage::

    python -m bsve.calibration.inspect <artifact_path>

Example::

    python -m bsve.calibration.inspect calibration_artifacts/reactive_jpy_v1.json

No calibration logic lives here.  This module only loads and displays
an existing artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_artifact(path: Path) -> dict:
    if not path.exists():
        print(f"Error: artifact not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def display_artifact(artifact: dict) -> None:
    """Print a human-readable summary of *artifact* to stdout."""
    ontology_id = artifact.get("ontology_id", "unknown")
    ontology_version = artifact.get("ontology_version", "unknown")
    outcome = artifact.get("outcome", "unknown")
    calibration_mode = artifact.get("calibration_mode")

    print(f"Ontology: {ontology_id}")
    print(f"Version:  {ontology_version}")
    if calibration_mode is not None:
        print(f"Mode:     {calibration_mode}")
    print()
    print(f"Outcome: {outcome}")
    print()

    if outcome == "null":
        null_reason = artifact.get("null_reason", "unspecified")
        print(f"Null reason: {null_reason}")
        return

    thresholds = artifact.get("thresholds", {})
    extreme = thresholds.get("extreme_threshold_net_pct")
    young = thresholds.get("young_boundary_bars")
    mature = thresholds.get("mature_boundary_bars")

    if extreme is not None:
        print(f"Extreme threshold: {extreme}")
    if young is not None:
        print(f"Young boundary:    {young}")
    if mature is not None:
        print(f"Mature boundary:   {mature}")

    diagnostics = artifact.get("diagnostics", {})
    episode_count = diagnostics.get("episode_count")
    crossover = diagnostics.get("hazard_crossover_bar")

    if episode_count is not None or crossover is not None:
        print()
        if episode_count is not None:
            print(f"Episode count:    {episode_count}")
        if crossover is not None:
            print(f"Hazard crossover: {crossover}")

    provenance = artifact.get("threshold_provenance")
    if provenance:
        print()
        print("Threshold provenance:")
        for threshold_name, meta in provenance.items():
            method = meta.get("method", "unknown")
            parts = [f"method={method}"]
            for k, v in meta.items():
                if k != "method":
                    parts.append(f"{k}={v}")
            print(f"  {threshold_name}: {', '.join(parts)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a BSVE calibration artifact JSON file.",
        prog="python -m bsve.calibration.inspect",
    )
    parser.add_argument(
        "artifact",
        metavar="ARTIFACT_PATH",
        help="Path to the calibration artifact JSON file.",
    )
    args = parser.parse_args(argv)

    path = Path(args.artifact)
    artifact = _load_artifact(path)
    display_artifact(artifact)


if __name__ == "__main__":
    main()
