"""Behavioral Surface Registry — Promotion Workflow.

Promotion is an **explicit manual step**.

Completed experiment reports do *not* automatically update the registry.
This script is the sole mechanism by which scientific evidence is formally
promoted into the Behavioral Surface Registry.

Usage
-----
::

    python analysis/registry/promote.py \\
        --surface reactive_jpy \\
        --experiments analysis/output/exp_2026_01_01 \\
        --author "your.name" \\
        --recommendation "Repeat characterization with additional training." \\
        --scientific-interest medium \\
        --scientific-confidence low \\
        --notes "Initial characterization.  Entropy high across all states."

Multiple experiment directories may be promoted in a single step::

    python analysis/registry/promote.py \\
        --surface reactive_jpy \\
        --experiments analysis/output/exp_A analysis/output/exp_B \\
        --author "your.name" \\
        --recommendation "Proceed to walk-forward validation." \\
        --scientific-interest high \\
        --scientific-confidence medium \\
        --notes "Two characterization experiments show consistent high agreement."

Each promotion
- references one or more completed experiment directories
- appends supporting evidence to the registry entry
- updates current_recommendation
- updates scientific_interest
- updates scientific_confidence
- records the author and timestamp
- appends a timestamped entry to promotion_history

The registry remains fully version-controlled.  Every change is recorded in
promotion_history and can be audited via git log.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INTEREST = frozenset({"low", "medium", "high"})
_VALID_CONFIDENCE = frozenset({"low", "medium", "high"})
_VALID_STAGES = frozenset({
    "Characterization",
    "Predictive Validation",
    "Trading Validation",
    "Integrated",
    "Retired",
})

# Maps a requested lifecycle_stage to the prerequisite (stage_key, required_status) that
# must hold in the current registry entry before the transition is permitted.
# Promotions within the same stage, or to "Retired", have no hard prerequisite here.
_STAGE_PREREQUISITES: dict[str, tuple[str, str]] = {
    "Predictive Validation": ("stage1", "complete"),
    "Trading Validation": ("stage2", "complete"),
    "Integrated": ("stage3", "complete"),
}


def _registry_root(repo_root: Path) -> Path:
    return repo_root / "registry" / "surfaces"


def _surface_path(registry_root: Path, surface_id: str) -> Path:
    return registry_root / f"{surface_id}.yaml"


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_registry_entry(path: Path) -> dict:
    """Load a registry YAML entry from *path*.

    Raises
    ------
    FileNotFoundError
        If the registry entry does not exist.
    ValueError
        If the file cannot be parsed as valid YAML or is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Registry entry not found: {path}\n"
            "Create the entry file before promoting into it."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Registry entry at {path} must be a YAML mapping, got {type(data).__name__}.")
    return data


def save_registry_entry(path: Path, data: dict) -> None:
    """Write *data* to *path* as YAML.

    Writes with explicit block style and preserves key ordering (insertion
    order is preserved by PyYAML when using default_flow_style=False).
    """
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_entry(data: dict) -> list[str]:
    """Return a list of validation error messages for *data*.

    An empty list means the entry is valid.
    """
    errors: list[str] = []
    required = [
        "surface_id", "ontology_version", "lifecycle_stage", "current_status",
        "scientific_interest", "scientific_confidence", "current_recommendation",
        "supporting_experiments", "promotion_history",
    ]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field!r}")

    if "scientific_interest" in data and data["scientific_interest"] not in _VALID_INTEREST:
        errors.append(
            f"scientific_interest must be one of {sorted(_VALID_INTEREST)}, "
            f"got {data['scientific_interest']!r}"
        )
    if "scientific_confidence" in data and data["scientific_confidence"] not in _VALID_CONFIDENCE:
        errors.append(
            f"scientific_confidence must be one of {sorted(_VALID_CONFIDENCE)}, "
            f"got {data['scientific_confidence']!r}"
        )
    if "lifecycle_stage" in data and data["lifecycle_stage"] not in _VALID_STAGES:
        errors.append(
            f"lifecycle_stage must be one of {sorted(_VALID_STAGES)}, "
            f"got {data['lifecycle_stage']!r}"
        )
    if "supporting_experiments" in data and not isinstance(data["supporting_experiments"], list):
        errors.append("supporting_experiments must be a list")
    if "promotion_history" in data and not isinstance(data["promotion_history"], list):
        errors.append("promotion_history must be a list")

    return errors


# ---------------------------------------------------------------------------
# Core promotion logic
# ---------------------------------------------------------------------------

def promote(
    *,
    surface_id: str,
    experiment_dirs: list[str],
    author: str,
    recommendation: str,
    scientific_interest: str,
    scientific_confidence: str,
    notes: str,
    lifecycle_stage: str | None = None,
    repo_root: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Promote experiment evidence into the Behavioral Surface Registry.

    Parameters
    ----------
    surface_id:
        Identifier of the Behavioral Surface (matches registry YAML filename).
    experiment_dirs:
        One or more paths to completed experiment output directories.
    author:
        Author identifier (e.g. GitHub username or full name).
    recommendation:
        Updated recommended next research step.
    scientific_interest:
        Updated Scientific Interest assessment (``"low"``, ``"medium"``, or ``"high"``).
    scientific_confidence:
        Updated Scientific Confidence assessment (``"low"``, ``"medium"``, or ``"high"``).
    notes:
        Human-readable description of what evidence is being promoted and why.
    lifecycle_stage:
        Optional updated lifecycle stage.  If not provided, the current stage is
        preserved unchanged.
    repo_root:
        Repository root.  Defaults to the parent of the directory containing this
        file, resolved relative to the script location.
    dry_run:
        If ``True``, validate and print the updated entry without writing to disk.
    force:
        If ``True``, bypass the lifecycle-stage consistency check.  Use only when
        you have deliberately set stage statuses out of the normal sequence.

    Returns
    -------
    dict
        The updated registry entry after promotion.

    Raises
    ------
    ValueError
        If any argument is invalid or the registry entry fails validation.
    FileNotFoundError
        If the registry entry or an experiment directory does not exist.
    """
    if scientific_interest not in _VALID_INTEREST:
        raise ValueError(
            f"scientific_interest must be one of {sorted(_VALID_INTEREST)}, "
            f"got {scientific_interest!r}"
        )
    if scientific_confidence not in _VALID_CONFIDENCE:
        raise ValueError(
            f"scientific_confidence must be one of {sorted(_VALID_CONFIDENCE)}, "
            f"got {scientific_confidence!r}"
        )
    if lifecycle_stage is not None and lifecycle_stage not in _VALID_STAGES:
        raise ValueError(
            f"lifecycle_stage must be one of {sorted(_VALID_STAGES)}, "
            f"got {lifecycle_stage!r}"
        )
    if not experiment_dirs:
        raise ValueError("At least one experiment directory must be specified.")
    if not author.strip():
        raise ValueError("author must not be empty.")

    resolved_root = repo_root or (Path(__file__).parent.parent.parent)

    # Check that every experiment directory exists before touching the registry.
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        if not exp_path.is_absolute():
            exp_path = resolved_root / exp_dir
        if not exp_path.exists():
            raise FileNotFoundError(
                f"Experiment directory not found: {exp_dir!r}\n"
                "Ensure the experiment has completed and the path is correct "
                "before promoting it into the registry."
            )

    registry_root = _registry_root(resolved_root)
    path = _surface_path(registry_root, surface_id)

    data = load_registry_entry(path)

    # Validate existing entry before modifying
    errors = validate_entry(data)
    if errors:
        raise ValueError(
            f"Registry entry at {path} is malformed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # Check lifecycle_stage consistency against stage statuses unless --force.
    if lifecycle_stage is not None and not force and lifecycle_stage in _STAGE_PREREQUISITES:
        stage_key, required_status = _STAGE_PREREQUISITES[lifecycle_stage]
        current_stage_status = (data.get(stage_key) or {}).get("status")
        if current_stage_status != required_status:
            raise ValueError(
                f"Cannot promote to lifecycle_stage={lifecycle_stage!r}: "
                f"{stage_key}.status is {current_stage_status!r}, expected {required_status!r}. "
                f"Complete {stage_key} work before advancing the lifecycle stage, "
                f"or pass force=True (--force on the CLI) to override."
            )

    promoted_at = datetime.now(timezone.utc).isoformat()

    # Build supporting experiment records
    new_experiment_records: list[dict] = []
    for exp_dir in experiment_dirs:
        new_experiment_records.append({
            "experiment_dir": str(exp_dir),
            "promoted_at": promoted_at,
            "promoted_by": author,
            "notes": notes,
        })

    # Append to supporting_experiments
    existing_exps: list[dict] = data.get("supporting_experiments") or []
    existing_exps.extend(new_experiment_records)
    data["supporting_experiments"] = existing_exps

    # Update current scientific assessment
    data["scientific_interest"] = scientific_interest
    data["scientific_confidence"] = scientific_confidence
    data["current_recommendation"] = recommendation

    if lifecycle_stage is not None:
        data["lifecycle_stage"] = lifecycle_stage

    # Append promotion history record
    history_record: dict = {
        "promoted_at": promoted_at,
        "promoted_by": author,
        "lifecycle_stage": data["lifecycle_stage"],
        "scientific_interest": scientific_interest,
        "scientific_confidence": scientific_confidence,
        "recommendation": recommendation,
        "experiments_added": list(experiment_dirs),
        "notes": notes,
    }
    history: list[dict] = data.get("promotion_history") or []
    history.append(history_record)
    data["promotion_history"] = history

    # Also update stage1.characterization_experiments if this is a Stage 1 promotion
    if data.get("lifecycle_stage") == "Characterization":
        stage1 = data.get("stage1") or {}
        char_exps: list[str] = stage1.get("characterization_experiments") or []
        for exp_dir in experiment_dirs:
            if str(exp_dir) not in char_exps:
                char_exps.append(str(exp_dir))
        stage1["characterization_experiments"] = char_exps
        stage1["status"] = "in_progress"
        data["stage1"] = stage1

    # Final validation of updated entry
    errors = validate_entry(data)
    if errors:
        raise ValueError(
            "Updated registry entry failed validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    if dry_run:
        print("--- DRY RUN: updated registry entry (not written) ---")
        print(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))
    else:
        save_registry_entry(path, data)
        print(f"Promoted {len(experiment_dirs)} experiment(s) into {path}")
        print(f"  surface_id:            {surface_id}")
        print(f"  scientific_interest:   {scientific_interest}")
        print(f"  scientific_confidence: {scientific_confidence}")
        print(f"  promoted_at:           {promoted_at}")
        print(f"  promoted_by:           {author}")

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="promote.py",
        description=(
            "Promote experiment evidence into the Behavioral Surface Registry.\n\n"
            "Promotion is an explicit manual step.  Completed experiment reports do\n"
            "not automatically update the registry.  Scientific interpretation\n"
            "therefore remains a deliberate research decision."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--surface",
        required=True,
        help="Surface ID matching the registry YAML filename (e.g. 'reactive_jpy').",
    )
    parser.add_argument(
        "--experiments",
        required=True,
        nargs="+",
        metavar="DIR",
        help="One or more completed experiment output directories to promote.",
    )
    parser.add_argument(
        "--author",
        required=True,
        help="Author identifier (e.g. GitHub username or full name).",
    )
    parser.add_argument(
        "--recommendation",
        required=True,
        help="Updated recommended next research step.",
    )
    parser.add_argument(
        "--scientific-interest",
        required=True,
        choices=sorted(_VALID_INTEREST),
        dest="scientific_interest",
        help="Updated Scientific Interest: low | medium | high.",
    )
    parser.add_argument(
        "--scientific-confidence",
        required=True,
        choices=sorted(_VALID_CONFIDENCE),
        dest="scientific_confidence",
        help="Updated Scientific Confidence: low | medium | high.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Human-readable description of the evidence being promoted.",
    )
    parser.add_argument(
        "--lifecycle-stage",
        dest="lifecycle_stage",
        default=None,
        choices=sorted(_VALID_STAGES),
        help="Updated lifecycle stage (optional; current stage preserved if omitted).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root directory.  Defaults to three levels above this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate and print the updated entry without writing to disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Bypass the lifecycle-stage consistency check.  "
            "Use only when stage statuses have been set out of the normal sequence."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else None
    try:
        promote(
            surface_id=args.surface,
            experiment_dirs=args.experiments,
            author=args.author,
            recommendation=args.recommendation,
            scientific_interest=args.scientific_interest,
            scientific_confidence=args.scientific_confidence,
            notes=args.notes,
            lifecycle_stage=args.lifecycle_stage,
            repo_root=repo_root,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
