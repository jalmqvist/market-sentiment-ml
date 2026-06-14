# bsve/calibration/validate_calibrations.py
"""
BSVE calibration validation harness.

Loads committed calibration artifacts, verifies their integrity,
checks that all placeholder thresholds have been replaced, and
confirms sign-off conditions are met before a canonical validation
run is permitted to proceed.

This is the gate between Step 1 (calibration) and Step 2 (dry run)
in the BSVE validation protocol.

Usage:
    python -m bsve.calibration.validate_calibrations \
        --calibration-dir bsve/calibrations \
        --environment reactive_jpy reactive_chf
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationValidationReport:
    environment_id: str
    artifact_path: str
    hash_verified: bool
    placeholders_resolved: bool
    sign_off_conditions_met: bool
    warnings: list[str]
    errors: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.hash_verified
            and self.placeholders_resolved
            and self.sign_off_conditions_met
            and len(self.errors) == 0
        )


# ---------------------------------------------------------------------------
# Sign-off conditions per environment
# ---------------------------------------------------------------------------

# These encode the expected direction of findings from RESEARCH_STATE.md.
# A calibration artifact that contradicts these conditions should not
# be committed — it means either the data doesn't support the hypothesis
# or the calibration procedure has a bug.

JPY_SIGN_OFF_CONDITIONS = {
    "reversal_rate_young_gt_mature": {
        "description": "Reversal rate in young states > mature states",
        "check": lambda r: r["reversal_rate_young"] > r["reversal_rate_mature"],
        "expected": "reversal_rate_young > reversal_rate_mature",
    },
    "young_boundary_lt_mature_boundary": {
        "description": "Young boundary < mature boundary",
        "check": lambda r: r["young_boundary_bars"] < r["mature_boundary_bars"],
        "expected": "young_boundary_bars < mature_boundary_bars",
    },
    "sufficient_episodes": {
        "description": "At least 50 episodes for reliable calibration",
        "check": lambda r: r["n_episodes_total"] >= 50,
        "expected": "n_episodes_total >= 50",
    },
    "censoring_rate_acceptable": {
        "description": "Censoring rate below 30%",
        "check": lambda r: r["censoring_rate"] < 0.30,
        "expected": "censoring_rate < 0.30",
    },
    "reversal_rate_young_meaningful": {
        "description": "Young reversal rate is non-trivial (> 15%)",
        "check": lambda r: r["reversal_rate_young"] > 0.15,
        "expected": "reversal_rate_young > 0.15",
    },
}

CHF_SIGN_OFF_CONDITIONS = {
    "low_threshold_lt_high_threshold": {
        "description": "Low vol threshold < high vol threshold",
        "check": lambda r: (
            r["low_vol_threshold_atr_pct"] < r["high_vol_threshold_atr_pct"]
        ),
        "expected": "low_vol_threshold_atr_pct < high_vol_threshold_atr_pct",
    },
    "persistence_low_gt_high": {
        "description": "Median persistence in low vol > high vol",
        "check": lambda r: (
            r["median_persistence_low_vol"] > r["median_persistence_high_vol"]
        ),
        "expected": "median_persistence_low_vol > median_persistence_high_vol",
    },
    "persistence_ratio_meaningful": {
        "description": "Low vol persistence at least 25% longer than high vol",
        "check": lambda r: (
            r["median_persistence_high_vol"] > 0
            and r["median_persistence_low_vol"]
            / r["median_persistence_high_vol"] >= 1.25
        ),
        "expected": "persistence_ratio >= 1.25",
    },
    "ks_pvalue_significant": {
        "description": "KS test rejects equal persistence distributions",
        "check": lambda r: r["ks_pvalue_low_vs_high"] < 0.05,
        "expected": "ks_pvalue_low_vs_high < 0.05",
    },
    "effect_size_meaningful": {
        "description": "Cohen's d effect size >= 0.3",
        "check": lambda r: abs(r["effect_size_low_vs_high"]) >= 0.3,
        "expected": "abs(effect_size_low_vs_high) >= 0.3",
    },
    "pair_threshold_agreement": {
        "description": "USDCHF and EURCHF thresholds are consistent",
        "check": lambda r: r["pair_threshold_agreement"] is True,
        "expected": "pair_threshold_agreement == True",
    },
    "regime_distribution_balanced": {
        "description": "No single regime captures more than 70% of bars",
        "check": lambda r: all(
            count / max(
                r["n_low_vol_bars"]
                + r["n_medium_vol_bars"]
                + r["n_high_vol_bars"], 1
            ) <= 0.70
            for count in [
                r["n_low_vol_bars"],
                r["n_medium_vol_bars"],
                r["n_high_vol_bars"],
            ]
        ),
        "expected": "each regime <= 70% of total bars",
    },
}

SIGN_OFF_CONDITIONS = {
    "reactive_jpy": JPY_SIGN_OFF_CONDITIONS,
    "reactive_chf": CHF_SIGN_OFF_CONDITIONS,
}


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------

def verify_artifact_hash(artifact: dict) -> bool:
    """
    Recompute the calibration hash and verify it matches the stored value.

    The hash is computed over all fields except calibration_hash itself,
    using the same deterministic JSON serialization used at write time.
    """
    stored_hash = artifact.get("calibration_hash")
    if not stored_hash:
        return False

    payload = {k: v for k, v in artifact.items() if k != "calibration_hash"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

    return recomputed == stored_hash


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

# These are the sentinel values written into environment specs before
# empirical calibration. Any artifact that still contains them has not
# been properly calibrated.
PLACEHOLDER_SENTINELS = {
    "extreme_threshold_net_pct": None,
    "young_boundary_bars": None,
    "mature_boundary_bars": None,
    "low_vol_threshold_atr_pct": None,
    "high_vol_threshold_atr_pct": None,
}

# Fields that must be present and non-null in a committed artifact
REQUIRED_FIELDS = {
    "reactive_jpy": [
        "environment_id",
        "calibration_version",
        "dataset_version",
        "pairs",
        "calibration_window_start",
        "calibration_window_end",
        "extreme_threshold_net_pct",
        "young_boundary_bars",
        "mature_boundary_bars",
        "n_episodes_total",
        "reversal_rate_young",
        "reversal_rate_mature",
        "hazard_crossover_bar",
        "censoring_rate",
        "calibration_hash",
    ],
    "reactive_chf": [
        "environment_id",
        "calibration_version",
        "dataset_version",
        "pairs",
        "calibration_window_start",
        "calibration_window_end",
        "low_vol_threshold_atr_pct",
        "high_vol_threshold_atr_pct",
        "pair_threshold_agreement",
        "n_low_vol_bars",
        "n_medium_vol_bars",
        "n_high_vol_bars",
        "median_persistence_low_vol",
        "median_persistence_high_vol",
        "ks_pvalue_low_vs_high",
        "effect_size_low_vs_high",
        "calibration_hash",
    ],
}


def check_placeholders_resolved(
    artifact: dict,
    environment_id: str,
) -> tuple[bool, list[str]]:
    """
    Verify all required fields are present and non-null.

    Returns:
        (all_resolved, list_of_unresolved_fields)
    """
    required = REQUIRED_FIELDS.get(environment_id, [])
    unresolved = []

    for field in required:
        value = artifact.get(field)
        if value is None:
            unresolved.append(f"{field} is null")
        elif isinstance(value, float) and value != value:
            # NaN check
            unresolved.append(f"{field} is NaN")

    return len(unresolved) == 0, unresolved


# ---------------------------------------------------------------------------
# Per-environment validation
# ---------------------------------------------------------------------------

def validate_calibration_artifact(
    artifact_path: Path,
    environment_id: str,
) -> CalibrationValidationReport:
    """
    Full validation of a single calibration artifact.

    Checks:
        1. File exists and is valid JSON
        2. environment_id matches expected
        3. Hash integrity verified
        4. All required fields present and non-null
        5. All sign-off conditions pass

    Args:
        artifact_path: Path to the calibration JSON artifact.
        environment_id: Expected environment ('reactive_jpy' or 'reactive_chf').

    Returns:
        CalibrationValidationReport
    """
    warnings_list = []
    errors_list = []

    # Load artifact
    if not artifact_path.exists():
        return CalibrationValidationReport(
            environment_id=environment_id,
            artifact_path=str(artifact_path),
            hash_verified=False,
            placeholders_resolved=False,
            sign_off_conditions_met=False,
            warnings=[],
            errors=[f"Artifact not found: {artifact_path}"],
        )

    try:
        with open(artifact_path) as f:
            artifact = json.load(f)
    except json.JSONDecodeError as e:
        return CalibrationValidationReport(
            environment_id=environment_id,
            artifact_path=str(artifact_path),
            hash_verified=False,
            placeholders_resolved=False,
            sign_off_conditions_met=False,
            warnings=[],
            errors=[f"Invalid JSON: {e}"],
        )

    # Check environment_id matches
    stored_env = artifact.get("environment_id")
    if stored_env != environment_id:
        errors_list.append(
            f"environment_id mismatch: expected '{environment_id}', "
            f"got '{stored_env}'"
        )

    # SNB era warning for CHF
    if environment_id == "reactive_chf" and artifact.get("snb_era_flag"):
        warnings_list.append(
            "SNB era flag is set. Calibration window includes pre-2015 "
            "EURCHF data. Verify thresholds are not distorted by the "
            "SNB floor removal event."
        )

    # Hash verification
    hash_ok = verify_artifact_hash(artifact)
    if not hash_ok:
        errors_list.append(
            "Hash verification failed. Artifact may have been "
            "manually edited after calibration. Re-run calibration "
            "to generate a fresh artifact."
        )

    # Placeholder check
    placeholders_ok, unresolved = check_placeholders_resolved(
        artifact, environment_id
    )
    if not placeholders_ok:
        for field in unresolved:
            errors_list.append(f"Unresolved placeholder: {field}")

    # Sign-off conditions
    conditions = SIGN_OFF_CONDITIONS.get(environment_id, {})
    sign_off_failures = []

    for condition_id, condition in conditions.items():
        try:
            passed = condition["check"](artifact)
        except Exception as e:
            passed = False
            warnings_list.append(
                f"Sign-off condition '{condition_id}' raised an "
                f"exception: {e}"
            )

        if not passed:
            sign_off_failures.append(
                f"FAILED: {condition['description']} "
                f"(expected: {condition['expected']})"
            )

    sign_off_ok = len(sign_off_failures) == 0
    if not sign_off_ok:
        errors_list.extend(sign_off_failures)

    return CalibrationValidationReport(
        environment_id=environment_id,
        artifact_path=str(artifact_path),
        hash_verified=hash_ok,
        placeholders_resolved=placeholders_ok,
        sign_off_conditions_met=sign_off_ok,
        warnings=warnings_list,
        errors=errors_list,
    )


# ---------------------------------------------------------------------------
# Multi-environment runner
# ---------------------------------------------------------------------------

def validate_all_calibrations(
    calibration_dir: Path,
    environments: list[str],
    calibration_version: str = "v1",
) -> dict[str, CalibrationValidationReport]:
    """
    Validate calibration artifacts for all requested environments.

    Expects artifacts named:
        <environment_id>_calibration_<version>.json

    Args:
        calibration_dir: Directory containing calibration artifacts.
        environments: List of environment IDs to validate.
        calibration_version: Version tag to look for.

    Returns:
        Dict of environment_id → CalibrationValidationReport
    """
    reports = {}

    for env_id in environments:
        artifact_path = (
            calibration_dir
            / f"{env_id}_calibration_{calibration_version}.json"
        )
        report = validate_calibration_artifact(artifact_path, env_id)
        reports[env_id] = report

    return reports


def print_validation_summary(
    reports: dict[str, CalibrationValidationReport],
) -> bool:
    """
    Print a human-readable validation summary.

    Returns:
        True if all environments passed, False otherwise.
    """
    all_passed = True
    separator = "-" * 60

    print("\n[BSVE] Calibration Validation Summary")
    print(separator)

    for env_id, report in reports.items():
        status = "✓ PASSED" if report.passed else "✗ FAILED"
        print(f"\n{status}  {env_id}")
        print(f"  artifact : {report.artifact_path}")
        print(f"  hash ok  : {report.hash_verified}")
        print(f"  fields ok: {report.placeholders_resolved}")
        print(f"  sign-off : {report.sign_off_conditions_met}")

        if report.warnings:
            for w in report.warnings:
                print(f"  ⚠ {w}")

        if report.errors:
            for e in report.errors:
                print(f"  ✗ {e}")
            all_passed = False

    print(f"\n{separator}")
    if all_passed:
        print(
            "[BSVE] All calibrations validated. "
            "Proceed to Step 2: state machine dry run.\n"
        )
    else:
        print(
            "[BSVE] Validation FAILED. Do not proceed to dry run.\n"
            "Resolve all errors above and re-run calibration.\n"
        )

    return all_passed


def assert_calibrations_valid(
    calibration_dir: Path,
    environments: list[str],
    calibration_version: str = "v1",
) -> dict[str, dict]:
    """
    Gate function for use at the start of a BSVE validation run.

    Loads and validates all calibration artifacts. Raises RuntimeError
    if any environment fails validation, preventing the validation run
    from proceeding with uncommitted or invalid thresholds.

    Args:
        calibration_dir: Directory containing calibration artifacts.
        environments: List of environment IDs required for this run.
        calibration_version: Version tag to look for.

    Returns:
        Dict of environment_id → raw artifact dict (for threshold injection
        into state machine at runtime).

    Raises:
        RuntimeError: If any calibration fails validation.
    """
    reports = validate_all_calibrations(
        calibration_dir, environments, calibration_version
    )
    all_passed = print_validation_summary(reports)

    if not all_passed:
        failed = [env for env, r in reports.items() if not r.passed]
        raise RuntimeError(
            f"[BSVE] Calibration validation failed for: {failed}. "
            "Cannot proceed with validation run. "
            "Re-run calibration and resolve all sign-off failures."
        )

    # Load and return raw artifacts for threshold injection
    artifacts = {}
    for env_id in environments:
        artifact_path = (
            calibration_dir
            / f"{env_id}_calibration_{calibration_version}.json"
        )
        with open(artifact_path) as f:
            artifacts[env_id] = json.load(f)

    return artifacts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="BSVE calibration validation harness"
    )
    parser.add_argument(
        "--calibration-dir",
        default="bsve/calibrations",
    )
    parser.add_argument(
        "--environment",
        nargs="+",
        default=["reactive_jpy", "reactive_chf"],
        dest="environments",
    )
    parser.add_argument(
        "--calibration-version",
        default="v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reports = validate_all_calibrations(
        calibration_dir=Path(args.calibration_dir),
        environments=args.environments,
        calibration_version=args.calibration_version,
    )
    passed = print_validation_summary(reports)
    raise SystemExit(0 if passed else 1)
