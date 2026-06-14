"""
BSVE Calibration Artifact Contract.

Defines the calibration artifact schema, required metadata, validation
helpers, and load/write utilities.

Null calibrations are first-class artifacts.  A missing file must never
be used to represent calibration failure — every calibration run writes
an artifact, regardless of outcome.

Artifact outcomes:
    "success" — thresholds were estimated and stored.
    "null"    — calibration could not produce thresholds (e.g., insufficient
                data, hypothesis not supported).  Downstream consumers must
                explicitly handle null artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

CALIBRATION_SCHEMA_VERSION = "1.0.0"

CalibrationOutcome = Literal["success", "null"]

# Required top-level metadata keys for every calibration artifact.
REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "calibration_id",
        "ontology_id",
        "ontology_version",
        "calibration_timestamp",
        "calibration_window_start",
        "calibration_window_end",
        "dataset_version",
        "calibration_method",
        "artifact_hash",
        "outcome",
    }
)


# ---------------------------------------------------------------------------
# Artifact type alias
# ---------------------------------------------------------------------------

# A CalibrationArtifact is a plain dict so it can be directly serialised to
# JSON without custom encoders.  Typed helpers (below) enforce the schema.
CalibrationArtifact = dict[str, Any]


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _compute_artifact_hash(artifact: CalibrationArtifact) -> str:
    """Compute SHA-256 over all fields except *artifact_hash* itself."""
    payload = {k: v for k, v in artifact.items() if k != "artifact_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _verify_artifact_hash(artifact: CalibrationArtifact) -> bool:
    """Return True if the stored *artifact_hash* matches recomputed value."""
    stored = artifact.get("artifact_hash")
    if not stored:
        return False
    return stored == _compute_artifact_hash(artifact)


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------


def build_calibration_artifact(
    *,
    calibration_id: str,
    ontology_id: str,
    ontology_version: str,
    calibration_window_start: str,
    calibration_window_end: str,
    dataset_version: str,
    calibration_method: str,
    outcome: CalibrationOutcome,
    thresholds: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    null_reason: str | None = None,
) -> CalibrationArtifact:
    """
    Build a complete calibration artifact dict with hash.

    Args:
        calibration_id: Unique identifier for this calibration run.
        ontology_id: Identifier of the behavioral ontology (e.g. ``reactive_jpy``).
        ontology_version: Version string of the ontology spec used.
        calibration_window_start: ISO-8601 date/datetime of calibration start.
        calibration_window_end: ISO-8601 date/datetime of calibration end.
        dataset_version: Version of the master research dataset consumed.
        calibration_method: Short label for the calibration algorithm used.
        outcome: ``"success"`` or ``"null"``.
        thresholds: Ontology-specific threshold values (may be empty for null).
        diagnostics: Optional diagnostic statistics to store alongside thresholds.
        null_reason: Human-readable explanation when ``outcome == "null"``.

    Returns:
        Artifact dict ready for :func:`write_calibration_artifact`.
    """
    artifact: CalibrationArtifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_id": calibration_id,
        "ontology_id": ontology_id,
        "ontology_version": ontology_version,
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "calibration_window_start": calibration_window_start,
        "calibration_window_end": calibration_window_end,
        "dataset_version": dataset_version,
        "calibration_method": calibration_method,
        "outcome": outcome,
        "thresholds": thresholds or {},
        "diagnostics": diagnostics or {},
    }

    if outcome == "null":
        artifact["null_reason"] = null_reason or "unspecified"

    # Compute and attach hash last.
    artifact["artifact_hash"] = _compute_artifact_hash(artifact)
    return artifact


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_calibration_artifact(
    artifact: CalibrationArtifact,
    *,
    strict: bool = True,
    evaluation_window_start: str | None = None,
    evaluation_window_end: str | None = None,
) -> list[str]:
    """
    Validate a calibration artifact against the PR2 contract.

    Checks performed:
    * All required metadata keys are present.
    * ``schema_version`` matches :data:`CALIBRATION_SCHEMA_VERSION`.
    * ``outcome`` is one of ``"success"`` or ``"null"``.
    * ``artifact_hash`` is present and verifies correctly.
    * Calibration window is a valid date range (start < end).
    * If evaluation window args are provided, calibration and evaluation
      windows must not overlap.
    * For ``"success"`` artifacts: ``thresholds`` must be a non-empty dict.
    * For ``"null"`` artifacts: ``null_reason`` must be present.

    Args:
        artifact: Artifact dict to validate.
        strict: When True, raise :class:`ValueError` on the first violation.
                When False, collect all violations and return them.
        evaluation_window_start: Optional ISO-8601 start of evaluation window.
        evaluation_window_end: Optional ISO-8601 end of evaluation window.

    Returns:
        List of violation messages (empty when valid).

    Raises:
        ValueError: If *strict* is True and any violation is found.
    """
    violations: list[str] = []

    def _fail(msg: str) -> None:
        if strict:
            raise ValueError(f"CalibrationArtifact validation failed: {msg}")
        violations.append(msg)

    # --- Required keys ---
    missing = REQUIRED_METADATA_KEYS - set(artifact.keys())
    if missing:
        _fail(f"missing required metadata keys: {sorted(missing)}")
        if strict:
            return violations

    # --- Schema version ---
    sv = artifact.get("schema_version")
    if sv != CALIBRATION_SCHEMA_VERSION:
        _fail(
            f"schema_version mismatch: expected {CALIBRATION_SCHEMA_VERSION!r}, "
            f"got {sv!r}"
        )

    # --- Outcome ---
    outcome = artifact.get("outcome")
    if outcome not in ("success", "null"):
        _fail(f"outcome must be 'success' or 'null', got {outcome!r}")

    # --- Hash integrity ---
    if not _verify_artifact_hash(artifact):
        _fail(
            "artifact_hash verification failed — artifact may have been modified "
            "after creation"
        )

    # --- Calibration window ---
    cal_start_str = artifact.get("calibration_window_start", "")
    cal_end_str = artifact.get("calibration_window_end", "")
    cal_start = _parse_date_str(cal_start_str)
    cal_end = _parse_date_str(cal_end_str)

    if cal_start is None:
        _fail(
            f"calibration_window_start is not a valid date/datetime: {cal_start_str!r}"
        )
    if cal_end is None:
        _fail(
            f"calibration_window_end is not a valid date/datetime: {cal_end_str!r}"
        )
    if cal_start is not None and cal_end is not None and cal_start >= cal_end:
        _fail(
            "calibration_window_start must be before calibration_window_end"
        )

    # --- Calibration/evaluation window overlap ---
    if evaluation_window_start is not None and evaluation_window_end is not None:
        eval_start = _parse_date_str(evaluation_window_start)
        eval_end = _parse_date_str(evaluation_window_end)
        if (
            cal_start is not None
            and cal_end is not None
            and eval_start is not None
            and eval_end is not None
        ):
            if _windows_overlap(cal_start, cal_end, eval_start, eval_end):
                _fail(
                    "calibration and evaluation windows must not overlap: "
                    f"calibration=[{cal_start_str}, {cal_end_str}], "
                    f"evaluation=[{evaluation_window_start}, {evaluation_window_end}]"
                )

    # --- Outcome-specific rules ---
    if outcome == "success":
        thresholds = artifact.get("thresholds")
        if not isinstance(thresholds, dict) or not thresholds:
            _fail(
                "outcome is 'success' but 'thresholds' is missing or empty — "
                "successful calibrations must record threshold values"
            )

    if outcome == "null":
        if not artifact.get("null_reason"):
            _fail(
                "outcome is 'null' but 'null_reason' is missing or empty"
            )

    return violations


# ---------------------------------------------------------------------------
# Load / write
# ---------------------------------------------------------------------------


def load_calibration_artifact(
    path: str | Path,
    *,
    strict: bool = True,
) -> CalibrationArtifact:
    """
    Load and validate a calibration artifact from a JSON file.

    Args:
        path: Path to the JSON artifact file.
        strict: Passed through to :func:`validate_calibration_artifact`.

    Returns:
        Validated artifact dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON is malformed or the artifact is invalid
                    (when *strict* is True).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration artifact not found: {path}")

    try:
        artifact: CalibrationArtifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in calibration artifact {path}: {exc}") from exc

    validate_calibration_artifact(artifact, strict=strict)
    return artifact


def write_calibration_artifact(
    artifact: CalibrationArtifact,
    path: str | Path,
    *,
    strict: bool = True,
) -> Path:
    """
    Validate and write a calibration artifact to a JSON file.

    The parent directory is created if it does not exist.

    Args:
        artifact: Artifact dict produced by :func:`build_calibration_artifact`.
        path: Destination path for the JSON file.
        strict: Passed through to :func:`validate_calibration_artifact`.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If the artifact is invalid (when *strict* is True).
    """
    validate_calibration_artifact(artifact, strict=strict)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date_str(value: str) -> datetime | None:
    """Try parsing an ISO-8601 date or datetime string; return None on failure."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _windows_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> bool:
    """Return True if two half-open intervals [a_start, a_end) and [b_start, b_end) overlap."""
    return a_start < b_end and b_start < a_end
