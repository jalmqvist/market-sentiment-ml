"""BSVE artifact validation/writing wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from schemas.bsve_artifact_schema import (
    BSVE_SCHEMA_VERSION,
    validate_bsve_artifact as _validate_bsve_artifact,
    write_bsve_artifact as _write_bsve_artifact,
)


def validate_bsve_artifact(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    spec_resolver=None,
    calibration_resolver=None,
) -> list[str]:
    """Validate a BSVE artifact DataFrame using the centralized schema."""
    metadata = metadata or {"schema_version": BSVE_SCHEMA_VERSION}
    return _validate_bsve_artifact(
        df,
        metadata=metadata,
        strict=strict,
        spec_resolver=spec_resolver,
        calibration_resolver=calibration_resolver,
    )


def write_bsve_artifact(
    df: pd.DataFrame,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    spec_resolver=None,
    calibration_resolver=None,
    artifact_metadata: dict[str, Any] | None = None,
) -> Path:
    """Validate and write a BSVE artifact parquet."""
    metadata = metadata or {"schema_version": BSVE_SCHEMA_VERSION}
    return _write_bsve_artifact(
        df,
        output_path,
        metadata=metadata,
        strict=strict,
        spec_resolver=spec_resolver,
        calibration_resolver=calibration_resolver,
        artifact_metadata=artifact_metadata,
    )
