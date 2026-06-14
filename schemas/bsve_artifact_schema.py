"""Centralized BSVE artifact schema constants, validation, and write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

BSVE_SCHEMA_VERSION = "1.0.0"

BSVE_ENTRY_TIME_COL = "entry_time"
BSVE_AVAILABLE_TS_COL = "prediction_available_timestamp"
BSVE_PAIR_COL = "pair"
BSVE_ENVIRONMENT_COL = "environment_id"
BSVE_STATE_ID_COL = "state_id"
BSVE_STATE_VERSION_COL = "state_version"
BSVE_MATURITY_BARS_COL = "maturity_bars"
BSVE_MATURITY_CLASS_COL = "maturity_class"
BSVE_STATE_CONFIDENCE_COL = "state_confidence"
BSVE_TRANSITION_EVENT_COL = "transition_event"
BSVE_SPEC_ID_COL = "spec_id"
BSVE_CALIBRATION_ID_COL = "calibration_id"

BSVE_REQUIRED_ARTIFACT_COLS = [
    BSVE_ENTRY_TIME_COL,
    BSVE_AVAILABLE_TS_COL,
    BSVE_PAIR_COL,
    BSVE_ENVIRONMENT_COL,
    BSVE_STATE_ID_COL,
    BSVE_STATE_VERSION_COL,
    BSVE_MATURITY_BARS_COL,
    BSVE_MATURITY_CLASS_COL,
    BSVE_STATE_CONFIDENCE_COL,
    BSVE_TRANSITION_EVENT_COL,
    BSVE_SPEC_ID_COL,
    BSVE_CALIBRATION_ID_COL,
]

BSVE_MATURITY_CLASS_VALUES = {"young", "maturing", "mature", "n_a"}
BSVE_TRANSITION_EVENT_VALUES = {
    "entry",
    "continuation",
    "exit_reversal",
    "exit_threshold",
    "exit_late_reversal",
    "exit_unknown",
}

SpecResolver = Callable[[str], bool]
CalibrationResolver = Callable[[str], bool]


def validate_bsve_artifact(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    spec_resolver: SpecResolver | None = None,
    calibration_resolver: CalibrationResolver | None = None,
) -> list[str]:
    """Validate BSVE artifact data against the PR1 output contract."""
    violations: list[str] = []

    def _fail(msg: str) -> None:
        if strict:
            raise ValueError(f"BSVE artifact validation failed: {msg}")
        violations.append(msg)

    missing_cols = [c for c in BSVE_REQUIRED_ARTIFACT_COLS if c not in df.columns]
    if missing_cols:
        _fail(f"missing required columns: {sorted(missing_cols)}")
        if strict:
            return violations

    if metadata is not None:
        sv = metadata.get("schema_version")
        if sv is None:
            _fail("metadata missing 'schema_version'")
        elif sv != BSVE_SCHEMA_VERSION:
            _fail(
                f"schema_version mismatch: expected {BSVE_SCHEMA_VERSION!r}, got {sv!r}"
            )

    for col in BSVE_REQUIRED_ARTIFACT_COLS:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            _fail(f"column '{col}' contains {n_null} null value(s)")

    for col in [BSVE_ENTRY_TIME_COL, BSVE_AVAILABLE_TS_COL]:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], errors="coerce")
        if series.isna().any():
            _fail(f"column '{col}' contains non-datetime value(s)")
            continue
        if series.dt.tz is not None:
            _fail(
                f"column '{col}' must be tz-naive (UTC), but has tz={series.dt.tz!r}"
            )

    if BSVE_ENTRY_TIME_COL in df.columns and BSVE_AVAILABLE_TS_COL in df.columns:
        entry = pd.to_datetime(df[BSVE_ENTRY_TIME_COL], errors="coerce")
        available = pd.to_datetime(df[BSVE_AVAILABLE_TS_COL], errors="coerce")
        mask = entry.notna() & available.notna() & (entry >= available)
        if mask.any():
            n = int(mask.sum())
            _fail(
                f"causal ordering violated: {n} row(s) have entry_time >= "
                "prediction_available_timestamp"
            )

    if BSVE_MATURITY_BARS_COL in df.columns:
        maturity = pd.to_numeric(df[BSVE_MATURITY_BARS_COL], errors="coerce")
        bad = maturity.isna() | (maturity < 0)
        if bad.any():
            _fail("maturity_bars must be numeric and >= 0")

    if BSVE_MATURITY_CLASS_COL in df.columns:
        bad = set(df[BSVE_MATURITY_CLASS_COL].dropna().astype(str)) - BSVE_MATURITY_CLASS_VALUES
        if bad:
            _fail(
                f"maturity_class contains invalid values: {sorted(bad)}; "
                f"allowed={sorted(BSVE_MATURITY_CLASS_VALUES)}"
            )

    if BSVE_STATE_CONFIDENCE_COL in df.columns:
        confidence = pd.to_numeric(df[BSVE_STATE_CONFIDENCE_COL], errors="coerce")
        bad = confidence.isna() | (confidence < 0.0) | (confidence > 1.0)
        if bad.any():
            _fail("state_confidence must be numeric within [0, 1]")

    if BSVE_TRANSITION_EVENT_COL in df.columns:
        bad = set(df[BSVE_TRANSITION_EVENT_COL].dropna().astype(str)) - BSVE_TRANSITION_EVENT_VALUES
        if bad:
            _fail(
                f"transition_event contains invalid values: {sorted(bad)}; "
                f"allowed={sorted(BSVE_TRANSITION_EVENT_VALUES)}"
            )

    key_cols = [BSVE_PAIR_COL, BSVE_ENVIRONMENT_COL, BSVE_ENTRY_TIME_COL]
    if all(c in df.columns for c in key_cols):
        dupes = int(df[key_cols].duplicated().sum())
        if dupes > 0:
            _fail(
                f"found {dupes} duplicate (pair, environment_id, entry_time) row(s)"
            )

        ambiguous = (
            df.groupby(key_cols, dropna=False)[BSVE_STATE_ID_COL]
            .nunique(dropna=False)
            .reset_index(name="state_count")
        )
        if (ambiguous["state_count"] > 1).any():
            _fail("ambiguous state assignments detected for identical key rows")

    if BSVE_SPEC_ID_COL in df.columns:
        ids = sorted({str(v) for v in df[BSVE_SPEC_ID_COL].dropna().unique()})
        if spec_resolver:
            unresolved = [sid for sid in ids if not spec_resolver(sid)]
            if unresolved:
                _fail(f"spec_id not resolvable: {unresolved[:3]}")

    if BSVE_CALIBRATION_ID_COL in df.columns:
        ids = sorted({str(v) for v in df[BSVE_CALIBRATION_ID_COL].dropna().unique()})
        if calibration_resolver:
            unresolved = [cid for cid in ids if not calibration_resolver(cid)]
            if unresolved:
                _fail(f"calibration_id not resolvable: {unresolved[:3]}")

    return violations


def write_bsve_artifact(
    df: pd.DataFrame,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    spec_resolver: SpecResolver | None = None,
    calibration_resolver: CalibrationResolver | None = None,
    artifact_metadata: dict[str, Any] | None = None,
) -> Path:
    """Validate then write BSVE artifact parquet with optional metadata."""
    metadata = metadata or {"schema_version": BSVE_SCHEMA_VERSION}
    validate_bsve_artifact(
        df,
        metadata=metadata,
        strict=strict,
        spec_resolver=spec_resolver,
        calibration_resolver=calibration_resolver,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged_metadata = {
        "schema_version": metadata.get("schema_version", BSVE_SCHEMA_VERSION),
    }
    if artifact_metadata:
        merged_metadata.update({k: str(v) for k, v in artifact_metadata.items()})

    try:
        import pyarrow.parquet as pq

        df.to_parquet(path, index=False)
        table = pq.read_table(path)
        existing = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else str(v)
            )
            for k, v in (table.schema.metadata or {}).items()
        }
        extra = {k: str(v) for k, v in merged_metadata.items()}
        table = table.replace_schema_metadata({**existing, **extra})
        pq.write_table(table, path)
    except Exception:
        df.to_parquet(path, index=False)

    return path
