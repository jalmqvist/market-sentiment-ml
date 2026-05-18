"""
schemas/dl_artifact_schema.py
==============================
Centralized DL artifact schema constants and validation for the MSML → MPML
interface contract (v2).

This module is the **single source of truth** for:

- Schema version string
- Column name constants (with explicit timestamp semantics)
- ``validate_dl_artifact()`` — fail-fast validator called before writing

Import these constants from here. Do not hardcode string column names anywhere
else in the codebase.

Timestamp semantics (v2 contract)
----------------------------------
Four timestamp columns are exported with exactly one meaning each:

``entry_time`` / ``DL_TIMESTAMP_COL``
    The H1 bar open timestamp being predicted (UTC, tz-naive).
    Backward-compat name: ``entry_time``.  MPML consumers may alias this as
    ``timestamp``.  ``DL_TIMESTAMP_COL = "timestamp"`` is the canonical
    logical name; ``entry_time`` is the physical parquet column.

``prediction_available_timestamp`` / ``DL_AVAILABLE_TS_COL``
    The earliest simulated historical timestamp at which this prediction
    **could have been known**.  Used by MPML for causality checks.
    CONTRACT: ``prediction_available_timestamp <= entry_time`` (i.e. the
    prediction must not post-date the bar it predicts).
    MUST NOT be a wall-clock ``pd.Timestamp.now()`` value.

``prediction_generated_timestamp`` / ``DL_GENERATED_TS_COL``
    Optional per-row wall-clock time at which the prediction was generated.
    For internal diagnostics only.
    MUST NOT be used for causality checks.

``artifact_created_timestamp`` / ``DL_ARTIFACT_CREATED_COL``
    Wall-clock time at which the parquet artifact was created/exported.
    Same value for all rows within one artifact.
    For provenance/auditing only.
    MUST NOT be used for causality checks.

See also
--------
- ``docs/integration/dl_artifact_contract.md`` — full contract specification
- ``docs/integration/DL_SIGNAL_SCHEMA.md``     — schema column reference
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

DL_SCHEMA_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Column name constants
# (single source of truth — import from here, never hardcode strings)
# ---------------------------------------------------------------------------

# The canonical logical name for the market bar timestamp.
# Physical parquet column name: "entry_time" (backward compat).
# MPML consumers alias entry_time → timestamp; see contract docs.
DL_TIMESTAMP_COL = "timestamp"

# Earliest simulated historical timestamp the prediction could have been
# observed.  Used by MPML for causality checks.
# CONTRACT: prediction_available_timestamp <= entry_time (= timestamp).
# MUST NOT be a wall-clock time.
DL_AVAILABLE_TS_COL = "prediction_available_timestamp"

# Optional per-row wall-clock time at which the prediction was generated.
# For internal diagnostics ONLY — MUST NOT be used for causality checks.
DL_GENERATED_TS_COL = "prediction_generated_timestamp"

# Wall-clock time at which the parquet artifact was created/exported.
# Same for all rows within a single artifact.
# For provenance/auditing ONLY — MUST NOT be used for causality checks.
DL_ARTIFACT_CREATED_COL = "artifact_created_timestamp"

# FX pair column (normalized lowercase xxx-yyy).
DL_PAIR_COL = "pair"

# ---------------------------------------------------------------------------
# Required artifact columns (used by validate_dl_artifact)
# ---------------------------------------------------------------------------

_REQUIRED_ARTIFACT_COLS = [
    DL_PAIR_COL,
    "entry_time",
    DL_AVAILABLE_TS_COL,
]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dl_artifact(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
) -> list[str]:
    """
    Validate a DL artifact DataFrame (and optional manifest metadata).

    Call this before writing any artifact to disk.  When ``strict=True``
    (the default) the function raises ``ValueError`` on the first violation
    found, providing fail-fast enforcement on the MSML export side.

    Checks performed
    ----------------
    1. Required columns present (``pair``, ``entry_time``,
       ``prediction_available_timestamp``).
    2. Schema version present and correct (when ``metadata`` is supplied).
    3. Null checks for key columns.
    4. Pair normalization — values must match ``^[a-z]{3}-[a-z]{3}$``.
    5. No duplicate ``(pair, entry_time)`` rows.
    6. ``entry_time`` monotonically non-decreasing within each pair.
    7. Timezone consistency — ``entry_time`` and timestamp columns must be
       tz-naive (UTC).
    8. Causal ordering — ``prediction_available_timestamp <= entry_time``.

    Parameters
    ----------
    df:
        DataFrame to validate.
    metadata:
        Optional manifest / provenance dict.  When supplied,
        ``schema_version`` is verified against ``DL_SCHEMA_VERSION``.
    strict:
        If ``True`` (default) raises ``ValueError`` on the first violation.
        If ``False`` collects all violations and returns them as a list.

    Returns
    -------
    list[str]
        Violation messages.  Empty when all checks pass.

    Raises
    ------
    ValueError
        If ``strict=True`` and any validation check fails.
    """
    violations: list[str] = []

    def _fail(msg: str) -> None:
        if strict:
            raise ValueError(f"DL artifact validation failed: {msg}")
        violations.append(msg)

    # ------------------------------------------------------------------
    # 1. Required columns
    # ------------------------------------------------------------------
    missing_cols = [c for c in _REQUIRED_ARTIFACT_COLS if c not in df.columns]
    if missing_cols:
        _fail(f"missing required columns: {sorted(missing_cols)}")
        if strict:
            return violations  # unreachable; _fail raised

    # ------------------------------------------------------------------
    # 2. Schema version (metadata)
    # ------------------------------------------------------------------
    if metadata is not None:
        sv = metadata.get("schema_version")
        if sv is None:
            _fail("metadata missing 'schema_version'")
        elif sv != DL_SCHEMA_VERSION:
            _fail(
                f"schema_version mismatch: expected {DL_SCHEMA_VERSION!r}, "
                f"got {sv!r}"
            )

    # ------------------------------------------------------------------
    # 3. Null checks for key columns
    # ------------------------------------------------------------------
    for col in [DL_PAIR_COL, "entry_time", DL_AVAILABLE_TS_COL]:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            _fail(f"column '{col}' contains {n_null} null value(s)")

    # ------------------------------------------------------------------
    # 4. Pair normalization (lowercase xxx-yyy)
    # ------------------------------------------------------------------
    if DL_PAIR_COL in df.columns:
        bad = df[DL_PAIR_COL].dropna()
        bad = bad[~bad.str.match(r"^[a-z]{3}-[a-z]{3}$")]
        if len(bad) > 0:
            sample = bad.unique()[:3].tolist()
            _fail(
                f"pair column contains non-normalized values "
                f"(expected lowercase xxx-yyy). Sample: {sample}"
            )

    # ------------------------------------------------------------------
    # 5. Duplicate (pair, entry_time) check
    # ------------------------------------------------------------------
    if DL_PAIR_COL in df.columns and "entry_time" in df.columns:
        dupes = int(df[[DL_PAIR_COL, "entry_time"]].duplicated().sum())
        if dupes > 0:
            _fail(
                f"found {dupes} duplicate (pair, entry_time) row(s); "
                "each (pair, entry_time) must be unique within an artifact"
            )

    # ------------------------------------------------------------------
    # 6. Monotonicity within pair
    # ------------------------------------------------------------------
    if DL_PAIR_COL in df.columns and "entry_time" in df.columns:
        for pair, grp in df.groupby(DL_PAIR_COL):
            et = grp["entry_time"].dropna()
            if len(et) > 1 and not et.is_monotonic_increasing:
                _fail(
                    f"entry_time is not monotonically non-decreasing "
                    f"for pair {pair!r}"
                )

    # ------------------------------------------------------------------
    # 7. Timezone consistency (must be tz-naive)
    # ------------------------------------------------------------------
    for col in ["entry_time", DL_AVAILABLE_TS_COL, DL_GENERATED_TS_COL]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if series.dt.tz is not None:
            _fail(
                f"column '{col}' must be tz-naive (UTC), but has "
                f"tz={series.dt.tz!r}. Strip timezone with "
                ".dt.tz_localize(None) before export."
            )

    # ------------------------------------------------------------------
    # 8. Causal ordering: prediction_available_timestamp <= entry_time
    # ------------------------------------------------------------------
    if DL_AVAILABLE_TS_COL in df.columns and "entry_time" in df.columns:
        avail = df[DL_AVAILABLE_TS_COL]
        bar = df["entry_time"]
        mask = avail.notna() & bar.notna() & (avail > bar)
        if mask.any():
            n = int(mask.sum())
            sample = df.loc[
                mask, [DL_PAIR_COL, "entry_time", DL_AVAILABLE_TS_COL]
            ].head(3)
            _fail(
                f"causal ordering violated: {n} row(s) have "
                f"prediction_available_timestamp > entry_time (timestamp). "
                f"Sample:\n{sample.to_string(index=False)}"
            )

    return violations
