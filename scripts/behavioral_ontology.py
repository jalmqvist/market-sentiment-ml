"""
behavioral_ontology.py
======================
Single authoritative source for Behavioral Surface ontology definitions
within the producer pipeline.

Defines the canonical set of known Behavioral Surfaces and their associated
Behavioral States, as documented in:

    docs/behavioral/behavioral_surface_contract.md

This module is the single source of truth for behavioral identity validation
within the producer pipeline.  Do not duplicate these definitions elsewhere.

Usage::

    from behavioral_ontology import (
        BEHAVIORAL_SURFACES,
        is_known_surface,
        is_known_state,
        validate_behavioral_identity,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Behavioral Surface Registry
# Derived from docs/behavioral/behavioral_surface_contract.md
# ---------------------------------------------------------------------------

# Maps surface_id → frozenset of canonical state_ids for that surface.
# Includes legacy compatibility aliases where documented in the contract.
BEHAVIORAL_SURFACES: dict[str, frozenset[str]] = {
    "trend_vol": frozenset(
        {
            "LVTF",   # Low-volatility trend-following
            "HVTF",   # High-volatility trend-following
            "LVR",    # Low-volatility ranging
            "HVR",    # High-volatility ranging
            # Legacy aliases (contract §Compatibility)
            "LVMR",   # compatibility alias → LVR
            "HVMR",   # compatibility alias → HVR
            "MIXED",  # multi-surface consolidation placeholder
        }
    ),
    "reactive_jpy": frozenset(
        {
            "JPY_NON_EXTREME",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURING",
            "JPY_CONSENSUS_MATURE",
        }
    ),
}

# The "unknown" sentinel is a valid passthrough value; never flagged as an
# error by validation.
_UNKNOWN_SENTINEL = "unknown"


def is_known_surface(surface_id: str) -> bool:
    """Return ``True`` if *surface_id* is registered in the ontology."""
    return surface_id == _UNKNOWN_SENTINEL or surface_id in BEHAVIORAL_SURFACES


def is_known_state(surface_id: str, state_id: str) -> bool:
    """Return ``True`` if *state_id* is valid for *surface_id*."""
    if state_id == _UNKNOWN_SENTINEL or surface_id == _UNKNOWN_SENTINEL:
        return True
    states = BEHAVIORAL_SURFACES.get(surface_id)
    if states is None:
        return False
    return state_id in states


def validate_behavioral_identity(
    surface_id: str,
    state_id: str,
) -> list[str]:
    """
    Validate a ``(surface_id, state_id)`` pair against the ontology.

    Returns a list of warning strings.  An empty list means the identity is
    consistent with the registered ontology.

    Parameters
    ----------
    surface_id:
        Behavioral Surface identifier, e.g. ``"trend_vol"`` or
        ``"reactive_jpy"``.
    state_id:
        Behavioral State identifier, e.g. ``"LVTF"`` or
        ``"JPY_CONSENSUS_YOUNG"``.

    Returns
    -------
    list[str]
        Warning messages.  Empty if the identity is valid.
    """
    issues: list[str] = []
    if not is_known_surface(surface_id):
        issues.append(
            f"surface_id={surface_id!r} is not in the Behavioral Surface Registry. "
            "If this is a new surface, register it in "
            "docs/behavioral/behavioral_surface_contract.md."
        )
    elif not is_known_state(surface_id, state_id):
        issues.append(
            f"state_id={state_id!r} is not a known state for surface "
            f"{surface_id!r}. "
            "If this is a new state, register it in "
            "docs/behavioral/behavioral_surface_contract.md."
        )
    return issues
