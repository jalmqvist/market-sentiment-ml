from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_ERROR_KEYWORDS = frozenset(["error", "fail", "fatal", "corrupt"])
_WARNING_KEYWORDS = frozenset(["warn", "missing", "incomplete", "mismatch"])


def _classify_message(text: str) -> str:
    """Classify a message string as 'error', 'warning', or 'note'."""
    lower = text.lower()
    if any(k in lower for k in _ERROR_KEYWORDS):
        return "error"
    if any(k in lower for k in _WARNING_KEYWORDS):
        return "warning"
    return "note"


def extract_manifest_messages(
    manifest: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Return (notes, warnings, errors) extracted from a manifest dict."""
    notes: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    raw = manifest.get("warnings", [])
    all_messages: list[str] = []
    if isinstance(raw, list):
        all_messages.extend(str(item) for item in raw)
    elif raw:
        all_messages.append(str(raw))

    missing = manifest.get("missing_provenance_counts", {})
    if isinstance(missing, dict):
        missing_fields = [k for k, v in missing.items() if int(v or 0) > 0]
        if missing_fields:
            all_messages.append(f"missing_provenance: {', '.join(sorted(missing_fields))}")

    for msg in all_messages:
        kind = _classify_message(msg)
        if kind == "error":
            errors.append(msg)
        elif kind == "warning":
            warnings.append(msg)
        else:
            notes.append(msg)

    return notes, warnings, errors


def extract_manifest_warnings(manifest: dict) -> list[str]:
    """Legacy helper: return all manifest messages as a flat list.

    Preserved for backwards compatibility with callers that do not yet
    distinguish notes / warnings / errors.
    """
    notes, warnings, errors = extract_manifest_messages(manifest)
    return notes + warnings + errors


def summarize_manifests(manifest_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(manifest_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        provenance = payload.get("provenance", {})
        notes, warnings, errors = extract_manifest_messages(payload)
        all_messages = notes + warnings + errors
        rows.append(
            {
                "manifest_file": path.name,
                "model": identity.get("model"),
                "dl_regime": identity.get("dl_regime"),
                "target_horizon": identity.get("target_horizon"),
                "feature_set": identity.get("feature_set"),
                "dataset_version": provenance.get("dataset_version"),
                "dataset_variant": provenance.get("dataset_variant"),
                "surface_id": provenance.get("surface_id"),
                "state_id": provenance.get("state_id"),
                "notes": " | ".join(notes),
                "warnings": " | ".join(warnings),
                "errors": " | ".join(errors),
                "note_count": len(notes),
                "warning_count": len(warnings),
                "error_count": len(errors),
                # Legacy field: all messages concatenated
                "all_messages": " | ".join(all_messages),
            }
        )
    return pd.DataFrame(rows)
