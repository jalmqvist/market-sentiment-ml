from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def extract_manifest_warnings(manifest: dict) -> list[str]:
    warnings: list[str] = []

    raw = manifest.get("warnings", [])
    if isinstance(raw, list):
        warnings.extend(str(item) for item in raw)
    elif raw:
        warnings.append(str(raw))

    missing = manifest.get("missing_provenance_counts", {})
    if isinstance(missing, dict):
        missing_fields = [k for k, v in missing.items() if int(v or 0) > 0]
        if missing_fields:
            warnings.append(f"missing_provenance: {', '.join(sorted(missing_fields))}")

    return warnings


def summarize_manifests(manifest_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(manifest_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        provenance = payload.get("provenance", {})
        warnings = extract_manifest_warnings(payload)
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
                "warnings": " | ".join(warnings),
                "warning_count": len(warnings),
            }
        )
    return pd.DataFrame(rows)
