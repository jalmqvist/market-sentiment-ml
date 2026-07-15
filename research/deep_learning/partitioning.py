from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import config as cfg


def validate_partition_args(
    regime: str | None,
    surface: str | None,
    state: str | None,
) -> None:
    has_regime = regime is not None
    has_surface = surface is not None
    has_state = state is not None

    if has_regime and (has_surface or has_state):
        provided = ["--regime"]
        if has_surface:
            provided.append("--surface")
        if has_state:
            provided.append("--state")
        raise ValueError(
            "Conflicting partition arguments: "
            f"{', '.join(provided)}. "
            "Use either --regime or both --surface and --state."
        )

    if has_surface and not has_state:
        raise ValueError(
            "Missing required argument: --state must be provided when --surface is set."
        )

    if has_state and not has_surface:
        raise ValueError(
            "Missing required argument: --surface must be provided when --state is set."
        )


def resolve_partition(regime: str | None, surface: str | None, state: str | None) -> dict[str, Any]:
    if regime is not None:
        return {
            "mode": "regime",
            "regime": regime,
            "surface": "trend_vol",
            "state": regime,
            "surface_version": None,
            "dl_regime": regime,
            "log_tag": regime.strip().lower(),
        }
    if surface is not None and state is not None:
        return {
            "mode": "behavioral",
            "regime": None,
            "surface": surface,
            "state": state,
            "surface_version": None,
            "dl_regime": f"{surface}:{state}",
            "log_tag": f"{surface.strip().lower()}-{state.strip().lower()}",
        }
    return {
        "mode": "none",
        "regime": None,
        "surface": "trend_vol",
        "state": "MIXED",
        "surface_version": None,
        "dl_regime": "MIXED",
        "log_tag": "all",
    }


def apply_partition_filter(df: pd.DataFrame, partition: dict[str, Any]) -> pd.DataFrame:
    if partition["mode"] == "regime":
        return df[df["regime"] == partition["regime"]]
    if partition["mode"] == "behavioral":
        return df[
            (df["surface_id"] == partition["surface"]) &
            (df["state_id"] == partition["state"])
        ]
    return df


def resolve_behavioral_provenance(
    df: pd.DataFrame,
    dataset_version: str,
    dataset_variant: str,
    *,
    selected_surface_id: str,
    selected_state_id: str,
) -> dict[str, Any]:
    ontology_version: str | None = None

    if "ontology_version" in df.columns:
        non_null = df["ontology_version"].dropna()
        if not non_null.empty:
            ontology_version = str(non_null.iloc[0])

    if ontology_version is None:
        ontology_version = _read_ontology_version_from_manifests(
            dataset_version=dataset_version,
            surface_id=selected_surface_id,
        )

    return {
        "surface_id": selected_surface_id,
        "surface_version": ontology_version,
        "state_id": selected_state_id,
        "dataset_variant": dataset_variant,
        "dataset_version": dataset_version,
        "ontology_version": ontology_version,
    }


def _read_ontology_version_from_manifests(
    *,
    dataset_version: str,
    surface_id: str,
) -> str | None:
    manifest_dir = Path(cfg.OUTPUT_DIR) / dataset_version
    if not manifest_dir.exists():
        return None

    for path in sorted(manifest_dir.glob("DATASET_MANIFEST*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        behavioral_surface = manifest.get("behavioral_surface")
        if not isinstance(behavioral_surface, dict):
            continue

        ontology_id = behavioral_surface.get("ontology_id")
        if ontology_id is not None and str(ontology_id) != str(surface_id):
            continue

        ontology_version = behavioral_surface.get("ontology_version")
        if ontology_version is not None:
            return str(ontology_version)

    return None
