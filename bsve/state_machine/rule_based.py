"""Thin orchestration entrypoint for Behavioral Surface Generation (PR4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter
from bsve.calibration.calibration_contract import load_calibration_artifact
from bsve.calibration.calibration_runner import load_state_spec
from bsve.state_machine.engine import (
    build_behavioral_surface_manifest,
    generate_behavioral_surface,
)
from bsve.state_machine.plugins import ReactiveJPYPlugin

_DEFAULT_CALIBRATION_ARTIFACT = Path("bsve/calibration_artifacts/reactive_jpy_calibration_v1.json")
_DEFAULT_STATE_SPEC = Path("bsve/state_specs/reactive_jpy_v1.yaml")


def _resolve_pairs(
    *,
    adapter: MasterResearchDatasetAdapter,
    spec: dict[str, Any],
    pairs: list[str] | None,
) -> list[str]:
    if pairs:
        resolved = [adapter.normalize_pair(p) for p in pairs]
    else:
        from_spec = spec.get("environment", {}).get("pairs", [])
        resolved = [adapter.normalize_pair(p) for p in from_spec]

    if not resolved:
        raise ValueError("No pairs configured. Pass --pairs or add pairs to state spec.")
    return sorted(set(resolved))


def run_behavioral_surface_pipeline(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    calibration_artifact_path: str | Path = _DEFAULT_CALIBRATION_ARTIFACT,
    state_spec_path: str | Path = _DEFAULT_STATE_SPEC,
    dataset_version: str = "unknown",
    pairs: list[str] | None = None,
) -> tuple[Path, Path]:
    """Run PR4 pipeline: load calibration+dataset, generate surface, export surface+manifest."""
    artifact = load_calibration_artifact(calibration_artifact_path, strict=True)
    spec = load_state_spec(state_spec_path)

    adapter = MasterResearchDatasetAdapter.from_artifact(dataset_path)
    resolved_pairs = _resolve_pairs(adapter=adapter, spec=spec, pairs=pairs)

    ds = adapter.get_sentiment_observations(
        pairs=resolved_pairs,
        columns=["net_sentiment", "crowd_side"],
    )

    if ds.empty:
        raise ValueError(f"No dataset rows available for pairs: {resolved_pairs}")

    plugin = ReactiveJPYPlugin()
    surface = generate_behavioral_surface(
        ds,
        plugin=plugin,
        calibration_artifact=artifact,
        dataset_version=dataset_version,
        pair_col=adapter.config.pair_col,
        timestamp_col=adapter.config.timestamp_col,
        crowd_side_col="crowd_side",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = surface.attrs.get("provenance", {})
    ontology_id = provenance.get("ontology_id", "ontology")
    ontology_version = provenance.get("ontology_version", "1.0.0")

    surface_path = output_dir / f"behavioral_surface_{ontology_id}_{ontology_version}.parquet"
    surface.to_parquet(surface_path, index=False)

    manifest = build_behavioral_surface_manifest(surface)
    manifest_path = output_dir / "behavioral_surface_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return surface_path, manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BSVE Behavioral Surface Generator (deterministic, causal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", required=True, help="Path to dataset CSV/parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs")
    parser.add_argument(
        "--calibration-artifact",
        default=str(_DEFAULT_CALIBRATION_ARTIFACT),
        help="Validated calibration artifact JSON path",
    )
    parser.add_argument(
        "--state-spec",
        default=str(_DEFAULT_STATE_SPEC),
        help="Ontology YAML path",
    )
    parser.add_argument(
        "--dataset-version",
        default="unknown",
        help="Dataset version for provenance",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional pair subset",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    surface_path, manifest_path = run_behavioral_surface_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        calibration_artifact_path=args.calibration_artifact,
        state_spec_path=args.state_spec,
        dataset_version=args.dataset_version,
        pairs=args.pairs,
    )
    print(f"[BSVE] Behavioral surface written: {surface_path}")
    print(f"[BSVE] Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
