"""
BSVE Calibration Runner Framework.

Provides an ontology-agnostic runner that:
    1. Loads the state specification for the target ontology.
    2. Loads the dataset through the :class:`MasterResearchDatasetAdapter`.
    3. Looks up the registered calibration plugin.
    4. Invokes the plugin's ``calibrate`` method.
    5. Validates the returned artifact.
    6. Writes the artifact to the configured output directory.

No JPY-specific logic.  No CHF-specific logic.  The runner is fully driven
by the registry and the state-spec contract.

Usage::

    runner = CalibrationRunner(registry=registry, output_dir="bsve/calibration_artifacts")
    artifact_path = runner.run(
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        state_spec_path="bsve/state_specs/reactive_jpy_v1.yaml",
        dataset_adapter=adapter,
        calibration_params={
            "calibration_id": "reactive_jpy_v1_20240101",
            "dataset_version": "1.3.2",
            "calibration_method": "hazard_analysis",
            "calibration_window_start": "2019-01-01",
            "calibration_window_end": "2023-12-31",
        },
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsve.calibration.calibration_contract import (
    CalibrationArtifact,
    validate_calibration_artifact,
    write_calibration_artifact,
)
from bsve.calibration.registry import CalibrationRegistry, get_default_registry


# ---------------------------------------------------------------------------
# State-spec loader
# ---------------------------------------------------------------------------


def load_state_spec(path: str | Path) -> dict[str, Any]:
    """
    Load a BSVE state-spec YAML file.

    Args:
        path: Path to the YAML spec file (e.g. ``bsve/state_specs/reactive_jpy_v1.yaml``).

    Returns:
        Parsed spec as a plain dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as valid YAML.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load state specs. "
            "Install it with: pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"state spec not found: {p}")

    try:
        with p.open(encoding="utf-8") as fh:
            spec: dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ValueError(f"failed to parse state spec {p}: {exc}") from exc

    return spec


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CalibrationRunner:
    """
    Ontology-agnostic calibration runner.

    The runner never inspects ``ontology_id`` to branch on environment
    identity.  All ontology-specific logic lives in the registered plugin.

    Args:
        registry: :class:`~bsve.calibration.registry.CalibrationRegistry`
            instance to use for plugin lookup.  Defaults to the module-level
            singleton returned by :func:`~bsve.calibration.registry.get_default_registry`.
        output_dir: Directory where artifact JSON files are written.
            Defaults to ``"bsve/calibration_artifacts"``.
        strict_validation: When True (default), validation failures raise
            :class:`ValueError` immediately.  Set to False to collect violations
            without raising (not recommended for production use).
    """

    def __init__(
        self,
        *,
        registry: CalibrationRegistry | None = None,
        output_dir: str | Path = "bsve/calibration_artifacts",
        strict_validation: bool = True,
    ) -> None:
        self._registry = registry if registry is not None else get_default_registry()
        self._output_dir = Path(output_dir)
        self._strict_validation = strict_validation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        ontology_id: str,
        ontology_version: str,
        state_spec_path: str | Path,
        dataset_adapter: Any,
        calibration_params: dict[str, Any],
        evaluation_window_start: str | None = None,
        evaluation_window_end: str | None = None,
    ) -> Path:
        """
        Execute one calibration run end-to-end.

        Steps:
            1. Load state spec from *state_spec_path*.
            2. Look up the plugin for *(ontology_id, ontology_version)*.
            3. Call ``plugin.calibrate(dataset_adapter, state_spec, calibration_params)``.
            4. Validate the returned artifact (fail-fast by default).
            5. Write the artifact to ``<output_dir>/<calibration_id>.json``.

        Args:
            ontology_id: Behavioral ontology identifier (e.g. ``"reactive_jpy"``).
            ontology_version: Ontology version string (e.g. ``"1.0.0"``).
            state_spec_path: Path to the YAML state-spec file for this ontology.
            dataset_adapter: A
                :class:`~bsve.adapters.dataset_adapter.MasterResearchDatasetAdapter`
                instance (or any object satisfying that interface).
            calibration_params: Dict of parameters forwarded to the plugin.
                Must include ``calibration_id`` — used as the output filename stem.
            evaluation_window_start: Optional ISO-8601 date.  If provided along
                with *evaluation_window_end*, the runner checks that calibration
                and evaluation windows do not overlap.
            evaluation_window_end: Optional ISO-8601 date.  See above.

        Returns:
            :class:`~pathlib.Path` of the written artifact file.

        Raises:
            KeyError: If no plugin is registered for the given ontology/version.
            ValueError: If the artifact produced by the plugin fails validation.
            FileNotFoundError: If *state_spec_path* does not exist.
        """
        calibration_id = calibration_params.get("calibration_id")
        if not calibration_id:
            raise ValueError(
                "calibration_params must include a non-empty 'calibration_id'"
            )

        # Step 1 — Load state spec
        state_spec = load_state_spec(state_spec_path)

        # Step 2 — Resolve plugin (raises KeyError if not registered)
        plugin = self._registry.lookup(ontology_id, ontology_version)

        # Step 3 — Run calibration
        artifact: CalibrationArtifact = plugin.calibrate(
            dataset_adapter, state_spec, calibration_params
        )

        # Step 4 — Validate
        validate_calibration_artifact(
            artifact,
            strict=self._strict_validation,
            evaluation_window_start=evaluation_window_start,
            evaluation_window_end=evaluation_window_end,
        )

        # Step 5 — Write
        output_path = self._output_dir / f"{calibration_id}.json"
        return write_calibration_artifact(
            artifact, output_path, strict=self._strict_validation
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def registry(self) -> CalibrationRegistry:
        """The :class:`CalibrationRegistry` used by this runner."""
        return self._registry

    @property
    def output_dir(self) -> Path:
        """Resolved output directory for artifact files."""
        return self._output_dir
