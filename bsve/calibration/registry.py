"""
BSVE Calibration Plugin Interface and Registry.

Provides:
    * :class:`CalibrationPlugin` — ``Protocol`` that all calibration plugins
      must satisfy.
    * :class:`CalibrationRegistry` — registration patterns for ontology-specific
      plugin lookup with version tracking.
    * :func:`get_default_registry` — module-level singleton registry.

Design:
    Plugins are registered by ``(ontology_id, ontology_version)`` key.
    The runner looks up the plugin at call time — no hardcoded branching on
    ontology identity.

    Registration:
        registry.register("reactive_jpy", "1.0.0", MyJPYPlugin())

    Lookup:
        plugin = registry.lookup("reactive_jpy", "1.0.0")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from bsve.calibration.calibration_contract import CalibrationArtifact


# ---------------------------------------------------------------------------
# Plugin Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CalibrationPlugin(Protocol):
    """
    Contract that all ontology-specific calibration plugins must satisfy.

    Framework callers interact exclusively through this interface.  No
    plugin is imported or referenced directly in framework code.

    Method:
        calibrate(dataset_adapter, state_spec, calibration_params)
            → CalibrationArtifact

    The returned artifact must be produced by
    :func:`~bsve.calibration.calibration_contract.build_calibration_artifact`
    so that it carries all required metadata and a valid hash.
    """

    def calibrate(
        self,
        dataset_adapter: Any,
        state_spec: dict[str, Any],
        calibration_params: dict[str, Any],
    ) -> CalibrationArtifact:
        """
        Run calibration and return a complete, validated artifact.

        Args:
            dataset_adapter: A
                :class:`~bsve.adapters.dataset_adapter.MasterResearchDatasetAdapter`
                instance providing normalized feature access.
            state_spec: Parsed state-spec YAML dict for the target ontology.
            calibration_params: Arbitrary parameters controlling the
                calibration (window dates, dataset_version, etc.).

        Returns:
            A :data:`~bsve.calibration.calibration_contract.CalibrationArtifact`
            with ``outcome`` set to ``"success"`` or ``"null"``.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CalibrationRegistry:
    """
    Ontology-agnostic plugin registry with version tracking.

    Stores plugins keyed by ``(ontology_id, ontology_version)``.  The runner
    resolves the correct plugin at call time through :meth:`lookup`, avoiding
    any ``if ontology == "..."`` branching in framework code.

    Usage::

        registry = CalibrationRegistry()
        registry.register("reactive_jpy", "1.0.0", JPYPlugin())

        plugin = registry.lookup("reactive_jpy", "1.0.0")
        artifact = plugin.calibrate(adapter, spec, params)
    """

    def __init__(self) -> None:
        # Keyed by (ontology_id, ontology_version) → plugin instance.
        self._plugins: dict[tuple[str, str], CalibrationPlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        ontology_id: str,
        ontology_version: str,
        plugin: CalibrationPlugin,
    ) -> None:
        """
        Register a calibration plugin for an ontology/version pair.

        Overwrites any previous registration for the same key without error,
        allowing hot-reload scenarios in development.

        Args:
            ontology_id: Identifier of the behavioral ontology (e.g.
                ``"reactive_jpy"``).
            ontology_version: Version string of the ontology spec (e.g.
                ``"1.0.0"``).
            plugin: Object satisfying the :class:`CalibrationPlugin` protocol.

        Raises:
            TypeError: If *plugin* does not satisfy :class:`CalibrationPlugin`.
        """
        if not isinstance(plugin, CalibrationPlugin):
            raise TypeError(
                f"plugin {plugin!r} does not satisfy the CalibrationPlugin protocol"
            )
        self._plugins[(ontology_id, ontology_version)] = plugin

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        ontology_id: str,
        ontology_version: str,
    ) -> CalibrationPlugin:
        """
        Retrieve the plugin for a given ontology/version pair.

        Args:
            ontology_id: Ontology identifier.
            ontology_version: Ontology version string.

        Returns:
            The registered :class:`CalibrationPlugin`.

        Raises:
            KeyError: If no plugin has been registered for the given key.
        """
        key = (ontology_id, ontology_version)
        if key not in self._plugins:
            registered = sorted(self._plugins.keys())
            raise KeyError(
                f"no calibration plugin registered for "
                f"ontology_id={ontology_id!r}, ontology_version={ontology_version!r}. "
                f"Registered keys: {registered}"
            )
        return self._plugins[key]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_keys(self) -> list[tuple[str, str]]:
        """Return a sorted list of ``(ontology_id, ontology_version)`` keys."""
        return sorted(self._plugins.keys())

    def is_registered(self, ontology_id: str, ontology_version: str) -> bool:
        """Return True if a plugin exists for the given ontology/version."""
        return (ontology_id, ontology_version) in self._plugins

    def versions_for(self, ontology_id: str) -> list[str]:
        """Return all registered versions for a given *ontology_id*."""
        return sorted(
            v for oid, v in self._plugins if oid == ontology_id
        )

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        keys = self.registered_keys()
        return f"CalibrationRegistry({keys!r})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry = CalibrationRegistry()


def get_default_registry() -> CalibrationRegistry:
    """Return the module-level singleton :class:`CalibrationRegistry`."""
    return _default_registry
