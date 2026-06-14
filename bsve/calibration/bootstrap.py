"""
BSVE Calibration Plugin Bootstrap.

Single registration point for all calibration plugins.  Import this module
(or call :func:`register_all_plugins`) to populate the default registry
before running any calibration.

Usage::

    from bsve.calibration.bootstrap import register_all_plugins
    from bsve.calibration.registry import get_default_registry

    register_all_plugins()
    registry = get_default_registry()

    # registry is now populated; pass it to CalibrationRunner.

Design:
    All plugin imports and registry.register() calls live here.  No plugin
    registration is scattered across individual plugin modules or runner
    entry points.  This avoids implicit import-order dependencies and makes
    the registered plugin set easy to audit in one place.

Currently registered plugins:
    * reactive_jpy / 1.0.0  → JPYMaturityCalibrationPlugin
"""

from __future__ import annotations

from bsve.calibration.registry import CalibrationRegistry, get_default_registry


def register_all_plugins(
    registry: CalibrationRegistry | None = None,
) -> CalibrationRegistry:
    """
    Register all production calibration plugins with *registry*.

    If *registry* is ``None`` the module-level singleton returned by
    :func:`~bsve.calibration.registry.get_default_registry` is used.

    This function is idempotent — calling it multiple times overwrites the
    previous registration for each key (no duplicates accumulate).

    Args:
        registry: Target registry.  Defaults to the module-level singleton.

    Returns:
        The populated :class:`~bsve.calibration.registry.CalibrationRegistry`.
    """
    if registry is None:
        registry = get_default_registry()

    # ------------------------------------------------------------------
    # reactive_jpy / 1.0.0
    # ------------------------------------------------------------------
    # Delayed import keeps the bootstrap importable even if optional
    # scientific dependencies (scipy, numpy) are unavailable at import time.
    from bsve.calibration.jpy_maturity_calibration import (
        JPYMaturityCalibrationPlugin,
    )

    registry.register(
        "reactive_jpy",
        "1.0.0",
        JPYMaturityCalibrationPlugin(),
    )

    return registry
