"""Behavioral Surface dataset augmentation.

Loads a frozen Behavioral Surface artifact produced by BSVE and performs a
strict left join onto an existing master research dataset variant, producing
a behaviorally-augmented dataset variant.

No calibration logic, state assignment logic, or behavioral interpretation
is implemented here.  The Behavioral Surface is treated as read-only input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Required manifest keys per the public Behavioral Surface contract.
_REQUIRED_MANIFEST_KEYS = {
    "ontology_id",
    "ontology_version",
    "calibration_id",
    "calibration_hash",
    "behavioral_surface_schema_version",
    "dataset_version",
}

# Required columns in the Behavioral Surface Parquet (full schema contract).
_REQUIRED_SURFACE_COLUMNS = [
    "timestamp",
    "pair",
    "surface_id",
    "surface_version",
    "state_id",
    "episode_id",
    "maturity_bars",
    "crowd_side",
    "transition_event",
]

# Canonical behavioral columns appended verbatim from the surface.
BEHAVIORAL_COLUMNS = [
    "surface_id",
    "surface_version",
    "state_id",
    "episode_id",
    "maturity_bars",
    "crowd_side",
    "transition_event",
]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _locate_manifest(surface_path: Path) -> Path:
    """Return the expected manifest path next to *surface_path*."""
    return surface_path.parent / "behavioral_surface_manifest.json"


def load_behavioral_surface_manifest(surface_path: Path) -> dict[str, Any]:
    """Load and structurally validate the manifest alongside *surface_path*.

    Raises:
        FileNotFoundError: if the manifest JSON is absent.
        ValueError: if required manifest keys are missing.
    """
    manifest_path = _locate_manifest(surface_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Behavioral Surface manifest not found at {manifest_path}. "
            "The manifest must reside beside the Parquet artifact."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Behavioral Surface manifest is not valid JSON: {exc}") from exc

    missing = _REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(
            f"Behavioral Surface manifest is missing required keys: {sorted(missing)}"
        )

    return manifest


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def validate_behavioral_surface_schema(surface: pd.DataFrame) -> None:
    """Verify the Behavioral Surface contains all required columns.

    Raises:
        ValueError: if any required column is missing.
    """
    missing = [c for c in _REQUIRED_SURFACE_COLUMNS if c not in surface.columns]
    if missing:
        raise ValueError(
            f"Behavioral Surface is missing required columns: {missing}"
        )


def load_behavioral_surface(surface_path: Path) -> pd.DataFrame:
    """Load the Behavioral Surface Parquet from *surface_path* (read-only)."""
    if not surface_path.exists():
        raise FileNotFoundError(f"Behavioral Surface not found: {surface_path}")

    surface = pd.read_parquet(surface_path)

    # Validate full schema before any further processing.
    validate_behavioral_surface_schema(surface)

    # Normalise timestamp dtype for reliable join.
    surface["timestamp"] = pd.to_datetime(surface["timestamp"], errors="coerce")
    nat_count = surface["timestamp"].isna().sum()
    if nat_count:
        raise ValueError(
            f"Behavioral Surface contains {nat_count} invalid timestamp(s) "
            "that could not be parsed. All timestamps must be valid."
        )

    return surface


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_dataset_version(manifest: dict[str, Any], dataset_version: str) -> None:
    """Fail fast if the surface was generated from a different dataset version.

    Raises:
        ValueError: on mismatch.
    """
    surface_version = manifest["dataset_version"]
    if surface_version != dataset_version:
        raise ValueError(
            f"Behavioral Surface dataset_version mismatch: "
            f"surface={surface_version!r}, current build={dataset_version!r}. "
            "Behavioral Surfaces may only be joined onto the dataset version "
            "from which they were generated."
        )


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def _prepare_dataset_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a 'timestamp' column aligned to entry_time."""
    out = df.copy()
    if "timestamp" not in out.columns:
        if "entry_time" not in out.columns:
            raise ValueError(
                "Dataset must contain 'timestamp' or 'entry_time' column "
                "for behavioral surface join."
            )
        out["timestamp"] = pd.to_datetime(out["entry_time"], errors="coerce")
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    nat_count = out["timestamp"].isna().sum()
    if nat_count:
        raise ValueError(
            f"Dataset contains {nat_count} invalid timestamp(s) "
            "that could not be parsed. All timestamps must be valid."
        )

    return out


def augment_with_behavioral_surface(
    dataset: pd.DataFrame,
    surface: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Left-join *surface* behavioral columns onto *dataset* on (timestamp, pair).

    The join is strict:
    - Row count and ordering of *dataset* are preserved.
    - Duplicate ``(timestamp, pair)`` keys in either side raise an error.
    - Pairs outside the Behavioral Surface ontology receive null behavioral columns.

    Args:
        dataset: Master research dataset (or variant).
        surface: Behavioral Surface DataFrame as loaded from Parquet.
        manifest: Validated behavioral surface manifest dict.

    Returns:
        Tuple of (augmented_df, join_stats) where join_stats contains:
            rows_loaded, rows_matched, rows_unmatched, behavioral_columns_added.

    Raises:
        ValueError: on duplicate join keys, one-to-many / many-to-one joins,
            or row-count change after join.
    """
    original_len = len(dataset)

    # Prepare dataset with a 'timestamp' column (validates timestamps).
    ds = _prepare_dataset_timestamp(dataset)

    # Check for duplicate join keys in the surface.
    surface_dups = surface.duplicated(subset=["timestamp", "pair"])
    if surface_dups.any():
        n = int(surface_dups.sum())
        raise ValueError(
            f"Behavioral Surface contains {n} duplicate (timestamp, pair) key(s). "
            "This indicates an invalid artifact."
        )

    # Check for duplicate join keys in the dataset.
    ds_dups = ds.duplicated(subset=["timestamp", "pair"])
    if ds_dups.any():
        n = int(ds_dups.sum())
        raise ValueError(
            f"Dataset contains {n} duplicate (timestamp, pair) key(s). "
            "Cannot perform unambiguous behavioral surface join."
        )

    # Select only the public Behavioral Surface contract columns.
    # Schema validation has already guaranteed that every required column exists.
    surface_join_cols = ["timestamp", "pair"] + BEHAVIORAL_COLUMNS
    surface_subset = surface[surface_join_cols].copy()

    # Preserve original ordering explicitly rather than relying on pandas'
    # current merge implementation.
    ds["_row_order"] = range(len(ds))

    augmented = (
        ds.merge(
            surface_subset,
            on=["timestamp", "pair"],
            how="left",
            validate="1:1",
        )
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )

    # Enforce strict row-count preservation.
    if len(augmented) != original_len:
        raise ValueError(
            f"Row count changed after behavioral surface join: "
            f"before={original_len}, after={len(augmented)}. "
            "The join must be lossless."
        )

    # Compute join stats.
    behavioral_col = BEHAVIORAL_COLUMNS[0]  # surface_id
    if behavioral_col in augmented.columns:
        rows_matched = int(augmented[behavioral_col].notna().sum())
    else:
        rows_matched = 0

    rows_unmatched = original_len - rows_matched
    behavioral_columns_added = sum(
        1 for c in BEHAVIORAL_COLUMNS if c in augmented.columns
    )

    stats = {
        "dataset_rows": original_len,
        "surface_rows": len(surface),
        "rows_matched": rows_matched,
        "rows_unmatched": rows_unmatched,
        "behavioral_columns_added": behavioral_columns_added,
    }

    return augmented, stats


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _ontology_slug(ontology_id: str, ontology_version: str) -> str:
    """Build a filesystem-safe identifier from ontology_id and version.

    Example: 'reactive_jpy', '1.0.0' → 'reactive_jpy_v1'
    """
    major = ontology_version.split(".")[0]
    return f"{ontology_id}_v{major}"


def behavioral_variant_filename(
    base_stem: str,
    ontology_id: str,
    ontology_version: str,
    tag: str | None = None,
) -> str:
    """Return the filename for a behavioral dataset variant.

    Example::

        behavioral_variant_filename(
            "master_research_dataset",
            "reactive_jpy",
            "1.0.0",
            "core",
        )
        # → "master_research_dataset_reactive_jpy_v1_core.csv"
    """
    slug = _ontology_slug(ontology_id, ontology_version)
    if tag:
        return f"{base_stem}_{slug}_{tag}.csv"
    return f"{base_stem}_{slug}.csv"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_bsve_summary(
    manifest: dict[str, Any],
    stats: dict[str, int],
) -> None:
    """Emit the canonical BSVE augmentation summary to the logger."""
    ontology_id = manifest.get("ontology_id", "")
    ontology_version = manifest.get("ontology_version", "")
    calibration_id = manifest.get("calibration_id", "")

    logger.info(
        "\n[BSVE]\n\n"
        "Behavioral Surface:\n"
        "    %s v%s\n\n"
        "Calibration:\n"
        "    %s\n\n"
        "Dataset Rows:\n"
        "    %d\n\n"
        "Behavioral Surface rows:\n"
        "    %d\n\n"
        "Rows matched:\n"
        "    %d\n\n"
        "Rows unmatched:\n"
        "    %d\n\n"
        "Behavioral columns added:\n"
        "    %d",
        ontology_id,
        ontology_version,
        calibration_id,
        stats["dataset_rows"],
        stats["surface_rows"],
        stats["rows_matched"],
        stats["rows_unmatched"],
        stats["behavioral_columns_added"],
    )


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------

def write_behavioral_dataset_manifest(
    output_dir: Path,
    base_manifest: dict[str, Any],
    bsve_manifest: dict[str, Any],
    surface_path: Path,
    bsve_manifest_path: Path,
    ontology_id: str,
    ontology_version: str,
    variant_paths: dict[str, str],
) -> Path:
    """Write a provenance manifest for the behavioral dataset variants.

    Preserves the existing dataset metadata and additionally records
    Behavioral Surface provenance, copied directly from the BSVE manifest.

    Args:
        output_dir: Directory to write the manifest into.
        base_manifest: Existing DATASET_MANIFEST.json content.
        bsve_manifest: Validated Behavioral Surface manifest.
        surface_path: Path to the Behavioral Surface Parquet.
        bsve_manifest_path: Path to the Behavioral Surface manifest JSON.
        ontology_id: Ontology family identifier.
        ontology_version: Ontology version string.
        variant_paths: Mapping of variant name → file path strings.

    Returns:
        Path to the written manifest file.
    """
    slug = _ontology_slug(ontology_id, ontology_version)
    manifest_filename = f"DATASET_MANIFEST_{slug}.json"
    manifest_path = output_dir / manifest_filename

    behavioral_manifest: dict[str, Any] = dict(base_manifest)
    behavioral_manifest["dataset_variants"] = variant_paths
    behavioral_manifest["behavioral_surface"] = {
        "surface_filename": surface_path.name,
        "manifest_filename": bsve_manifest_path.name,
        "ontology_id": bsve_manifest["ontology_id"],
        "ontology_version": bsve_manifest["ontology_version"],
        "calibration_id": bsve_manifest["calibration_id"],
        "calibration_hash": bsve_manifest["calibration_hash"],
        "behavioral_surface_schema_version": bsve_manifest[
            "behavioral_surface_schema_version"
        ],
        "ontology_slug": slug,
    }

    manifest_path.write_text(
        json.dumps(behavioral_manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved behavioral dataset manifest to: %s", manifest_path.resolve())
    return manifest_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_behavioral_augmentation(
    *,
    surface_path: Path,
    dataset_version: str,
    output_dir: Path,
    base_dataset_paths: dict[str, Path],
    base_manifest_path: Path | None = None,
) -> dict[str, Path]:
    """Load, validate, join, and export behavioral dataset variants.

    Args:
        surface_path: Path to the Behavioral Surface Parquet.
        dataset_version: Version string of the current dataset build.
        output_dir: Directory where output CSVs will be written.
        base_dataset_paths: Mapping of variant label → CSV path for the base
            datasets (e.g. {"full": ..., "core": ..., "extended": ...}).
        base_manifest_path: Optional path to the existing DATASET_MANIFEST.json.

    Returns:
        Mapping of variant label → written behavioral CSV path.
    """
    surface_path = Path(surface_path)

    # Load and validate manifest.
    bsve_manifest = load_behavioral_surface_manifest(surface_path)
    validate_dataset_version(bsve_manifest, dataset_version)

    # Load surface (read-only).
    surface = load_behavioral_surface(surface_path)

    ontology_id = bsve_manifest["ontology_id"]
    ontology_version = bsve_manifest["ontology_version"]
    base_stem = "master_research_dataset"

    output_paths: dict[str, Path] = {}
    last_stats: dict[str, int] = {}

    for variant_label, base_path in base_dataset_paths.items():
        if not base_path.exists():
            logger.debug("Skipping missing base dataset variant: %s", base_path)
            continue

        base_df = pd.read_csv(base_path)
        augmented, stats = augment_with_behavioral_surface(base_df, surface)

        # Determine tag: None for the "full" variant (no suffix), else the label
        # itself (e.g. "core" → "_core", "extended" → "_extended").
        # Convention: the caller passes "full" as the label for the un-filtered dataset.
        tag = None if variant_label == "full" else variant_label
        out_filename = behavioral_variant_filename(base_stem, ontology_id, ontology_version, tag)
        out_path = output_dir / out_filename

        augmented.to_csv(out_path, index=False)
        logger.info("Saved behavioral dataset variant '%s' to: %s", variant_label, out_path)
        output_paths[variant_label] = out_path
        last_stats = stats

    # Log summary once (using stats from last processed variant; counts are
    # surface-level and identical across variants).
    if last_stats:
        log_bsve_summary(bsve_manifest, last_stats)

    # Write behavioral provenance manifest.
    base_manifest: dict[str, Any] = {}
    if base_manifest_path and base_manifest_path.exists():
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))

    bsve_manifest_path = _locate_manifest(surface_path)
    version_prefix = f"data/output/{dataset_version}"
    variant_path_strings = {
        label: f"{version_prefix}/{p.name}"
        for label, p in output_paths.items()
    }

    write_behavioral_dataset_manifest(
        output_dir=output_dir,
        base_manifest=base_manifest,
        bsve_manifest=bsve_manifest,
        surface_path=surface_path,
        bsve_manifest_path=bsve_manifest_path,
        ontology_id=ontology_id,
        ontology_version=ontology_version,
        variant_paths=variant_path_strings,
    )

    return output_paths
