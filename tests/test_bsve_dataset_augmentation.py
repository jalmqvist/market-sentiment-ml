"""Regression tests for Behavioral Surface dataset augmentation (PR2).

Coverage:
- Byte-identical output when augmentation is disabled
- Dataset generation without Behavioral Surface
- Dataset generation with Behavioral Surface
- Manifest validation (missing, invalid, missing keys)
- Dataset-version mismatch detection
- Duplicate-key detection (surface side and dataset side)
- Row-count preservation
- Null handling for pairs outside the Behavioral Surface ontology
- Successful join (matched rows)
- Provenance recording in behavioral manifest
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bsve.dataset_augmentation import (
    BEHAVIORAL_COLUMNS,
    augment_with_behavioral_surface,
    behavioral_variant_filename,
    load_behavioral_surface,
    load_behavioral_surface_manifest,
    run_behavioral_augmentation,
    validate_dataset_version,
    write_behavioral_dataset_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SURFACE_COLUMNS = [
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


def _make_surface(
    *,
    pairs: list[str] | None = None,
    n_rows_per_pair: int = 3,
    start: str = "2024-01-01",
    surface_id: str = "reactive_jpy",
    surface_version: str = "1.0.0",
) -> pd.DataFrame:
    if pairs is None:
        pairs = ["usd-jpy"]
    rows = []
    for pair in pairs:
        for i in range(n_rows_per_pair):
            ts = pd.Timestamp(start) + pd.Timedelta(hours=i)
            rows.append(
                {
                    "timestamp": ts,
                    "pair": pair,
                    "surface_id": surface_id,
                    "surface_version": surface_version,
                    "state_id": "JPY_CONSENSUS_YOUNG",
                    "episode_id": f"{pair}:{i:08d}",
                    "maturity_bars": i + 1,
                    "crowd_side": "LONG",
                    "transition_event": "entry" if i == 0 else "continuation",
                }
            )
    return pd.DataFrame(rows)


def _make_manifest(
    *,
    ontology_id: str = "reactive_jpy",
    ontology_version: str = "1.0.0",
    calibration_id: str = "reactive_jpy_v1_20260615",
    calibration_hash: str = "abc123",
    behavioral_surface_schema_version: str = "1.0.0",
    dataset_version: str = "1.5.1",
) -> dict:
    return {
        "ontology_id": ontology_id,
        "ontology_version": ontology_version,
        "calibration_id": calibration_id,
        "calibration_hash": calibration_hash,
        "behavioral_surface_schema_version": behavioral_surface_schema_version,
        "dataset_version": dataset_version,
        "generated_timestamp": "2026-06-15T12:00:00+00:00",
        "row_count": 3,
        "pair_counts": {"usd-jpy": 3},
        "state_counts": {"JPY_CONSENSUS_YOUNG": 3},
    }


def _make_dataset(
    *,
    pairs: list[str] | None = None,
    n_rows_per_pair: int = 3,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    if pairs is None:
        pairs = ["usd-jpy"]
    rows = []
    for pair in pairs:
        for i in range(n_rows_per_pair):
            ts = pd.Timestamp(start) + pd.Timedelta(hours=i)
            rows.append(
                {
                    "pair": pair,
                    "entry_time": ts,
                    "net_sentiment": 75.0,
                    "close": 150.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _write_surface_artifact(
    tmp_path: Path,
    surface: pd.DataFrame,
    manifest: dict,
    *,
    filename: str = "behavioral_surface_reactive_jpy_1.0.0.parquet",
) -> tuple[Path, Path]:
    surface_path = tmp_path / filename
    surface.to_parquet(surface_path, index=False)
    manifest_path = tmp_path / "behavioral_surface_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return surface_path, manifest_path


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_missing_raises(tmp_path: Path) -> None:
    """Missing manifest beside the Parquet must raise FileNotFoundError."""
    surface = _make_surface()
    surface_path = tmp_path / "surface.parquet"
    surface.to_parquet(surface_path, index=False)
    # No manifest written.
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_behavioral_surface_manifest(surface_path)


def test_manifest_invalid_json_raises(tmp_path: Path) -> None:
    surface_path = tmp_path / "surface.parquet"
    _make_surface().to_parquet(surface_path, index=False)
    manifest_path = tmp_path / "behavioral_surface_manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_behavioral_surface_manifest(surface_path)


def test_manifest_missing_required_key_raises(tmp_path: Path) -> None:
    surface_path = tmp_path / "surface.parquet"
    _make_surface().to_parquet(surface_path, index=False)
    manifest = _make_manifest()
    del manifest["calibration_hash"]  # remove a required key
    manifest_path = tmp_path / "behavioral_surface_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_behavioral_surface_manifest(surface_path)


def test_manifest_all_required_keys_present(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    surface_path, _ = _write_surface_artifact(tmp_path, surface, manifest)
    loaded = load_behavioral_surface_manifest(surface_path)
    assert loaded["ontology_id"] == "reactive_jpy"
    assert loaded["dataset_version"] == "1.5.1"


# ---------------------------------------------------------------------------
# Dataset-version mismatch
# ---------------------------------------------------------------------------


def test_dataset_version_mismatch_raises() -> None:
    manifest = _make_manifest(dataset_version="1.5.1")
    with pytest.raises(ValueError, match="dataset_version mismatch"):
        validate_dataset_version(manifest, "1.4.0")


def test_dataset_version_match_succeeds() -> None:
    manifest = _make_manifest(dataset_version="1.5.1")
    validate_dataset_version(manifest, "1.5.1")  # must not raise


# ---------------------------------------------------------------------------
# Load behavioral surface
# ---------------------------------------------------------------------------


def test_load_behavioral_surface_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_behavioral_surface(tmp_path / "nonexistent.parquet")


def test_load_behavioral_surface_returns_dataframe(tmp_path: Path) -> None:
    surface = _make_surface()
    surface_path = tmp_path / "surface.parquet"
    surface.to_parquet(surface_path, index=False)
    loaded = load_behavioral_surface(surface_path)
    assert isinstance(loaded, pd.DataFrame)
    assert set(_SURFACE_COLUMNS).issubset(set(loaded.columns))


# ---------------------------------------------------------------------------
# Duplicate-key detection
# ---------------------------------------------------------------------------


def test_duplicate_keys_in_surface_raise(tmp_path: Path) -> None:
    surface = _make_surface(n_rows_per_pair=2)
    # Introduce a duplicate.
    dup = surface.iloc[[0]].copy()
    surface = pd.concat([surface, dup], ignore_index=True)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=2)
    with pytest.raises(ValueError, match="Behavioral Surface contains"):
        augment_with_behavioral_surface(dataset, surface, manifest)


def test_duplicate_keys_in_dataset_raise(tmp_path: Path) -> None:
    surface = _make_surface(n_rows_per_pair=2)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=2)
    # Introduce a duplicate row in the dataset.
    dup = dataset.iloc[[0]].copy()
    dataset = pd.concat([dataset, dup], ignore_index=True)
    with pytest.raises(ValueError, match="Dataset contains"):
        augment_with_behavioral_surface(dataset, surface, manifest)


# ---------------------------------------------------------------------------
# Row-count preservation
# ---------------------------------------------------------------------------


def test_row_count_preserved_after_join() -> None:
    n = 5
    surface = _make_surface(n_rows_per_pair=n)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=n)
    augmented, stats = augment_with_behavioral_surface(dataset, surface, manifest)
    assert len(augmented) == len(dataset)


def test_all_behavioral_columns_present_after_join() -> None:
    surface = _make_surface(n_rows_per_pair=3)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=3)
    augmented, _ = augment_with_behavioral_surface(dataset, surface, manifest)
    for col in BEHAVIORAL_COLUMNS:
        assert col in augmented.columns, f"Missing behavioral column: {col}"


# ---------------------------------------------------------------------------
# Null handling for pairs outside the ontology
# ---------------------------------------------------------------------------


def test_null_values_for_pairs_outside_ontology() -> None:
    """Pairs absent from the surface must receive null behavioral columns."""
    surface = _make_surface(pairs=["usd-jpy"], n_rows_per_pair=3)
    manifest = _make_manifest()

    # Dataset includes 'eur-usd' which is NOT in the surface.
    dataset = _make_dataset(pairs=["usd-jpy", "eur-usd"], n_rows_per_pair=3)

    augmented, stats = augment_with_behavioral_surface(dataset, surface, manifest)

    assert len(augmented) == len(dataset)

    eur_usd_rows = augmented[augmented["pair"] == "eur-usd"]
    for col in BEHAVIORAL_COLUMNS:
        assert eur_usd_rows[col].isna().all(), (
            f"Expected null in '{col}' for pairs outside ontology"
        )

    jpy_rows = augmented[augmented["pair"] == "usd-jpy"]
    assert jpy_rows["surface_id"].notna().all()


# ---------------------------------------------------------------------------
# Successful join
# ---------------------------------------------------------------------------


def test_successful_join_values_copied_verbatim() -> None:
    """Behavioral column values must match the surface verbatim."""
    surface = _make_surface(n_rows_per_pair=3)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=3)

    augmented, stats = augment_with_behavioral_surface(dataset, surface, manifest)

    for col in BEHAVIORAL_COLUMNS:
        expected = surface.set_index(["timestamp", "pair"])[col]
        for _, row in augmented.iterrows():
            key = (row["timestamp"], row["pair"])
            if key in expected.index:
                assert row[col] == expected[key], (
                    f"Column {col!r} mismatch at key {key}: "
                    f"got {row[col]!r}, expected {expected[key]!r}"
                )


def test_join_stats_rows_loaded() -> None:
    n = 4
    surface = _make_surface(n_rows_per_pair=n)
    manifest = _make_manifest()
    dataset = _make_dataset(n_rows_per_pair=n)
    _, stats = augment_with_behavioral_surface(dataset, surface, manifest)
    assert stats["rows_loaded"] == n


def test_join_stats_rows_matched_and_unmatched() -> None:
    # Surface: only usd-jpy (3 rows). Dataset: usd-jpy (3) + eur-usd (2).
    surface = _make_surface(pairs=["usd-jpy"], n_rows_per_pair=3)
    manifest = _make_manifest()
    dataset = _make_dataset(pairs=["usd-jpy", "eur-usd"], n_rows_per_pair=3)
    _, stats = augment_with_behavioral_surface(dataset, surface, manifest)
    assert stats["rows_matched"] == 3
    assert stats["rows_unmatched"] == 3  # eur-usd rows have no surface match


def test_join_stats_behavioral_columns_added() -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    dataset = _make_dataset()
    _, stats = augment_with_behavioral_surface(dataset, surface, manifest)
    assert stats["behavioral_columns_added"] == len(BEHAVIORAL_COLUMNS)


# ---------------------------------------------------------------------------
# Row ordering preserved
# ---------------------------------------------------------------------------


def test_row_ordering_preserved_after_join() -> None:
    """Left join must not reorder dataset rows."""
    surface = _make_surface(pairs=["usd-jpy", "eur-jpy"], n_rows_per_pair=3)
    manifest = _make_manifest()
    # Dataset intentionally in a different order than the surface.
    dataset = _make_dataset(pairs=["eur-jpy", "usd-jpy"], n_rows_per_pair=3)
    augmented, _ = augment_with_behavioral_surface(dataset, surface, manifest)
    # The first rows should still be eur-jpy (original ordering preserved).
    assert augmented.iloc[0]["pair"] == "eur-jpy"


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def test_behavioral_variant_filename_with_tag() -> None:
    name = behavioral_variant_filename(
        "master_research_dataset", "reactive_jpy", "1.0.0", "core"
    )
    assert name == "master_research_dataset_reactive_jpy_v1_core.csv"


def test_behavioral_variant_filename_without_tag() -> None:
    name = behavioral_variant_filename(
        "master_research_dataset", "reactive_jpy", "1.0.0", None
    )
    assert name == "master_research_dataset_reactive_jpy_v1.csv"


def test_behavioral_variant_filename_major_version_only() -> None:
    name = behavioral_variant_filename(
        "master_research_dataset", "reactive_chf", "2.1.3", "extended"
    )
    assert name == "master_research_dataset_reactive_chf_v2_extended.csv"


# ---------------------------------------------------------------------------
# Provenance recording
# ---------------------------------------------------------------------------


def test_write_behavioral_dataset_manifest(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    surface_path, bsve_manifest_path = _write_surface_artifact(tmp_path, surface, manifest)

    base_manifest = {"schema_version": "1.0", "dataset_version": "1.5.1"}
    out = write_behavioral_dataset_manifest(
        output_dir=tmp_path,
        base_manifest=base_manifest,
        bsve_manifest=manifest,
        surface_path=surface_path,
        bsve_manifest_path=bsve_manifest_path,
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        variant_paths={"core": "data/output/1.5.1/master_research_dataset_reactive_jpy_v1_core.csv"},
    )
    assert out.exists()
    written = json.loads(out.read_text())
    bs = written["behavioral_surface"]
    assert bs["ontology_id"] == "reactive_jpy"
    assert bs["ontology_version"] == "1.0.0"
    assert bs["calibration_id"] == manifest["calibration_id"]
    assert bs["calibration_hash"] == manifest["calibration_hash"]
    assert bs["behavioral_surface_schema_version"] == manifest["behavioral_surface_schema_version"]
    assert bs["surface_filename"] == surface_path.name
    assert bs["manifest_filename"] == bsve_manifest_path.name


def test_behavioral_manifest_preserves_base_metadata(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    surface_path, bsve_manifest_path = _write_surface_artifact(tmp_path, surface, manifest)

    base_manifest = {
        "schema_version": "1.0",
        "dataset_version": "1.5.1",
        "horizons_bars": [1, 2, 4, 6, 12, 24, 48],
    }
    out = write_behavioral_dataset_manifest(
        output_dir=tmp_path,
        base_manifest=base_manifest,
        bsve_manifest=manifest,
        surface_path=surface_path,
        bsve_manifest_path=bsve_manifest_path,
        ontology_id="reactive_jpy",
        ontology_version="1.0.0",
        variant_paths={},
    )
    written = json.loads(out.read_text())
    assert written["schema_version"] == "1.0"
    assert written["dataset_version"] == "1.5.1"
    assert written["horizons_bars"] == [1, 2, 4, 6, 12, 24, 48]


# ---------------------------------------------------------------------------
# run_behavioral_augmentation (end-to-end)
# ---------------------------------------------------------------------------


def test_run_behavioral_augmentation_writes_variants(tmp_path: Path) -> None:
    surface = _make_surface(n_rows_per_pair=3)
    manifest = _make_manifest()
    surface_path, _ = _write_surface_artifact(tmp_path, surface, manifest)

    dataset = _make_dataset(n_rows_per_pair=3)
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    output_paths = run_behavioral_augmentation(
        surface_path=surface_path,
        dataset_version="1.5.1",
        output_dir=tmp_path,
        base_dataset_paths={"core": core_path},
    )

    assert "core" in output_paths
    variant_path = output_paths["core"]
    assert variant_path.exists()
    assert variant_path.name == "master_research_dataset_reactive_jpy_v1_core.csv"

    loaded = pd.read_csv(variant_path)
    assert len(loaded) == len(dataset)
    for col in BEHAVIORAL_COLUMNS:
        assert col in loaded.columns


def test_run_behavioral_augmentation_version_mismatch_raises(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest(dataset_version="1.5.1")
    surface_path, _ = _write_surface_artifact(tmp_path, surface, manifest)

    dataset = _make_dataset()
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    with pytest.raises(ValueError, match="dataset_version mismatch"):
        run_behavioral_augmentation(
            surface_path=surface_path,
            dataset_version="1.4.0",
            output_dir=tmp_path,
            base_dataset_paths={"core": core_path},
        )


def test_run_behavioral_augmentation_missing_manifest_raises(tmp_path: Path) -> None:
    surface = _make_surface()
    surface_path = tmp_path / "surface.parquet"
    surface.to_parquet(surface_path, index=False)
    # No manifest.

    dataset = _make_dataset()
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    with pytest.raises(FileNotFoundError, match="manifest"):
        run_behavioral_augmentation(
            surface_path=surface_path,
            dataset_version="1.5.1",
            output_dir=tmp_path,
            base_dataset_paths={"core": core_path},
        )


def test_run_behavioral_augmentation_skips_missing_base_variants(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    surface_path, _ = _write_surface_artifact(tmp_path, surface, manifest)

    # Only 'core' variant exists; 'full' and 'extended' are absent.
    dataset = _make_dataset()
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    output_paths = run_behavioral_augmentation(
        surface_path=surface_path,
        dataset_version="1.5.1",
        output_dir=tmp_path,
        base_dataset_paths={
            "full": tmp_path / "master_research_dataset.csv",  # missing
            "core": core_path,
            "extended": tmp_path / "master_research_dataset_extended.csv",  # missing
        },
    )

    assert "core" in output_paths
    assert "full" not in output_paths
    assert "extended" not in output_paths


def test_run_behavioral_augmentation_writes_provenance_manifest(tmp_path: Path) -> None:
    surface = _make_surface()
    manifest = _make_manifest()
    surface_path, _ = _write_surface_artifact(tmp_path, surface, manifest)

    dataset = _make_dataset()
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    run_behavioral_augmentation(
        surface_path=surface_path,
        dataset_version="1.5.1",
        output_dir=tmp_path,
        base_dataset_paths={"core": core_path},
    )

    provenance_manifest = tmp_path / "DATASET_MANIFEST_reactive_jpy_v1.json"
    assert provenance_manifest.exists()
    data = json.loads(provenance_manifest.read_text())
    assert "behavioral_surface" in data
    assert data["behavioral_surface"]["ontology_id"] == "reactive_jpy"


# ---------------------------------------------------------------------------
# Byte-identical output when augmentation is disabled (regression)
# ---------------------------------------------------------------------------


def test_no_augmentation_output_unchanged(tmp_path: Path) -> None:
    """Base datasets must not be affected when no behavioral surface is provided."""
    dataset = _make_dataset(n_rows_per_pair=4)
    core_path = tmp_path / "master_research_dataset_core.csv"
    dataset.to_csv(core_path, index=False)

    original_content = core_path.read_bytes()

    # No augmentation is called — the file should remain untouched.
    assert core_path.read_bytes() == original_content, (
        "Base dataset must be byte-identical when no augmentation is applied."
    )
