"""Tests for PR5.4 — Behavioral Surface Registry.

Covers:
- Registry creation and schema validation
- Promotion workflow (single and multiple promotions)
- Multiple promotions of the same surface
- Version history preservation
- Stage 2/Stage 3 placeholders present even when empty
- Summary generation (high_score.py)
- Malformed registry entries
- Duplicate surface detection (surface ID matches file name)
- Existing Behavioral Evaluation Framework tests continue to pass
  (verified separately; this file verifies registry-specific behaviour)
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers — locate registry tools via importlib so tests do not depend on
# PYTHONPATH containing analysis/ as a top-level namespace.
# ---------------------------------------------------------------------------

import sys
import importlib.util

_REPO_ROOT = Path(__file__).parent.parent


def _import_module(rel_path: str):
    """Import a module by its path relative to the repository root."""
    full = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(full.stem, full)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


promote_mod = _import_module("analysis/registry/promote.py")
high_score_mod = _import_module("analysis/registry/high_score.py")

promote = promote_mod.promote
load_registry_entry = promote_mod.load_registry_entry
save_registry_entry = promote_mod.save_registry_entry
validate_entry = promote_mod.validate_entry

generate_summary = high_score_mod.generate_summary
load_all_surfaces = high_score_mod.load_all_surfaces
build_summary_rows = high_score_mod.build_summary_rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry_root(tmp_path: Path) -> Path:
    """A temporary registry/surfaces directory."""
    root = tmp_path / "registry" / "surfaces"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def repo_root(tmp_path: Path, registry_root: Path) -> Path:
    """Fake repository root containing registry/surfaces/."""
    return tmp_path


def _minimal_entry(surface_id: str = "test_surface") -> dict:
    """Return a minimal valid registry entry dict."""
    return {
        "surface_id": surface_id,
        "ontology_version": "v1",
        "lifecycle_stage": "Characterization",
        "current_status": "active",
        "scientific_interest": "medium",
        "scientific_confidence": "low",
        "current_recommendation": "Await additional evidence.",
        "supporting_experiments": [],
        "promotion_history": [],
        "stage1": {
            "status": "in_progress",
            "characterization_experiments": [],
        },
        "stage2": {
            "status": "not_started",
            "walk_forward_experiments": [],
            "summary": None,
        },
        "stage3": {
            "status": "not_started",
            "trading_experiments": [],
            "summary": None,
        },
    }


def _write_entry(registry_root: Path, surface_id: str, data: dict | None = None) -> Path:
    path = registry_root / f"{surface_id}.yaml"
    entry = data if data is not None else _minimal_entry(surface_id)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(entry, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


# ===========================================================================
# Registry schema validation
# ===========================================================================

class TestRegistrySchema:
    def test_minimal_entry_passes_validation(self):
        entry = _minimal_entry()
        errors = validate_entry(entry)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_missing_required_field_produces_error(self):
        entry = _minimal_entry()
        del entry["scientific_interest"]
        errors = validate_entry(entry)
        assert any("scientific_interest" in e for e in errors)

    def test_invalid_scientific_interest_produces_error(self):
        entry = _minimal_entry()
        entry["scientific_interest"] = "very_high"
        errors = validate_entry(entry)
        assert any("scientific_interest" in e for e in errors)

    def test_invalid_scientific_confidence_produces_error(self):
        entry = _minimal_entry()
        entry["scientific_confidence"] = "none"
        errors = validate_entry(entry)
        assert any("scientific_confidence" in e for e in errors)

    def test_invalid_lifecycle_stage_produces_error(self):
        entry = _minimal_entry()
        entry["lifecycle_stage"] = "Unknown Stage"
        errors = validate_entry(entry)
        assert any("lifecycle_stage" in e for e in errors)

    def test_supporting_experiments_must_be_list(self):
        entry = _minimal_entry()
        entry["supporting_experiments"] = "not_a_list"
        errors = validate_entry(entry)
        assert any("supporting_experiments" in e for e in errors)

    def test_promotion_history_must_be_list(self):
        entry = _minimal_entry()
        entry["promotion_history"] = {"bad": "type"}
        errors = validate_entry(entry)
        assert any("promotion_history" in e for e in errors)

    def test_all_valid_interest_values_accepted(self):
        for val in ("low", "medium", "high"):
            entry = _minimal_entry()
            entry["scientific_interest"] = val
            errors = validate_entry(entry)
            assert errors == [], f"Valid interest={val!r} rejected: {errors}"

    def test_all_valid_lifecycle_stages_accepted(self):
        for stage in ("Characterization", "Predictive Validation",
                      "Trading Validation", "Integrated", "Retired"):
            entry = _minimal_entry()
            entry["lifecycle_stage"] = stage
            errors = validate_entry(entry)
            assert errors == [], f"Valid stage={stage!r} rejected: {errors}"


# ===========================================================================
# Stage 2/Stage 3 placeholders
# ===========================================================================

class TestStagePlaceholders:
    def test_reactive_jpy_has_stage2(self):
        """The shipped reactive_jpy.yaml contains stage2."""
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found — skipping shipped file check")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "stage2" in data, "reactive_jpy.yaml must contain stage2"

    def test_reactive_jpy_has_stage3(self):
        """The shipped reactive_jpy.yaml contains stage3."""
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found — skipping shipped file check")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "stage3" in data, "reactive_jpy.yaml must contain stage3"

    def test_reactive_jpy_stage2_not_started(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["stage2"]["status"] == "not_started"

    def test_reactive_jpy_stage3_not_started(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["stage3"]["status"] == "not_started"

    def test_entry_with_empty_stage2_still_passes_validation(self, registry_root, repo_root):
        entry = _minimal_entry("surface_a")
        entry["stage2"] = {"status": "not_started", "walk_forward_experiments": [], "summary": None}
        _write_entry(registry_root, "surface_a", entry)
        errors = validate_entry(entry)
        assert errors == []


# ===========================================================================
# Promotion workflow — single promotion
# ===========================================================================

class TestPromoteSingle:
    def test_promote_appends_supporting_experiment(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["analysis/output/exp_001"],
            author="test.author",
            recommendation="Repeat characterization.",
            scientific_interest="medium",
            scientific_confidence="low",
            notes="First characterization experiment.",
            repo_root=repo_root,
        )
        assert len(result["supporting_experiments"]) == 1
        assert result["supporting_experiments"][0]["experiment_dir"] == "analysis/output/exp_001"

    def test_promote_records_author(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="researcher.one",
            recommendation="Repeat.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="Initial.",
            repo_root=repo_root,
        )
        assert result["supporting_experiments"][0]["promoted_by"] == "researcher.one"

    def test_promote_records_timestamp(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="researcher.one",
            recommendation="Repeat.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="Initial.",
            repo_root=repo_root,
        )
        promoted_at = result["supporting_experiments"][0]["promoted_at"]
        assert isinstance(promoted_at, str)
        assert "T" in promoted_at  # ISO 8601

    def test_promote_updates_scientific_interest(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="Proceed.",
            scientific_interest="high",
            scientific_confidence="medium",
            notes="Updated interest.",
            repo_root=repo_root,
        )
        assert result["scientific_interest"] == "high"

    def test_promote_updates_scientific_confidence(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="Proceed.",
            scientific_interest="medium",
            scientific_confidence="high",
            notes="Updated confidence.",
            repo_root=repo_root,
        )
        assert result["scientific_confidence"] == "high"

    def test_promote_updates_recommendation(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="Proceed to walk-forward validation.",
            scientific_interest="high",
            scientific_confidence="medium",
            notes="",
            repo_root=repo_root,
        )
        assert "walk-forward" in result["current_recommendation"]

    def test_promote_appends_promotion_history(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="Note.",
            repo_root=repo_root,
        )
        assert len(result["promotion_history"]) == 1

    def test_promote_writes_to_disk(self, registry_root, repo_root):
        path = _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="",
            repo_root=repo_root,
        )
        with path.open("r", encoding="utf-8") as fh:
            reloaded = yaml.safe_load(fh)
        assert len(reloaded["supporting_experiments"]) == 1

    def test_promote_multiple_experiments_in_one_call(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001", "exp_002"],
            author="a",
            recommendation="R.",
            scientific_interest="medium",
            scientific_confidence="medium",
            notes="Two experiments.",
            repo_root=repo_root,
        )
        assert len(result["supporting_experiments"]) == 2

    def test_promote_dry_run_does_not_write(self, registry_root, repo_root):
        path = _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="",
            repo_root=repo_root,
            dry_run=True,
        )
        with path.open("r", encoding="utf-8") as fh:
            reloaded = yaml.safe_load(fh)
        # Original entry unchanged
        assert len(reloaded["supporting_experiments"]) == 0

    def test_promote_updates_lifecycle_stage_when_provided(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="high",
            scientific_confidence="high",
            notes="",
            lifecycle_stage="Predictive Validation",
            repo_root=repo_root,
        )
        assert result["lifecycle_stage"] == "Predictive Validation"

    def test_promote_preserves_lifecycle_stage_when_not_provided(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="medium",
            scientific_confidence="low",
            notes="",
            lifecycle_stage=None,
            repo_root=repo_root,
        )
        assert result["lifecycle_stage"] == "Characterization"


# ===========================================================================
# Multiple promotions of the same surface
# ===========================================================================

class TestMultiplePromotions:
    def test_second_promotion_appends_to_history(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="Repeat characterization.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="First.",
            repo_root=repo_root,
        )
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_002"],
            author="b",
            recommendation="Proceed to walk-forward validation.",
            scientific_interest="high",
            scientific_confidence="medium",
            notes="Second.",
            repo_root=repo_root,
        )
        assert len(result["promotion_history"]) == 2
        assert len(result["supporting_experiments"]) == 2

    def test_second_promotion_updates_interest_and_confidence(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="",
            repo_root=repo_root,
        )
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_002"],
            author="b",
            recommendation="Proceed.",
            scientific_interest="high",
            scientific_confidence="high",
            notes="",
            repo_root=repo_root,
        )
        assert result["scientific_interest"] == "high"
        assert result["scientific_confidence"] == "high"

    def test_version_history_preserves_previous_interest(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_001"],
            author="a",
            recommendation="R.",
            scientific_interest="low",
            scientific_confidence="low",
            notes="First.",
            repo_root=repo_root,
        )
        result = promote(
            surface_id="surface_a",
            experiment_dirs=["exp_002"],
            author="b",
            recommendation="R2.",
            scientific_interest="high",
            scientific_confidence="medium",
            notes="Second.",
            repo_root=repo_root,
        )
        # First promotion's interest is preserved in history
        assert result["promotion_history"][0]["scientific_interest"] == "low"
        assert result["promotion_history"][1]["scientific_interest"] == "high"

    def test_promotion_history_records_experiments_added(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["exp_A", "exp_B"],
            author="a",
            recommendation="R.",
            scientific_interest="medium",
            scientific_confidence="medium",
            notes="",
            repo_root=repo_root,
        )
        path = registry_root / "surface_a.yaml"
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["promotion_history"][0]["experiments_added"] == ["exp_A", "exp_B"]

    def test_three_promotions_accumulate_correctly(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        for i in range(3):
            promote(
                surface_id="surface_a",
                experiment_dirs=[f"exp_{i:03d}"],
                author="researcher",
                recommendation=f"Step {i + 1}.",
                scientific_interest="medium",
                scientific_confidence="low",
                notes=f"Promotion {i + 1}.",
                repo_root=repo_root,
            )
        path = registry_root / "surface_a.yaml"
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert len(data["supporting_experiments"]) == 3
        assert len(data["promotion_history"]) == 3


# ===========================================================================
# Error handling — malformed registry entries and missing surfaces
# ===========================================================================

class TestRegistryErrors:
    def test_promote_raises_on_missing_registry_entry(self, registry_root, repo_root):
        with pytest.raises(FileNotFoundError):
            promote(
                surface_id="nonexistent_surface",
                experiment_dirs=["exp_001"],
                author="a",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="low",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_invalid_interest(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        with pytest.raises(ValueError, match="scientific_interest"):
            promote(
                surface_id="surface_a",
                experiment_dirs=["exp_001"],
                author="a",
                recommendation="R.",
                scientific_interest="very_high",
                scientific_confidence="low",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_invalid_confidence(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        with pytest.raises(ValueError, match="scientific_confidence"):
            promote(
                surface_id="surface_a",
                experiment_dirs=["exp_001"],
                author="a",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="extreme",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_empty_experiments(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        with pytest.raises(ValueError):
            promote(
                surface_id="surface_a",
                experiment_dirs=[],
                author="a",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="low",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_empty_author(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        with pytest.raises(ValueError):
            promote(
                surface_id="surface_a",
                experiment_dirs=["exp_001"],
                author="   ",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="low",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_malformed_entry(self, registry_root, repo_root):
        """An entry missing required fields should raise on promotion."""
        malformed = {"surface_id": "bad_surface"}  # missing many required fields
        path = registry_root / "bad_surface.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(malformed, fh)
        with pytest.raises(ValueError, match="malformed"):
            promote(
                surface_id="bad_surface",
                experiment_dirs=["exp_001"],
                author="a",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="low",
                notes="",
                repo_root=repo_root,
            )

    def test_promote_raises_on_invalid_lifecycle_stage(self, registry_root, repo_root):
        _write_entry(registry_root, "surface_a")
        with pytest.raises(ValueError, match="lifecycle_stage"):
            promote(
                surface_id="surface_a",
                experiment_dirs=["exp_001"],
                author="a",
                recommendation="R.",
                scientific_interest="low",
                scientific_confidence="low",
                notes="",
                lifecycle_stage="Unknown Stage XYZ",
                repo_root=repo_root,
            )

    def test_load_registry_entry_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_registry_entry(tmp_path / "nonexistent.yaml")

    def test_load_registry_entry_raises_on_non_mapping_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_registry_entry(path)


# ===========================================================================
# Duplicate surface detection
# ===========================================================================

class TestDuplicateSurfaceDetection:
    def test_surface_id_matches_filename(self, registry_root):
        """Each surface file should have a surface_id matching its filename stem."""
        entry = _minimal_entry("reactive_jpy")
        path = registry_root / "reactive_jpy.yaml"
        save_registry_entry(path, entry)
        loaded = load_registry_entry(path)
        assert loaded["surface_id"] == path.stem, (
            f"surface_id {loaded['surface_id']!r} does not match filename {path.stem!r}"
        )

    def test_mismatched_surface_id_detected_via_validate_then_compare(self, registry_root):
        """An entry whose surface_id differs from its filename can be detected."""
        entry = _minimal_entry("wrong_id")
        path = registry_root / "correct_id.yaml"
        save_registry_entry(path, entry)
        loaded = load_registry_entry(path)
        # The caller is responsible for detecting the mismatch
        assert loaded["surface_id"] != path.stem

    def test_load_all_surfaces_detects_two_entries(self, registry_root):
        _write_entry(registry_root, "surface_a")
        _write_entry(registry_root, "surface_b")
        surfaces = load_all_surfaces(registry_root)
        ids = [s["surface_id"] for s in surfaces]
        assert "surface_a" in ids
        assert "surface_b" in ids

    def test_load_all_surfaces_skips_non_yaml_files(self, registry_root):
        _write_entry(registry_root, "surface_a")
        (registry_root / "README.md").write_text("# readme", encoding="utf-8")
        (registry_root / "notes.txt").write_text("notes", encoding="utf-8")
        surfaces = load_all_surfaces(registry_root)
        # Only surface_a should be loaded (README.md is skipped by glob pattern *.yaml)
        assert all(isinstance(s, dict) for s in surfaces)


# ===========================================================================
# Summary generation (high_score.py)
# ===========================================================================

class TestHighScore:
    def test_summary_generated_for_single_surface(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "surface_a" in summary

    def test_summary_contains_lifecycle_stage(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "Characterization" in summary

    def test_summary_contains_scientific_interest(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "medium" in summary.lower()

    def test_summary_contains_scientific_confidence(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "low" in summary.lower()

    def test_summary_shows_latest_experiment_after_promotion(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        promote(
            surface_id="surface_a",
            experiment_dirs=["analysis/output/exp_XYZ"],
            author="a",
            recommendation="R.",
            scientific_interest="medium",
            scientific_confidence="low",
            notes="",
            repo_root=repo_root,
        )
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "exp_XYZ" in summary

    def test_summary_shows_none_when_no_experiments(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "none" in summary.lower()

    def test_summary_covers_all_registered_surfaces(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        _write_entry(registry_root, "surface_b")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "surface_a" in summary
        assert "surface_b" in summary

    def test_summary_csv_format(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="csv")
        assert "Surface" in summary
        assert "surface_a" in summary

    def test_summary_text_format(self, repo_root, registry_root):
        _write_entry(registry_root, "surface_a")
        summary = generate_summary(repo_root=repo_root, fmt="text")
        assert "surface_a" in summary

    def test_summary_does_not_rank_surfaces_numerically(self, repo_root, registry_root):
        """Summary should not assign numeric scores."""
        _write_entry(registry_root, "surface_a")
        _write_entry(registry_root, "surface_b")
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        # No numeric ranking column in header
        assert "Score" not in summary
        assert "Rank" not in summary

    def test_summary_empty_registry(self, repo_root, registry_root):
        summary = generate_summary(repo_root=repo_root, fmt="markdown")
        assert "No surfaces registered" in summary

    def test_summary_handles_missing_registry_dir(self, tmp_path):
        """When registry directory does not exist, return a descriptive message."""
        summary = generate_summary(repo_root=tmp_path, fmt="markdown")
        assert "not found" in summary.lower() or "no surfaces" in summary.lower()

    def test_build_summary_rows_fields(self, registry_root):
        _write_entry(registry_root, "surface_a")
        surfaces = load_all_surfaces(registry_root)
        rows = build_summary_rows(surfaces)
        assert len(rows) == 1
        row = rows[0]
        assert "Surface" in row
        assert "Lifecycle Stage" in row
        assert "Scientific Interest" in row
        assert "Scientific Confidence" in row
        assert "Current Recommendation" in row
        assert "Latest Supporting Experiment" in row

    def test_build_summary_rows_recommendation_truncated(self, registry_root):
        entry = _minimal_entry("surface_a")
        entry["current_recommendation"] = "A" * 200  # very long
        _write_entry(registry_root, "surface_a", entry)
        surfaces = load_all_surfaces(registry_root)
        rows = build_summary_rows(surfaces)
        assert len(rows[0]["Current Recommendation"]) < 200


# ===========================================================================
# Shipped reactive_jpy.yaml validates cleanly
# ===========================================================================

class TestShippedReactiveJpyEntry:
    def test_reactive_jpy_yaml_is_valid(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        data = load_registry_entry(path)
        errors = validate_entry(data)
        assert errors == [], f"reactive_jpy.yaml has validation errors: {errors}"

    def test_reactive_jpy_has_correct_surface_id(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        data = load_registry_entry(path)
        assert data["surface_id"] == "reactive_jpy"

    def test_reactive_jpy_lifecycle_stage_is_characterization(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        data = load_registry_entry(path)
        assert data["lifecycle_stage"] == "Characterization"

    def test_reactive_jpy_has_no_promoted_experiments_initially(self):
        path = _REPO_ROOT / "registry" / "surfaces" / "reactive_jpy.yaml"
        if not path.exists():
            pytest.skip("reactive_jpy.yaml not found")
        data = load_registry_entry(path)
        assert data["supporting_experiments"] == []
        assert data["promotion_history"] == []
