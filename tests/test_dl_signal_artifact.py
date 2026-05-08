"""
tests/test_dl_signal_artifact.py
=================================
Unit tests for the DL signal artifact builder (scripts/build_dl_signal_artifact.py).

Tests cover:
1. _normalize_pair: pair string normalisation.
2. _normalize_entry_time: timestamp parsing and tz-stripping.
3. _build_artifact: schema construction, signal_strength derivation,
   provenance defaults, dtype enforcement.
4. _run_qa: uniqueness, monotonic, range, regime taxonomy checks.
5. build_dl_signal_artifact: end-to-end integration with temp CSV files.
6. Error paths: missing input directory, missing required columns,
   invalid pred_prob_up, duplicate unique keys.

Run with::

    python -m pytest tests/test_dl_signal_artifact.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root and scripts dir are on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_dl_signal_artifact import (
    OUTPUT_COLS,
    SCHEMA_VERSION,
    SURFACE_GRAIN_COLS,
    UNIQUE_KEY_COLS,
    VALID_DL_REGIMES,
    _build_artifact,
    _normalize_entry_time,
    _normalize_pair,
    _run_qa,
    _write_manifest,
    build_dl_signal_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_raw(n: int = 4) -> pd.DataFrame:
    """Return a minimal valid raw DataFrame as loaded from a CSV."""
    times = pd.date_range("2023-01-02", periods=n, freq="h")
    return pd.DataFrame(
        {
            "entry_time": times,
            "pair": ["eur-usd"] * n,
            "pred_prob_up": np.linspace(0.4, 0.7, n),
            "model": ["MLP"] * n,
            "dl_regime": ["LVTF"] * n,
            "target_horizon": [24] * n,  # numeric (Int64)
            "feature_set": ["price_vol_sentiment"] * n,
            "dataset_version": ["1.1.0"] * n,
            "model_version": ["v1.0"] * n,
        }
    )


def _make_artifact(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None:
        raw = _minimal_raw()
    return _build_artifact(raw)


# ---------------------------------------------------------------------------
# _normalize_pair
# ---------------------------------------------------------------------------


class TestNormalizePair:
    def test_lowercase_hyphen(self):
        assert _normalize_pair("EUR-USD") == "eur-usd"

    def test_slash_separator(self):
        assert _normalize_pair("EUR/USD") == "eur-usd"

    def test_underscore_separator(self):
        assert _normalize_pair("EUR_USD") == "eur-usd"

    def test_no_separator_6char(self):
        assert _normalize_pair("EURUSD") == "eur-usd"

    def test_already_normalized(self):
        assert _normalize_pair("eur-usd") == "eur-usd"

    def test_strips_whitespace(self):
        assert _normalize_pair("  EUR-USD  ") == "eur-usd"


# ---------------------------------------------------------------------------
# _normalize_entry_time
# ---------------------------------------------------------------------------


class TestNormalizeEntryTime:
    def test_naive_string(self):
        s = pd.Series(["2023-01-02 00:00:00", "2023-01-02 01:00:00"])
        result = _normalize_entry_time(s)
        assert np.issubdtype(result.dtype, np.datetime64)
        assert result.dt.tz is None

    def test_utc_aware_string(self):
        s = pd.Series(["2023-01-02 00:00:00+00:00"])
        result = _normalize_entry_time(s)
        assert result.dt.tz is None

    def test_already_datetime(self):
        s = pd.to_datetime(pd.Series(["2023-01-02", "2023-01-03"]))
        result = _normalize_entry_time(s)
        assert np.issubdtype(result.dtype, np.datetime64)


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    def test_output_columns_present(self):
        art = _make_artifact()
        for col in OUTPUT_COLS:
            assert col in art.columns, f"Missing column: {col}"

    def test_schema_version_constant(self):
        art = _make_artifact()
        assert (art["schema_version"] == SCHEMA_VERSION).all()

    def test_signal_strength_formula(self):
        raw = _minimal_raw()
        raw["pred_prob_up"] = [0.0, 0.5, 0.75, 1.0]
        art = _build_artifact(raw)
        expected = pd.Series([-1.0, 0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(
            art["signal_strength"].reset_index(drop=True),
            expected,
            check_names=False,
        )

    def test_signal_strength_in_range(self):
        art = _make_artifact()
        assert art["signal_strength"].between(-1, 1).all()

    def test_pred_direction_derived_if_absent(self):
        raw = _minimal_raw().drop(columns=["model"])  # keep pred_direction absent
        raw["pred_prob_up"] = [0.6, 0.4, 0.8, 0.2]
        art = _build_artifact(raw)
        # tri-state: >0.5 => +1, <0.5 => -1, ==0.5 => 0
        assert set(art["pred_direction"].dropna().unique()).issubset({1, -1, 0})

    def test_confidence_null_if_absent(self):
        art = _make_artifact()
        assert art["confidence"].isna().all()

    def test_confidence_kept_if_supplied(self):
        raw = _minimal_raw()
        raw["confidence"] = [0.9, 0.8, 0.7, 0.6]
        art = _build_artifact(raw)
        assert not art["confidence"].isna().all()

    def test_pair_normalized(self):
        raw = _minimal_raw()
        raw["pair"] = ["EUR/USD", "EURUSD", "eur_usd", "EUR-USD"]
        art = _build_artifact(raw)
        assert (art["pair"] == "eur-usd").all()

    def test_provenance_defaults_when_absent(self):
        raw = _minimal_raw()[["entry_time", "pair", "pred_prob_up"]]
        art = _build_artifact(raw)
        assert (art["model"] == "unknown").all()
        assert (art["dl_regime"] == "unknown").all()
        # target_horizon: Int64 nullable; pd.NA when absent
        assert art["target_horizon"].isna().all()
        assert (art["feature_set"] == "unknown").all()

    def test_sorted_by_pair_entry_time(self):
        raw = _minimal_raw()
        # Shuffle
        raw = raw.sample(frac=1, random_state=42).reset_index(drop=True)
        art = _build_artifact(raw)
        for _, grp in art.groupby("pair"):
            assert grp["entry_time"].is_monotonic_increasing

    def test_invalid_pred_prob_up_raises(self):
        raw = _minimal_raw()
        raw.loc[0, "pred_prob_up"] = 1.5
        with pytest.raises(ValueError, match="invalid pred_prob_up"):
            _build_artifact(raw)

    def test_negative_pred_prob_up_raises(self):
        raw = _minimal_raw()
        raw.loc[0, "pred_prob_up"] = -0.1
        with pytest.raises(ValueError, match="invalid pred_prob_up"):
            _build_artifact(raw)


# ---------------------------------------------------------------------------
# _run_qa
# ---------------------------------------------------------------------------


class TestRunQA:
    def test_passes_on_valid(self):
        art = _make_artifact()
        _run_qa(art)  # should not raise

    def test_fails_on_duplicates(self):
        art = _make_artifact()
        duped = pd.concat([art, art], ignore_index=True)
        with pytest.raises(AssertionError, match="duplicate"):
            _run_qa(duped)

    def test_fails_on_null_entry_time(self):
        art = _make_artifact()
        art.loc[0, "entry_time"] = pd.NaT
        with pytest.raises(AssertionError, match="null entry_time"):
            _run_qa(art)

    def test_fails_on_out_of_range_pred_prob_up(self):
        art = _make_artifact()
        art.loc[0, "pred_prob_up"] = 1.5
        with pytest.raises(AssertionError, match="pred_prob_up outside"):
            _run_qa(art)

    def test_fails_on_out_of_range_signal_strength(self):
        art = _make_artifact()
        art.loc[0, "signal_strength"] = 2.0
        with pytest.raises(AssertionError, match="signal_strength outside"):
            _run_qa(art)

    def test_fails_on_non_monotonic_entry_time(self):
        art = _make_artifact()
        # Reverse the entry_time order to break monotonicity
        art = art.iloc[::-1].reset_index(drop=True)
        # Re-insert same pair so it's treated as one surface
        with pytest.raises(AssertionError, match="not monotonically increasing"):
            _run_qa(art)

    def test_passes_on_multiple_surfaces_independent_monotonicity(self):
        """Two surfaces can have interleaved timestamps as long as each is monotonic."""
        times = pd.date_range("2023-01-02", periods=4, freq="h")
        raw1 = _minimal_raw(4)
        raw1["dl_regime"] = "HVTF"
        raw2 = _minimal_raw(4)
        raw2["dl_regime"] = "LVTF"
        raw2["entry_time"] = times + pd.Timedelta(hours=10)
        combined = pd.concat([raw1, raw2], ignore_index=True)
        art = _build_artifact(combined)
        _run_qa(art)  # should not raise

    def test_surface_grain_monotonicity_checked(self):
        """Monotonicity is per (pair, model, dl_regime, target_horizon, feature_set)."""
        art = _make_artifact()
        # Reverse one surface
        art = art.iloc[::-1].reset_index(drop=True)
        with pytest.raises(AssertionError, match="not monotonically increasing"):
            _run_qa(art)


# ---------------------------------------------------------------------------
# end-to-end: build_dl_signal_artifact
# ---------------------------------------------------------------------------


class TestBuildDlSignalArtifact:
    def _write_csv(self, tmp_path: Path, df: pd.DataFrame, name: str = "preds.csv") -> Path:
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p

    def test_writes_parquet(self, tmp_path):
        raw = _minimal_raw()
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "preds.csv").write_text(raw.to_csv(index=False))

        out_pq = tmp_path / "out.parquet"
        out_mf = tmp_path / "manifest.json"
        build_dl_signal_artifact(input_dir, out_pq, out_mf)

        assert out_pq.exists()
        loaded = pd.read_parquet(out_pq)
        assert len(loaded) == len(raw)

    def test_writes_manifest(self, tmp_path):
        raw = _minimal_raw()
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "preds.csv").write_text(raw.to_csv(index=False))

        out_pq = tmp_path / "out.parquet"
        out_mf = tmp_path / "manifest.json"
        build_dl_signal_artifact(input_dir, out_pq, out_mf)

        assert out_mf.exists()
        manifest = json.loads(out_mf.read_text())
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert "generated_at_utc" in manifest
        assert "export_frequency" in manifest
        assert manifest["export_frequency"] == "H1"
        assert "unique_key" in manifest
        assert "signal_definition" in manifest
        assert "git_commit" in manifest
        assert "total_rows" in manifest
        assert "pair_stats" in manifest
        assert "signal_stats" in manifest
        # v1 tighten-ups
        assert "calibration" in manifest
        assert manifest["calibration"]["method"] == "none"
        assert "train_period" in manifest
        assert "warnings" in manifest
        assert "missing_provenance_counts" in manifest

    def test_multiple_csvs_concatenated(self, tmp_path):
        raw1 = _minimal_raw(n=3)
        raw1["pair"] = "eur-usd"
        raw2 = _minimal_raw(n=3)
        raw2["pair"] = "usd-jpy"
        # Offset entry_time so they don't collide on unique key
        raw2["entry_time"] = raw2["entry_time"] + pd.Timedelta(hours=10)

        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "eur_usd.csv").write_text(raw1.to_csv(index=False))
        (input_dir / "usd_jpy.csv").write_text(raw2.to_csv(index=False))

        out_pq = tmp_path / "out.parquet"
        out_mf = tmp_path / "manifest.json"
        result = build_dl_signal_artifact(input_dir, out_pq, out_mf)

        assert result["pair"].nunique() == 2
        assert len(result) == 6

    def test_missing_input_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            build_dl_signal_artifact(
                tmp_path / "nonexistent",
                tmp_path / "out.parquet",
                tmp_path / "manifest.json",
            )

    def test_empty_input_dir_raises(self, tmp_path):
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        with pytest.raises(ValueError, match="No CSV files found"):
            build_dl_signal_artifact(
                input_dir,
                tmp_path / "out.parquet",
                tmp_path / "manifest.json",
            )

    def test_missing_required_columns_raises(self, tmp_path):
        df = pd.DataFrame({"entry_time": ["2023-01-01"], "pair": ["eur-usd"]})
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "bad.csv").write_text(df.to_csv(index=False))

        with pytest.raises(ValueError, match="missing required columns"):
            build_dl_signal_artifact(
                input_dir,
                tmp_path / "out.parquet",
                tmp_path / "manifest.json",
            )

    def test_no_standardisation_of_signal(self, tmp_path):
        """signal_strength must preserve [-1,1] semantics without normalisation."""
        raw = _minimal_raw()
        raw["pred_prob_up"] = [0.0, 0.25, 0.75, 1.0]
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "preds.csv").write_text(raw.to_csv(index=False))

        out_pq = tmp_path / "out.parquet"
        out_mf = tmp_path / "manifest.json"
        result = build_dl_signal_artifact(input_dir, out_pq, out_mf)

        expected = pd.Series([-1.0, -0.5, 0.5, 1.0])
        pd.testing.assert_series_equal(
            result["signal_strength"].reset_index(drop=True),
            expected,
            check_names=False,
        )

    def test_emits_deprecation_warning(self, tmp_path):
        """build_dl_signal_artifact must emit a DeprecationWarning."""
        raw = _minimal_raw()
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "preds.csv").write_text(raw.to_csv(index=False))
        with pytest.warns(DeprecationWarning, match="deprecated"):
            build_dl_signal_artifact(
                input_dir,
                tmp_path / "out.parquet",
                tmp_path / "manifest.json",
            )


# ---------------------------------------------------------------------------
# Schema v1 tighten-ups
# ---------------------------------------------------------------------------


class TestSchemaV1Updates:
    """Tests for v1 schema contract tighten-ups."""

    def test_target_horizon_is_int64(self):
        """target_horizon must be Int64 (nullable integer), not a string."""
        art = _make_artifact()
        assert art["target_horizon"].dtype == "Int64"
        assert art["target_horizon"].iloc[0] == 24

    def test_target_horizon_na_when_absent(self):
        """target_horizon must be pd.NA (not 'unknown') when not supplied."""
        raw = _minimal_raw()[["entry_time", "pair", "pred_prob_up"]]
        art = _build_artifact(raw)
        assert art["target_horizon"].isna().all()

    def test_target_horizon_coerces_numeric_strings(self):
        """String '24' should be coerced to Int64(24)."""
        raw = _minimal_raw()
        raw["target_horizon"] = ["24"] * len(raw)
        art = _build_artifact(raw)
        assert art["target_horizon"].dtype == "Int64"
        assert (art["target_horizon"] == 24).all()

    def test_pred_direction_neutral_at_half(self):
        """pred_prob_up == 0.5 must produce pred_direction == 0 (neutral)."""
        raw = _minimal_raw()
        raw["pred_prob_up"] = [0.5, 0.5, 0.5, 0.5]
        art = _build_artifact(raw)
        assert (art["pred_direction"] == 0).all()

    def test_pred_direction_tristate_values(self):
        """pred_direction must be in {-1, 0, +1} only."""
        raw = _minimal_raw()
        raw["pred_prob_up"] = [0.6, 0.5, 0.4, 0.5]
        art = _build_artifact(raw)
        expected = pd.array([1, 0, -1, 0], dtype="Int64")
        pd.testing.assert_extension_array_equal(
            art["pred_direction"].values, expected
        )

    def test_prediction_timestamp_null_when_absent(self):
        """prediction_timestamp must be NaT when not provided."""
        art = _make_artifact()
        assert "prediction_timestamp" in art.columns
        assert art["prediction_timestamp"].isna().all()

    def test_prediction_timestamp_preserved_when_supplied(self):
        """prediction_timestamp must be kept when provided."""
        raw = _minimal_raw()
        raw["prediction_timestamp"] = pd.date_range("2023-01-02 00:30", periods=4, freq="h")
        art = _build_artifact(raw)
        assert not art["prediction_timestamp"].isna().all()

    def test_output_cols_include_prediction_timestamp(self):
        """prediction_timestamp must be in the output column list."""
        from build_dl_signal_artifact import OUTPUT_COLS as cols
        assert "prediction_timestamp" in cols

    def test_surface_grain_cols_defined(self):
        """SURFACE_GRAIN_COLS constant must be defined and contain expected cols."""
        expected = {"pair", "model", "dl_regime", "target_horizon", "feature_set"}
        assert expected == set(SURFACE_GRAIN_COLS)

