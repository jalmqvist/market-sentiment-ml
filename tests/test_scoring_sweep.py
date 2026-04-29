"""
tests/test_scoring_sweep.py
===========================
Minimal tests for scoring.py and sweep.py.

Run with::

    python -m pytest tests/test_scoring_sweep.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.abm.scoring import compute_score, extract_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comparison_df() -> pd.DataFrame:
    """Synthetic comparison DataFrame matching compare_to_data schema."""
    return pd.DataFrame({
        "statistic": ["mean", "std", "abs_mean", "autocorr", "extreme_freq", "long_frac"],
        "simulated": [5.0, 30.0, 35.0, 0.5, 0.1, 0.55],
        "real":      [0.0, 25.0, 30.0, 0.4, 0.08, 0.5],
        "abs_diff":  [5.0,  5.0,  5.0, 0.1, 0.02, 0.05],
        "rel_diff":  [5.0 / 1e-12, 5.0 / 25.0, 5.0 / 30.0, 0.1 / 0.4, 0.02 / 0.08, 0.05 / 0.5],
    })


def _make_sweep_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset mimicking the real research dataset columns."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "pair":          ["eur-usd"] * n,
        "entry_time":    times,
        "snapshot_time": times,
        "entry_close":   1.10 + rng.standard_normal(n) * 0.001,
        "net_sentiment": rng.uniform(-80, 80, size=n),
    })


# ---------------------------------------------------------------------------
# scoring.extract_metric
# ---------------------------------------------------------------------------

class TestExtractMetric:
    def test_extracts_known_metric(self):
        df = _make_comparison_df()
        val = extract_metric(df, "std")
        assert val == pytest.approx(5.0 / 25.0)

    def test_returns_inf_for_missing_metric(self):
        df = _make_comparison_df()
        assert extract_metric(df, "nonexistent") == float("inf")

    def test_returns_inf_for_nan_rel_diff(self):
        df = pd.DataFrame({
            "statistic": ["std"],
            "rel_diff":  [float("nan")],
        })
        assert extract_metric(df, "std") == float("inf")


# ---------------------------------------------------------------------------
# scoring.compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:
    def test_returns_float(self):
        score = compute_score(_make_comparison_df())
        assert isinstance(score, float)

    def test_finite_for_valid_input(self):
        # Use a comparison df where rel_diff values are small and finite
        df = pd.DataFrame({
            "statistic": ["std", "abs_mean", "autocorr", "extreme_freq"],
            "rel_diff":  [0.2, 0.1, 0.3, 0.05],
        })
        score = compute_score(df)
        assert math.isfinite(score)

    def test_returns_inf_when_metric_missing(self):
        # Missing "autocorr" → inf
        df = pd.DataFrame({
            "statistic": ["std", "abs_mean", "extreme_freq"],
            "rel_diff":  [0.2, 0.1, 0.05],
        })
        assert compute_score(df) == float("inf")

    def test_returns_inf_when_metric_is_nan(self):
        df = pd.DataFrame({
            "statistic": ["std", "abs_mean", "autocorr", "extreme_freq"],
            "rel_diff":  [0.2, float("nan"), 0.3, 0.05],
        })
        assert compute_score(df) == float("inf")

    def test_weighted_sum_correct(self):
        df = pd.DataFrame({
            "statistic": ["std", "abs_mean", "autocorr", "extreme_freq"],
            "rel_diff":  [1.0, 1.0, 1.0, 1.0],
        })
        score = compute_score(df)
        expected = 0.3 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0 + 0.1 * 1.0
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sweep.run_sweep
# ---------------------------------------------------------------------------

class TestSweep:
    def test_sweep_runs_at_least_one_config(self):
        from research.abm.sweep import run_sweep

        df = _make_sweep_dataset()
        result = run_sweep(df, pair="eur-usd", n_steps=20, seed=0)
        assert len(result) >= 1

    def test_best_score_is_finite(self):
        from research.abm.sweep import run_sweep

        df = _make_sweep_dataset()
        result = run_sweep(df, pair="eur-usd", n_steps=20, seed=0)
        assert not result.empty
        best_score = result.iloc[0]["score"]
        assert math.isfinite(best_score)

    def test_result_has_required_columns(self):
        from research.abm.sweep import run_sweep

        df = _make_sweep_dataset()
        result = run_sweep(df, pair="eur-usd", n_steps=20, seed=0)
        for col in ("trend_ratio", "persistence", "threshold", "score",
                    "std_diff", "autocorr_diff"):
            assert col in result.columns, f"Missing column: {col}"

    def test_results_sorted_by_score(self):
        from research.abm.sweep import run_sweep

        df = _make_sweep_dataset()
        result = run_sweep(df, pair="eur-usd", n_steps=20, seed=0)
        scores = result["score"].values
        assert list(scores) == sorted(scores), "Results should be sorted ascending by score"

    def test_module_constants_restored_after_sweep(self):
        """Sweep must restore _PERSISTENCE_WEIGHT and _INERTIA_THRESHOLD."""
        from research.abm import agents as agents_module
        from research.abm.sweep import run_sweep

        orig_persistence = agents_module._PERSISTENCE_WEIGHT
        orig_threshold = agents_module._INERTIA_THRESHOLD

        df = _make_sweep_dataset()
        run_sweep(df, pair="eur-usd", n_steps=10, seed=0)

        assert agents_module._PERSISTENCE_WEIGHT == orig_persistence
        assert agents_module._INERTIA_THRESHOLD == orig_threshold
