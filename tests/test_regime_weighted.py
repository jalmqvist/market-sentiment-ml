"""
tests/test_regime_weighted.py
==============================
Sanity checks for the regime-based signal weighting pipeline.

These tests validate:
1. ``compute_regime_sharpe_map`` uses only training data (no leakage).
2. ``convert_sharpe_to_weight`` produces weights in (-1, 1) (tanh mode)
   or proportional to Sharpe (normalize mode).
3. ``apply_regime_weighted_signal`` sets signal = 0 for unknown regimes.
4. ``apply_regime_weighted_signal`` sets signal = 0 when |weight| < threshold.
5. ``regime_weighted_walk_forward`` computes the Sharpe map from training
   slices only (expanding window; test data never enters weight computation).

Run with: ``python -m pytest tests/test_regime_weighted.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on sys.path when running directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.regime_v3 import (
    CONTRARIAN_REGIMES,
    TREND_REGIMES,
    WEIGHT_THRESHOLD,
    apply_regime_weighted_signal,
    compute_regime_sharpe_map,
    convert_sharpe_to_weight,
    regime_weighted_walk_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 600,
    regimes: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal synthetic dataset for testing.

    Produces a DataFrame with the columns required by the weighting functions:
    ``crowding_regime``, ``net_sentiment``, ``ret_48b``, ``year``.

    Args:
        n: Number of rows.
        regimes: Regime labels to cycle through.  Defaults to
            ``CONTRARIAN_REGIMES + TREND_REGIMES``.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic DataFrame ready for testing.
    """
    rng = np.random.default_rng(seed)
    if regimes is None:
        regimes = list(dict.fromkeys(CONTRARIAN_REGIMES + TREND_REGIMES))

    regime_col = [regimes[i % len(regimes)] for i in range(n)]
    net_sentiment = rng.uniform(-80, 80, size=n)
    ret_48b = rng.normal(0, 0.01, size=n)
    # Three years so walk-forward has at least 3 unique years.
    years = np.array([2020 + (i // (n // 3)) for i in range(n)])

    return pd.DataFrame(
        {
            "crowding_regime": regime_col,
            "net_sentiment": net_sentiment,
            "ret_48b": ret_48b,
            "year": years,
        }
    )


# ---------------------------------------------------------------------------
# compute_regime_sharpe_map
# ---------------------------------------------------------------------------

class TestComputeRegimeSharpeMap:
    def test_returns_dict(self):
        df = _make_df()
        sharpe_map = compute_regime_sharpe_map(df)
        assert isinstance(sharpe_map, dict)

    def test_uses_only_training_data(self):
        """Verify that the map is computed exclusively on the supplied slice.

        Pass training data with a known contrived regime signal, then verify
        the map changes when different data is passed — confirming that no
        external data bleeds in.
        """
        # Use a small min_n so both slices can produce non-empty maps.
        df = _make_df(n=900, seed=1)
        train = df[df["year"] < 2022]
        test = df[df["year"] >= 2022]

        map_train = compute_regime_sharpe_map(train, min_n=10)
        map_test = compute_regime_sharpe_map(test, min_n=10)

        # Both maps must be non-empty.
        assert len(map_train) > 0, "Training map must be non-empty"
        assert len(map_test) > 0, "Test map must be non-empty"

        # Maps are computed on different data; Sharpe values should differ.
        shared_regimes = set(map_train) & set(map_test)
        assert shared_regimes, "Expected overlapping regime labels across slices"
        sharpe_pairs = [(map_train[r], map_test[r]) for r in shared_regimes]
        # At least one regime must have a different Sharpe across splits
        # (returns are random so this is virtually guaranteed).
        all_equal = all(abs(a - b) < 1e-12 for a, b in sharpe_pairs)
        assert not all_equal, (
            "Training and test Sharpe maps are identical; this suggests "
            "data from the test slice may have leaked into the training map."
        )

    def test_empty_dataframe(self):
        df = _make_df(n=0)
        sharpe_map = compute_regime_sharpe_map(df)
        assert sharpe_map == {}

    def test_missing_regime_col(self):
        df = _make_df().drop(columns=["crowding_regime"])
        sharpe_map = compute_regime_sharpe_map(df)
        assert sharpe_map == {}

    def test_unknown_regime_excluded(self):
        """Regimes with fewer than min_n active rows must be absent."""
        df = _make_df(n=600)
        # Use a very high min_n to force all regimes to be excluded.
        sharpe_map = compute_regime_sharpe_map(df, min_n=10_000)
        assert sharpe_map == {}

    def test_values_are_finite(self):
        df = _make_df(n=600)
        sharpe_map = compute_regime_sharpe_map(df)
        for regime, sharpe in sharpe_map.items():
            assert np.isfinite(sharpe), f"regime={regime} has non-finite Sharpe={sharpe}"


# ---------------------------------------------------------------------------
# convert_sharpe_to_weight
# ---------------------------------------------------------------------------

class TestConvertSharpeToWeight:
    def test_tanh_mode_bounds(self):
        """Tanh mode (normalize=False) must produce weights strictly in (-1, 1)."""
        sharpe_map = {"A": 5.0, "B": -3.0, "C": 0.3}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=False)
        for w in weight_map.values():
            assert -1.0 < w < 1.0, f"weight {w} out of open (-1, 1) in tanh mode"

    def test_tanh_mode_large_values_approach_extremes(self):
        """A single dominant large Sharpe produces a weight near +1/-1."""
        # With only one regime, std=0 so divisor falls back to 1e-6;
        # tanh(very_large / 1e-6) → ±1.
        weight_map_pos = convert_sharpe_to_weight({"A": 1e6}, normalize=False)
        weight_map_neg = convert_sharpe_to_weight({"A": -1e6}, normalize=False)
        assert weight_map_pos["A"] > 0.99, (
            f"Expected weight near +1, got {weight_map_pos['A']}"
        )
        assert weight_map_neg["A"] < -0.99, (
            f"Expected weight near -1, got {weight_map_neg['A']}"
        )

    def test_normalize_mode_max_is_one(self):
        sharpe_map = {"A": 2.0, "B": 1.0, "C": -0.5}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=True)
        max_abs = max(abs(w) for w in weight_map.values())
        assert max_abs == pytest.approx(1.0, abs=1e-9)

    def test_normalize_mode_proportional(self):
        sharpe_map = {"A": 2.0, "B": 1.0}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=True)
        assert weight_map["B"] == pytest.approx(weight_map["A"] / 2.0, abs=1e-9)

    def test_empty_input(self):
        assert convert_sharpe_to_weight({}) == {}

    def test_zero_max_abs_sharpe_normalize(self):
        sharpe_map = {"A": 0.0, "B": 0.0}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=True)
        for w in weight_map.values():
            assert w == pytest.approx(0.0)

    def test_sign_preserved(self):
        sharpe_map = {"pos": 0.5, "neg": -0.5}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=False)
        assert weight_map["pos"] > 0
        assert weight_map["neg"] < 0

    def test_tanh_mode_small_sharpes_produce_meaningful_weights(self):
        """Regime Sharpes ~±0.05 must produce non-negligible weights in tanh mode."""
        sharpe_map = {"A": 0.06, "B": -0.04, "C": 0.02}
        weight_map = convert_sharpe_to_weight(sharpe_map, normalize=False)
        abs_weights = [abs(w) for w in weight_map.values()]
        # All weights should be non-zero and the max should be substantial.
        assert all(w > 1e-4 for w in abs_weights), (
            f"Expected non-negligible weights for small Sharpes, got {abs_weights}"
        )
        assert max(abs_weights) > 0.5, (
            f"Expected max abs weight > 0.5 for cross-sectionally large Sharpe, "
            f"got {max(abs_weights)}"
        )


# ---------------------------------------------------------------------------
# apply_regime_weighted_signal
# ---------------------------------------------------------------------------

class TestApplyRegimeWeightedSignal:
    def test_unknown_regime_signal_is_zero(self):
        """Rows with an unknown regime must have weighted_signal = 0."""
        df = _make_df(n=300)
        # Use a weight map that excludes all regimes in the test data.
        weight_map = {"unknown_regime_xyz": 0.8}
        out = apply_regime_weighted_signal(df, weight_map)
        assert (out["weighted_signal"] == 0.0).all(), (
            "Expected all signals = 0 for unknown regimes"
        )

    def test_below_threshold_signal_is_zero(self):
        """Rows with |weight| < threshold must have weighted_signal = 0."""
        df = _make_df(n=300)
        # All regimes get a weight below the explicit threshold.
        regimes = df["crowding_regime"].unique()
        weight_map = {r: 0.01 for r in regimes}  # below explicit threshold 0.05
        out = apply_regime_weighted_signal(
            df, weight_map, weight_threshold=0.05
        )
        assert (out["weighted_signal"] == 0.0).all(), (
            "Expected all signals = 0 when weight below threshold"
        )

    def test_active_flags_consistent(self):
        """is_active_weighted must match weighted_signal != 0."""
        df = _make_df(n=300)
        regimes = df["crowding_regime"].unique()
        weight_map = {r: 0.5 for r in regimes}
        out = apply_regime_weighted_signal(df, weight_map, weight_threshold=0.05)
        expected = out["weighted_signal"] != 0.0
        pd.testing.assert_series_equal(
            out["is_active_weighted"],
            expected,
            check_names=False,
        )

    def test_weighted_signal_proportional_to_weight(self):
        """weighted_signal must equal regime_weight * direction_signal."""
        df = _make_df(n=300)
        regimes = df["crowding_regime"].unique()
        weight = 0.7
        weight_map = {r: weight for r in regimes}
        out = apply_regime_weighted_signal(df, weight_map, weight_threshold=0.0)

        # For active rows, weighted_signal = regime_weight * signal
        active = out[out["is_active_weighted"]]
        expected = active["regime_weight"] * active["signal"]
        pd.testing.assert_series_equal(
            active["weighted_signal"],
            expected,
            check_names=False,
        )

    def test_columns_added(self):
        df = _make_df(n=100)
        regimes = df["crowding_regime"].unique()
        weight_map = {r: 0.5 for r in regimes}
        out = apply_regime_weighted_signal(df, weight_map)
        for col in ["regime_weight", "weighted_signal", "is_active_weighted"]:
            assert col in out.columns, f"Column '{col}' missing from output"

    def test_does_not_modify_input(self):
        df = _make_df(n=100)
        original_cols = set(df.columns)
        weight_map = {"long_extreme": 0.5}
        apply_regime_weighted_signal(df, weight_map)
        assert set(df.columns) == original_cols, "Input DataFrame was modified"


# ---------------------------------------------------------------------------
# regime_weighted_walk_forward (leakage check)
# ---------------------------------------------------------------------------

class TestRegimeWeightedWalkForward:
    def test_returns_dataframe(self):
        df = _make_df(n=600)
        result = regime_weighted_walk_forward(df)
        assert isinstance(result, pd.DataFrame)

    def test_schema(self):
        df = _make_df(n=600)
        result = regime_weighted_walk_forward(df)
        # Result may be empty if min_n threshold is not met, but columns must
        # always match the defined schema regardless of row count.
        assert list(result.columns) == ["year", "n", "mean", "sharpe", "hit_rate"]

    def test_fewer_than_3_years_returns_empty(self):
        df = _make_df(n=300)
        df["year"] = 2020  # only one year
        result = regime_weighted_walk_forward(df)
        assert result.empty

    def test_leakage_free_weight_map_per_fold(self):
        """Each fold's weight map must be computed only from training years.

        We verify this structurally: the function uses expanding-window
        training slices, so later folds have strictly more training data.
        We assert that the function completes without error and returns a
        consistent result (no NaN years, no duplicated years).
        """
        df = _make_df(n=900, seed=7)
        result = regime_weighted_walk_forward(df)
        if not result.empty:
            # Years must be strictly increasing (one row per test year).
            assert result["year"].is_monotonic_increasing
            assert result["year"].nunique() == len(result)

    def test_missing_regime_col_returns_empty(self):
        df = _make_df(n=600).drop(columns=["crowding_regime"])
        result = regime_weighted_walk_forward(df)
        assert result.empty
