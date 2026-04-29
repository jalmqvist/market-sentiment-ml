"""
tests/test_abm.py
=================
Unit tests for the retail FX sentiment agent-based model.

Tests cover:
1. Agent construction and boundary values.
2. Agent position updates (directional correctness).
3. FXSentimentSimulation output schema and invariants.
4. FXSentimentSimulation with an external price series.
5. Calibration helpers (calibrate_from_dataset, compare_to_data).

Run with::

    python -m pytest tests/test_abm.py -v
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

from research.abm.agents import Contrarian, NoiseTrader, RetailTrader, TrendFollower
from research.abm.calibration import calibrate_from_dataset, compare_to_data
from research.abm.simulation import FXSentimentSimulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_agents(n_trend=10, n_contrarian=10, n_noise=5, seed=0):
    rng = _rng(seed)
    agents = (
        [TrendFollower(rng) for _ in range(n_trend)]
        + [Contrarian(rng) for _ in range(n_contrarian)]
        + [NoiseTrader(rng) for _ in range(n_noise)]
    )
    return agents, rng


def _make_real_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset mimicking the real research dataset columns."""
    rng = np.random.default_rng(seed)
    net_sentiment = rng.uniform(-90, 90, size=n)
    pairs = np.array(["eur-usd", "usd-jpy"])[rng.integers(0, 2, size=n)]
    times = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "pair": pairs,
        "net_sentiment": net_sentiment,
        "snapshot_time": times,
    })


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestTrendFollower:
    def test_construction_defaults(self):
        agent = TrendFollower(_rng())
        assert agent.momentum_window == 12
        assert agent.position in {-1, 0, 1}

    def test_invalid_momentum_window(self):
        with pytest.raises(ValueError):
            TrendFollower(_rng(), momentum_window=0)

    def test_goes_long_after_up_move(self):
        """With enough up-moves the trend-follower must go long (no noise)."""
        agent = TrendFollower(_rng(1), momentum_window=3, noise_scale=0.0, crowd_weight=0.0)
        prices = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
        agent.update(prices, crowd_sentiment=0.0)
        assert agent.position == 1

    def test_goes_short_after_down_move(self):
        agent = TrendFollower(_rng(2), momentum_window=3, noise_scale=0.0, crowd_weight=0.0)
        prices = np.array([1.04, 1.03, 1.02, 1.01, 1.00])
        agent.update(prices, crowd_sentiment=0.0)
        assert agent.position == -1

    def test_no_update_with_single_price(self):
        agent = TrendFollower(_rng())
        initial_pos = agent.position
        agent.update(np.array([1.0]), crowd_sentiment=0.0)
        assert agent.position == initial_pos


class TestContrarian:
    def test_construction_defaults(self):
        agent = Contrarian(_rng())
        assert agent.momentum_window == 12
        assert agent.position in {-1, 0, 1}

    def test_fades_up_move(self):
        """Contrarian must go short after an up-move (no noise)."""
        agent = Contrarian(_rng(1), momentum_window=3, noise_scale=0.0, crowd_weight=0.0)
        prices = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
        agent.update(prices, crowd_sentiment=0.0)
        assert agent.position == -1

    def test_fades_down_move(self):
        agent = Contrarian(_rng(2), momentum_window=3, noise_scale=0.0, crowd_weight=0.0)
        prices = np.array([1.04, 1.03, 1.02, 1.01, 1.00])
        agent.update(prices, crowd_sentiment=0.0)
        assert agent.position == 1

    def test_invalid_momentum_window(self):
        with pytest.raises(ValueError):
            Contrarian(_rng(), momentum_window=0)


class TestNoiseTrader:
    def test_construction_defaults(self):
        agent = NoiseTrader(_rng())
        assert agent.position in {-1, 0, 1}

    def test_price_signal_is_zero(self):
        agent = NoiseTrader(_rng())
        # _price_signal is always 0; position is driven by noise only.
        assert agent._price_signal(np.array([1.0, 1.1])) == 0.0


class TestRetailTraderAbstract:
    def test_cannot_call_price_signal_directly(self):
        """RetailTrader._price_signal must raise NotImplementedError."""
        agent = RetailTrader.__new__(RetailTrader)
        agent.rng = _rng()
        agent.crowd_weight = 0.0
        agent.noise_scale = 0.1
        agent.position = 0
        with pytest.raises(NotImplementedError):
            agent._price_signal(np.array([1.0, 1.1]))


# ---------------------------------------------------------------------------
# FXSentimentSimulation tests
# ---------------------------------------------------------------------------

class TestFXSentimentSimulation:
    def _sim(self, **kwargs):
        agents, rng = _make_agents()
        defaults = dict(agents=agents, rng=rng, warmup_steps=5)
        defaults.update(kwargs)
        return FXSentimentSimulation(**defaults)

    def test_output_schema(self):
        sim = self._sim()
        df = sim.run(n_steps=50)
        expected_cols = {"step", "price", "net_sentiment", "abs_sentiment",
                         "crowd_side", "n_long", "n_short", "n_flat"}
        assert expected_cols.issubset(df.columns)

    def test_output_row_count(self):
        sim = self._sim()
        n = 100
        df = sim.run(n_steps=n)
        assert len(df) == n

    def test_step_index_zero_based(self):
        sim = self._sim()
        df = sim.run(n_steps=20)
        assert df["step"].iloc[0] == 0
        assert df["step"].iloc[-1] == 19

    def test_net_sentiment_bounded(self):
        sim = self._sim()
        df = sim.run(n_steps=200)
        assert (df["net_sentiment"] >= -100).all()
        assert (df["net_sentiment"] <= 100).all()

    def test_abs_sentiment_is_abs(self):
        sim = self._sim()
        df = sim.run(n_steps=100)
        pd.testing.assert_series_equal(
            df["abs_sentiment"],
            df["net_sentiment"].abs(),
            check_names=False,
        )

    def test_crowd_side_sign(self):
        sim = self._sim()
        df = sim.run(n_steps=100)
        expected = df["net_sentiment"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        ).astype(int)
        pd.testing.assert_series_equal(
            df["crowd_side"].astype(int),
            expected,
            check_names=False,
        )

    def test_agent_counts_sum_to_total(self):
        agents, rng = _make_agents(n_trend=10, n_contrarian=10, n_noise=5)
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=5)
        df = sim.run(n_steps=50)
        totals = df["n_long"] + df["n_short"] + df["n_flat"]
        assert (totals == sim.n_agents).all()

    def test_n_agents_property(self):
        agents, rng = _make_agents(n_trend=3, n_contrarian=2, n_noise=1)
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=2)
        assert sim.n_agents == 6

    def test_raises_on_empty_agents(self):
        with pytest.raises(ValueError):
            FXSentimentSimulation(agents=[])

    def test_raises_on_zero_steps(self):
        sim = self._sim()
        with pytest.raises(ValueError):
            sim.run(n_steps=0)

    def test_raises_on_negative_volatility(self):
        agents, rng = _make_agents()
        with pytest.raises(ValueError):
            FXSentimentSimulation(agents, rng=rng, volatility=-0.01)

    def test_raises_on_non_positive_initial_price(self):
        agents, rng = _make_agents()
        with pytest.raises(ValueError):
            FXSentimentSimulation(agents, rng=rng, initial_price=0.0)

    def test_with_external_price_series(self):
        agents, rng = _make_agents()
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=5)
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=100, price_series=prices)
        assert len(df) == 100

    def test_external_price_series_too_short_raises(self):
        agents, rng = _make_agents()
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=10)
        prices = np.ones(5)  # way too short
        with pytest.raises(ValueError):
            sim.run(n_steps=100, price_series=prices)

    def test_reproducible_with_same_seed(self):
        agents1, rng1 = _make_agents(seed=7)
        agents2, rng2 = _make_agents(seed=7)
        sim1 = FXSentimentSimulation(agents1, rng=rng1, warmup_steps=5)
        sim2 = FXSentimentSimulation(agents2, rng=rng2, warmup_steps=5)
        df1 = sim1.run(n_steps=50)
        df2 = sim2.run(n_steps=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        agents1, rng1 = _make_agents(seed=1)
        agents2, rng2 = _make_agents(seed=2)
        sim1 = FXSentimentSimulation(agents1, rng=rng1, warmup_steps=5)
        sim2 = FXSentimentSimulation(agents2, rng=rng2, warmup_steps=5)
        df1 = sim1.run(n_steps=100)
        df2 = sim2.run(n_steps=100)
        # Net sentiment should not be identical across different seeds.
        assert not df1["net_sentiment"].equals(df2["net_sentiment"])


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibrateFromDataset:
    def test_returns_dict(self):
        df = _make_real_df()
        result = calibrate_from_dataset(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_real_df()
        result = calibrate_from_dataset(df)
        for key in ("mean", "std", "abs_mean", "autocorr", "extreme_freq", "long_frac", "n_rows"):
            assert key in result, f"Missing key: {key}"

    def test_n_rows_matches(self):
        df = _make_real_df(n=200)
        result = calibrate_from_dataset(df)
        assert result["n_rows"] == 200

    def test_pair_filter(self):
        df = _make_real_df(n=300)
        n_eurusd = (df["pair"] == "eur-usd").sum()
        result = calibrate_from_dataset(df, pair="eur-usd")
        assert result["n_rows"] == n_eurusd
        assert result["pair"] == "eur-usd"

    def test_unknown_pair_raises(self):
        df = _make_real_df()
        with pytest.raises(ValueError):
            calibrate_from_dataset(df, pair="xxx-yyy")

    def test_missing_columns_raises(self):
        df = _make_real_df().drop(columns=["net_sentiment"])
        with pytest.raises(ValueError):
            calibrate_from_dataset(df)

    def test_extreme_freq_in_range(self):
        df = _make_real_df()
        result = calibrate_from_dataset(df)
        assert 0.0 <= result["extreme_freq"] <= 1.0

    def test_long_frac_in_range(self):
        df = _make_real_df()
        result = calibrate_from_dataset(df)
        assert 0.0 <= result["long_frac"] <= 1.0

    def test_abs_mean_non_negative(self):
        df = _make_real_df()
        result = calibrate_from_dataset(df)
        assert result["abs_mean"] >= 0.0


class TestCompareToData:
    def _targets(self) -> dict:
        return calibrate_from_dataset(_make_real_df())

    def _sim_df(self, n: int = 200) -> pd.DataFrame:
        agents, rng = _make_agents(seed=99)
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=5)
        return sim.run(n_steps=n)

    def test_returns_dataframe(self):
        result = compare_to_data(self._sim_df(), self._targets())
        assert isinstance(result, pd.DataFrame)

    def test_schema(self):
        result = compare_to_data(self._sim_df(), self._targets())
        for col in ("statistic", "simulated", "real", "abs_diff", "rel_diff"):
            assert col in result.columns, f"Missing column: {col}"

    def test_all_stats_present(self):
        result = compare_to_data(self._sim_df(), self._targets())
        expected_stats = {"mean", "std", "abs_mean", "autocorr", "extreme_freq", "long_frac"}
        assert expected_stats.issubset(set(result["statistic"]))

    def test_abs_diff_non_negative(self):
        result = compare_to_data(self._sim_df(), self._targets())
        finite = result["abs_diff"].dropna()
        assert (finite >= 0).all()

    def test_missing_sim_col_raises(self):
        sim_df = self._sim_df().drop(columns=["net_sentiment"])
        with pytest.raises(ValueError):
            compare_to_data(sim_df, self._targets())

    def test_empty_sim_df_raises(self):
        sim_df = self._sim_df().iloc[:0]
        with pytest.raises(ValueError):
            compare_to_data(sim_df, self._targets())
