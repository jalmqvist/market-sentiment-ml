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
6. run_abm CLI: validation, config JSON, log file, output CSV columns.

Run with::

    python -m pytest tests/test_abm.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

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
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=50, price_series=prices)
        expected_cols = {"step", "price", "net_sentiment", "abs_sentiment",
                         "crowd_side", "n_long", "n_short", "n_flat"}
        assert expected_cols.issubset(df.columns)

    def test_output_row_count(self):
        sim = self._sim()
        n = 100
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=n, price_series=prices)
        assert len(df) == n

    def test_step_index_zero_based(self):
        sim = self._sim()
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=20, price_series=prices)
        assert df["step"].iloc[0] == 0
        assert df["step"].iloc[-1] == 19

    def test_net_sentiment_bounded(self):
        sim = self._sim()
        prices = np.linspace(1.0, 1.1, 400)
        df = sim.run(n_steps=200, price_series=prices)
        assert (df["net_sentiment"] >= -100).all()
        assert (df["net_sentiment"] <= 100).all()

    def test_abs_sentiment_is_abs(self):
        sim = self._sim()
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=100, price_series=prices)
        pd.testing.assert_series_equal(
            df["abs_sentiment"],
            df["net_sentiment"].abs(),
            check_names=False,
        )

    def test_crowd_side_sign(self):
        sim = self._sim()
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=100, price_series=prices)
        expected = np.sign(df["net_sentiment"]).astype(int)
        pd.testing.assert_series_equal(
            df["crowd_side"].astype(int),
            expected,
            check_names=False,
        )

    def test_agent_counts_sum_to_total(self):
        agents, rng = _make_agents(n_trend=10, n_contrarian=10, n_noise=5)
        sim = FXSentimentSimulation(agents, rng=rng, warmup_steps=5)
        prices = np.linspace(1.0, 1.1, 200)
        df = sim.run(n_steps=50, price_series=prices)
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
        prices = np.linspace(1.0, 1.1, 200)
        with pytest.raises(ValueError):
            sim.run(n_steps=0, price_series=prices)

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
        prices = np.linspace(1.0, 1.1, 200)
        df1 = sim1.run(n_steps=50, price_series=prices)
        df2 = sim2.run(n_steps=50, price_series=prices)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        agents1, rng1 = _make_agents(seed=1)
        agents2, rng2 = _make_agents(seed=2)
        sim1 = FXSentimentSimulation(agents1, rng=rng1, warmup_steps=5)
        sim2 = FXSentimentSimulation(agents2, rng=rng2, warmup_steps=5)
        # Use a volatile random-walk price series so that agents generate
        # non-zero positions; a flat linspace produces signals too weak to
        # cross the decision threshold and makes both series identically zero.
        # Seed 99 is chosen to be independent of the agent seeds (1 and 2).
        price_rng = np.random.default_rng(99)
        returns = price_rng.normal(0, 0.005, 299)
        prices = np.concatenate([[1.0], (1.0 + returns).cumprod()])
        df1 = sim1.run(n_steps=100, price_series=prices)
        df2 = sim2.run(n_steps=100, price_series=prices)
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
        prices = np.linspace(1.0, 1.1, n + 10)
        return sim.run(n_steps=n, price_series=prices)

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


# ---------------------------------------------------------------------------
# run_abm CLI tests
# ---------------------------------------------------------------------------

def _make_minimal_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a minimal dataframe that _load_real_data would return."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "pair": ["eur-usd"] * n,
        "entry_time": times,
        "snapshot_time": times,
        "entry_close": 1.10 + rng.standard_normal(n) * 0.001,
        "net_sentiment": rng.uniform(-80, 80, size=n),
    })


class TestRunAbmCLI:
    """Tests for the run_abm entry-point."""

    def test_fails_without_pair(self):
        """main() must fail when --pair is missing."""
        from research.abm.run_abm import main

        with pytest.raises(SystemExit):
            main(["--version", "1.0.0"])

    def test_fails_without_version(self):
        """main() must fail when --version is missing."""
        from research.abm.run_abm import main

        with pytest.raises(SystemExit):
            main(["--pair", "eur-usd"])

    def test_fails_with_zero_steps(self, tmp_path):
        """main() must raise ValueError when --steps 0 is passed."""
        from research.abm.run_abm import main

        with pytest.raises(ValueError, match="steps must be > 0"):
            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_dataset(), Path("/fake/dataset.csv")),
            ):
                main([
                    "--version", "1.0.0",
                    "--pair", "eur-usd",
                    "--steps", "0",
                    "--no-log-file",
                ])

    def test_config_json_is_written(self, tmp_path):
        """A JSON config snapshot must appear in the log directory."""
        import config as cfg
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_dataset(), tmp_path / "fake.csv"),
            ):
                main([
                    "--version", "1.0.0",
                    "--pair", "eur-usd",
                    "--steps", "10",
                    "--seed", "0",
                ])

        json_files = list((tmp_path / "logs").glob("abm_eur-usd_*.json"))
        assert len(json_files) == 1, "Expected exactly one JSON config snapshot"

        payload = json.loads(json_files[0].read_text())
        for key in ("experiment_type", "cli_command", "dataset_path", "dataset_version",
                    "variant", "pair", "seed", "steps",
                    "n_trend", "n_contrarian", "n_noise", "momentum_window"):
            assert key in payload, f"Config JSON missing key: {key}"

    def test_log_file_is_created(self, tmp_path):
        """A .log file must appear in the log directory."""
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_dataset(), tmp_path / "fake.csv"),
            ):
                main([
                    "--version", "1.0.0",
                    "--pair", "eur-usd",
                    "--steps", "10",
                ])

        log_files = list((tmp_path / "logs").glob("abm_eur-usd_*.log"))
        assert len(log_files) == 1, "Expected exactly one log file"

    def test_no_log_file_flag(self, tmp_path):
        """--no-log-file must suppress log file creation."""
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_dataset(), tmp_path / "fake.csv"),
            ):
                main([
                    "--version", "1.0.0",
                    "--pair", "eur-usd",
                    "--steps", "10",
                    "--no-log-file",
                ])

        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("*.log")) if log_dir.exists() else []
        assert len(log_files) == 0, "--no-log-file should not create any .log files"

    def test_output_csv_columns(self, tmp_path):
        """Output CSV must contain exactly the required columns."""
        from research.abm.run_abm import main, _OUTPUT_COLUMNS

        output_path = tmp_path / "out.csv"

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_dataset(), tmp_path / "fake.csv"),
            ):
                main([
                    "--version", "1.0.0",
                    "--pair", "eur-usd",
                    "--steps", "10",
                    "--no-log-file",
                    "--output", str(output_path),
                ])

        assert output_path.exists(), "Output CSV was not created"
        out_df = pd.read_csv(output_path)
        for col in _OUTPUT_COLUMNS:
            assert col in out_df.columns, f"Output CSV missing column: {col}"
        assert len(out_df) > 0, "Output CSV is empty"
