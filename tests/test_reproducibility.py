"""
tests/test_reproducibility.py
==============================
Minimal tests for standardized experiment logging and config JSON snapshots.

Validates:
- log file is created with the correct naming pattern
- config JSON is created alongside the log file
- config JSON contains ``cli_command``

Run with::

    python -m pytest tests/test_reproducibility.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_abm_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "pair": ["eur-usd"] * n,
        "entry_time": times,
        "snapshot_time": times,
        "entry_close": 1.10 + rng.standard_normal(n) * 0.001,
        "net_sentiment": rng.uniform(-80, 80, size=n),
    })


# ---------------------------------------------------------------------------
# utils/logging tests
# ---------------------------------------------------------------------------

class TestSetupExperimentLogging:
    def test_log_file_created(self, tmp_path):
        import config as cfg
        from utils.logging import setup_experiment_logging

        with patch.object(cfg, "REPO_ROOT", tmp_path):
            log_file = setup_experiment_logging(
                experiment_type="mlp",
                tag="price-only",
                log_level="WARNING",
                no_log_file=False,
            )

        assert log_file is not None
        assert log_file.exists()
        assert log_file.name.startswith("mlp_price-only_")
        assert log_file.suffix == ".log"

    def test_no_log_file_flag(self, tmp_path):
        import config as cfg
        from utils.logging import setup_experiment_logging

        with patch.object(cfg, "REPO_ROOT", tmp_path):
            log_file = setup_experiment_logging(
                experiment_type="lstm",
                tag="seq-v1",
                log_level="WARNING",
                no_log_file=True,
            )

        assert log_file is None
        assert not (tmp_path / "logs").exists() or not list((tmp_path / "logs").glob("*.log"))

    def test_json_path_derivable(self, tmp_path):
        import config as cfg
        from utils.logging import setup_experiment_logging

        with patch.object(cfg, "REPO_ROOT", tmp_path):
            log_file = setup_experiment_logging(
                experiment_type="abm",
                tag="eur-usd",
                log_level="WARNING",
                no_log_file=False,
            )

        assert log_file is not None
        json_path = log_file.with_suffix(".json")
        assert json_path.name.startswith("abm_eur-usd_")
        assert json_path.suffix == ".json"


# ---------------------------------------------------------------------------
# ABM reproducibility tests
# ---------------------------------------------------------------------------

class TestAbmReproducibility:
    def test_log_file_created(self, tmp_path):
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_abm_dataset(), tmp_path / "fake.csv"),
            ):
                main(["--version", "1.0.0", "--pair", "eur-usd", "--steps", "10"])

        log_files = list((tmp_path / "logs").glob("abm_eur-usd_*.log"))
        assert len(log_files) == 1

    def test_config_json_created(self, tmp_path):
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_abm_dataset(), tmp_path / "fake.csv"),
            ):
                main(["--version", "1.0.0", "--pair", "eur-usd", "--steps", "10"])

        json_files = list((tmp_path / "logs").glob("abm_eur-usd_*.json"))
        assert len(json_files) == 1

    def test_cli_command_in_json(self, tmp_path):
        from research.abm.run_abm import main

        with patch("research.abm.run_abm.cfg") as mock_cfg:
            mock_cfg.LOG_LEVEL = "WARNING"
            mock_cfg.REPO_ROOT = tmp_path
            mock_cfg.OUTPUT_DIR = tmp_path / "data" / "output"

            with patch(
                "research.abm.run_abm._load_real_data",
                return_value=(_make_minimal_abm_dataset(), tmp_path / "fake.csv"),
            ):
                main(["--version", "1.0.0", "--pair", "eur-usd", "--steps", "10"])

        json_files = list((tmp_path / "logs").glob("abm_eur-usd_*.json"))
        payload = json.loads(json_files[0].read_text())
        assert "cli_command" in payload
        assert isinstance(payload["cli_command"], str)
        assert "experiment_type" in payload
        assert payload["experiment_type"] == "abm"
