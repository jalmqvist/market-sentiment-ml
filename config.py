"""
config.py
=========
Centralized configuration for the market-sentiment-ml research pipeline.

All hardcoded values (paths, thresholds, horizons, filters) are defined here.
Override via environment variables or CLI arguments where appropriate.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

SENTIMENT_DIR = INPUT_DIR / "sentiment"
PRICE_DIR = INPUT_DIR / "fx"
REGIME_DIR = INPUT_DIR / "regimes"

MASTER_DATASET_PATH = OUTPUT_DIR / "master_research_dataset.csv"
MASTER_DATASET_CORE_PATH = OUTPUT_DIR / "master_research_dataset_core.csv"
MASTER_DATASET_EXTENDED_PATH = OUTPUT_DIR / "master_research_dataset_extended.csv"
MASTER_DATASET_WITH_REGIME_PATH = OUTPUT_DIR / "master_research_dataset_with_regime.csv"
PAIR_COVERAGE_PATH = OUTPUT_DIR / "pair_coverage_summary.csv"

# ---------------------------------------------------------------------------
# Canonical dataset aliases
# ---------------------------------------------------------------------------

# DATA_PATH: canonical base dataset used for signal discovery, walk-forward
# evaluation, portfolio construction, and sanity checks.
DATA_PATH: Path = MASTER_DATASET_PATH  # data/output/master_research_dataset.csv

# DATA_PATH_REGIME: regime-enriched dataset.  Use ONLY for regime experiments,
# conditioning tests, and feature engineering that requires regime columns
# (phase, is_trending, is_high_vol).  Scripts that require this dataset will
# fail clearly if those columns are absent.
DATA_PATH_REGIME: Path = MASTER_DATASET_WITH_REGIME_PATH  # data/output/master_research_dataset_with_regime.csv
DATASET_MANIFEST_PATH = OUTPUT_DIR / "DATASET_MANIFEST.json"

FEATURES_DIR = OUTPUT_DIR / "features"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Walk-forward results
WF_RESULTS_PATH = ANALYSIS_DIR / "jpy_walk_forward_results.csv"

# Regime labels (from companion repo) — override via REGIME_PARQUET_PATH env var
_default_regime_parquet = (
    REPO_ROOT / "../market-phase-ml/data/output/regimes/phase_labels_d1.parquet"
).resolve()
REGIME_PARQUET_DEFAULT = Path(
    os.environ.get("REGIME_PARQUET_PATH", str(_default_regime_parquet))
)

# ---------------------------------------------------------------------------
# Return / forward-return horizons (number of hourly bars)
# ---------------------------------------------------------------------------

HORIZONS: list[int] = [1, 2, 4, 6, 12, 24, 48]
EVAL_HORIZONS: list[int] = [12, 48]  # horizons used in evaluation/experiments

# ---------------------------------------------------------------------------
# Dataset build configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
MERGE_TOLERANCE = "90min"

# Timezone offsets for raw data
SENTIMENT_ASSUMED_UTC_OFFSET = "+02:00"
PRICE_ASSUMED_UTC_OFFSET = "+01:00"
SNAPSHOT_SHIFT = "-1h"

# Coverage thresholds for dataset variant selection
CORE_MIN_ELIGIBLE_MATCH_RATIO: float = 0.95
EXTENDED_MIN_ELIGIBLE_MATCH_RATIO: float = 0.90

# Return sanity filter: drop returns > 20% (likely bad data)
MAX_RETURN_THRESHOLD: float = 0.20

# Corrupted pairs to exclude from the dataset
EXCLUDED_PAIRS: frozenset[str] = frozenset({"eur-mxn", "gbp-zar"})

# ---------------------------------------------------------------------------
# Signal configuration
# ---------------------------------------------------------------------------

# Canonical signal thresholds
SIGNAL_EXTREME_STREAK_MIN: int = 3
SIGNAL_PERSISTENCE_BUCKETS: list[str] = ["high", "medium"]

# Regime V2 signal
REGIME_V2_PAIR_GROUP: str = "JPY_cross"
REGIME_V2_PERSISTENCE_BUCKET: str = "high"
REGIME_V2_ACCELERATION_BUCKET: str = "decreasing"

# Holdout split year
HOLDOUT_SPLIT_YEAR: int = 2022

# Minimum number of signals required to compute meaningful stats
MIN_SIGNALS_FOR_STATS: int = 20
MIN_SIGNALS_FOR_PORTFOLIO: int = 50

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

USE_TREND_FILTER: bool = True
MAX_SIGNALS_PER_DAY: int = 2
USE_EQUAL_WEIGHT: bool = True
SURVIVOR_MIN_SIGNALS: int = 100
SURVIVOR_MIN_AFTER_DEDUP: int = 50
SURVIVOR_MIN_SHARPE: float = 0.08

# ---------------------------------------------------------------------------
# Pair grouping
# ---------------------------------------------------------------------------

JPY_PAIR_PATTERN: str = "JPY"  # case-insensitive substring match

# ---------------------------------------------------------------------------
# Trend strength bucket quantile labels
# ---------------------------------------------------------------------------

TREND_STRENGTH_LABELS: list[str] = ["weak", "medium", "strong", "extreme"]
TREND_STRENGTH_QUANTILES: int = 4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Regime attachment
# ---------------------------------------------------------------------------

VALID_PHASES: frozenset[str] = frozenset(
    {"HV_Trend", "HV_Ranging", "LV_Trend", "LV_Ranging", "Unknown"}
)
REGIME_WARN_MATCH_THRESHOLD: float = 0.99
