"""
research/deep_learning/dataset_loader.py
=========================================
PyTorch-ready dataset loader for the versioned master research dataset.

Responsibilities
----------------
- Load a dataset variant by version:   ``load_dataset(version)``
- Select features by group:            ``get_features(df, feature_set)``
- Chronological train/test split:      ``train_test_split(df, test_ratio)``
- Return numpy arrays or torch tensors: ``to_tensors(X, y)``

Usage::

    from research.deep_learning.dataset_loader import (
        load_dataset,
        get_features,
        train_test_split,
        to_tensors,
    )

    df = load_dataset("1.1.0")
    X, y = get_features(df, "price_sentiment")
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, df)
    X_train_t, y_train_t = to_tensors(X_train, y_train)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from research.deep_learning.feature_sets import FEATURE_SETS, TARGET

logger = logging.getLogger(__name__)

FeatureSet = Literal["price_only", "price_sentiment"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    version: str,
    variant: str = "core",
) -> pd.DataFrame:
    """Load a versioned dataset from ``data/output/<version>/``.

    Args:
        version: Dataset version string, e.g. ``"1.1.0"``.
        variant: One of ``"full"``, ``"core"``, or ``"extended"``.
                 Defaults to ``"core"`` (highest coverage quality).

    Returns:
        DataFrame sorted by ``snapshot_time`` with rows missing the
        target column already dropped.

    Raises:
        FileNotFoundError: If the requested file does not exist.
        ValueError: If ``variant`` is not recognised.
    """
    _valid_variants = {"full", "core", "extended"}
    if variant not in _valid_variants:
        raise ValueError(f"variant must be one of {_valid_variants}, got {variant!r}")

    suffix = "" if variant == "full" else f"_{variant}"
    filename = f"master_research_dataset{suffix}.csv"
    path = cfg.OUTPUT_DIR / version / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Build it first with: python scripts/build_dataset.py --version {version}"
        )

    logger.info("Loading dataset version=%s variant=%s from %s", version, variant, path)
    df = pd.read_csv(path, parse_dates=["snapshot_time", "entry_time"])
    df = df.sort_values("snapshot_time").reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped:
        logger.debug("Dropped %d rows missing target '%s'", dropped, TARGET)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def get_features(
    df: pd.DataFrame,
    feature_set: FeatureSet = "price_sentiment",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y.

    Args:
        df:          DataFrame returned by :func:`load_dataset`.
        feature_set: ``"price_only"`` or ``"price_sentiment"``.

    Returns:
        Tuple of ``(X, y)`` as float32 numpy arrays.

    Raises:
        ValueError: If ``feature_set`` is not recognised or columns are missing.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {list(FEATURE_SETS)}, got {feature_set!r}"
        )

    columns = FEATURE_SETS[feature_set]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in DataFrame: {missing}")

    X = df[columns].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    logger.debug("Feature matrix shape: %s  target shape: %s", X.shape, y.shape)
    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    time_col: str = "snapshot_time",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Chronological train/test split (no random shuffling).

    The split boundary is chosen so that the most recent ``test_ratio``
    fraction of rows (by time) forms the test set.

    Args:
        X:          Feature matrix aligned with *df*.
        y:          Target vector aligned with *df*.
        df:         Source DataFrame (used for time ordering).
        test_ratio: Fraction of rows reserved for test (default 0.2).
        time_col:   Column in *df* used for ordering; defaults to
                    ``"snapshot_time"``.

    Returns:
        ``((X_train, y_train), (X_test, y_test))``

    Raises:
        ValueError: If ``test_ratio`` is not in (0, 1).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    n = len(df)
    split_idx = int(n * (1.0 - test_ratio))

    # df is already sorted by time from load_dataset; use sequential index order
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, n)

    split_time = df[time_col].iloc[split_idx] if time_col in df.columns else "unknown"
    logger.info(
        "Train/test split: %d train / %d test rows (split at %s)",
        len(train_idx),
        len(test_idx),
        split_time,
    )

    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
):
    """Convert numpy arrays to PyTorch float tensors.

    Requires ``torch`` to be installed.  Returns plain numpy arrays when
    torch is not available (graceful degradation for non-DL environments).

    Args:
        X: Feature matrix (float32 numpy array).
        y: Target vector (float32 numpy array).

    Returns:
        ``(X_tensor, y_tensor)`` as ``torch.Tensor`` objects, or the
        original numpy arrays if torch is unavailable.
    """
    try:
        import torch
        return torch.from_numpy(X), torch.from_numpy(y)
    except ImportError:
        logger.warning(
            "torch is not installed; returning numpy arrays instead of tensors."
        )
        return X, y
