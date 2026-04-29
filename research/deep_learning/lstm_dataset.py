"""
research/deep_learning/lstm_dataset.py
=======================================
Convert a tabular dataset into fixed-length sequences for LSTM training.

Responsibilities
----------------
- Build rolling-window sequences per currency pair with no cross-pair leakage.
- Preserve strict chronological order; no shuffling.
- Drop any sequence that contains NaN values.
- Provide a time-based (non-random) train/test split for sequences.

Usage::

    from research.deep_learning.lstm_dataset import (
        build_sequences,
        train_test_split_sequences,
    )

    X, y = build_sequences(df, features=PRICE_FEATURES, target="ret_48b", seq_len=48)
    (X_train, y_train), (X_test, y_test) = train_test_split_sequences(X, y, split_ratio=0.8)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_sequences(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build rolling-window sequences for LSTM training.

    Each sample ``(X_seq, y)`` consists of a window of ``seq_len`` consecutive
    feature rows ending at time *t*, paired with the target value at time *t*.
    Sequences are built independently per currency pair so there is no
    cross-pair leakage.

    Args:
        df:       DataFrame containing at least the ``features``, ``target``,
                  ``"pair"``, and ``"snapshot_time"`` columns.
        features: Ordered list of feature column names to include in *X*.
        target:   Name of the target column.
        seq_len:  Number of time steps per sequence window.

    Returns:
        ``(X, y)`` where

        - ``X`` has shape ``(N, seq_len, n_features)``, dtype ``float32``.
        - ``y`` has shape ``(N,)``, dtype ``float32``.

        Sequences containing any NaN value (in features or target) are dropped.

    Raises:
        ValueError: If required columns are absent from *df* or ``seq_len < 1``.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    required_cols = ["pair", "snapshot_time"] + features + [target]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    all_x: list[np.ndarray] = []
    all_y: list[float] = []
    all_t = []

    for pair, group in df.groupby("pair", sort=True):
        group = group.sort_values("snapshot_time")

        feat_vals = group[features].to_numpy(dtype=np.float32)  # (T, n_features)
        tgt_vals = group[target].to_numpy(dtype=np.float32)     # (T,)

        n_rows = len(group)
        if n_rows < seq_len:
            logger.debug(
                "Pair %s skipped: only %d rows, need at least %d (seq_len)",
                pair, n_rows, seq_len,
            )
            continue

        for end in range(seq_len - 1, n_rows):
            start = end - seq_len + 1
            X_seq = feat_vals[start : end + 1]  # (seq_len, n_features)
            y_val = tgt_vals[end]

            # Drop sequences containing any NaN (features or target)
            if not np.isfinite(X_seq).all() or not np.isfinite(y_val):
                continue

            all_x.append(X_seq)
            all_y.append(y_val)
            all_t.append(group["snapshot_time"].iloc[end])

    if not all_x:
        logger.warning("build_sequences produced 0 valid sequences.")
        n_features = len(features)
        return (
            np.empty((0, seq_len, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X = np.stack(all_x, axis=0)  # (N, seq_len, n_features)
    y = np.array(all_y, dtype=np.float32)  # (N,)

    t = np.array(all_t)
    order = np.argsort(t)
    X = X[order]
    y = y[order]

    logger.info("Sequences sorted globally by timestamp")
    logger.info(
        "build_sequences: %d sequences  shape X=%s  y=%s",
        len(y), X.shape, y.shape,
    )
    return X, y


def train_test_split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    split_ratio: float = 0.8,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split sequences chronologically into train and test sets.

    The split is positional (the first ``split_ratio`` fraction of sequences
    become the training set, the remainder the test set).  Because
    :func:`build_sequences` preserves chronological order within each pair and
    appends pairs in sorted order, this provides a reasonable time-based split
    without any random shuffling.

    Args:
        X:           Sequence array of shape ``(N, seq_len, n_features)``.
        y:           Target array of shape ``(N,)``.
        split_ratio: Fraction of sequences used for training (default ``0.8``).

    Returns:
        ``((X_train, y_train), (X_test, y_test))``

    Raises:
        ValueError: If ``split_ratio`` is not in ``(0, 1)`` or ``X``/``y``
                    lengths differ.
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length; got {len(X)} vs {len(y)}"
        )

    n = len(X)
    split_idx = int(n * split_ratio)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    logger.info(
        "train_test_split_sequences: %d train / %d test  (split_ratio=%.2f)",
        len(y_train), len(y_test), split_ratio,
    )
    return (X_train, y_train), (X_test, y_test)
