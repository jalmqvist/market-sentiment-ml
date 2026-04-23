"""
pipeline/features.py
====================
Feature engineering logic extracted from build_fx_sentiment_dataset.py.

Functions here are pure transformations: they accept a DataFrame and return
a new DataFrame with additional columns.  No I/O is performed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level streak helpers
# ---------------------------------------------------------------------------

def compute_streak_from_boolean(series: pd.Series) -> pd.Series:
    """Return a consecutive-True streak counter for a boolean series."""
    out = np.zeros(len(series), dtype=np.int64)
    count = 0
    vals = series.fillna(False).to_numpy()
    for i, v in enumerate(vals):
        if v:
            count += 1
        else:
            count = 0
        out[i] = count
    return pd.Series(out, index=series.index)


def compute_same_value_streak(series: pd.Series) -> pd.Series:
    """Return a consecutive-same-value streak counter."""
    out = np.ones(len(series), dtype=np.int64)
    vals = series.to_numpy()
    if len(vals) == 0:
        return pd.Series([], dtype="int64", index=series.index)
    count = 1
    for i in range(1, len(vals)):
        if pd.isna(vals[i]) or pd.isna(vals[i - 1]):
            count = 1
        elif vals[i] == vals[i - 1]:
            count += 1
        else:
            count = 1
        out[i] = count
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Crowd-side
# ---------------------------------------------------------------------------

def compute_crowd_side(net_sentiment: pd.Series) -> pd.Series:
    """Convert signed net sentiment into crowd-side labels (+1 long, -1 short, 0 neutral)."""
    return np.select(
        [net_sentiment > 0, net_sentiment < 0],
        [1, -1],
        default=0,
    ).astype(int)


def add_crowd_side(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``crowd_side`` column derived from ``net_sentiment``."""
    out = df.copy()
    out["crowd_side"] = compute_crowd_side(out["net_sentiment"])
    return out


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal (backward-looking) trend features for each eval horizon.

    For each horizon h in [12, 48]:
      - ``trend_{h}b``: past h-bar return
      - ``trend_dir_{h}b``: sign of that return
      - ``trend_alignment_{h}b``: crowd_side * trend_dir (contrarian = -1)
      - ``trend_strength_{h}b``: |trend_{h}b|
    """
    out = df.copy()
    out = out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    for h in [12, 48]:
        out[f"trend_{h}b"] = out.groupby("pair")["entry_close"].pct_change(h)

        out[f"trend_dir_{h}b"] = np.sign(out[f"trend_{h}b"])
        out.loc[out[f"trend_dir_{h}b"] == 0, f"trend_dir_{h}b"] = np.nan

        out[f"trend_alignment_{h}b"] = out["crowd_side"] * out[f"trend_dir_{h}b"]
        out[f"trend_strength_{h}b"] = out[f"trend_{h}b"].abs()

    logger.debug("Trend feature columns added: %s", [c for c in out.columns if c.startswith("trend_")][:6])
    return out


# ---------------------------------------------------------------------------
# Crowd persistence bucket
# ---------------------------------------------------------------------------

def _bucket_crowd_persistence(streak) -> str | None:
    if pd.isna(streak):
        return None
    elif streak == 0:
        return "none"
    elif streak <= 2:
        return "low"
    elif streak <= 5:
        return "medium"
    else:
        return "high"


def add_crowd_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``crowd_persistence_bucket_70`` derived from ``extreme_streak_70``."""
    out = df.copy()
    out["crowd_persistence_bucket_70"] = out["extreme_streak_70"].apply(_bucket_crowd_persistence)
    return out


# ---------------------------------------------------------------------------
# Acceleration bucket
# ---------------------------------------------------------------------------

def add_acceleration_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``acceleration_bucket`` from 6h sentiment change quantiles."""
    out = df.copy()
    out["sentiment_change_6h"] = out.groupby("pair")["net_sentiment"].diff(6)

    q_low = out["sentiment_change_6h"].quantile(0.33)
    q_high = out["sentiment_change_6h"].quantile(0.66)

    def _bucket(x):
        if pd.isna(x):
            return None
        elif x <= q_low:
            return "decreasing"
        elif x >= q_high:
            return "increasing"
        else:
            return "stable"

    out["acceleration_bucket"] = out["sentiment_change_6h"].apply(_bucket).fillna("unknown")
    return out


# ---------------------------------------------------------------------------
# Saturation bucket
# ---------------------------------------------------------------------------

def _bucket_saturation(x) -> str | None:
    if pd.isna(x):
        return None
    elif x < 60:
        return "normal"
    elif x < 75:
        return "elevated"
    elif x < 85:
        return "extreme"
    else:
        return "panic"


def add_saturation_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``saturation_bucket`` derived from ``abs_sentiment``."""
    out = df.copy()
    out["saturation_bucket"] = out["abs_sentiment"].apply(_bucket_saturation)
    return out


# ---------------------------------------------------------------------------
# Trend strength buckets
# ---------------------------------------------------------------------------

def add_trend_strength_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``trend_strength_bucket_{h}b`` for h in [12, 48] via quartile cut."""
    out = df.copy()
    for h in [12, 48]:
        col = f"trend_strength_{h}b"
        bucket_col = f"trend_strength_bucket_{h}b"
        valid = out[col].notna()
        out.loc[valid, bucket_col] = pd.qcut(
            out.loc[valid, col],
            q=cfg.TREND_STRENGTH_QUANTILES,
            labels=cfg.TREND_STRENGTH_LABELS,
        )
    return out


# ---------------------------------------------------------------------------
# Macro regime label
# ---------------------------------------------------------------------------

def add_macro_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``macro_regime`` column: ``'pre_2022'`` or ``'post_2022'``."""
    out = df.copy()
    out["year"] = pd.to_datetime(out["entry_time"]).dt.year
    out["macro_regime"] = out["year"].apply(
        lambda y: "pre_2022" if y <= 2021 else "post_2022"
    )
    return out


# ---------------------------------------------------------------------------
# Pair group
# ---------------------------------------------------------------------------

def add_pair_group(df: pd.DataFrame, pattern: str | None = None) -> pd.DataFrame:
    """Add ``pair_group`` column: ``'JPY_cross'`` or ``'other'``.

    Args:
        df: DataFrame with a ``pair`` column.
        pattern: Substring to match (case-insensitive).  Defaults to
                 ``config.JPY_PAIR_PATTERN``.
    """
    if pattern is None:
        pattern = cfg.JPY_PAIR_PATTERN
    out = df.copy()
    out["pair_group"] = np.where(
        out["pair"].str.contains(pattern, case=False, na=False),
        "JPY_cross",
        "other",
    )
    return out


# ---------------------------------------------------------------------------
# Fight/follow trend flags
# ---------------------------------------------------------------------------

def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean ``fight_trend`` and ``follow_trend`` columns.

    Uses ``trend_alignment_12b``:
      - ``fight_trend`` = True when alignment == -1 (contrarian trade)
      - ``follow_trend`` = True when alignment == +1 (momentum trade)
    """
    out = df.copy()
    out["fight_trend"] = out["trend_alignment_12b"] == -1
    out["follow_trend"] = out["trend_alignment_12b"] == 1
    return out
