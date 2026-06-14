"""Volatility-oriented behavioral feature computations."""

from __future__ import annotations

import pandas as pd


def compute_volatility_regime(
    volatility: pd.Series,
    *,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    unknown_label: str = "unclassified",
) -> pd.Series:
    """Classify volatility into low/medium/high with threshold placeholders."""
    if low_threshold is None or high_threshold is None:
        return pd.Series([unknown_label] * len(volatility), index=volatility.index)

    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be less than high_threshold")

    regime = pd.Series(["medium"] * len(volatility), index=volatility.index)
    regime.loc[volatility < low_threshold] = "low"
    regime.loc[volatility >= high_threshold] = "high"
    return regime


def compute_volatility_regime_persistence(
    df: pd.DataFrame,
    *,
    pair_col: str = "pair",
    regime_col: str = "volatility_regime",
) -> pd.Series:
    """Compute consecutive-bar persistence for volatility regimes."""
    if regime_col not in df.columns:
        raise ValueError(f"missing volatility regime column: {regime_col}")

    change = df[regime_col].ne(df[regime_col].shift()) | df[pair_col].ne(df[pair_col].shift())
    grp = change.groupby(df[pair_col], sort=False).cumsum()
    return df.groupby([df[pair_col], grp], sort=False).cumcount() + 1
