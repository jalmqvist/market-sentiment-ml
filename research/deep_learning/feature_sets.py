"""
research/deep_learning/feature_sets.py
=======================================
Canonical feature set definitions for ML models.

All columns listed here are causal (backward-looking only) and free of
target leakage.  Forward-looking columns (ret_*, contrarian_ret_*,
future_close_*) are intentionally excluded.

Usage::

    from research.deep_learning.feature_sets import PRICE_FEATURES, SENTIMENT_FEATURES, TARGET
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

TARGET: str = "ret_48b"

# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------

# Past return over the last N bars (causal: computed from entry_close history)
# and associated directional / strength signals.
PRICE_FEATURES: list[str] = [
    "trend_12b",            # 12-bar past return (pct_change)
    "trend_48b",            # 48-bar past return (pct_change)
    "trend_dir_12b",        # sign of 12-bar past return
    "trend_dir_48b",        # sign of 48-bar past return
    "trend_strength_12b",   # |trend_12b|
    "trend_strength_48b",   # |trend_48b|
    "entry_tick_volume",    # volume at entry bar
]

# ---------------------------------------------------------------------------
# Sentiment features
# ---------------------------------------------------------------------------

SENTIMENT_FEATURES: list[str] = [
    "net_sentiment",         # signed % (long positive, short negative)
    "abs_sentiment",         # |net_sentiment|
    "crowd_side",            # +1 net long / -1 net short / 0 neutral
    "side_streak",           # consecutive bars on same crowd side
    "extreme_streak_70",     # bars with abs_sentiment >= 70 in a row
    "extreme_streak_80",     # bars with abs_sentiment >= 80 in a row
    "sentiment_change",      # change vs. previous snapshot
]

# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------

PRICE_VOL_FEATURES: list[str] = PRICE_FEATURES + [
    "vol_12b",   # 12-bar realised volatility
    "vol_48b",   # 48-bar realised volatility
]

PRICE_VOL_SENTIMENT_FEATURES: list[str] = PRICE_VOL_FEATURES + SENTIMENT_FEATURES

# ---------------------------------------------------------------------------
# Trend + regime features
# ---------------------------------------------------------------------------

# Volatility-adjusted trend strength and regime flags (requires vol_12b + trend_12b)
PRICE_TREND_FEATURES: list[str] = [
    "trend_12b",        # 12-bar past return (pct_change)
    "trend_strength",   # abs(trend_12b) / (vol_12b + 1e-8)
    "is_trending",      # trend_strength > TREND_THRESHOLD
    "is_high_vol",      # vol_12b > vol_12b.median() (per pair)
]

PRICE_TREND_SENTIMENT_FEATURES: list[str] = PRICE_TREND_FEATURES + SENTIMENT_FEATURES

# ---------------------------------------------------------------------------
# Convenience groupings
# ---------------------------------------------------------------------------

FEATURE_SETS: dict[str, list[str]] = {
    "price_only": PRICE_FEATURES,
    "price_sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
    "price_vol": PRICE_VOL_FEATURES,
    "price_vol_sentiment": PRICE_VOL_SENTIMENT_FEATURES,
    "price_trend": PRICE_TREND_FEATURES,
    "price_trend_sentiment": PRICE_TREND_SENTIMENT_FEATURES,
}
