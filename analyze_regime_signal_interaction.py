#!/usr/bin/env python3
"""
Analyze regime × sentiment signal interaction

Input:
    data/output/master_research_dataset_with_regime.csv

Focus:
    - contrarian_ret_12b / 48b
    - phase (HV/LV × Trend/Ranging)
    - trend_alignment
    - trend_strength_bucket
    - JPY vs non-JPY
"""

import pandas as pd
import numpy as np


INPUT_PATH = "data/output/master_research_dataset_with_regime.csv"


# -----------------------------
# Helpers
# -----------------------------

def add_features(df):
    df = df.copy()

    # Only keep rows with regime
    df = df[df["phase"].notna()].copy()

    # Pair group (JPY vs non-JPY)
    df["pair_group"] = np.where(df["pair"].str.contains("jpy"), "JPY_cross", "non_JPY")

    # Trend bucket
    df["trend_bucket"] = np.where(df["trend_alignment_12b"] == 1, "fight_trend", "follow_trend")

    return df


def summarize(df, group_cols, ret_col):
    df = df.copy()

    # Create hit indicator first
    df["_hit"] = (df[ret_col] > 0).astype(float)

    g = df.groupby(group_cols)

    out = g.agg(
        n=(ret_col, "count"),
        mean=(ret_col, "mean"),
        median=(ret_col, "median"),
        std=(ret_col, "std"),
        hit_rate=("_hit", "mean"),
    )

    return out.reset_index()


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------
# Core analysis
# -----------------------------

def run():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])

    df = add_features(df)
    df = df.dropna(subset=["contrarian_ret_12b"])

    print(f"\nRows with regime: {len(df):,}")

    # --------------------------------------------------
    # 1) Phase-only baseline
    # --------------------------------------------------
    print_section("PHASE BASELINE (12b)")

    res = summarize(df, ["phase"], "contrarian_ret_12b")
    print(res.to_string(index=False))


    # --------------------------------------------------
    # 2) Phase × trend alignment
    # --------------------------------------------------
    print_section("PHASE × TREND ALIGNMENT (12b)")

    res = summarize(df, ["phase", "trend_bucket"], "contrarian_ret_12b")
    print(res.to_string(index=False))


    # --------------------------------------------------
    # 3) JPY hypothesis under regimes
    # --------------------------------------------------
    print_section("JPY × PHASE × TREND (12b)")

    subset = df[
        (df["pair_group"] == "JPY_cross") &
        (df["extreme_streak_70"] > 0)   # proxy for persistence
    ]

    res = summarize(
        subset,
        ["phase", "trend_bucket", "trend_strength_bucket_12b"],
        "contrarian_ret_12b"
    )

    print(res.sort_values(["phase", "trend_bucket", "trend_strength_bucket_12b"]).to_string(index=False))


    # --------------------------------------------------
    # 4) Strength monotonicity check (JPY focus)
    # --------------------------------------------------
    print_section("JPY × STRENGTH PROFILE (fight_trend, persistent)")

    subset = df[
        (df["pair_group"] == "JPY_cross") &
        (df["trend_bucket"] == "fight_trend") &
        (df["extreme_streak_70"] > 0)
    ]

    res = summarize(
        subset,
        ["trend_strength_bucket_12b"],
        "contrarian_ret_12b"
    )

    print(res.to_string(index=False))


    # --------------------------------------------------
    # 5) Horizon comparison
    # --------------------------------------------------
    print_section("HORIZON COMPARISON (JPY, fight_trend)")

    rows = []

    for h, col in [(12, "contrarian_ret_12b"), (48, "contrarian_ret_48b")]:
        tmp = summarize(
            subset,
            ["trend_strength_bucket_12b"],
            col
        )
        tmp["horizon"] = h
        rows.append(tmp)

    out = pd.concat(rows)
    print(out.to_string(index=False))


    # --------------------------------------------------
    # 6) Phase interaction (most important)
    # --------------------------------------------------
    print_section("PHASE × JPY SIGNAL (fight_trend, persistent)")

    subset = df[
        (df["pair_group"] == "JPY_cross") &
        (df["trend_bucket"] == "fight_trend") &
        (df["extreme_streak_70"] > 0)
    ]

    res = summarize(
        subset,
        ["phase", "trend_strength_bucket_12b"],
        "contrarian_ret_12b"
    )

    print(res.sort_values(["phase", "trend_strength_bucket_12b"]).to_string(index=False))


if __name__ == "__main__":
    run()
