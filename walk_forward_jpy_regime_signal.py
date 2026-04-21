#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset_with_regime.csv"


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare(df):
    df = df.copy()

    # Keep only rows with valid regime
    df = df[df["phase"].notna()].copy()

    # Keep only JPY crosses
    df = df[df["pair"].str.contains("jpy")].copy()

    # Behavioral filters
    df = df[df["extreme_streak_70"] > 0].copy()  # persistent sentiment
    df["fight_trend"] = df["trend_alignment_12b"] == 1

    # Regime flags
    df["is_hv"] = df["phase"].str.contains("HV")
    df["is_trend"] = df["phase"].str.contains("Trend")

    # Strength (RELAXED: strong + extreme)
    df["is_strong_plus"] = df["trend_strength_bucket_12b"].isin(["strong", "extreme"])

    # Drop NaNs in returns
    df = df.dropna(subset=["contrarian_ret_12b", "contrarian_ret_48b"])

    return df


# --------------------------------------------------
# Signal definition (UPDATED)
# --------------------------------------------------

def apply_signal(df):
    return df[
        (df["fight_trend"]) &
        (df["is_strong_plus"])
    ]


# --------------------------------------------------
# Stats
# --------------------------------------------------

def compute_stats(df, ret_col):
    if len(df) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, sharpe=np.nan, hit_rate=np.nan)

    mean = df[ret_col].mean()
    std = df[ret_col].std()

    return dict(
        n=len(df),
        mean=mean,
        std=std,
        sharpe=mean / std if std > 0 else np.nan,
        hit_rate=(df[ret_col] > 0).mean(),
    )


# --------------------------------------------------
# Walk-forward
# --------------------------------------------------

def walk_forward(df, ret_col):
    results = []

    df = df.copy()
    df["year"] = df["entry_time"].dt.year

    years = sorted(df["year"].unique())

    for y in years:
        test = df[df["year"] == y]

        if len(test) == 0:
            continue

        test_sig = apply_signal(test)

        stats = compute_stats(test_sig, ret_col)
        stats["year"] = y

        results.append(stats)

    return pd.DataFrame(results)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])

    df = prepare(df)

    print(f"\nFiltered dataset size (after base filters): {len(df):,}")

    # Show how many signals total (sanity check)
    total_signals = len(apply_signal(df))
    print(f"Total signal observations: {total_signals:,}")

    for horizon, col in [(12, "contrarian_ret_12b"), (48, "contrarian_ret_48b")]:
        print(f"\n=== WALK-FORWARD (horizon={horizon}) ===")

        wf = walk_forward(df, col)

        print(wf)

        print("\nSummary (mean over folds):")
        print(wf[["mean", "hit_rate", "sharpe"]].mean())

        print("\nTrades per year:")
        print(wf[["year", "n"]])


if __name__ == "__main__":
    main()