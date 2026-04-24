#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset_with_regime.csv"


# --------------------------------------------------
# Prepare
# --------------------------------------------------

def prepare(df):
    df = df.copy()

    df = df[df["phase"].notna()].copy()

    df["pair_group"] = np.where(df["pair"].str.contains("jpy"), "JPY_cross", "non_JPY")

    df["fight_trend"] = df["trend_alignment_12b"] == 1
    df["is_hv"] = df["phase"].str.contains("HV")

    df = df[df["extreme_streak_70"] > 0].copy()

    df = df.dropna(subset=["contrarian_ret_12b", "contrarian_ret_48b"])

    return df


# --------------------------------------------------
# Signal
# --------------------------------------------------

def apply_signal(df):
    return df[
        (df["pair"].str.contains("jpy")) &
        (df["fight_trend"]) &
        (df["trend_strength_bucket_12b"] == "extreme") &
        (df["crowd_persistence_bucket_70"] == "high")
    ]


# --------------------------------------------------
# Stats
# --------------------------------------------------

def stats(df, col):
    if len(df) == 0:
        return dict(n=0, mean=np.nan, hit_rate=np.nan, sharpe=np.nan)

    mean = df[col].mean()
    std = df[col].std()

    return dict(
        n=len(df),
        mean=mean,
        hit_rate=(df[col] > 0).mean(),
        sharpe=mean / std if std > 0 else np.nan,
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(df):
    signal = apply_signal(df)

    print("\n=== GLOBAL SIGNAL ===")
    print(stats(signal, "contrarian_ret_12b"))

    print("\n=== BY PHASE ===")
    for phase, g in signal.groupby("phase"):
        print(phase, stats(g, "contrarian_ret_12b"))

    print("\n=== HV vs LV ===")
    for name, g in signal.groupby("is_hv"):
        label = "HV" if name else "LV"
        print(label, stats(g, "contrarian_ret_12b"))

    print("\n=== BY YEAR (diagnostic only) ===")
    signal["year"] = signal["entry_time"].dt.year
    for y, g in signal.groupby("year"):
        print(y, stats(g, "contrarian_ret_12b"))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])

    df = prepare(df)

    print(f"Dataset size: {len(df):,}")

    evaluate(df)


if __name__ == "__main__":
    main()