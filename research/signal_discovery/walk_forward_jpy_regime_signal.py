# Legacy experiment — not part of current validated approach\n#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset_with_regime.csv"

# --------------------------------------------------
# Helpers
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
# Prepare dataset
# --------------------------------------------------

def prepare(df):
    df = df.copy()

    df = df[df["phase"].notna()].copy()

    df["pair_group"] = np.where(df["pair"].str.contains("jpy"), "JPY_cross", "non_JPY")
    df["fight_trend"] = df["trend_alignment_12b"] == -1

    df = df.dropna(subset=["contrarian_ret_12b", "contrarian_ret_48b"])

    df["year"] = df["entry_time"].dt.year

    return df


# --------------------------------------------------
# Signal definition (LOCKED)
# --------------------------------------------------

def apply_signal(df):
    return df[
        (df["pair_group"] == "JPY_cross") &
        (df["fight_trend"]) &
        (df["trend_strength_bucket_12b"] == "extreme")
    ].copy()


# --------------------------------------------------
# Enforce non-overlapping trades
# --------------------------------------------------

def enforce_spacing(df, min_gap_hours=48):
    df = df.sort_values("entry_time").copy()

    selected = []
    last_time = None

    for _, row in df.iterrows():
        if last_time is None:
            selected.append(row)
            last_time = row["entry_time"]
        else:
            diff = (row["entry_time"] - last_time).total_seconds() / 3600
            if diff >= min_gap_hours:
                selected.append(row)
                last_time = row["entry_time"]

    if len(selected) == 0:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(selected)


# --------------------------------------------------
# Walk-forward (expanding window)
# --------------------------------------------------

def walk_forward(df, ret_col):
    results = []

    years = sorted(df["year"].unique())

    for i in range(2, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train = df[df["year"].isin(train_years)]
        test = df[df["year"] == test_year]

        if len(test) == 0:
            continue

        # Apply signal (no tuning here — fixed)
        test_sig = apply_signal(test)

        # Enforce spacing
        test_sig = enforce_spacing(test_sig, min_gap_hours=48)

        stats = compute_stats(test_sig, ret_col)
        stats["year"] = test_year

        results.append(stats)

    return pd.DataFrame(results)

def run_for_subset(df, label):
    print("\n" + "=" * 80)
    print(f"MACRO REGIME: {label}")
    print("=" * 80)

    print(f"Dataset size: {len(df):,}")

    signal = apply_signal(df)
    print(f"Raw signal count: {len(signal):,}")

    signal_spaced = enforce_spacing(signal)
    print(f"Non-overlapping signals: {len(signal_spaced):,}")

    print("\nSignals per year (after spacing):")
    tmp = signal_spaced.copy()
    tmp["year"] = tmp["entry_time"].dt.year
    print(tmp.groupby("year").size())

    for horizon, col in [(12, "contrarian_ret_12b"), (48, "contrarian_ret_48b")]:
        print("\n" + "-" * 60)
        print(f"WALK-FORWARD (horizon={horizon})")
        print("-" * 60)

        wf = walk_forward(df, col)

        print(wf)

        print("\nSummary:")
        print(wf[["mean", "hit_rate", "sharpe"]].mean())
# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])

    df = prepare(df)

    # ------------------------------------------
    # FULL DATA (baseline)
    # ------------------------------------------
    run_for_subset(df, "ALL DATA")

    # ------------------------------------------
    # PRE-2022
    # ------------------------------------------
    df_pre = df[df["macro_regime"] == "pre_2022"].copy()
    run_for_subset(df_pre, "PRE 2022")

    # ------------------------------------------
    # POST-2022
    # ------------------------------------------
    df_post = df[df["macro_regime"] == "post_2022"].copy()
    run_for_subset(df_post, "POST 2022")


if __name__ == "__main__":
    main()