# Legacy experiment — not part of current validated approach\n#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats

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
# Prepare
# --------------------------------------------------

def prepare(df):
    df = df.copy()

    df = df[df["phase"].notna()].copy()

    df["pair_group"] = np.where(df["pair"].str.contains("jpy"), "JPY_cross", "non_JPY")
    df["fight_trend"] = df["trend_alignment_12b"] == -1

    df = df.dropna(subset=["contrarian_ret_12b", "contrarian_ret_48b"])

    return df


# --------------------------------------------------
# Signal (LOCKED)
# --------------------------------------------------

def apply_signal(df):
    return df[
        (df["pair_group"] == "JPY_cross") &
        (df["fight_trend"]) &
        (df["trend_strength_bucket_12b"] == "extreme")
    ].copy()


# --------------------------------------------------
# Main comparison
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])

    df = prepare(df)

    # ------------------------------------------
    # Apply signal + spacing ONCE
    # ------------------------------------------
    signal = apply_signal(df)
    signal = enforce_spacing(signal)

    print(f"\nTotal non-overlapping signals: {len(signal)}")

    # ------------------------------------------
    # Split regimes
    # ------------------------------------------
    pre = signal[signal["macro_regime"] == "pre_2022"]
    post = signal[signal["macro_regime"] == "post_2022"]

    print(f"\nPRE 2022 signals:  {len(pre)}")
    print(f"POST 2022 signals: {len(post)}")

    # ------------------------------------------
    # Evaluate both horizons
    # ------------------------------------------
    for col, label in [
        ("contrarian_ret_12b", "12b"),
        ("contrarian_ret_48b", "48b"),
    ]:
        print("\n" + "=" * 80)
        print(f"HORIZON: {label}")
        print("=" * 80)

        stats_pre = compute_stats(pre, col)
        stats_post = compute_stats(post, col)

        print("\nPRE 2022:")
        print(stats_pre)

        print("\nPOST 2022:")
        print(stats_post)

        # --------------------------------------
        # Statistical test (difference in means)
        # --------------------------------------
        if len(pre) > 1 and len(post) > 1:
            t_stat, p_val = stats.ttest_ind(
                pre[col],
                post[col],
                equal_var=False,  # Welch t-test
                nan_policy="omit",
            )

            print("\nDifference in means test (POST - PRE):")
            print(f"t-stat: {t_stat:.4f}")
            print(f"p-value: {p_val:.6f}")

            diff = stats_post["mean"] - stats_pre["mean"]
            print(f"mean_diff: {diff:.6f}")

        # --------------------------------------
        # Year breakdown (diagnostic)
        # --------------------------------------
        print("\nPOST-2022 yearly breakdown:")
        tmp = post.copy()
        tmp["year"] = tmp["entry_time"].dt.year

        yearly = tmp.groupby("year")[col].agg(["count", "mean", "std"])
        yearly["sharpe"] = yearly["mean"] / yearly["std"]

        print(yearly)


if __name__ == "__main__":
    main()