import pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset.csv"


# =========================
# 1. Load + prepare
# =========================
def load_data():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    # Robust timestamp parsing
    df["timestamp"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["year"] = df["timestamp"].dt.year

    # Robust pair grouping
    df["pair_group"] = np.where(
        df["pair"].str.contains("JPY", case=False, na=False),
        "JPY_cross",
        "other"
    )

    return df


# =========================
# 2. Signal definition (V2 baseline)
# =========================
def apply_signal(df):
    return df[
        (df["pair_group"] == "JPY_cross") &
        (df["crowd_persistence_bucket_70"] == "high") &
        (df["acceleration_bucket"] == "decreasing")
    ].copy()


# =========================
# 3. Enforce spacing (no overlap)
# =========================
def enforce_spacing(signal_df, horizon):
    if len(signal_df) == 0:
        return signal_df.copy()

    signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

    selected = []
    last_time = None

    for _, row in signal_df.iterrows():
        if last_time is None:
            selected.append(row)
            last_time = row["timestamp"]
        else:
            if (row["timestamp"] - last_time).total_seconds() >= horizon * 3600:
                selected.append(row)
                last_time = row["timestamp"]

    if len(selected) == 0:
        return signal_df.iloc[0:0].copy()

    return pd.DataFrame(selected)


# =========================
# 4. Metrics
# =========================
def compute_stats(df, ret_col):
    if len(df) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan
        }

    if ret_col not in df.columns:
        raise ValueError(f"Missing column: {ret_col}")

    mean = df[ret_col].mean()
    std = df[ret_col].std()

    return {
        "n": len(df),
        "mean": mean,
        "std": std,
        "sharpe": mean / std if std != 0 else np.nan,
        "hit_rate": (df[ret_col] > 0).mean()
    }


# =========================
# 5. Yearly evaluation
# =========================
def yearly_eval(signal_df, horizon):

    if len(signal_df) == 0:
        return pd.DataFrame()

    ret_col = f"contrarian_ret_{horizon}b"

    results = []

    years = sorted(signal_df["year"].unique())

    for year in years:
        if year < 2021:
            continue  # enforce forward logic

        test = signal_df[signal_df["year"] == year]

        stats = compute_stats(test, ret_col)
        stats["year"] = year

        results.append(stats)

    return pd.DataFrame(results)


def holdout_test(signal_df, horizon):
    if len(signal_df) == 0:
        return None

    ret_col = f"contrarian_ret_{horizon}b"

    train = signal_df[signal_df["year"] <= 2022]
    test = signal_df[signal_df["year"] >= 2023]

    def stats(df):
        if len(df) == 0:
            return {"n": 0, "mean": np.nan, "sharpe": np.nan}
        if ret_col not in df.columns:
            raise ValueError(f"Missing column: {ret_col}")
        mean = df[ret_col].mean()
        std = df[ret_col].std()
        return {
            "n": len(df),
            "mean": mean,
            "sharpe": mean / std if std != 0 else np.nan
        }

    return {
        "train": stats(train),
        "test": stats(test)
    }


# =========================
# 6. Diagnostics
# =========================
def debug_checks(df):

    print("\n=== DEBUG CHECKS ===")

    print("\npair_group counts:")
    print(df["pair_group"].value_counts())

    print("\nsaturation_bucket counts:")
    print(df["saturation_bucket"].value_counts(dropna=False))

    print("\ncrowd_persistence_bucket_70 counts:")
    print(df["crowd_persistence_bucket_70"].value_counts(dropna=False))

    debug = df[
        df["pair_group"].eq("JPY_cross") &
        df["saturation_bucket"].isin(["extreme", "panic"]) &
        df["crowd_persistence_bucket_70"].isin(["medium", "high"])
    ]

    print("\nSignal overlap check:")
    print("count:", len(debug))


def tradeability_check(signal_df, horizon):
    ret_col = f"contrarian_ret_{horizon}b"

    if ret_col not in signal_df.columns:
        raise ValueError(f"Missing column: {ret_col}")

    if len(signal_df) == 0:
        return None

    mean = signal_df[ret_col].mean()
    std = signal_df[ret_col].std()

    return {
        "mean_return": mean,
        "std": std,
        "mean_bps": mean * 10000,
        "sharpe": mean / std if std != 0 else np.nan
    }


def frequency_check(signal_df):
    if len(signal_df) == 0:
        return None

    signal_df = signal_df.copy()
    signal_df["date"] = signal_df["timestamp"].dt.date

    per_day = signal_df.groupby("date").size()

    return {
        "signals_total": len(signal_df),
        "avg_per_day": per_day.mean(),
        "median_per_day": per_day.median(),
        "max_per_day": per_day.max()
    }


def distribution_check(signal_df):
    if len(signal_df) == 0:
        return None

    return {
        "pairs": signal_df["pair"].value_counts().head(5).to_dict(),
        "years": signal_df["year"].value_counts().to_dict()
    }


def remove_top_pairs(signal_df, top_n=1):
    if len(signal_df) == 0:
        return signal_df.copy()

    top_pairs = signal_df["pair"].value_counts().head(top_n).index.tolist()

    print(f"\nRemoving top {top_n} pairs:", top_pairs)

    return signal_df[~signal_df["pair"].isin(top_pairs)].copy()


# =========================
# 7. Evaluation wrapper
# =========================
def run_evaluation(signal, label):

    print("\n" + "=" * 80)
    print(f"EVALUATION: {label}")
    print("=" * 80)

    print(f"Signals: {len(signal):,}")

    for horizon in [12, 48]:

        print("\n--- Horizon:", horizon, "---")

        # ✅ Correct spacing per horizon
        signal_spaced = enforce_spacing(signal, horizon)

        wf = yearly_eval(signal_spaced, horizon)

        if wf.empty:
            print("No signals")
            continue

        print(wf[["year", "n", "mean", "sharpe"]])

        # ✅ Correct pooled Sharpe
        ret_col = f"contrarian_ret_{horizon}b"
        pooled = signal_spaced[ret_col]
        summary_sharpe = pooled.mean() / pooled.std()

        print("Summary Sharpe:", summary_sharpe)

        hold = holdout_test(signal_spaced, horizon)
        if hold:
            print("Holdout:", hold)

        trade = tradeability_check(signal_spaced, horizon)
        if trade:
            print("Tradeability:", trade["mean_bps"], "bps")


# =========================
# 8. Main
# =========================
def run():

    df = load_data()

    print(f"Dataset size: {len(df):,}")

    debug_checks(df)

    signal_raw = apply_signal(df)

    print(f"\nRaw signal count: {len(signal_raw):,}")

    # =========================
    # PER-PAIR DIAGNOSTICS
    # =========================
    print("\n=== PER-PAIR PERFORMANCE (48b) ===")

    pair_stats = signal_raw.groupby("pair").apply(
        lambda x: compute_stats(x, "contrarian_ret_48b")
    ).apply(pd.Series)

    print(pair_stats.sort_values("sharpe", ascending=False))

    # =========================
    # BASE SIGNAL
    # =========================
    run_evaluation(signal_raw, "BASE")

    # =========================
    # REMOVE TOP PAIRS (before spacing!)
    # =========================
    signal_no_top1 = remove_top_pairs(signal_raw, top_n=1)
    run_evaluation(signal_no_top1, "REMOVE TOP 1")

    signal_no_top2 = remove_top_pairs(signal_raw, top_n=2)
    run_evaluation(signal_no_top2, "REMOVE TOP 2")

    print("\n--- FREQUENCY ---")
    freq = frequency_check(signal_raw)
    if freq:
        print(freq)

    print("\n--- DISTRIBUTION ---")
    dist = distribution_check(signal_raw)
    if dist:
        print(dist)


if __name__ == "__main__":
    run()