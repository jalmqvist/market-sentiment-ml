import pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset.csv"


# =========================
# Load + prepare
# =========================
def load_data():
    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year

    df["pair_group"] = np.where(
        df["pair"].str.contains("JPY", case=False, na=False),
        "JPY_cross",
        "other"
    )

    return df


# =========================
# Signal builder
# =========================
def build_signal(df, streak_threshold, use_persistence):

    signal = df[
        (df["pair_group"] == "JPY_cross") &
        (df["extreme_streak_70"] >= streak_threshold) &
        (df["acceleration_bucket"] == "decreasing")
    ]

    if use_persistence:
        signal = signal[
            signal["crowd_persistence_bucket_70"] == "high"
        ]

    return signal.copy()


# =========================
# Spacing
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
# Metrics
# =========================
def compute_stats(df, ret_col):
    if len(df) == 0:
        return np.nan, np.nan, np.nan

    mean = df[ret_col].mean()
    std = df[ret_col].std()

    sharpe = mean / std if std != 0 else np.nan
    hit = (df[ret_col] > 0).mean()

    return mean, sharpe, hit


# =========================
# Walk-forward (true OOS)
# =========================
def walk_forward(signal_df, horizon):

    if len(signal_df) == 0:
        return np.nan, np.nan, np.nan

    ret_col = f"contrarian_ret_{horizon}b"

    years = sorted(signal_df["year"].unique())

    means = []
    sharpes = []
    hits = []

    for year in years:
        if year < 2021:
            continue

        test = signal_df[signal_df["year"] == year]

        mean, sharpe, hit = compute_stats(test, ret_col)

        means.append(mean)
        sharpes.append(sharpe)
        hits.append(hit)

    return np.nanmean(means), np.nanmean(sharpes), np.nanmean(hits)


# =========================
# Holdout test
# =========================
def holdout(signal_df, horizon):

    if len(signal_df) == 0:
        return np.nan, np.nan

    ret_col = f"contrarian_ret_{horizon}b"

    pre = signal_df[signal_df["year"] <= 2022]
    post = signal_df[signal_df["year"] >= 2023]

    mean_pre, sharpe_pre, _ = compute_stats(pre, ret_col)
    mean_post, sharpe_post, _ = compute_stats(post, ret_col)

    return sharpe_pre, sharpe_post


# =========================
# Main experiment loop
# =========================
def run():

    df = load_data()

    results = []

    for streak in [2, 3, 4]:
        for use_persistence in [False, True]:

            signal = build_signal(df, streak, use_persistence)

            raw_n = len(signal)

            signal = enforce_spacing(signal, horizon=12)

            spaced_n = len(signal)

            wf12 = walk_forward(signal, 12)
            wf48 = walk_forward(signal, 48)

            hold12 = holdout(signal, 12)
            hold48 = holdout(signal, 48)

            results.append({
                "streak": streak,
                "persistence": use_persistence,
                "raw_n": raw_n,
                "spaced_n": spaced_n,

                "wf12_mean": wf12[0],
                "wf12_sharpe": wf12[1],
                "wf12_hit": wf12[2],

                "wf48_mean": wf48[0],
                "wf48_sharpe": wf48[1],
                "wf48_hit": wf48[2],

                "hold12_pre_sharpe": hold12[0],
                "hold12_post_sharpe": hold12[1],

                "hold48_pre_sharpe": hold48[0],
                "hold48_post_sharpe": hold48[1],
            })

    results_df = pd.DataFrame(results)

    print("\n=== EXPERIMENT RESULTS ===")
    print(results_df.sort_values("wf48_sharpe", ascending=False))

    return results_df


if __name__ == "__main__":
    run()