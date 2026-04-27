# Legacy experiment — not part of current validated approach\nimport pandas as pd
import numpy as np

DATA_PATH = "data/output/master_research_dataset.csv"

HORIZONS = [12, 48]
HOLDOUT_SPLIT_YEAR = 2022


# =========================
# LOAD + PREP
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    # timestamp
    df["timestamp"] = pd.to_datetime(df["time"], format="mixed")
    df = df.sort_values("timestamp")

    # year
    df["year"] = df["timestamp"].dt.year

    return df


# =========================
# SIGNAL DEFINITION (FIXED)
# =========================
def apply_signal(df):
    signal = df[
        (df["extreme_streak_70"] >= 3) &
        (df["crowd_persistence_bucket_70"].isin(["high", "medium"]))
    ].copy()

    return signal


# =========================
# NON-OVERLAP FILTER
# =========================
def enforce_non_overlap(df, horizon):
    df = df.sort_values("timestamp").copy()

    selected = []
    last_time = {}

    for _, row in df.iterrows():
        pair = row["pair"]
        t = row["timestamp"]

        if pair not in last_time:
            selected.append(row)
            last_time[pair] = t
        else:
            delta = (t - last_time[pair]).total_seconds() / 3600
            if delta >= horizon:
                selected.append(row)
                last_time[pair] = t

    return pd.DataFrame(selected)


# =========================
# METRICS
# =========================
def compute_metrics(df, horizon):
    col = f"contrarian_ret_{horizon}b"

    if col not in df.columns or df.empty:
        return None

    r = df[col].dropna()

    if len(r) < 20:
        return None

    return {
        "n": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan,
        "hit_rate": (r > 0).mean()
    }


# =========================
# HOLDOUT
# =========================
def holdout_test(df, horizon):
    col = f"contrarian_ret_{horizon}b"

    train = df[df["year"] <= HOLDOUT_SPLIT_YEAR]
    test = df[df["year"] >= HOLDOUT_SPLIT_YEAR + 1]

    def stats(x):
        if len(x) < 20:
            return None
        r = x[col].dropna()
        return {
            "n": len(r),
            "mean": r.mean(),
            "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan
        }

    return {
        "train": stats(train),
        "test": stats(test)
    }


# =========================
# WALK-FORWARD (simple yearly)
# =========================
def walk_forward(df, horizon):
    col = f"contrarian_ret_{horizon}b"

    results = []

    years = sorted(df["year"].unique())

    for y in years:
        test = df[df["year"] == y]

        if len(test) < 20:
            continue

        r = test[col].dropna()

        results.append({
            "year": y,
            "n": len(r),
            "mean": r.mean(),
            "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan
        })

    return pd.DataFrame(results)


# =========================
# PER-PAIR ANALYSIS
# =========================
def analyze_pairs(signal, horizon):
    results = []

    for pair, df_pair in signal.groupby("pair"):
        df_pair = enforce_non_overlap(df_pair, horizon)

        metrics = compute_metrics(df_pair, horizon)
        if metrics is None:
            continue

        holdout = holdout_test(df_pair, horizon)

        results.append({
            "pair": pair,
            "n": metrics["n"],
            "mean": metrics["mean"],
            "sharpe": metrics["sharpe"],
            "hit_rate": metrics["hit_rate"],
            "train_sharpe": holdout["train"]["sharpe"] if holdout["train"] else np.nan,
            "test_sharpe": holdout["test"]["sharpe"] if holdout["test"] else np.nan
        })

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# =========================
# MAIN
# =========================
def run():
    print("Loading dataset...")
    df = load_data()
    print("Dataset size:", len(df))

    signal = apply_signal(df)
    print("Raw signal count:", len(signal))

    if signal.empty:
        print("No signals found.")
        return

    print("\n=== SIGNAL DISTRIBUTION ===")
    print(signal["pair"].value_counts().head(10))

    for horizon in HORIZONS:
        print("\n" + "="*80)
        print(f"PER-PAIR DISCOVERY (horizon={horizon})")
        print("="*80)

        results = analyze_pairs(signal, horizon)

        if results.empty:
            print("No valid pairs.")
            continue

        print(results)

        print("\nTop 5:")
        print(results.head(5))

        print("\nBottom 5:")
        print(results.tail(5))


if __name__ == "__main__":
    run()