# Legacy experiment — not part of current validated approach\nimport pandas as pd
import numpy as np

DATA_PATH = "data/output/master_research_dataset.csv"

HORIZONS = [12, 48]
HOLDOUT_SPLIT_YEAR = 2022

# =========================
# CONFIG
# =========================
USE_TREND_FILTER = True
MAX_SIGNALS_PER_DAY = 2
USE_EQUAL_WEIGHT = True


# =========================
# LOAD
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["time"], format="mixed")
    df = df.sort_values("timestamp")
    df["year"] = df["timestamp"].dt.year

    return df


# =========================
# SIGNAL
# =========================
def apply_signal(df):
    return df[
        (df["extreme_streak_70"] >= 3) &
        (df["crowd_persistence_bucket_70"].isin(["high", "medium"]))
    ].copy()


# =========================
# NON-OVERLAP
# =========================
def enforce_non_overlap(df, horizon):
    df = df.sort_values("timestamp").copy()

    selected_idx = []
    last_time = {}

    for idx, row in df.iterrows():
        pair = row["pair"]
        t = row["timestamp"]

        if pair not in last_time:
            selected_idx.append(idx)
            last_time[pair] = t
        else:
            delta = (t - last_time[pair]).total_seconds() / 3600
            if delta >= horizon:
                selected_idx.append(idx)
                last_time[pair] = t

    return df.loc[selected_idx].copy()

# =========================
# DAILY CAP
# =========================
def cap_signals_per_day(df):
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp")

    df = df.groupby("date", group_keys=False).head(MAX_SIGNALS_PER_DAY)

    return df.drop(columns="date")


# =========================
# EQUAL WEIGHT
# =========================
def apply_equal_weight(df, horizon):
    df = df.copy()
    col = f"contrarian_ret_{horizon}b"

    # True equal weight per trade
    df["pair_weight"] = 1.0
    df["weighted_ret"] = df[col]

    return df


# =========================
# METRICS (PORTFOLIO LEVEL)
# =========================
def compute_metrics(df, horizon):
    col = "weighted_ret" if USE_EQUAL_WEIGHT else f"contrarian_ret_{horizon}b"

    r = df[col].dropna()

    return {
        "n": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan,
        "hit_rate": (r > 0).mean()
    }


# =========================
# HOLDOUT (RAW RETURNS ONLY)
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
# WALK-FORWARD (RAW RETURNS)
# =========================
def walk_forward(df, horizon):
    col = f"contrarian_ret_{horizon}b"

    results = []

    for year in sorted(df["year"].unique()):
        test = df[df["year"] == year]

        if len(test) < 20:
            continue

        r = test[col].dropna()

        results.append({
            "year": year,
            "n": len(r),
            "mean": r.mean(),
            "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan
        })

    return pd.DataFrame(results)


# =========================
# SURVIVOR FILTER
# =========================
def select_survivors(df, horizon):
    col = f"contrarian_ret_{horizon}b"

    survivors = []

    for pair, g in df.groupby("pair"):
        g = enforce_non_overlap(g, horizon)

        if len(g) < 100:
            continue

        r = g[col].dropna()

        if len(r) < 50:
            continue

        sharpe = r.mean() / r.std() if r.std() > 0 else 0

        hold = holdout_test(g, horizon)

        if hold["test"] is None:
            continue

        test_sharpe = hold["test"]["sharpe"]

        if sharpe > 0.08 and test_sharpe > 0:
            survivors.append(pair)

    return survivors


# =========================
# PORTFOLIO
# =========================
def build_portfolio(signal, survivors, horizon):
    df = signal[signal["pair"].isin(survivors)].copy()
    print("\n[DEBUG] After survivor filter:", len(df))

    if USE_TREND_FILTER:
        if "trend_alignment_48b" not in df.columns:
            print("[DEBUG] trend column missing!")
        else:
            print("[DEBUG] trend counts:")
            print(df["trend_alignment_48b"].value_counts(dropna=False))

            df = df[df["trend_alignment_48b"] == -1]
            print("[DEBUG] After trend filter:", len(df))

    df = enforce_non_overlap(df, horizon)
    print("[DEBUG] After non-overlap:", len(df))

    df = cap_signals_per_day(df)
    print("[DEBUG] After daily cap:", len(df))

    if USE_EQUAL_WEIGHT:
        df = apply_equal_weight(df, horizon)

    return df


# =========================
# MAIN
# =========================
def run():
    print("Loading dataset...")
    df = load_data()
    print("Dataset size:", len(df))

    signal = apply_signal(df)
    print("Raw signal count:", len(signal))

    for horizon in HORIZONS:
        print("\n" + "="*80)
        print(f"PORTFOLIO (horizon={horizon})")
        print("="*80)

        survivors = select_survivors(signal, horizon)

        print("\nSurvivor pairs:")
        print(survivors)

        if len(survivors) == 0:
            print("No survivors.")
            continue

        portfolio = build_portfolio(signal, survivors, horizon)

        print("\nPortfolio size:", len(portfolio))

        # ===== METRICS =====
        m = compute_metrics(portfolio, horizon)
        print("\n--- Overall ---")
        print(m)

        # ===== WALK-FORWARD =====
        wf = walk_forward(portfolio, horizon)
        print("\n--- Walk-forward ---")
        print(wf)

        if not wf.empty:
            print("\nWF Sharpe mean:", wf["sharpe"].mean())

        # ===== HOLDOUT =====
        hold = holdout_test(portfolio, horizon)
        print("\n--- Holdout ---")
        print(hold)

        # ===== TRADEABILITY =====
        print("\n--- Tradeability ---")
        print({
            "bps": m["mean"] * 10000,
            "sharpe": m["sharpe"]
        })


if __name__ == "__main__":
    run()