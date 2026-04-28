# Legacy experiment — not part of current validated approach\nimport pandas as pd

INPUT_PATH = "data/output/master_research_dataset_core.csv"
OUTPUT_PATH = "data/output/analysis/trend_alignment_summary.csv"

HORIZONS = [12, 48]


def load_data():
    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_time"])
    return df


def filter_base(df):
    return df[
        (df["abs_sentiment"] >= 70)
        & (df["extreme_streak_70"] >= 3)
        & df["trend_alignment_12b"].isin([-1, 1])
    ].copy()


def summarize(df, label):
    rows = []

    for h in HORIZONS:
        col = f"contrarian_ret_{h}b"

        sub = df[col].dropna()

        if len(sub) == 0:
            continue

        # --- HARD FILTER (stronger than winsor) ---
        sub = sub[(sub > -0.1) & (sub < 0.1)]
        # -----------------------------------------

        rows.append({
            "subset": label,
            "horizon": h,
            "n": len(sub),
            "mean": sub.mean(),
            "median": sub.median(),
            "std": sub.std(),
            "hit_rate": (sub > 0).mean()
        })

    return rows


def main():
    df = load_data()
    df = filter_base(df)

    results = []

    # Crowd fighting trend (early reversal)
    df_fight = df[df["trend_alignment_12b"] == -1]
    results += summarize(df_fight, "fight_trend")

    # Crowd following trend (late chasing)
    df_follow = df[df["trend_alignment_12b"] == 1]
    results += summarize(df_follow, "follow_trend")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    print(out)


if __name__ == "__main__":
    main()