# Legacy experiment — not part of current validated approach\nimport pandas as pd
import numpy as np

INPUT_PATH = "data/output/master_research_dataset_core.csv"
OUTPUT_PATH = "data/output/analysis/jpy_walk_forward_results.csv"

HORIZONS = [12, 48]


# --------------------------------------------------
# Signal definition (LOCKED)
# --------------------------------------------------

def compute_signal(df, h):
    out = df.copy()

    trend_align = out[f"trend_alignment_{h}b"]
    strength = out[f"trend_strength_bucket_{h}b"]

    signal = np.zeros(len(out))

    # Regime A: fight trend (early reversal)
    cond_a = (
        (trend_align == -1) &
        (strength.isin(["medium", "strong"]))
    )

    # Regime B: follow trend (late chasing)
    cond_b = (
        (trend_align == 1) &
        (strength == "extreme")
    )

    signal[cond_a | cond_b] = 1

    out["signal"] = signal
    return out


# --------------------------------------------------
# Filtering (LOCKED)
# --------------------------------------------------

def apply_filters(df):
    out = df.copy()

    # Derive JPY_cross from pair
    out["pair_lower"] = out["pair"].str.lower()
    out["is_jpy"] = out["pair_lower"].str.contains("jpy")

    return out[
        (out["is_jpy"]) &
        (out["abs_sentiment"] >= 70) &
        (out["extreme_streak_70"] >= 3)
    ].copy()


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def evaluate(df, ret_col):
    if len(df) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan
        }

    r = df[ret_col]

    return {
        "n": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "sharpe": r.mean() / r.std() if r.std() > 0 else np.nan,
        "hit_rate": (r > 0).mean()
    }


# --------------------------------------------------
# Walk-forward splits
# --------------------------------------------------

def add_time_folds(df):
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    # Use yearly folds (simple + interpretable)
    df["fold"] = df["entry_time"].dt.year

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    df = add_time_folds(df)
    df = apply_filters(df)

    results = []

    for h in HORIZONS:
        print(f"Processing horizon {h}...")

        df_h = compute_signal(df, h)

        ret_col = f"contrarian_ret_{h}b"

        for fold in sorted(df_h["fold"].dropna().unique()):

            fold_df = df_h[df_h["fold"] == fold]

            # Only evaluate where signal is active
            active = fold_df[fold_df["signal"] == 1]

            stats = evaluate(active, ret_col)

            results.append({
                "fold": fold,
                "horizon": h,
                **stats
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("\nWalk-forward results:")
    print(out_df)


if __name__ == "__main__":
    main()