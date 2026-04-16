import pandas as pd
import numpy as np

INPUT_PATH = "data/output/analysis/trend_behavior_summary.csv"

STRENGTH_ORDER = ["weak", "medium", "strong", "extreme"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def order_strength(df):
    return df.set_index("trend_strength").loc[STRENGTH_ORDER].reset_index()


def detect_pattern(values):
    """
    Detect whether pattern is monotonic or peak-shaped
    """
    arr = np.array(values)

    if np.all(np.diff(arr) >= 0):
        return "monotonic_increasing"

    if np.all(np.diff(arr) <= 0):
        return "monotonic_decreasing"

    peak_idx = np.argmax(arr)
    if peak_idx == 2:  # "strong"
        return "peak_at_strong"

    if peak_idx == 3:  # "extreme"
        return "peak_at_extreme"

    return "non_monotonic"


def print_block(title, df):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    df = order_strength(df)

    print(df[[
        "trend_strength",
        "n",
        "mean",
        "median",
        "std",
        "hit_rate"
    ]])

    # pattern detection
    mean_pattern = detect_pattern(df["mean"].values)
    hit_pattern = detect_pattern(df["hit_rate"].values)

    print("\nPattern detection:")
    print(f"mean      → {mean_pattern}")
    print(f"hit_rate  → {hit_pattern}")


# --------------------------------------------------
# Core analysis
# --------------------------------------------------

def analyze_core(df):

    # Focus on most important slice first
    base = df[
        (df["analysis"] == "full_interaction_strength") &
        (df["pair_group"] == "JPY_cross") &
        (df["persistence"] == "persistent")
    ]

    for h in [12, 48]:
        for trend in ["fight_trend", "follow_trend"]:

            subset = base[
                (base["horizon"] == h) &
                (base["trend_bucket"] == trend)
            ]

            if len(subset) == 0:
                continue

            print_block(
                f"JPY × persistent × {trend} × horizon={h}",
                subset
            )


# --------------------------------------------------
# Compare JPY vs non-JPY
# --------------------------------------------------

def analyze_pair_group(df):

    base = df[
        (df["analysis"] == "full_interaction_strength") &
        (df["persistence"] == "persistent") &
        (df["trend_bucket"] == "fight_trend") &
        (df["horizon"] == 12)
    ]

    for pg in ["JPY_cross", "non_JPY"]:

        subset = base[base["pair_group"] == pg]

        print_block(
            f"{pg} × persistent × fight_trend × horizon=12",
            subset
        )


# --------------------------------------------------
# Variance analysis
# --------------------------------------------------

def analyze_variance(df):

    base = df[
        (df["analysis"] == "full_interaction_strength") &
        (df["pair_group"] == "JPY_cross") &
        (df["persistence"] == "persistent") &
        (df["trend_bucket"] == "fight_trend") &
        (df["horizon"] == 12)
    ]

    base = order_strength(base)

    print("\n" + "=" * 80)
    print("Variance behavior (JPY × persistent × fight_trend × 12b)")
    print("=" * 80)

    for _, row in base.iterrows():
        print(f"{row['trend_strength']}: std = {row['std']:.6f}")


# --------------------------------------------------
# Horizon comparison
# --------------------------------------------------

def analyze_horizon(df):

    base = df[
        (df["analysis"] == "full_interaction_strength") &
        (df["pair_group"] == "JPY_cross") &
        (df["persistence"] == "persistent") &
        (df["trend_bucket"] == "fight_trend")
    ]

    for strength in STRENGTH_ORDER:

        subset = base[base["trend_strength"] == strength]

        if len(subset) == 0:
            continue

        print("\n" + "=" * 80)
        print(f"Horizon comparison for strength={strength}")
        print("=" * 80)

        print(subset[[
            "horizon",
            "mean",
            "median",
            "std",
            "hit_rate"
        ]].sort_values("horizon"))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    # Only rows with strength
    df = df[df["trend_strength"].notna()]

    # Ensure ordering works
    df["trend_strength"] = pd.Categorical(
        df["trend_strength"],
        categories=STRENGTH_ORDER,
        ordered=True
    )

    print("\n=== CORE ANALYSIS ===")
    analyze_core(df)

    print("\n=== JPY vs NON-JPY ===")
    analyze_pair_group(df)

    print("\n=== VARIANCE ANALYSIS ===")
    analyze_variance(df)

    print("\n=== HORIZON COMPARISON ===")
    analyze_horizon(df)


if __name__ == "__main__":
    main()