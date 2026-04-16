import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "data/output/master_research_dataset_core.csv"
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [12, 48]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def classify_trend_alignment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mapping = {
        -1: "fight_trend",
         1: "follow_trend"
    }

    out["trend_behavior"] = out["trend_alignment_12b"].map(mapping)

    return out


def classify_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["persistence_bucket"] = np.where(
        out["extreme_streak_70"] >= 3, "persistent",
        np.where(out["extreme_streak_70"] >= 1, "non_persistent", "none")
    )

    return out


def classify_jpy_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_jpy_cross"] = out["pair"].str.endswith("-jpy")
    out["pair_group_simple"] = np.where(out["is_jpy_cross"], "JPY_cross", "non_JPY")

    return out

def ensure_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure trend strength bucket exists (defensive).
    """
    out = df.copy()

    if "trend_strength_bucket_12b" not in out.columns:
        raise ValueError("Missing trend_strength_bucket_12b — rebuild dataset")

    return out

def summarize(df: pd.DataFrame, ret_col: str) -> dict:
    x = df[ret_col].dropna()

    # --- HARD FILTER (consistent across all analysis scripts) ---
    x = x[(x > -0.1) & (x < 0.1)]
    # -----------------------------------------------------------

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "hit_rate": np.nan,
        }

    return {
        "n": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "std": x.std(),
        "hit_rate": (x > 0).mean(),
    }


# --------------------------------------------------
# Core analysis
# --------------------------------------------------

def run_analysis(df: pd.DataFrame):

    results = []

    for h in HORIZONS:
        ret_col = f"contrarian_ret_{h}b"

        # 1. Trend only
        for trend_bucket in ["follow_trend", "fight_trend"]:
            subset = df[df["trend_behavior"] == trend_bucket]

            stats = summarize(subset, ret_col)

            results.append({
                "horizon": h,
                "analysis": "trend_only",
                "trend_bucket": trend_bucket,
                "persistence": "all",
                "pair_group": "all",
                **stats
            })

        # 2. Trend × persistence
        for trend_bucket in ["follow_trend", "fight_trend"]:
            for persistence in ["persistent", "non_persistent"]:
                subset = df[
                    (df["trend_behavior"] == trend_bucket) &
                    (df["persistence_bucket"] == persistence)
                ]

                stats = summarize(subset, ret_col)

                results.append({
                    "horizon": h,
                    "analysis": "trend_x_persistence",
                    "trend_bucket": trend_bucket,
                    "persistence": persistence,
                    "pair_group": "all",
                    **stats
                })

        # 3. Trend × JPY vs non-JPY
        for trend_bucket in ["follow_trend", "fight_trend"]:
            for pg in ["JPY_cross", "non_JPY"]:
                subset = df[
                    (df["trend_behavior"] == trend_bucket) &
                    (df["pair_group_simple"] == pg)
                ]

                stats = summarize(subset, ret_col)

                results.append({
                    "horizon": h,
                    "analysis": "trend_x_pair_group",
                    "trend_bucket": trend_bucket,
                    "persistence": "all",
                    "pair_group": pg,
                    **stats
                })

        # 4. Full interaction (🔥 most important)
        for trend_bucket in ["follow_trend", "fight_trend"]:
            for persistence in ["persistent", "non_persistent"]:
                for pg in ["JPY_cross", "non_JPY"]:

                    subset = df[
                        (df["trend_behavior"] == trend_bucket) &
                        (df["persistence_bucket"] == persistence) &
                        (df["pair_group_simple"] == pg)
                    ]

                    stats = summarize(subset, ret_col)

                    results.append({
                        "horizon": h,
                        "analysis": "full_interaction",
                        "trend_bucket": trend_bucket,
                        "persistence": persistence,
                        "pair_group": pg,
                        **stats
                    })
        # 5. Trend × strength (NEW)
        for strength in ["weak", "medium", "strong", "extreme"]:
            for trend_bucket in ["follow_trend", "fight_trend"]:

                subset = df[
                    (df["trend_behavior"] == trend_bucket) &
                    (df["trend_strength_bucket_12b"] == strength)
                ]

                stats = summarize(subset, ret_col)

                results.append({
                    "horizon": h,
                    "analysis": "trend_x_strength",
                    "trend_bucket": trend_bucket,
                    "trend_strength": strength,
                    "persistence": "all",
                    "pair_group": "all",
                    **stats
                })

        # 6. Full interaction × strength (🔥)
        for strength in ["weak", "medium", "strong", "extreme"]:
            for trend_bucket in ["follow_trend", "fight_trend"]:
                for persistence in ["persistent", "non_persistent"]:
                    for pg in ["JPY_cross", "non_JPY"]:

                        subset = df[
                            (df["trend_behavior"] == trend_bucket) &
                            (df["trend_strength_bucket_12b"] == strength) &
                            (df["persistence_bucket"] == persistence) &
                            (df["pair_group_simple"] == pg)
                        ]

                        stats = summarize(subset, ret_col)

                        results.append({
                            "horizon": h,
                            "analysis": "full_interaction_strength",
                            "trend_bucket": trend_bucket,
                            "trend_strength": strength,
                            "persistence": persistence,
                            "pair_group": pg,
                            **stats
                        })
    return pd.DataFrame(results)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    # Feature classification
    df = classify_trend_alignment(df)
    df = classify_persistence(df)
    df = classify_jpy_group(df)
    df = ensure_trend_strength(df)

    # Only keep rows with valid trend
    df = df[
        df["trend_behavior"].notna() &
        df["trend_alignment_12b"].isin([-1, 1])
        ]

    print(f"Rows after filtering: {len(df)}")

    result_df = run_analysis(df)

    output_path = OUTPUT_DIR / "trend_behavior_summary.csv"
    result_df.to_csv(output_path, index=False)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()