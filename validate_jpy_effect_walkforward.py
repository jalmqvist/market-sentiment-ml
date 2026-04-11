from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================

DATASET_PATH = Path("data/output/analysis/master_research_dataset_core_cleaned.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMNS = ["snapshot_time", "entry_time"]

JPY_CROSSES = {
    "aud-jpy",
    "cad-jpy",
    "chf-jpy",
    "eur-jpy",
    "gbp-jpy",
    "nzd-jpy",
}

CROSS_PAIRS = {
    "eur-gbp",
    "eur-jpy",
    "eur-chf",
    "eur-cad",
    "eur-aud",
    "eur-nzd",
    "gbp-jpy",
    "gbp-chf",
    "gbp-cad",
    "gbp-aud",
    "gbp-nzd",
    "aud-jpy",
    "aud-chf",
    "aud-cad",
    "aud-nzd",
    "cad-jpy",
    "cad-chf",
    "chf-jpy",
    "nzd-jpy",
    "nzd-chf",
}

HORIZONS = [12, 48]

# Fixed discovered condition
MIN_ABS_SENTIMENT = 70
MIN_EXTREME_STREAK_70 = 3

# Require at least this many years before first test year
MIN_TRAIN_YEARS = 2


# =========================
# Helpers
# =========================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def collapse_to_entry_bar(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["pair", "entry_time", "snapshot_time"])
        .groupby(["pair", "entry_time"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["pair"].isin(CROSS_PAIRS)].copy()
    df["is_jpy_cross"] = df["pair"].isin(JPY_CROSSES)
    df["year"] = df["snapshot_time"].dt.year

    df = df[
        (df["abs_sentiment"] >= MIN_ABS_SENTIMENT) &
        (df["extreme_streak_70"] >= MIN_EXTREME_STREAK_70)
    ].copy()

    return df


def summarize_group(df: pd.DataFrame, group_name: str, horizon: int) -> dict:
    col = f"contrarian_ret_{horizon}b"
    s = df[col].dropna()

    return {
        "group": group_name,
        "horizon_bars": horizon,
        "n": int(len(s)),
        "mean_contrarian_ret": float(s.mean()) if len(s) else pd.NA,
        "median_contrarian_ret": float(s.median()) if len(s) else pd.NA,
        "std_contrarian_ret": float(s.std()) if len(s) else pd.NA,
        "hit_rate": float((s > 0).mean()) if len(s) else pd.NA,
    }


def summarize_period(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        jpy = df[df["is_jpy_cross"]]
        non_jpy = df[~df["is_jpy_cross"]]

        rows.append({
            "period": label,
            **summarize_group(jpy, "JPY_cross", h),
        })
        rows.append({
            "period": label,
            **summarize_group(non_jpy, "non_JPY_cross", h),
        })

        jpy_s = jpy[f"contrarian_ret_{h}b"].dropna()
        non_jpy_s = non_jpy[f"contrarian_ret_{h}b"].dropna()

        rows.append({
            "period": label,
            "group": "JPY_minus_nonJPY",
            "horizon_bars": h,
            "n": pd.NA,
            "mean_contrarian_ret": (float(jpy_s.mean()) - float(non_jpy_s.mean())) if len(jpy_s) and len(non_jpy_s) else pd.NA,
            "median_contrarian_ret": pd.NA,
            "std_contrarian_ret": pd.NA,
            "hit_rate": ((jpy_s > 0).mean() - (non_jpy_s > 0).mean()) if len(jpy_s) and len(non_jpy_s) else pd.NA,
        })

    return pd.DataFrame(rows)


def build_walkforward_summary(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].dropna().unique().tolist())

    if len(years) < MIN_TRAIN_YEARS + 1:
        raise ValueError("Not enough years for walk-forward validation.")

    rows = []

    for i in range(MIN_TRAIN_YEARS, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_df = df[df["year"].isin(train_years)].copy()
        test_df = df[df["year"] == test_year].copy()

        for h in HORIZONS:
            train_jpy = train_df[train_df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()
            train_non = train_df[~train_df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()

            test_jpy = test_df[test_df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()
            test_non = test_df[~test_df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()

            rows.append({
                "train_start_year": min(train_years),
                "train_end_year": max(train_years),
                "test_year": test_year,
                "horizon_bars": h,

                "train_jpy_n": len(train_jpy),
                "train_non_jpy_n": len(train_non),
                "train_jpy_mean": float(train_jpy.mean()) if len(train_jpy) else pd.NA,
                "train_non_jpy_mean": float(train_non.mean()) if len(train_non) else pd.NA,
                "train_diff_mean": (float(train_jpy.mean()) - float(train_non.mean())) if len(train_jpy) and len(train_non) else pd.NA,
                "train_jpy_hit": float((train_jpy > 0).mean()) if len(train_jpy) else pd.NA,
                "train_non_jpy_hit": float((train_non > 0).mean()) if len(train_non) else pd.NA,
                "train_diff_hit": ((train_jpy > 0).mean() - (train_non > 0).mean()) if len(train_jpy) and len(train_non) else pd.NA,

                "test_jpy_n": len(test_jpy),
                "test_non_jpy_n": len(test_non),
                "test_jpy_mean": float(test_jpy.mean()) if len(test_jpy) else pd.NA,
                "test_non_jpy_mean": float(test_non.mean()) if len(test_non) else pd.NA,
                "test_diff_mean": (float(test_jpy.mean()) - float(test_non.mean())) if len(test_jpy) and len(test_non) else pd.NA,
                "test_jpy_hit": float((test_jpy > 0).mean()) if len(test_jpy) else pd.NA,
                "test_non_jpy_hit": float((test_non > 0).mean()) if len(test_non) else pd.NA,
                "test_diff_hit": ((test_jpy > 0).mean() - (test_non > 0).mean()) if len(test_jpy) and len(test_non) else pd.NA,

                "test_same_sign_as_train_mean": (
                    ((float(train_jpy.mean()) - float(train_non.mean())) > 0) ==
                    ((float(test_jpy.mean()) - float(test_non.mean())) > 0)
                ) if len(train_jpy) and len(train_non) and len(test_jpy) and len(test_non) else pd.NA,

                "test_same_sign_as_train_hit": (
                    (((train_jpy > 0).mean() - (train_non > 0).mean()) > 0) ==
                    (((test_jpy > 0).mean() - (test_non > 0).mean()) > 0)
                ) if len(train_jpy) and len(train_non) and len(test_jpy) and len(test_non) else pd.NA,
            })

    return pd.DataFrame(rows)


def build_walkforward_aggregate(wf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        sub = wf[wf["horizon_bars"] == h].copy()

        rows.append({
            "horizon_bars": h,
            "n_test_windows": len(sub),
            "mean_test_diff_mean": sub["test_diff_mean"].mean(),
            "median_test_diff_mean": sub["test_diff_mean"].median(),
            "positive_test_diff_mean_rate": (sub["test_diff_mean"] > 0).mean(),
            "same_sign_as_train_mean_rate": sub["test_same_sign_as_train_mean"].mean(),

            "mean_test_diff_hit": sub["test_diff_hit"].mean(),
            "median_test_diff_hit": sub["test_diff_hit"].median(),
            "positive_test_diff_hit_rate": (sub["test_diff_hit"] > 0).mean(),
            "same_sign_as_train_hit_rate": sub["test_same_sign_as_train_hit"].mean(),
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset(DATASET_PATH)
    df = collapse_to_entry_bar(df)
    df = prepare_dataset(df)

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows after filtering: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")
    print(f"Years: {df['year'].min()} -> {df['year'].max()}")

    overall_summary = summarize_period(df, "all_years")
    wf_summary = build_walkforward_summary(df)
    wf_aggregate = build_walkforward_aggregate(wf_summary)

    overall_out = OUTPUT_DIR / "jpy_effect_walkforward_overall_summary.csv"
    wf_out = OUTPUT_DIR / "jpy_effect_walkforward_windows.csv"
    wf_agg_out = OUTPUT_DIR / "jpy_effect_walkforward_aggregate.csv"

    overall_summary.to_csv(overall_out, index=False)
    wf_summary.to_csv(wf_out, index=False)
    wf_aggregate.to_csv(wf_agg_out, index=False)

    print("\n=== Overall summary ===")
    print(overall_summary.to_string(index=False))

    print("\n=== Walk-forward windows ===")
    print(wf_summary.to_string(index=False))

    print("\n=== Walk-forward aggregate ===")
    print(wf_aggregate.to_string(index=False))

    print("\nSaved:")
    print(f"  {overall_out}")
    print(f"  {wf_out}")
    print(f"  {wf_agg_out}")


if __name__ == "__main__":
    main()
