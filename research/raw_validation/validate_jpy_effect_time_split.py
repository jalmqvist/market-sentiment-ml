'''
This is the first time-based holdout test.

It uses the current discovered effect:

- cleaned core dataset
- cross universe
- JPY-cross subgroup
- condition: `abs_sentiment >= 70` and `extreme_streak_70 >= 3`

It evaluates:
- JPY crosses vs non-JPY crosses
- by year
- and by a simple discovery/holdout split
'''


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
DISCOVERY_END_YEAR = 2023  # holdout begins in 2024


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

    # Focused persistence condition discovered in earlier analysis
    df = df[(df["abs_sentiment"] >= 70) & (df["extreme_streak_70"] >= 3)].copy()

    return df


def summarize_group(df: pd.DataFrame, label: str, horizon: int) -> dict:
    col = f"contrarian_ret_{horizon}b"
    s = df[col].dropna()

    return {
        "group": label,
        "horizon_bars": horizon,
        "n": int(len(s)),
        "mean_contrarian_ret": float(s.mean()) if len(s) else pd.NA,
        "median_contrarian_ret": float(s.median()) if len(s) else pd.NA,
        "std_contrarian_ret": float(s.std()) if len(s) else pd.NA,
        "hit_rate": float((s > 0).mean()) if len(s) else pd.NA,
    }


def summarize_period(df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        rows.append({
            "period": period_name,
            **summarize_group(df[df["is_jpy_cross"]], "JPY_cross", h),
        })
        rows.append({
            "period": period_name,
            **summarize_group(df[~df["is_jpy_cross"]], "non_JPY_cross", h),
        })

        jpy = df[df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()
        non_jpy = df[~df["is_jpy_cross"]][f"contrarian_ret_{h}b"].dropna()

        rows.append({
            "period": period_name,
            "group": "JPY_minus_nonJPY",
            "horizon_bars": h,
            "n": pd.NA,
            "mean_contrarian_ret": (float(jpy.mean()) - float(non_jpy.mean())) if len(jpy) and len(non_jpy) else pd.NA,
            "median_contrarian_ret": pd.NA,
            "std_contrarian_ret": pd.NA,
            "hit_rate": ((jpy > 0).mean() - (non_jpy > 0).mean()) if len(jpy) and len(non_jpy) else pd.NA,
        })

    return pd.DataFrame(rows)


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year, grp in sorted(df.groupby("year"), key=lambda x: x[0]):
        for h in HORIZONS:
            jpy = grp[grp["is_jpy_cross"]]
            non_jpy = grp[~grp["is_jpy_cross"]]

            rows.append({
                "year": year,
                "group": "JPY_cross",
                **summarize_group(jpy, "JPY_cross", h),
            })
            rows.append({
                "year": year,
                "group": "non_JPY_cross",
                **summarize_group(non_jpy, "non_JPY_cross", h),
            })

            jpy_s = jpy[f"contrarian_ret_{h}b"].dropna()
            non_jpy_s = non_jpy[f"contrarian_ret_{h}b"].dropna()

            rows.append({
                "year": year,
                "group": "JPY_minus_nonJPY",
                "horizon_bars": h,
                "n": pd.NA,
                "mean_contrarian_ret": (float(jpy_s.mean()) - float(non_jpy_s.mean())) if len(jpy_s) and len(non_jpy_s) else pd.NA,
                "median_contrarian_ret": pd.NA,
                "std_contrarian_ret": pd.NA,
                "hit_rate": ((jpy_s > 0).mean() - (non_jpy_s > 0).mean()) if len(jpy_s) and len(non_jpy_s) else pd.NA,
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

    discovery = df[df["year"] <= DISCOVERY_END_YEAR].copy()
    holdout = df[df["year"] > DISCOVERY_END_YEAR].copy()

    overall_summary = pd.concat([
        summarize_period(df, "all_years"),
        summarize_period(discovery, "discovery"),
        summarize_period(holdout, "holdout"),
    ], ignore_index=True)

    year_summary = summarize_by_year(df)

    overall_out = OUTPUT_DIR / "jpy_effect_time_split_summary.csv"
    yearly_out = OUTPUT_DIR / "jpy_effect_yearly_summary.csv"

    overall_summary.to_csv(overall_out, index=False)
    year_summary.to_csv(yearly_out, index=False)

    print("\n=== Time split summary ===")
    print(overall_summary.to_string(index=False))

    print("\n=== Yearly summary ===")
    print(year_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {overall_out}")
    print(f"  {yearly_out}")


if __name__ == "__main__":
    main()
