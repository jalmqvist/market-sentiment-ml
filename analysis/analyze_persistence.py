from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# Configuration
# =========================

DATASET_PATH = Path("data/output/analysis/master_research_dataset_core_cleaned.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 6, 12, 24, 48]
DATE_COLUMNS = ["snapshot_time", "entry_time"]


# =========================
# Pair grouping
# =========================

PAIR_GROUPS = {
    # majors
    "eur-usd": "major",
    "gbp-usd": "major",
    "usd-jpy": "major",
    "usd-chf": "major",
    "usd-cad": "major",
    "aud-usd": "major",
    "nzd-usd": "major",

    # G10 / liquid crosses
    "eur-gbp": "cross",
    "eur-jpy": "cross",
    "eur-chf": "cross",
    "eur-cad": "cross",
    "eur-aud": "cross",
    "eur-nzd": "cross",
    "gbp-jpy": "cross",
    "gbp-chf": "cross",
    "gbp-cad": "cross",
    "gbp-aud": "cross",
    "gbp-nzd": "cross",
    "aud-jpy": "cross",
    "aud-chf": "cross",
    "aud-cad": "cross",
    "aud-nzd": "cross",
    "cad-jpy": "cross",
    "cad-chf": "cross",
    "chf-jpy": "cross",
    "nzd-jpy": "cross",
    "nzd-chf": "cross",

    # thinner / exotic / regional pairs
    "aud-sgd": "thin_exotic",
    "usd-sgd": "thin_exotic",
    "gbp-sgd": "thin_exotic",
    "sgd-jpy": "thin_exotic",
    "eur-nok": "thin_exotic",
    "usd-nok": "thin_exotic",
    "gbp-nok": "thin_exotic",
    "chf-nok": "thin_exotic",
    "cad-nok": "thin_exotic",
    "eur-sek": "thin_exotic",
    "usd-sek": "thin_exotic",
    "gbp-sek": "thin_exotic",
    "nok-sek": "thin_exotic",
    "eur-pln": "thin_exotic",
    "usd-pln": "thin_exotic",
    "gbp-pln": "thin_exotic",
    "eur-huf": "thin_exotic",
    "usd-huf": "thin_exotic",
    "gbp-huf": "thin_exotic",
    "chf-huf": "thin_exotic",
    "eur-mxn": "thin_exotic",
    "usd-mxn": "thin_exotic",
    "gbp-mxn": "thin_exotic",
    "usd-zar": "thin_exotic",
    "gbp-zar": "thin_exotic",
}


# =========================
# Helpers
# =========================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, parse_dates=DATE_COLUMNS)

    required = {
        "pair",
        "snapshot_time",
        "entry_time",
        "abs_sentiment",
        "extreme_streak_70",
        "extreme_streak_80",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def collapse_to_entry_bar(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["pair", "entry_time", "snapshot_time"])
        .groupby(["pair", "entry_time"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def add_pair_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pair_group"] = out["pair"].map(PAIR_GROUPS).fillna("unclassified")
    return out


def summarize_subset(df: pd.DataFrame, subset_name: str, horizons: Iterable[int]) -> pd.DataFrame:
    rows = []

    for h in horizons:
        col = f"contrarian_ret_{h}b"
        s = df[col].dropna()

        rows.append({
            "subset": subset_name,
            "horizon_bars": h,
            "n": int(s.shape[0]),
            "mean_contrarian_ret": float(s.mean()) if len(s) else pd.NA,
            "median_contrarian_ret": float(s.median()) if len(s) else pd.NA,
            "std_contrarian_ret": float(s.std()) if len(s) else pd.NA,
            "hit_rate": float((s > 0).mean()) if len(s) else pd.NA,
        })

    return pd.DataFrame(rows)


def build_persistence_summary(
    df: pd.DataFrame,
    streak_col: str,
    streak_prefix: str,
    thresholds: list[int],
    horizons: Iterable[int],
) -> pd.DataFrame:
    parts = []

    for threshold in thresholds:
        base = df[df["abs_sentiment"] >= threshold].copy()

        s1 = base[base[streak_col] == 1].copy()
        s2 = base[base[streak_col] == 2].copy()
        s3p = base[base[streak_col] >= 3].copy()

        parts.append(summarize_subset(s1, f"{streak_prefix}_thr{threshold}_streak1", horizons))
        parts.append(summarize_subset(s2, f"{streak_prefix}_thr{threshold}_streak2", horizons))
        parts.append(summarize_subset(s3p, f"{streak_prefix}_thr{threshold}_streak3plus", horizons))

    return pd.concat(parts, ignore_index=True)


def run_for_group(df: pd.DataFrame, group_name: str, out_stem: str) -> None:
    print(f"\n=== Persistence analysis: {group_name} ===")
    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")

    summary_70 = build_persistence_summary(
        df=df,
        streak_col="extreme_streak_70",
        streak_prefix="ext70",
        thresholds=[70, 80],
        horizons=HORIZONS,
    )

    summary_80 = build_persistence_summary(
        df=df,
        streak_col="extreme_streak_80",
        streak_prefix="ext80",
        thresholds=[80],
        horizons=HORIZONS,
    )

    print("\n--- ext70 summary ---")
    print(summary_70.to_string(index=False))

    print("\n--- ext80 summary ---")
    print(summary_80.to_string(index=False))

    summary_70.to_csv(OUTPUT_DIR / f"persistence_summary_{out_stem}_ext70.csv", index=False)
    summary_80.to_csv(OUTPUT_DIR / f"persistence_summary_{out_stem}_ext80.csv", index=False)

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / f'persistence_summary_{out_stem}_ext70.csv'}")
    print(f"  {OUTPUT_DIR / f'persistence_summary_{out_stem}_ext80.csv'}")


def main() -> None:
    df = load_dataset(DATASET_PATH)
    df = collapse_to_entry_bar(df)
    df = add_pair_group(df)

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")

    # Full cleaned dataset
    run_for_group(df, "all_cleaned_core", "all_cleaned_core")

    # Crosses first: most interesting from threshold analysis
    cross_df = df[df["pair_group"] == "cross"].copy()
    run_for_group(cross_df, "cross", "cross")

    # Majors
    major_df = df[df["pair_group"] == "major"].copy()
    run_for_group(major_df, "major", "major")

    # Thin/exotic
    thin_df = df[df["pair_group"] == "thin_exotic"].copy()
    run_for_group(thin_df, "thin_exotic", "thin_exotic")


if __name__ == "__main__":
    main()
