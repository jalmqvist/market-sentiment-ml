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
THRESHOLDS = [70, 80, 90]
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

    required_cols = {"pair", "abs_sentiment", "snapshot_time", "entry_time"}
    missing = required_cols - set(df.columns)
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


def build_group_threshold_summary(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    parts = []

    for group_name, grp in df.groupby("pair_group", observed=True):
        parts.append(summarize_subset(grp, f"{group_name} / all", horizons))

        for threshold in THRESHOLDS:
            sub = grp[grp["abs_sentiment"] >= threshold].copy()
            parts.append(
                summarize_subset(
                    sub,
                    f"{group_name} / abs_sentiment>={threshold}",
                    horizons
                )
            )

    return pd.concat(parts, ignore_index=True)


def build_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["pair_group", "pair"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["pair_group", "rows", "pair"], ascending=[True, False, True])
    )
    return out


# =========================
# Main
# =========================

def main() -> None:
    df = load_dataset(DATASET_PATH)
    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")

    # collapse again just to be safe if the input file is not already collapsed
    df = collapse_to_entry_bar(df)
    df = add_pair_group(df)

    print("\nPair-group counts:")
    group_counts = build_group_counts(df)
    print(group_counts.to_string(index=False))

    unknown_pairs = sorted(df.loc[df["pair_group"] == "unclassified", "pair"].unique().tolist())
    if unknown_pairs:
        print("\nUnclassified pairs:")
        print(unknown_pairs)

    summary = build_group_threshold_summary(df, HORIZONS)

    print("\n=== Group threshold summary ===")
    print(summary.to_string(index=False))

    group_counts_out = OUTPUT_DIR / "pair_group_counts.csv"
    summary_out = OUTPUT_DIR / "threshold_summary_by_pair_group.csv"

    group_counts.to_csv(group_counts_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("\nSaved:")
    print(f"  {group_counts_out}")
    print(f"  {summary_out}")


if __name__ == "__main__":
    main()
