# Legacy experiment — not part of current validated approach\nfrom __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

DATASETS = {
    "full": Path("data/output/master_research_dataset.csv"),
    "core": Path("data/output/master_research_dataset_core.csv"),
    "core_cleaned": Path("data/output/analysis/master_research_dataset_core_cleaned.csv"),
    "extended": Path("data/output/master_research_dataset_extended.csv"),
}

OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 6, 12, 24, 48]
THRESHOLDS = [70, 80, 90]

DATE_COLUMNS = ["snapshot_time", "entry_time"]


# =========================
# Helpers
# =========================

def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else np.nan


def safe_median(s: pd.Series) -> float:
    return float(s.median()) if len(s) else np.nan


def safe_std(s: pd.Series) -> float:
    return float(s.std()) if len(s) else np.nan


def safe_hit_rate(s: pd.Series) -> float:
    s = s.dropna()
    return float((s > 0).mean()) if len(s) else np.nan


def load_master_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, parse_dates=DATE_COLUMNS)

    required_cols = {"pair", "snapshot_time", "entry_time", "abs_sentiment", "net_sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for h in HORIZONS:
        needed = {f"ret_{h}b", f"contrarian_ret_{h}b"}
        missing_h = needed - set(df.columns)
        if missing_h:
            raise ValueError(f"{path.name} is missing horizon columns: {missing_h}")

    return df


def collapse_to_entry_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the last sentiment snapshot for each (pair, entry_time).
    This removes repeated signals that map to the same entry bar.
    """
    out = (
        df.sort_values(["pair", "entry_time", "snapshot_time"])
        .groupby(["pair", "entry_time"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
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
            "mean_contrarian_ret": safe_mean(s),
            "median_contrarian_ret": safe_median(s),
            "std_contrarian_ret": safe_std(s),
            "hit_rate": safe_hit_rate(s),
        })

    return pd.DataFrame(rows)


def build_threshold_summary(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    parts = [summarize_subset(df, "all", horizons)]

    for threshold in THRESHOLDS:
        sub = df[df["abs_sentiment"] >= threshold].copy()
        parts.append(summarize_subset(sub, f"abs_sentiment>={threshold}", horizons))

    out = pd.concat(parts, ignore_index=True)
    return out


def build_decile_summary(df: pd.DataFrame, horizons: Iterable[int], bins: int = 10) -> pd.DataFrame:
    """
    Bucket abs_sentiment into quantile bins for a quick monotonicity check.
    """
    working = df.copy()

    # duplicates='drop' protects against insufficient unique values
    working["abs_sentiment_bin"] = pd.qcut(
        working["abs_sentiment"],
        q=bins,
        duplicates="drop"
    )

    rows = []
    for sentiment_bin, grp in working.groupby("abs_sentiment_bin", observed=True):
        for h in horizons:
            col = f"contrarian_ret_{h}b"
            s = grp[col].dropna()
            rows.append({
                "abs_sentiment_bin": str(sentiment_bin),
                "horizon_bars": h,
                "n": int(s.shape[0]),
                "mean_contrarian_ret": safe_mean(s),
                "median_contrarian_ret": safe_median(s),
                "hit_rate": safe_hit_rate(s),
            })

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def run_for_dataset(dataset_name: str, dataset_path: Path) -> None:
    print(f"\n\n##############################")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"##############################")

    df = load_master_dataset(dataset_path)

    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")
    print(f"Date range: {df['snapshot_time'].min()} -> {df['snapshot_time'].max()}")

    variants = {
        "raw": df,
        "collapsed": collapse_to_entry_bar(df),
    }

    for variant_name, variant_df in variants.items():
        print(f"\n--- Variant: {variant_name} ---")
        print(f"Rows: {len(variant_df):,}")
        print(f"Pairs: {variant_df['pair'].nunique():,}")

        threshold_summary = build_threshold_summary(variant_df, HORIZONS)
        decile_summary = build_decile_summary(variant_df, HORIZONS, bins=10)

        print_summary_table(threshold_summary, f"{dataset_name} / {variant_name} / threshold summary")

        threshold_out = OUTPUT_DIR / f"threshold_summary_{dataset_name}_{variant_name}.csv"
        decile_out = OUTPUT_DIR / f"decile_summary_{dataset_name}_{variant_name}.csv"

        threshold_summary.to_csv(threshold_out, index=False)
        decile_summary.to_csv(decile_out, index=False)

        print(f"\nSaved: {threshold_out}")
        print(f"Saved: {decile_out}")


def main() -> None:
    for dataset_name, dataset_path in DATASETS.items():
        if dataset_path.exists():
            run_for_dataset(dataset_name, dataset_path)
        else:
            print(f"\nSkipping missing dataset: {dataset_path}")


if __name__ == "__main__":
    main()
