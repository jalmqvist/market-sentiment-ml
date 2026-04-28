# Legacy experiment — not part of current validated approach\nfrom __future__ import annotations

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


def summarize_subset(df: pd.DataFrame, pair: str, subset_name: str, horizons: Iterable[int]) -> pd.DataFrame:
    rows = []

    for h in horizons:
        col = f"contrarian_ret_{h}b"
        s = df[col].dropna()

        rows.append({
            "pair": pair,
            "subset": subset_name,
            "horizon_bars": h,
            "n": int(s.shape[0]),
            "mean_contrarian_ret": float(s.mean()) if len(s) else pd.NA,
            "median_contrarian_ret": float(s.median()) if len(s) else pd.NA,
            "std_contrarian_ret": float(s.std()) if len(s) else pd.NA,
            "hit_rate": float((s > 0).mean()) if len(s) else pd.NA,
        })

    return pd.DataFrame(rows)


def build_pair_persistence_summary(
    df: pd.DataFrame,
    streak_col: str,
    threshold: int,
    pair_list: list[str],
    horizons: Iterable[int],
) -> pd.DataFrame:
    parts = []

    for pair in pair_list:
        grp = df[df["pair"] == pair].copy()
        base = grp[grp["abs_sentiment"] >= threshold].copy()

        s1 = base[base[streak_col] == 1].copy()
        s2 = base[base[streak_col] == 2].copy()
        s3p = base[base[streak_col] >= 3].copy()

        parts.append(summarize_subset(s1, pair, f"thr{threshold}_streak1", horizons))
        parts.append(summarize_subset(s2, pair, f"thr{threshold}_streak2", horizons))
        parts.append(summarize_subset(s3p, pair, f"thr{threshold}_streak3plus", horizons))

    return pd.concat(parts, ignore_index=True)


def build_focus_table(summary: pd.DataFrame, focus_horizons: list[int]) -> pd.DataFrame:
    """
    Create a compact table for easy inspection at selected horizons.
    """
    sub = summary[summary["horizon_bars"].isin(focus_horizons)].copy()

    pivot_mean = sub.pivot_table(
        index=["pair", "subset"],
        columns="horizon_bars",
        values="mean_contrarian_ret"
    )
    pivot_hit = sub.pivot_table(
        index=["pair", "subset"],
        columns="horizon_bars",
        values="hit_rate"
    )
    pivot_n = sub.pivot_table(
        index=["pair", "subset"],
        columns="horizon_bars",
        values="n"
    )

    pivot_mean.columns = [f"mean_{c}b" for c in pivot_mean.columns]
    pivot_hit.columns = [f"hit_{c}b" for c in pivot_hit.columns]
    pivot_n.columns = [f"n_{c}b" for c in pivot_n.columns]

    out = pd.concat([pivot_n, pivot_mean, pivot_hit], axis=1).reset_index()
    return out


def print_top_pairs(focus: pd.DataFrame, subset_name: str, metric_col: str, top_n: int = 10) -> None:
    view = focus[focus["subset"] == subset_name].copy()
    view = view.sort_values(metric_col, ascending=False)

    cols = [c for c in ["pair", "subset", "n_12b", "mean_12b", "hit_12b", "n_48b", "mean_48b", "hit_48b"] if c in view.columns]

    print(f"\nTop {top_n} pairs for {subset_name} by {metric_col}:")
    print(view[cols].head(top_n).to_string(index=False))


# =========================
# Main
# =========================

def main() -> None:
    df = load_dataset(DATASET_PATH)
    df = collapse_to_entry_bar(df)
    df = add_pair_group(df)

    cross_df = df[df["pair_group"] == "cross"].copy()
    cross_pairs = sorted(cross_df["pair"].unique().tolist())

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows after collapse: {len(df):,}")
    print(f"Cross rows: {len(cross_df):,}")
    print(f"Cross pairs ({len(cross_pairs)}): {cross_pairs}")

    # Main summaries
    summary_70 = build_pair_persistence_summary(
        df=cross_df,
        streak_col="extreme_streak_70",
        threshold=70,
        pair_list=cross_pairs,
        horizons=HORIZONS,
    )

    summary_80 = build_pair_persistence_summary(
        df=cross_df,
        streak_col="extreme_streak_80",
        threshold=80,
        pair_list=cross_pairs,
        horizons=HORIZONS,
    )

    summary_70_out = OUTPUT_DIR / "pair_persistence_cross_thr70.csv"
    summary_80_out = OUTPUT_DIR / "pair_persistence_cross_thr80.csv"

    summary_70.to_csv(summary_70_out, index=False)
    summary_80.to_csv(summary_80_out, index=False)

    # Compact inspection tables for selected horizons
    focus_horizons = [4, 12, 48]

    focus_70 = build_focus_table(summary_70, focus_horizons=focus_horizons)
    focus_80 = build_focus_table(summary_80, focus_horizons=focus_horizons)

    focus_70_out = OUTPUT_DIR / "pair_persistence_cross_thr70_focus.csv"
    focus_80_out = OUTPUT_DIR / "pair_persistence_cross_thr80_focus.csv"

    focus_70.to_csv(focus_70_out, index=False)
    focus_80.to_csv(focus_80_out, index=False)

    print("\nSaved:")
    print(f"  {summary_70_out}")
    print(f"  {summary_80_out}")
    print(f"  {focus_70_out}")
    print(f"  {focus_80_out}")

    # Print top pairs for the most interesting subsets
    if "mean_12b" in focus_70.columns:
        print_top_pairs(focus_70, "thr70_streak1", "mean_12b")
        print_top_pairs(focus_70, "thr70_streak3plus", "mean_12b")

    if "mean_48b" in focus_70.columns:
        print_top_pairs(focus_70, "thr70_streak3plus", "mean_48b")

    if "mean_12b" in focus_80.columns:
        print_top_pairs(focus_80, "thr80_streak2", "mean_12b")
        print_top_pairs(focus_80, "thr80_streak3plus", "mean_12b")


if __name__ == "__main__":
    main()
