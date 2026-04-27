# Legacy experiment — not part of current validated approach\nfrom __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

DATASET_PATH = Path("data/output/analysis/master_research_dataset_core_cleaned.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMNS = ["snapshot_time", "entry_time"]
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

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

# Focused tests
TESTS = [
    {"subset_name": "thr70_streak3plus", "horizon_bars": 12},
    {"subset_name": "thr70_streak3plus", "horizon_bars": 48},
    {"subset_name": "thr70_streak1", "horizon_bars": 12},
]


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


def prepare_cross_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["pair"].isin(CROSS_PAIRS)].copy()

    # Build persistence subset labels matching your earlier analysis
    df["subset_name"] = pd.NA

    mask_70 = df["abs_sentiment"] >= 70
    df.loc[mask_70 & (df["extreme_streak_70"] == 1), "subset_name"] = "thr70_streak1"
    df.loc[mask_70 & (df["extreme_streak_70"] == 2), "subset_name"] = "thr70_streak2"
    df.loc[mask_70 & (df["extreme_streak_70"] >= 3), "subset_name"] = "thr70_streak3plus"

    return df


def pair_level_metric_table(df: pd.DataFrame, subset_name: str, horizon_bars: int) -> pd.DataFrame:
    col = f"contrarian_ret_{horizon_bars}b"

    sub = df[df["subset_name"] == subset_name].copy()
    sub = sub[sub[col].notna()].copy()

    out = (
        sub.groupby("pair")[col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={
            "count": "n",
            "mean": "mean_contrarian_ret",
            "median": "median_contrarian_ret",
            "std": "std_contrarian_ret",
        })
    )

    out["hit_rate"] = (
        sub.groupby("pair")[col]
        .apply(lambda s: (s > 0).mean())
        .values
    )

    out["is_jpy_cross"] = out["pair"].isin(JPY_CROSSES)
    return out


def observed_group_difference(pair_metrics: pd.DataFrame) -> dict:
    jpy = pair_metrics[pair_metrics["is_jpy_cross"]]
    non_jpy = pair_metrics[~pair_metrics["is_jpy_cross"]]

    result = {
        "jpy_pair_count": len(jpy),
        "non_jpy_pair_count": len(non_jpy),
        "jpy_mean_of_pair_means": jpy["mean_contrarian_ret"].mean(),
        "non_jpy_mean_of_pair_means": non_jpy["mean_contrarian_ret"].mean(),
        "difference_mean_of_pair_means": jpy["mean_contrarian_ret"].mean() - non_jpy["mean_contrarian_ret"].mean(),
        "jpy_mean_hit_rate": jpy["hit_rate"].mean(),
        "non_jpy_mean_hit_rate": non_jpy["hit_rate"].mean(),
        "difference_mean_hit_rate": jpy["hit_rate"].mean() - non_jpy["hit_rate"].mean(),
    }
    return result


def permutation_test(pair_metrics: pd.DataFrame, n_permutations: int, rng: random.Random) -> pd.DataFrame:
    pairs = pair_metrics["pair"].tolist()
    n_jpy = pair_metrics["is_jpy_cross"].sum()

    observed = observed_group_difference(pair_metrics)
    obs_diff_mean = observed["difference_mean_of_pair_means"]
    obs_diff_hit = observed["difference_mean_hit_rate"]

    rows = []
    for i in range(n_permutations):
        pseudo_jpy = set(rng.sample(pairs, k=n_jpy))

        temp = pair_metrics.copy()
        temp["pseudo_jpy"] = temp["pair"].isin(pseudo_jpy)

        a = temp[temp["pseudo_jpy"]]
        b = temp[~temp["pseudo_jpy"]]

        diff_mean = a["mean_contrarian_ret"].mean() - b["mean_contrarian_ret"].mean()
        diff_hit = a["hit_rate"].mean() - b["hit_rate"].mean()

        rows.append({
            "perm_id": i,
            "diff_mean_of_pair_means": diff_mean,
            "diff_mean_hit_rate": diff_hit,
            "exceeds_observed_mean": diff_mean >= obs_diff_mean,
            "exceeds_observed_hit": diff_hit >= obs_diff_hit,
        })

    return pd.DataFrame(rows)


def summarize_permutation(observed: dict, perm_df: pd.DataFrame, subset_name: str, horizon_bars: int) -> pd.DataFrame:
    p_value_mean = perm_df["exceeds_observed_mean"].mean()
    p_value_hit = perm_df["exceeds_observed_hit"].mean()

    return pd.DataFrame([{
        "subset_name": subset_name,
        "horizon_bars": horizon_bars,
        "jpy_pair_count": observed["jpy_pair_count"],
        "non_jpy_pair_count": observed["non_jpy_pair_count"],
        "observed_diff_mean_of_pair_means": observed["difference_mean_of_pair_means"],
        "observed_diff_mean_hit_rate": observed["difference_mean_hit_rate"],
        "permutation_p_value_mean": p_value_mean,
        "permutation_p_value_hit": p_value_hit,
    }])


# =========================
# Main
# =========================

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    df = load_dataset(DATASET_PATH)
    df = collapse_to_entry_bar(df)
    df = prepare_cross_dataset(df)

    all_results = []

    for test in TESTS:
        subset_name = test["subset_name"]
        horizon_bars = test["horizon_bars"]

        pair_metrics = pair_level_metric_table(df, subset_name=subset_name, horizon_bars=horizon_bars)

        # Skip if too few pairs available
        if pair_metrics.empty:
            print(f"Skipping empty test: {subset_name}, {horizon_bars}b")
            continue

        observed = observed_group_difference(pair_metrics)
        perm_df = permutation_test(pair_metrics, n_permutations=N_PERMUTATIONS, rng=rng)
        summary = summarize_permutation(observed, perm_df, subset_name, horizon_bars)

        pair_metrics_out = OUTPUT_DIR / f"jpy_cluster_pair_metrics_{subset_name}_{horizon_bars}b.csv"
        perm_out = OUTPUT_DIR / f"jpy_cluster_permutations_{subset_name}_{horizon_bars}b.csv"
        summary_out = OUTPUT_DIR / f"jpy_cluster_summary_{subset_name}_{horizon_bars}b.csv"

        pair_metrics.to_csv(pair_metrics_out, index=False)
        perm_df.to_csv(perm_out, index=False)
        summary.to_csv(summary_out, index=False)

        all_results.append(summary)

        print(f"\n=== Test: {subset_name} / {horizon_bars}b ===")
        print(summary.to_string(index=False))
        print(f"Saved: {pair_metrics_out}")
        print(f"Saved: {perm_out}")
        print(f"Saved: {summary_out}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_out = OUTPUT_DIR / "jpy_cluster_permutation_summary_all.csv"
        combined.to_csv(combined_out, index=False)

        print("\n=== Combined permutation summary ===")
        print(combined.to_string(index=False))
        print(f"Saved: {combined_out}")


if __name__ == "__main__":
    main()
