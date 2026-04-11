'''

This script:

- loads the cleaned core dataset
- collapses to one row per `(pair, entry_time)`
- restricts to cross pairs
- applies the locked rule
- evaluates only the untouched period
- reports:
  - effect sizes
  - bootstrap confidence intervals
  - subwindow stability

'''

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

DATASET_PATH = Path("data/output/analysis/master_research_dataset_core_cleaned.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMNS = ["snapshot_time", "entry_time"]

# Locked subgroup
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

# Locked condition
MIN_ABS_SENTIMENT = 70
MIN_EXTREME_STREAK_70 = 3
HORIZONS = [12, 48]

# Locked untouched period
UNTOUCHED_START = "2026-01-01"
UNTOUCHED_END = None  # use latest available after start

# Bootstrap
N_BOOTSTRAPS = 5000
BOOTSTRAP_RANDOM_SEED = 42

# Stability window
STABILITY_FREQ = "Q"   # "Q" for quarter, "M" for month


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

    df = df[
        (df["abs_sentiment"] >= MIN_ABS_SENTIMENT) &
        (df["extreme_streak_70"] >= MIN_EXTREME_STREAK_70)
    ].copy()

    df["event_date"] = pd.to_datetime(df["snapshot_time"]).dt.floor("D")
    return df


def select_untouched_period(
    df: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    out = df[df["snapshot_time"] >= pd.Timestamp(start)].copy()
    if end is not None:
        out = out[out["snapshot_time"] < pd.Timestamp(end)].copy()
    return out


def compute_group_metrics(df: pd.DataFrame, horizon: int) -> dict:
    col = f"contrarian_ret_{horizon}b"

    jpy = df[df["is_jpy_cross"]][col].dropna()
    non = df[~df["is_jpy_cross"]][col].dropna()

    jpy_hit = (jpy > 0).mean() if len(jpy) else np.nan
    non_hit = (non > 0).mean() if len(non) else np.nan

    return {
        "horizon_bars": horizon,
        "jpy_n": int(len(jpy)),
        "non_jpy_n": int(len(non)),
        "jpy_mean": float(jpy.mean()) if len(jpy) else np.nan,
        "non_jpy_mean": float(non.mean()) if len(non) else np.nan,
        "diff_mean": float(jpy.mean() - non.mean()) if len(jpy) and len(non) else np.nan,
        "jpy_median": float(jpy.median()) if len(jpy) else np.nan,
        "non_jpy_median": float(non.median()) if len(non) else np.nan,
        "jpy_hit_rate": float(jpy_hit) if len(jpy) else np.nan,
        "non_jpy_hit_rate": float(non_hit) if len(non) else np.nan,
        "diff_hit_rate": float(jpy_hit - non_hit) if len(jpy) and len(non) else np.nan,
        "jpy_std": float(jpy.std()) if len(jpy) else np.nan,
        "non_jpy_std": float(non.std()) if len(non) else np.nan,
    }


def bootstrap_by_date_blocks(
    df: pd.DataFrame,
    horizon: int,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    col = f"contrarian_ret_{horizon}b"

    unique_dates = np.array(sorted(df["event_date"].dropna().unique()))
    if len(unique_dates) == 0:
        return pd.DataFrame(columns=["boot_id", "diff_mean", "diff_hit_rate"])

    sampled_rows = []

    date_to_df = {d: df[df["event_date"] == d].copy() for d in unique_dates}

    for b in range(n_bootstraps):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        boot_df = pd.concat([date_to_df[d] for d in sampled_dates], ignore_index=True)

        jpy = boot_df[boot_df["is_jpy_cross"]][col].dropna()
        non = boot_df[~boot_df["is_jpy_cross"]][col].dropna()

        if len(jpy) == 0 or len(non) == 0:
            diff_mean = np.nan
            diff_hit = np.nan
        else:
            diff_mean = float(jpy.mean() - non.mean())
            diff_hit = float((jpy > 0).mean() - (non > 0).mean())

        sampled_rows.append({
            "boot_id": b,
            "diff_mean": diff_mean,
            "diff_hit_rate": diff_hit,
        })

    return pd.DataFrame(sampled_rows)


def ci_from_bootstrap(series: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    s = series.dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    lo = float(s.quantile(alpha / 2))
    hi = float(s.quantile(1 - alpha / 2))
    return lo, hi


def build_bootstrap_summary(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = compute_group_metrics(df, horizon)
    boot = bootstrap_by_date_blocks(
        df=df,
        horizon=horizon,
        n_bootstraps=N_BOOTSTRAPS,
        seed=BOOTSTRAP_RANDOM_SEED + horizon,
    )

    diff_mean_lo, diff_mean_hi = ci_from_bootstrap(boot["diff_mean"])
    diff_hit_lo, diff_hit_hi = ci_from_bootstrap(boot["diff_hit_rate"])

    summary = pd.DataFrame([{
        **point,
        "diff_mean_ci_95_lo": diff_mean_lo,
        "diff_mean_ci_95_hi": diff_mean_hi,
        "diff_hit_rate_ci_95_lo": diff_hit_lo,
        "diff_hit_rate_ci_95_hi": diff_hit_hi,
    }])

    return summary, boot


def build_stability_summary(df: pd.DataFrame, horizons: Iterable[int], freq: str) -> pd.DataFrame:
    working = df.copy()
    working["stability_window"] = working["snapshot_time"].dt.to_period(freq).astype(str)

    rows = []
    for window, grp in sorted(working.groupby("stability_window"), key=lambda x: x[0]):
        for h in horizons:
            metrics = compute_group_metrics(grp, h)
            rows.append({
                "stability_window": window,
                **metrics,
            })

    return pd.DataFrame(rows)


# =========================
# Main
# =========================

def main() -> None:
    df = load_dataset(DATASET_PATH)
    df = collapse_to_entry_bar(df)
    df = prepare_dataset(df)

    untouched = select_untouched_period(df, UNTOUCHED_START, UNTOUCHED_END)

    if untouched.empty:
        raise ValueError("No rows found in untouched evaluation period.")

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows after locked filtering: {len(df):,}")
    print(f"Untouched rows: {len(untouched):,}")
    print(
        "Untouched period: "
        f"{untouched['snapshot_time'].min()} -> {untouched['snapshot_time'].max()}"
    )

    summary_parts = []
    bootstrap_outputs = {}

    for h in HORIZONS:
        summary_h, boot_h = build_bootstrap_summary(untouched, h)
        summary_parts.append(summary_h)
        bootstrap_outputs[h] = boot_h

    summary = pd.concat(summary_parts, ignore_index=True)
    stability = build_stability_summary(untouched, HORIZONS, STABILITY_FREQ)

    summary_out = OUTPUT_DIR / "jpy_effect_preregistered_summary.csv"
    stability_out = OUTPUT_DIR / "jpy_effect_preregistered_stability.csv"

    summary.to_csv(summary_out, index=False)
    stability.to_csv(stability_out, index=False)

    for h, boot_df in bootstrap_outputs.items():
        boot_out = OUTPUT_DIR / f"jpy_effect_preregistered_bootstrap_{h}b.csv"
        boot_df.to_csv(boot_out, index=False)

    print("\n=== Pre-registered summary ===")
    print(summary.to_string(index=False))

    print("\n=== Stability summary ===")
    print(stability.to_string(index=False))

    print("\nSaved:")
    print(f"  {summary_out}")
    print(f"  {stability_out}")
    for h in HORIZONS:
        print(f"  {OUTPUT_DIR / f'jpy_effect_preregistered_bootstrap_{h}b.csv'}")


if __name__ == "__main__":
    main()
