# Legacy experiment — not part of current validated approach\nfrom __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path("data/output/master_research_dataset_core.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 6, 12, 24, 48]
DATE_COLUMNS = ["snapshot_time", "entry_time"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=DATE_COLUMNS)
    return df


def summarize_return_quantiles(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    quantiles = [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]

    for col in cols:
        s = df[col].dropna()
        q = s.quantile(quantiles)
        row = {"column": col, "n": len(s)}
        for k, v in q.items():
            row[f"q_{k:.3f}"] = v
        row["mean"] = s.mean()
        row["median"] = s.median()
        row["std"] = s.std()
        rows.append(row)

    return pd.DataFrame(rows)


def extract_extreme_rows(
    df: pd.DataFrame,
    col: str,
    top_n: int = 30,
) -> pd.DataFrame:
    cols = [
        "pair",
        "snapshot_time",
        "entry_time",
        "net_sentiment",
        "abs_sentiment",
        "entry_close",
        col,
    ]

    future_close_col = col.replace("ret_", "future_close_").replace("b", "b")
    if future_close_col in df.columns:
        cols.append(future_close_col)

    existing = [c for c in cols if c in df.columns]

    extreme = (
        df.loc[df[col].notna(), existing]
        .assign(abs_value=df[col].abs())
        .sort_values("abs_value", ascending=False)
        .head(top_n)
        .drop(columns=["abs_value"])
    )

    return extreme


def pair_level_extremes(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    rows = []
    for pair, grp in df.groupby("pair"):
        s = grp[ret_col].dropna()
        if len(s) == 0:
            continue

        rows.append({
            "pair": pair,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "max_abs": s.abs().max(),
            "q_99": s.quantile(0.99),
            "q_01": s.quantile(0.01),
        })

    return pd.DataFrame(rows).sort_values("max_abs", ascending=False)


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    s = s.copy()
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def winsorized_threshold_summary(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    rows = []

    for threshold in [None] + thresholds:
        if threshold is None:
            sub = df.copy()
            subset_name = "all"
        else:
            sub = df[df["abs_sentiment"] >= threshold].copy()
            subset_name = f"abs_sentiment>={threshold}"

        for h in HORIZONS:
            col = f"contrarian_ret_{h}b"
            s = sub[col].dropna()
            sw = winsorize_series(s, 0.01, 0.99)

            rows.append({
                "subset": subset_name,
                "horizon_bars": h,
                "n": len(s),
                "raw_mean": s.mean(),
                "raw_median": s.median(),
                "raw_hit_rate": (s > 0).mean(),
                "winsor_mean_1_99": sw.mean(),
                "winsor_median_1_99": sw.median(),
                "winsor_hit_rate_1_99": (sw > 0).mean(),
            })

    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset(DATASET_PATH)

    ret_cols = [f"ret_{h}b" for h in HORIZONS]
    contrarian_cols = [f"contrarian_ret_{h}b" for h in HORIZONS]

    # 1. Quantiles
    ret_quantiles = summarize_return_quantiles(df, ret_cols)
    contrarian_quantiles = summarize_return_quantiles(df, contrarian_cols)

    ret_quantiles.to_csv(OUTPUT_DIR / "return_quantiles_core.csv", index=False)
    contrarian_quantiles.to_csv(OUTPUT_DIR / "contrarian_return_quantiles_core.csv", index=False)

    print("\n=== Raw return quantiles ===")
    print(ret_quantiles.to_string(index=False))

    print("\n=== Contrarian return quantiles ===")
    print(contrarian_quantiles.to_string(index=False))

    # 2. Extreme rows per horizon
    for h in HORIZONS:
        col = f"ret_{h}b"
        extremes = extract_extreme_rows(df, col, top_n=30)
        extremes.to_csv(OUTPUT_DIR / f"extreme_rows_{col}_core.csv", index=False)

    # 3. Pair-level extreme diagnostics
    for h in HORIZONS:
        col = f"ret_{h}b"
        pair_ext = pair_level_extremes(df, col)
        pair_ext.to_csv(OUTPUT_DIR / f"pair_extremes_{col}_core.csv", index=False)

    # 4. Winsorized threshold summary
    winsor_summary = winsorized_threshold_summary(df, thresholds=[70, 80, 90])
    winsor_summary.to_csv(OUTPUT_DIR / "winsorized_threshold_summary_core.csv", index=False)

    print("\n=== Winsorized threshold summary (1% / 99%) ===")
    print(winsor_summary.to_string(index=False))


if __name__ == "__main__":
    main()
