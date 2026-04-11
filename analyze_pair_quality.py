from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# Configuration
# =========================

DATASET_PATH = Path("data/output/master_research_dataset_core.csv")
OUTPUT_DIR = Path("data/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 6, 12, 24, 48]
DATE_COLUMNS = ["snapshot_time", "entry_time"]

# Pair flagging thresholds
# Start conservative: only remove clearly broken pairs
MAX_ABS_RETURN_THRESHOLD = {
    1: 0.10,
    2: 0.12,
    4: 0.15,
    6: 0.18,
    12: 0.25,
    24: 0.35,
    48: 0.50,
}

Q999_ABS_RETURN_THRESHOLD = {
    1: 0.03,
    2: 0.04,
    4: 0.05,
    6: 0.07,
    12: 0.10,
    24: 0.15,
    48: 0.20,
}


# =========================
# Helpers
# =========================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def summarize_pair_quality(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    rows = []

    for pair, grp in df.groupby("pair"):
        row = {
            "pair": pair,
            "rows": len(grp),
        }

        for h in horizons:
            col = f"ret_{h}b"
            s = grp[col].dropna()

            if len(s) == 0:
                row[f"n_{h}b"] = 0
                row[f"mean_{h}b"] = pd.NA
                row[f"median_{h}b"] = pd.NA
                row[f"std_{h}b"] = pd.NA
                row[f"max_abs_{h}b"] = pd.NA
                row[f"q001_{h}b"] = pd.NA
                row[f"q999_{h}b"] = pd.NA
                continue

            row[f"n_{h}b"] = int(len(s))
            row[f"mean_{h}b"] = float(s.mean())
            row[f"median_{h}b"] = float(s.median())
            row[f"std_{h}b"] = float(s.std())
            row[f"max_abs_{h}b"] = float(s.abs().max())
            row[f"q001_{h}b"] = float(s.quantile(0.001))
            row[f"q999_{h}b"] = float(s.quantile(0.999))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)


def flag_bad_pairs(quality: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    flagged = quality.copy()

    flag_cols = []

    for h in horizons:
        max_abs_col = f"flag_max_abs_{h}b"
        q999_col = f"flag_q999_{h}b"

        flagged[max_abs_col] = flagged[f"max_abs_{h}b"] > MAX_ABS_RETURN_THRESHOLD[h]
        flagged[q999_col] = flagged[f"q999_{h}b"].abs() > Q999_ABS_RETURN_THRESHOLD[h]

        flag_cols.extend([max_abs_col, q999_col])

    flagged["n_flags"] = flagged[flag_cols].sum(axis=1)
    flagged["is_flagged"] = flagged["n_flags"] > 0

    return flagged.sort_values(["is_flagged", "n_flags", "pair"], ascending=[False, False, True])


def explain_flags(flagged_row: pd.Series, horizons: Iterable[int]) -> str:
    reasons = []

    for h in horizons:
        if bool(flagged_row.get(f"flag_max_abs_{h}b", False)):
            reasons.append(
                f"max_abs_{h}b={flagged_row[f'max_abs_{h}b']:.4f}>{MAX_ABS_RETURN_THRESHOLD[h]:.4f}"
            )
        if bool(flagged_row.get(f"flag_q999_{h}b", False)):
            reasons.append(
                f"|q999_{h}b|={abs(flagged_row[f'q999_{h}b']):.4f}>{Q999_ABS_RETURN_THRESHOLD[h]:.4f}"
            )

    return "; ".join(reasons)


def build_bad_pair_report(flagged: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    bad = flagged[flagged["is_flagged"]].copy()
    bad["flag_reasons"] = bad.apply(lambda row: explain_flags(row, horizons), axis=1)

    cols = ["pair", "rows", "n_flags", "flag_reasons"]
    for h in horizons:
        cols.extend([
            f"max_abs_{h}b",
            f"q999_{h}b",
        ])

    return bad[cols].reset_index(drop=True)


def save_cleaned_dataset(df: pd.DataFrame, bad_pairs: list[str], output_path: Path) -> pd.DataFrame:
    cleaned = df[~df["pair"].isin(bad_pairs)].copy()
    cleaned.to_csv(output_path, index=False)
    return cleaned


# =========================
# Main
# =========================

def main() -> None:
    df = load_dataset(DATASET_PATH)

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique():,}")

    quality = summarize_pair_quality(df, HORIZONS)
    flagged = flag_bad_pairs(quality, HORIZONS)
    bad_report = build_bad_pair_report(flagged, HORIZONS)

    quality_out = OUTPUT_DIR / "pair_quality_summary.csv"
    flagged_out = OUTPUT_DIR / "pair_quality_flagged.csv"
    bad_pairs_out = OUTPUT_DIR / "bad_pairs.csv"
    cleaned_out = OUTPUT_DIR / "master_research_dataset_core_cleaned.csv"

    quality.to_csv(quality_out, index=False)
    flagged.to_csv(flagged_out, index=False)
    bad_report.to_csv(bad_pairs_out, index=False)

    bad_pairs = bad_report["pair"].tolist()
    cleaned = save_cleaned_dataset(df, bad_pairs, cleaned_out)

    print("\n=== Flagged pairs ===")
    if len(bad_report) == 0:
        print("No pairs flagged.")
    else:
        print(bad_report[["pair", "rows", "n_flags", "flag_reasons"]].to_string(index=False))

    print(f"\nBad pairs flagged: {len(bad_pairs)}")
    print(bad_pairs)

    print(f"\nCleaned dataset rows: {len(cleaned):,}")
    print(f"Cleaned dataset pairs: {cleaned['pair'].nunique():,}")

    print("\nSaved:")
    print(f"  {quality_out}")
    print(f"  {flagged_out}")
    print(f"  {bad_pairs_out}")
    print(f"  {cleaned_out}")


if __name__ == "__main__":
    main()
