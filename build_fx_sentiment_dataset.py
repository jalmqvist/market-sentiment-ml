from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

SENTIMENT_DIR = Path("data/input/sentiment")
PRICE_DIR = Path("data/input/fx")
OUTPUT_FILE = Path("data/output/master_research_dataset.csv")

HORIZONS = [1, 2, 4, 6, 12, 24, 48]


# =========================
# Helpers
# =========================

FILENAME_TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{4})\.csv$", re.IGNORECASE)


def normalize_sentiment_pair(pair: str) -> str:
    pair = str(pair).strip().lower()
    pair = pair.replace("/", "-").replace("_", "-")
    return pair


def normalize_price_pair_from_filename(path: Path) -> str:
    """
    Extract pair from filename and convert to sentiment-style format.
    Examples:
        EURUSD_H1.csv -> eur-usd
        EURUSD60.csv  -> eur-usd
        eurusd.csv    -> eur-usd
        GBPJPY.csv    -> gbp-jpy
    """
    stem = path.stem.upper()

    # Remove common timeframe suffixes
    stem = re.sub(r'(_?H1|_?60|_?M60)$', '', stem, flags=re.IGNORECASE)

    # Keep letters only
    letters = re.sub(r'[^A-Z]', '', stem)

    if len(letters) < 6:
        raise ValueError(f"Could not infer FX symbol from filename: {path.name}")

    symbol = letters[:6]

    return f"{symbol[:3].lower()}-{symbol[3:6].lower()}"


def parse_snapshot_time_from_filename(path: Path) -> pd.Timestamp:
    m = FILENAME_TS_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse timestamp from filename: {path.name}")

    yyyy, mm, dd, hhmm = m.groups()
    hh = hhmm[:2]
    minute = hhmm[2:]
    return pd.Timestamp(f"{yyyy}-{mm}-{dd} {hh}:{minute}:00")


def compute_streak_from_boolean(series: pd.Series) -> pd.Series:
    out = np.zeros(len(series), dtype=np.int64)
    count = 0
    vals = series.fillna(False).to_numpy()

    for i, v in enumerate(vals):
        if v:
            count += 1
        else:
            count = 0
        out[i] = count

    return pd.Series(out, index=series.index)


def compute_same_value_streak(series: pd.Series) -> pd.Series:
    out = np.ones(len(series), dtype=np.int64)
    vals = series.to_numpy()

    if len(vals) == 0:
        return pd.Series([], dtype="int64", index=series.index)

    count = 1
    for i in range(1, len(vals)):
        if pd.isna(vals[i]) or pd.isna(vals[i - 1]):
            count = 1
        elif vals[i] == vals[i - 1]:
            count += 1
        else:
            count = 1
        out[i] = count

    return pd.Series(out, index=series.index)


# =========================
# Sentiment loading
# =========================

def load_one_sentiment_file(path: Path) -> pd.DataFrame:
    # Sentiment snapshots are scraped in UTC+2, while the MT4 hourly price data is in UTC+1.
    # Align sentiment to price time by subtracting 1 hour.
    snapshot_time = parse_snapshot_time_from_filename(path)
    snapshot_time = snapshot_time - pd.Timedelta(hours=1)

    df = pd.read_csv(path)

    # Remove junk index column if present
    df = df.drop(columns=["Unnamed: 0", ""], errors="ignore")

    expected = {"pair", "perc", "direction", "time"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[["pair", "perc", "direction", "time"]].copy()
    df["pair"] = df["pair"].map(normalize_sentiment_pair)
    df["perc"] = pd.to_numeric(df["perc"], errors="coerce")
    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    df["raw_time"] = pd.to_datetime(df["time"], errors="coerce")

    df["snapshot_time"] = pd.to_datetime(snapshot_time, errors="coerce")
    df["source_file"] = path.name

    df = df[df["direction"].isin(["long", "short"])].copy()
    df = df.dropna(subset=["pair", "perc", "snapshot_time"])

    return df


def load_all_sentiment_files(sentiment_dir: Path) -> pd.DataFrame:
    files = sorted(sentiment_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No sentiment CSV files found in {sentiment_dir.resolve()}")

    frames = [load_one_sentiment_file(f) for f in files]
    sentiment = pd.concat(frames, ignore_index=True)

    sentiment["net_sentiment"] = np.where(
        sentiment["direction"].eq("long"),
        sentiment["perc"],
        -sentiment["perc"]
    ).astype(float)

    sentiment["abs_sentiment"] = sentiment["net_sentiment"].abs()
    sentiment["crowd_side"] = np.where(sentiment["net_sentiment"] >= 0, 1, -1)

    sentiment = sentiment.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)

    sentiment["prev_net_sentiment"] = sentiment.groupby("pair")["net_sentiment"].shift(1)
    sentiment["sentiment_change"] = sentiment["net_sentiment"] - sentiment["prev_net_sentiment"]

    sentiment["side_streak"] = (
        sentiment.groupby("pair")["crowd_side"]
        .transform(compute_same_value_streak)
    )

    sentiment["extreme_70"] = sentiment["abs_sentiment"] >= 70
    sentiment["extreme_80"] = sentiment["abs_sentiment"] >= 80

    sentiment["extreme_streak_70"] = (
        sentiment.groupby("pair")["extreme_70"]
        .transform(compute_streak_from_boolean)
    )
    sentiment["extreme_streak_80"] = (
        sentiment.groupby("pair")["extreme_80"]
        .transform(compute_streak_from_boolean)
    )

    return sentiment


# =========================
# MT4 hourly price loading
# =========================

def load_one_mt4_price_file(path: Path) -> pd.DataFrame:
    pair = normalize_price_pair_from_filename(path)

    df = pd.read_csv(path)

    required = {"time_utc", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[["time_utc", "open", "high", "low", "close", "tick_volume"]].copy()
    df = df.rename(columns={
        "time_utc": "timestamp",
        "tick_volume": "volume"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pair"] = pair
    df["source_price_file"] = path.name

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_all_mt4_prices(price_dir: Path) -> pd.DataFrame:
    files = sorted(price_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hourly price CSV files found in {price_dir.resolve()}")

    frames = [load_one_mt4_price_file(f) for f in files]
    prices = pd.concat(frames, ignore_index=True)

    prices = prices.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    prices = prices.drop_duplicates(subset=["pair", "timestamp"], keep="last")

    return prices


# =========================
# Merge logic
# =========================

def attach_entry_bar(sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each sentiment snapshot, attach the first hourly bar at or after snapshot_time.
    Done pair-by-pair to avoid merge_asof global sorting issues.
    Only match if the price bar is close enough in time.
    """
    out_frames = []

    sentiment = sentiment.copy()
    prices = prices.copy()

    sentiment = sentiment.dropna(subset=["pair", "snapshot_time"])
    prices = prices.dropna(subset=["pair", "timestamp"])

    common_pairs = sorted(set(sentiment["pair"]).intersection(set(prices["pair"])))

    for pair in common_pairs:
        left = (
            sentiment.loc[sentiment["pair"] == pair]
            .sort_values("snapshot_time")
            .copy()
        )

        right = (
            prices.loc[prices["pair"] == pair]
            .sort_values("timestamp")
            .copy()
        )

        if left.empty or right.empty:
            continue

        right = right.drop(columns=["pair"])

        merged_pair = pd.merge_asof(
            left,
            right,
            left_on="snapshot_time",
            right_on="timestamp",
            direction="forward",
            allow_exact_matches=True,
            tolerance=pd.Timedelta("90min")
        )

        merged_pair["pair"] = pair
        out_frames.append(merged_pair)

    if not out_frames:
        raise ValueError("No overlapping pairs could be merged between sentiment and price data.")

    merged = pd.concat(out_frames, ignore_index=True)

    merged = merged.rename(columns={
        "timestamp": "entry_time",
        "open": "entry_open",
        "high": "entry_high",
        "low": "entry_low",
        "close": "entry_close",
        "volume": "entry_tick_volume",
    })

    return merged


def add_forward_returns(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: Iterable[int]
) -> pd.DataFrame:
    """
    Compute forward returns using trading bars ahead, not wall-clock time.

    For each entry_time, find the close price h bars later within the same pair:
        ret_{h}b = future_close / entry_close - 1

    Also compute contrarian returns:
        contrarian_ret_{h}b = -sign(net_sentiment) * ret_{h}b

    Horizons are interpreted as number of hourly trading bars ahead.
    """
    out = df.copy()

    # Build per-pair lookup of future closes by bar index
    px = prices[["pair", "timestamp", "close"]].copy()
    px = px.sort_values(["pair", "timestamp"]).reset_index(drop=True)

    # For each pair, precompute shifted future closes
    future_tables = []
    for pair, grp in px.groupby("pair", sort=False):
        grp = grp.sort_values("timestamp").copy()

        for h in horizons:
            grp[f"future_close_{h}b"] = grp["close"].shift(-h)

        future_tables.append(grp)

    px_future = pd.concat(future_tables, ignore_index=True)

    # Merge future-close table onto entry bars
    merge_cols = ["pair", "timestamp"] + [f"future_close_{h}b" for h in horizons]

    out = out.merge(
        px_future[merge_cols],
        left_on=["pair", "entry_time"],
        right_on=["pair", "timestamp"],
        how="left"
    )

    # Clean up duplicate merge timestamp column
    out = out.drop(columns=["timestamp"], errors="ignore")

    # Compute returns
    for h in horizons:
        future_close_col = f"future_close_{h}b"
        ret_col = f"ret_{h}b"
        contrarian_col = f"contrarian_ret_{h}b"

        out[ret_col] = out[future_close_col] / out["entry_close"] - 1.0
        out[contrarian_col] = -np.sign(out["net_sentiment"]) * out[ret_col]

    return out


def build_master_dataset(
    sentiment_dir: Path,
    price_dir: Path,
    output_file: Optional[Path] = None,
    horizons: Iterable[int] = HORIZONS
) -> pd.DataFrame:
    sentiment = load_all_sentiment_files(sentiment_dir)
    print(f"Loaded sentiment rows: {len(sentiment):,}")
    print(f"Sentiment pairs: {sentiment['pair'].nunique():,}")

    prices = load_all_mt4_prices(price_dir)
    print(f"Loaded hourly price rows: {len(prices):,}")
    print(f"Price pairs: {prices['pair'].nunique():,}")

    price_coverage = (
        prices.groupby("pair")["timestamp"]
        .agg(["min", "max", "count"])
        .sort_values("count")
    )
    print("\nPrice coverage by pair:")
    print(price_coverage.head(15).to_string())

    pair_overlap = sorted(set(sentiment["pair"]).intersection(set(prices["pair"])))
    print(f"\nOverlapping pairs: {len(pair_overlap):,}")

    # Attach first valid hourly bar at or after each sentiment snapshot
    master = attach_entry_bar(sentiment, prices)

    # Add price window info for coverage diagnostics
    price_window = (
        prices.groupby("pair")["timestamp"]
        .agg(price_start="min", price_end="max", price_bars="count")
        .reset_index()
    )

    master = master.merge(price_window, on="pair", how="left")

    # Eligibility: weekday sentiment row that falls inside the available price-history window
    master["is_weekday"] = master["snapshot_time"].dt.dayofweek < 5
    master["within_price_window"] = (
        (master["snapshot_time"] >= master["price_start"]) &
        (master["snapshot_time"] <= master["price_end"])
    )
    master["eligible"] = master["is_weekday"] & master["within_price_window"]

    print(f"\nRows after entry alignment: {len(master):,}")
    print(f"Rows without valid entry bar: {master['entry_time'].isna().sum():,}")

    # Weekday/weekend diagnostic
    master["dayofweek"] = master["snapshot_time"].dt.dayofweek
    master["weekday"] = master["snapshot_time"].dt.day_name()

    week_summary = (
        master.assign(has_entry_bar=master["entry_time"].notna())
        .groupby(["dayofweek", "weekday"])["has_entry_bar"]
        .agg(total="count", matched="sum")
        .reset_index()
        .sort_values("dayofweek")
    )

    week_summary["unmatched"] = week_summary["total"] - week_summary["matched"]
    week_summary["match_ratio"] = week_summary["matched"] / week_summary["total"]

    print("\nWeekday/weekend match summary:")
    print(
        week_summary[["weekday", "total", "matched", "unmatched", "match_ratio"]]
        .to_string(index=False)
    )

    # Pair-level coverage summary
    coverage = (
        master.assign(has_entry_bar=master["entry_time"].notna())
        .groupby("pair")
        .agg(
            sentiment_rows=("pair", "size"),
            price_start=("price_start", "first"),
            price_end=("price_end", "first"),
            price_bars=("price_bars", "first"),
            eligible_rows=("eligible", "sum"),
            matched_rows=("has_entry_bar", "sum"),
        )
        .reset_index()
    )

    coverage["unmatched_rows"] = coverage["sentiment_rows"] - coverage["matched_rows"]
    coverage["raw_match_ratio"] = coverage["matched_rows"] / coverage["sentiment_rows"]
    coverage["eligible_match_ratio"] = coverage["matched_rows"] / coverage["eligible_rows"]

    coverage = coverage.sort_values(
        ["eligible_match_ratio", "sentiment_rows"],
        ascending=[True, False]
    )

    print("\nEligible coverage by pair:")
    print(coverage.to_string(index=False))

    coverage.to_csv("data/output/pair_coverage_summary.csv", index=False)
    print("\nSaved pair coverage summary to pair_coverage_summary.csv")

    # Coverage-based universes
    core_pairs = coverage.loc[coverage["eligible_match_ratio"] >= 0.95, "pair"]
    extended_pairs = coverage.loc[coverage["eligible_match_ratio"] >= 0.90, "pair"]

    print(f"\nCore pairs (eligible_match_ratio >= 0.95): {len(core_pairs):,}")
    print(sorted(core_pairs.tolist()))

    print(f"\nExtended pairs (eligible_match_ratio >= 0.90): {len(extended_pairs):,}")
    print(sorted(extended_pairs.tolist()))

    # Keep only rows with a valid entry bar before forward-return calculation
    master_valid = master.dropna(subset=["entry_time", "entry_close"]).copy()
    print(f"\nRows with valid entry bar: {len(master_valid):,}")

    # Build filtered datasets
    master_core = master_valid[master_valid["pair"].isin(core_pairs)].copy()
    master_extended = master_valid[master_valid["pair"].isin(extended_pairs)].copy()

    print(f"Core-universe rows: {len(master_core):,}")
    print(f"Core-universe pairs: {master_core['pair'].nunique():,}")

    print(f"Extended-universe rows: {len(master_extended):,}")
    print(f"Extended-universe pairs: {master_extended['pair'].nunique():,}")

    # Add forward returns
    master_valid = add_forward_returns(master_valid, prices, horizons=horizons)
    master_core = add_forward_returns(master_core, prices, horizons=horizons)
    master_extended = add_forward_returns(master_extended, prices, horizons=horizons)

    # Add convenience flags
    for df in (master_valid, master_core, master_extended):
        df["is_long_crowd"] = df["net_sentiment"] > 0
        df["is_short_crowd"] = df["net_sentiment"] < 0

    # Final sorting
    master_valid = master_valid.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)
    master_core = master_core.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)
    master_extended = master_extended.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)

    # Save default outputs
    master_valid.to_csv("data/output/master_research_dataset.csv", index=False)
    master_core.to_csv("data/output/master_research_dataset_core.csv", index=False)
    master_extended.to_csv("data/output/master_research_dataset_extended.csv", index=False)

    print("\nSaved:")
    print("  master_research_dataset.csv")
    print("  master_research_dataset_core.csv")
    print("  master_research_dataset_extended.csv")

    # Optional custom output_file writes the full valid dataset
    if output_file is not None:
        master_valid.to_csv(output_file, index=False)
        print(f"\nSaved master dataset to: {output_file.resolve()}")

    return master_valid


def quick_summary(master: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> None:
    print("\n=== Master dataset summary ===")
    print(f"Rows: {len(master):,}")
    print(f"Pairs: {master['pair'].nunique():,}")
    print(f"Date range: {master['snapshot_time'].min()} -> {master['snapshot_time'].max()}")

    print("\n=== Missing forward returns ===")
    for h in horizons:
        col = f"ret_{h}b"
        print(f"{col}: {master[col].isna().sum():,}")

    print("\n=== Mean contrarian return by horizon ===")
    for h in horizons:
        col = f"contrarian_ret_{h}b"
        print(f"{col}: {master[col].mean(skipna=True):.6f}")

    print("\n=== Mean contrarian return for abs_sentiment >= 70 ===")
    subset = master[master["abs_sentiment"] >= 70]
    for h in horizons:
        col = f"contrarian_ret_{h}b"
        print(f"{col}: {subset[col].mean(skipna=True):.6f}")


if __name__ == "__main__":
    master_df = build_master_dataset(
        sentiment_dir=SENTIMENT_DIR,
        price_dir=PRICE_DIR,
        output_file=OUTPUT_FILE,
        horizons=HORIZONS
    )

    quick_summary(master_df, horizons=HORIZONS)
