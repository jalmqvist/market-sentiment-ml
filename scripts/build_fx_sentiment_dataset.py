from __future__ import annotations

import hashlib
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

SENTIMENT_DIR = Path("data/input/sentiment")
PRICE_DIR = Path("data/input/fx")
DEFAULT_VERSION = "v1"

HORIZONS = [1, 2, 4, 6, 12, 24, 48]

SCHEMA_VERSION = "1.0"
CORE_MIN_ELIGIBLE_MATCH_RATIO = 0.95
EXTENDED_MIN_ELIGIBLE_MATCH_RATIO = 0.90
MERGE_TOLERANCE = "90min"
SENTIMENT_ASSUMED_UTC_OFFSET = "+02:00"
PRICE_ASSUMED_UTC_OFFSET = "+01:00"
SNAPSHOT_SHIFT = "-1h"
# Regime threshold (used in DL + ABM alignment)
TREND_THRESHOLD = 1.0
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

def compute_crowd_side(net_sentiment: pd.Series) -> pd.Series:
    """
    Convert signed net sentiment into crowd-side labels.

    Convention:
    - +1 => crowd net long
    - -1 => crowd net short
    -  0 => exactly neutral / zero sentiment
    """
    return np.select(
        [
            net_sentiment > 0,
            net_sentiment < 0,
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype(int)

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
    sentiment["crowd_side"] = compute_crowd_side(sentiment["net_sentiment"])

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

    # After computing returns: safety net
    for h in horizons:
        ret_col = f"ret_{h}b"
        contrarian_col = f"contrarian_ret_{h}b"

        mask = out[ret_col].abs() > 0.2
        out.loc[mask, [ret_col, contrarian_col]] = np.nan
    return out

# =========================
# Trend features
# =========================

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal (backward-looking) trend features.

    trend_12b / trend_48b: past returns
    trend_dir: sign of past return
    trend_alignment: crowd_side * trend_dir
    trend_strength: absolute return
    """
    out = df.copy()

    out = out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    for h in [12, 48]:
        # past return
        out[f"trend_{h}b"] = (
            out.groupby("pair")["entry_close"]
            .pct_change(h)
        )

        # direction
        out[f"trend_dir_{h}b"] = np.sign(out[f"trend_{h}b"])
        out.loc[out[f"trend_dir_{h}b"] == 0, f"trend_dir_{h}b"] = np.nan

        # alignment
        out[f"trend_alignment_{h}b"] = (
            out["crowd_side"] * out[f"trend_dir_{h}b"]
        )

        # strength
        out[f"trend_strength_{h}b"] = out[f"trend_{h}b"].abs()

    return out

### Manifest helpers

def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def add_crowd_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crowd_side"] = compute_crowd_side(out["net_sentiment"])
    return out


def align_dataset_columns(
    full_df: pd.DataFrame,
    core_df: pd.DataFrame,
    extended_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Guarantee identical column sets and column order across dataset variants.
    Columns missing from a variant are added as NA.
    """
    all_cols = list(full_df.columns)

    for extra_col in core_df.columns:
        if extra_col not in all_cols:
            all_cols.append(extra_col)

    for extra_col in extended_df.columns:
        if extra_col not in all_cols:
            all_cols.append(extra_col)

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in all_cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[all_cols]

    return _align(full_df), _align(core_df), _align(extended_df)


def compute_csv_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset_manifest(
    output_dir: Path,
    full_df: pd.DataFrame,
    core_df: pd.DataFrame,
    extended_df: pd.DataFrame,
    version: str,
    tag: Optional[str] = None,
    git_commit: str | None = None,
) -> None:
    versioned_prefix = f"data/output/{version}"
    canonical_dataset = f"{versioned_prefix}/master_research_dataset_core.csv"

    full_path = output_dir / "master_research_dataset.csv"
    dataset_hash = compute_csv_sha256(full_path) if full_path.exists() else None

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": version,
        "dataset_tag": tag,
        "dataset_hash": dataset_hash,
        "canonical_dataset": canonical_dataset,
        "dataset_variants": {
            "full": f"{versioned_prefix}/master_research_dataset.csv",
            "core": f"{versioned_prefix}/master_research_dataset_core.csv",
            "extended": f"{versioned_prefix}/master_research_dataset_extended.csv",
            "coverage_summary": f"{versioned_prefix}/pair_coverage_summary.csv",
        },
        "pair_normalization": "lowercase 3-3 with '-' separator",
        "return_definition": "trading bars ahead within pair series",
        "horizons_bars": list(HORIZONS),
        "merge": {
            "direction": "forward",
            "tolerance": MERGE_TOLERANCE,
            "allow_exact_matches": True,
        },
        "timezone_alignment": {
            "sentiment_assumed_utc_offset": SENTIMENT_ASSUMED_UTC_OFFSET,
            "price_assumed_utc_offset": PRICE_ASSUMED_UTC_OFFSET,
            "snapshot_shift": SNAPSHOT_SHIFT,
        },
        "universe_filters": {
            "core_min_eligible_match_ratio": CORE_MIN_ELIGIBLE_MATCH_RATIO,
            "extended_min_eligible_match_ratio": EXTENDED_MIN_ELIGIBLE_MATCH_RATIO,
        },
        "build": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "row_counts": {
                "full": int(len(full_df)),
                "core": int(len(core_df)),
                "extended": int(len(extended_df)),
            },
            "pairs": {
                "full": int(full_df["pair"].nunique()) if "pair" in full_df.columns else 0,
                "core": int(core_df["pair"].nunique()) if "pair" in core_df.columns else 0,
                "extended": int(extended_df["pair"].nunique()) if "pair" in extended_df.columns else 0,
            },
        },
    }

    manifest_path = output_dir / "DATASET_MANIFEST.json"
    ensure_output_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved dataset manifest to: {manifest_path.resolve()}")


def get_git_commit_hash() -> str | None:
    try:
        from subprocess import run, PIPE
        result = run(
            ["git", "rev-parse", "HEAD"],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None

def build_master_dataset(
    sentiment_dir: Path,
    price_dir: Path,
    output_file: Optional[Path] = None,
    horizons: Iterable[int] = HORIZONS,
    version: str = DEFAULT_VERSION,
    tag: Optional[str] = None,
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

    # Attach entry bars
    master = attach_entry_bar(sentiment, prices)

    # Add price window info for coverage diagnostics
    price_window = (
        prices.groupby("pair")["timestamp"]
        .agg(price_start="min", price_end="max", price_bars="count")
        .reset_index()
    )
    master = master.merge(price_window, on="pair", how="left")

    # Eligibility helpers
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

    coverage_path = Path(f"data/output/{version}/pair_coverage_summary.csv")
    ensure_output_dir(coverage_path)
    coverage.to_csv(coverage_path, index=False)
    print(f"\nSaved pair coverage summary to {coverage_path}")

    # Coverage-based universes
    core_pairs = coverage.loc[
        coverage["eligible_match_ratio"] >= CORE_MIN_ELIGIBLE_MATCH_RATIO, "pair"
    ]
    extended_pairs = coverage.loc[
        coverage["eligible_match_ratio"] >= EXTENDED_MIN_ELIGIBLE_MATCH_RATIO, "pair"
    ]

    print(f"\nCore pairs (eligible_match_ratio >= {CORE_MIN_ELIGIBLE_MATCH_RATIO:.2f}): {len(core_pairs):,}")
    print(sorted(core_pairs.tolist()))

    print(f"\nExtended pairs (eligible_match_ratio >= {EXTENDED_MIN_ELIGIBLE_MATCH_RATIO:.2f}): {len(extended_pairs):,}")
    print(sorted(extended_pairs.tolist()))

    # Keep only rows with a valid entry bar
    master_valid = master.dropna(subset=["entry_time", "entry_close"]).copy()
    print(f"\nRows with valid entry bar: {len(master_valid):,}")

    # ============================================
    # Remove known corrupted pairs (price scaling issues)
    # ============================================
    BAD_PAIRS = {
        "eur-mxn",
        "gbp-zar",  # optional but recommended
    }

    before_rows = len(master_valid)
    before_pairs = master_valid["pair"].nunique()

    master_valid = master_valid[~master_valid["pair"].isin(BAD_PAIRS)].copy()

    after_rows = len(master_valid)
    after_pairs = master_valid["pair"].nunique()

    print("\nRemoved corrupted pairs:")
    print(f"  pairs removed: {sorted(BAD_PAIRS)}")
    print(f"  rows removed: {before_rows - after_rows:,}")
    print(f"  remaining pairs: {after_pairs:,} (was {before_pairs:,})")

    # Add forward returns BEFORE splitting, so column parity stays easier to maintain
    master_valid = add_forward_returns(master_valid, prices, horizons=horizons)

    # Add trend features (analysis-only, uses forward returns)
    print("Trend feature columns added:",
          [c for c in master_valid.columns if c.startswith("trend_")][:5])

    # Stable sentiment side fields
    master_valid = add_crowd_side(master_valid)
    master_valid = add_trend_features(master_valid)

    # ============================================
    # Crowd persistence feature (behavioral regime)
    # ============================================

    def bucket_crowd_persistence(streak):
        if pd.isna(streak):
            return None
        elif streak == 0:
            return "none"
        elif streak <= 2:
            return "low"
        elif streak <= 5:
            return "medium"
        else:
            return "high"

    master_valid["crowd_persistence_bucket_70"] = (
        master_valid["extreme_streak_70"]
        .apply(bucket_crowd_persistence)
    )


    # ============================================
    # REGIME V2: ACCELERATION
    # ============================================

    master_valid["sentiment_change_6h"] = (
        master_valid.groupby("pair")["net_sentiment"].diff(6)
    )

    q_low = master_valid["sentiment_change_6h"].quantile(0.33)
    q_high = master_valid["sentiment_change_6h"].quantile(0.66)

    def bucket_acceleration(x):
        if pd.isna(x):
            return None
        elif x <= q_low:
            return "decreasing"
        elif x >= q_high:
            return "increasing"
        else:
            return "stable"

    master_valid["acceleration_bucket"] = (
        master_valid["sentiment_change_6h"]
        .apply(bucket_acceleration).fillna("unknown")
    )

    # ============================================
    # REGIME V2: SATURATION
    # ============================================

    def bucket_saturation(x):
        if pd.isna(x):
            return None
        elif x < 60:
            return "normal"
        elif x < 75:
            return "elevated"
        elif x < 85:
            return "extreme"
        else:
            return "panic"

    master_valid["saturation_bucket"] = (
        master_valid["abs_sentiment"]
        .apply(bucket_saturation)
    )


    master_valid["is_long_crowd"] = master_valid["crowd_side"] == 1
    master_valid["is_short_crowd"] = master_valid["crowd_side"] == -1

    # --------------------------------
    # Trend strength buckets (v1)
    # --------------------------------

    for h in [12, 48]:
        col = f"trend_strength_{h}b"
        bucket_col = f"trend_strength_bucket_{h}b"

        # avoid NaNs / extreme outliers
        valid = master_valid[col].notna()

        master_valid.loc[valid, bucket_col] = pd.qcut(
            master_valid.loc[valid, col],
            q=4,
            labels=["weak", "medium", "strong", "extreme"]
        )

    # Split into filtered datasets
    master_core = master_valid[master_valid["pair"].isin(core_pairs)].copy()
    master_extended = master_valid[master_valid["pair"].isin(extended_pairs)].copy()

    print(f"Core-universe rows: {len(master_core):,}")
    print(f"Core-universe pairs: {master_core['pair'].nunique():,}")

    print(f"Extended-universe rows: {len(master_extended):,}")
    print(f"Extended-universe pairs: {master_extended['pair'].nunique():,}")

    # Sorting
    master_valid = master_valid.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)
    master_core = master_core.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)
    master_extended = master_extended.sort_values(["pair", "snapshot_time"]).reset_index(drop=True)

    # --- macro regime (time-based) ---
    master_valid["year"] = pd.to_datetime(master_valid["entry_time"]).dt.year

    master_valid["macro_regime"] = master_valid["year"].apply(
        lambda y: "pre_2022" if y <= 2021 else "post_2022"
    )

    # Guarantee identical columns and order across outputs
    master_valid, master_core, master_extended = align_dataset_columns(
        master_valid,
        master_core,
        master_extended,
    )
    # =========================
    # REGIME V2 FEATURES
    # =========================

    # --- trend alignment flags (already exist, just make explicit flags) ---
    master_valid["fight_trend"] = master_valid["trend_alignment_12b"] == -1
    master_valid["follow_trend"] = master_valid["trend_alignment_12b"] == 1



    # Save dataset variants
    full_path = Path(f"data/output/{version}/master_research_dataset.csv")
    core_path = Path(f"data/output/{version}/master_research_dataset_core.csv")
    extended_path = Path(f"data/output/{version}/master_research_dataset_extended.csv")

    ensure_output_dir(full_path)
    master_valid.to_csv(full_path, index=False)
    master_core.to_csv(core_path, index=False)
    master_extended.to_csv(extended_path, index=False)

    print("\nSaved:")
    print(f"  {full_path}")
    print(f"  {core_path}")
    print(f"  {extended_path}")

    # Optional custom output_file writes the full valid dataset
    if output_file is not None:
        ensure_output_dir(output_file)
        master_valid.to_csv(output_file, index=False)
        print(f"\nSaved master dataset to: {output_file.resolve()}")

    # Manifest
    git_commit = get_git_commit_hash()
    write_dataset_manifest(
        output_dir=Path(f"data/output/{version}"),
        full_df=master_valid,
        core_df=master_core,
        extended_df=master_extended,
        version=version,
        tag=tag,
        git_commit=git_commit,
    )

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
        horizons=HORIZONS,
        version=DEFAULT_VERSION,
    )

# ============================================================
# REGIME FEATURES (ABM-aligned, strictly causal)
# ============================================================

import numpy as np


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["vol_12b", "trend_12b"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataset")

    out = df.copy()

    if "entry_time" not in out.columns:
        raise ValueError("entry_time column missing")

    out = out.sort_values(["pair", "entry_time"]).reset_index(drop=True)

    # Trend strength
    out["trend_vol_adj_strength"] = (
        out["trend_12b"].abs() / (out["vol_12b"] + 1e-8)
    )

    # Trending flag
    out["is_trending"] = (
            out["trend_vol_adj_strength"].fillna(0.0) > TREND_THRESHOLD
    ).astype(bool)

    # High vol (causal)
    out["vol_median"] = (
        out.groupby("pair")["vol_12b"]
        .transform(lambda x: x.expanding().median())
    )

    out["is_high_vol"] = (out["vol_12b"] > out["vol_median"]).astype(bool)

    # Regime
    conditions = [
        out["is_high_vol"] & out["is_trending"],
        ~out["is_high_vol"] & out["is_trending"],
        out["is_high_vol"] & ~out["is_trending"],
        ~out["is_high_vol"] & ~out["is_trending"],
    ]

    choices = ["HVTF", "LVTF", "HVR", "LVR"]

    out["regime"] = np.select(conditions, choices, default="LVR")

    # NaNs
    nan_mask = out["vol_12b"].isna() | out["trend_12b"].isna()
    out.loc[nan_mask, "regime"] = np.nan

    return out