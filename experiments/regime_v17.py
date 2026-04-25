import argparse
import logging
import numpy as np
import pandas as pd

from utils.io import read_csv
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)


# =========================
# Config
# =========================
TARGET_COL = "ret_48b"


# =========================
# Feature engineering
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- choose price column ---
    price_col = None
    for col in ["price_end", "entry_close"]:
        if col in df.columns:
            valid_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            logger.debug(f"price candidate={col} valid_ratio={valid_ratio:.3f}")
            if valid_ratio > 0.5:
                price_col = col
                break

    if price_col is None:
        raise ValueError("No valid price column found")

    logger.info(f"Using price column: {price_col}")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # --- returns ---
    df["ret_1b"] = df.groupby("pair")[price_col].pct_change()

    # --- momentum ---
    df["mom_48b"] = (
        df.groupby("pair")["ret_1b"]
        .rolling(48, min_periods=48)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --- z-scores ---
    def rolling_z(x, window=96):
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

    df["price_mom_z"] = df.groupby("pair")["mom_48b"].transform(rolling_z)

    if "net_sentiment" not in df.columns:
        raise ValueError("Missing net_sentiment")

    df["sentiment_z"] = df.groupby("pair")["net_sentiment"].transform(rolling_z)

    # --- divergence ---
    df["divergence"] = df["sentiment_z"] - df["price_mom_z"]

    # --- trend proxy ---
    df["trend"] = df["price_mom_z"]

    # --- interactions ---
    df["interaction_1"] = df["divergence"] * df["trend"]
    df["interaction_2"] = df["divergence"] * df["trend"].abs()

    return df


# =========================
# Ranking logic
# =========================
def compute_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def rank_group(g):
        for col in ["divergence", "interaction_1", "interaction_2"]:
            g[f"rank_{col}"] = g[col].rank(pct=True)
        return g

    df = df.groupby("time", group_keys=False).apply(rank_group)

    df["score"] = (
        df["rank_divergence"]
        + df["rank_interaction_1"]
        + df["rank_interaction_2"]
    )

    return df


# =========================
# Walk-forward
# =========================
def walk_forward(df: pd.DataFrame, top_frac: float = 0.2):
    df = df.copy()

    df["year"] = df["time"].dt.year

    results = []

    years = sorted(df["year"].dropna().unique())

    for year in years:
        if year < 2020:
            logger.debug(f"Skipping year={year}")
            continue

        train = df[df["year"] < year].copy()
        test = df[df["year"] == year].copy()

        if len(train) < 1000 or len(test) < 100:
            logger.debug(f"Skipping year={year} (insufficient data)")
            continue

        # --- rank using TRAIN distribution (important!) ---
        test = compute_cross_sectional_ranks(test)

        # --- drop NaNs ---
        test = test.dropna(subset=["score", TARGET_COL])

        if test.empty:
            logger.debug(f"No valid rows for year={year}")
            continue

        # --- select top fraction ---
        cutoff = test["score"].quantile(1 - top_frac)
        selected = test[test["score"] >= cutoff]

        if selected.empty:
            continue

        pnl = selected[TARGET_COL]

        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        hit = (pnl > 0).mean()

        logger.info(
            f"[{year}] n={len(selected)} | sharpe={sharpe:.4f} | hit={hit:.4f}"
        )

        results.append(
            {
                "year": year,
                "n": len(selected),
                "sharpe": sharpe,
                "hit_rate": hit,
            }
        )

    return pd.DataFrame(results)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    df = read_csv(args.data, required_columns=["time", TARGET_COL])
    df = parse_timestamps(df, "time", context="regime_v17")

    df = build_features(df)

    results = walk_forward(df, top_frac=args.top_frac)

    print("\n=== RESULTS ===")
    print(results)

    if not results.empty:
        print("\nMEAN SHARPE:", results["sharpe"].mean())


if __name__ == "__main__":
    main()
