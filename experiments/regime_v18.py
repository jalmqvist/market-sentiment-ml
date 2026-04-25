import argparse
import logging
import numpy as np
import pandas as pd

from utils.io import read_csv
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)

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

    # returns
    df["ret_1b"] = df.groupby("pair")[price_col].pct_change()

    # momentum
    df["mom_48b"] = (
        df.groupby("pair")["ret_1b"]
        .rolling(48, min_periods=48)
        .sum()
        .reset_index(level=0, drop=True)
    )

    def rolling_z(x, window=96):
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

    df["price_mom_z"] = df.groupby("pair")["mom_48b"].transform(rolling_z)
    df["sentiment_z"] = df.groupby("pair")["net_sentiment"].transform(rolling_z)

    df["divergence"] = df["sentiment_z"] - df["price_mom_z"]
    df["trend"] = df["price_mom_z"]

    df["interaction_1"] = df["divergence"] * df["trend"]
    df["interaction_2"] = df["divergence"] * df["trend"].abs()

    return df


# =========================
# Ranking + scoring
# =========================
def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" not in df.columns:
        df = df.reset_index()

    # --- ranks ---
    for col in ["divergence", "interaction_1", "interaction_2"]:
        df[f"rank_{col}"] = df.groupby("time")[col].rank(pct=True)

    df["score_raw"] = (
        df["rank_divergence"]
        + df["rank_interaction_1"]
        + df["rank_interaction_2"]
    )

    # --- simple centering (NO z-score) ---
    df["score_centered"] = df["score_raw"] - 1.5

    # --- position ---
    df["position"] = np.tanh(df["score_centered"])

    return df
# =========================
# Walk-forward
# =========================
def walk_forward(df: pd.DataFrame):
    df = df.copy()
    df["year"] = df["time"].dt.year

    results = []
    years = sorted(df["year"].dropna().unique())

    for year in years:
        if year < 2020:
            logger.debug(f"Skipping year={year}")
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year]

        if len(train) < 1000 or len(test) < 100:
            logger.debug(f"Skipping year={year} (insufficient data)")
            continue

        test = compute_score(test)
        test = test.dropna(subset=["position", TARGET_COL])

        if test.empty:
            continue

        pnl = test["position"] * test[TARGET_COL]

        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        hit = (pnl > 0).mean()

        logger.info(
            f"[{year}] n={len(test)} | sharpe={sharpe:.4f} | hit={hit:.4f}"
        )

        results.append(
            {
                "year": year,
                "n": len(test),
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
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    df = read_csv(args.data, required_columns=["time", TARGET_COL, "net_sentiment"])
    df = parse_timestamps(df, "time", context="regime_v18")

    df = build_features(df)

    results = walk_forward(df)

    print("\n=== RESULTS ===")
    print(results)

    if not results.empty:
        print("\nMEAN SHARPE:", results["sharpe"].mean())


if __name__ == "__main__":
    main()
