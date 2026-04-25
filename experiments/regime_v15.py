import argparse
import logging
import numpy as np
import pandas as pd

from utils.io import read_csv
from utils.validation import parse_timestamps

TARGET_COL = "ret_48b"


# ============================================================
# Utils
# ============================================================
def detect_price_column(df):
    for col in ["price", "price_end", "entry_close"]:
        if col in df.columns:
            tmp = pd.to_numeric(df[col], errors="coerce")
            if tmp.notna().mean() > 0.9:
                df[col] = tmp
                logging.info(f"Using price column: {col}")
                return col
    raise ValueError("No valid price column found")


def rolling_zscore(series, window=96):
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


def percentile_rank(series):
    return series.rank(pct=True)


# ============================================================
# Feature builder
# ============================================================
def build_features(df):
    df = df.copy()

    df["net_sentiment"] = pd.to_numeric(df["net_sentiment"], errors="coerce")

    df = df.dropna(subset=["price", "net_sentiment"])
    df = df.sort_values(["pair", "time"])

    def compute_group(grp):
        grp = grp.copy()

        grp["ret_1b"] = grp["price"].pct_change()
        grp["mom_48b"] = grp["ret_1b"].rolling(48, min_periods=48).sum()

        grp["sentiment_z"] = rolling_zscore(grp["net_sentiment"], 96)
        grp["price_mom_z"] = rolling_zscore(grp["mom_48b"], 96)

        grp["divergence"] = grp["sentiment_z"] - grp["price_mom_z"]

        grp["trend_z"] = rolling_zscore(grp["mom_48b"], 96)

        # === INTERACTIONS ===
        grp["interaction_1"] = grp["divergence"] * grp["trend_z"]
        grp["interaction_2"] = grp["divergence"] * grp["trend_z"].abs()

        # === COMPOSITE SCORE ===
        grp["score"] = (
            grp["divergence"]
            + 0.5 * grp["interaction_1"]
            + 0.25 * grp["interaction_2"]
        )

        return grp

    df = df.groupby("pair", group_keys=False).apply(compute_group)

    return df


# ============================================================
# Walk-forward (ranking-based)
# ============================================================
def walk_forward(df):
    results = []

    df = df.copy()
    df["year"] = df["time"].dt.year

    years = sorted(df["year"].unique())

    for year in years:
        train_mask = df["year"] < year
        test_mask = df["year"] == year

        if train_mask.sum() < 2000 or test_mask.sum() < 200:
            logging.debug(f"Skipping year={year}")
            continue

        fold_df = df.loc[train_mask | test_mask].copy()
        fold_df = build_features(fold_df)

        test = fold_df.loc[fold_df["year"] == year].copy()

        test = test.dropna(subset=["score", TARGET_COL])

        if len(test) == 0:
            continue

        # === RANKING ===
        test["rank"] = percentile_rank(test["score"])

        # [-1, +1]
        test["position"] = 2 * (test["rank"] - 0.5)

        # Optional nonlinear scaling
        test["position"] = np.tanh(2 * test["position"])

        pnl = test["position"] * test[TARGET_COL]

        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        hit_rate = (pnl > 0).mean()

        results.append({
            "year": year,
            "n": len(test),
            "sharpe": sharpe,
            "hit_rate": hit_rate
        })

        logging.info(
            f"[{year}] n={len(test)} | sharpe={sharpe:.4f} | hit={hit_rate:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    df = read_csv(args.data)
    df = parse_timestamps(df, "time", context="regime_v15")

    price_col = detect_price_column(df)
    df["price"] = df[price_col]

    fold_df = walk_forward(df)

    print("\n=== RESULTS ===")
    print(fold_df)

    if len(fold_df) > 0:
        print("\nMEAN SHARPE:", fold_df["sharpe"].mean())
    else:
        print("\nNo valid folds")


if __name__ == "__main__":
    main()
