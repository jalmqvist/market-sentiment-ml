import argparse
import logging
import numpy as np
import pandas as pd

from utils.io import read_csv
from utils.validation import parse_timestamps

TARGET_COL = "ret_48b"


# ============================================================
# Price detection
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


# ============================================================
# Rolling z-score
# ============================================================
def rolling_zscore(series, window=96):
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


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

        grp["score"] = pd.to_numeric(grp["divergence"], errors="coerce")

        return grp

    df = df.groupby("pair", group_keys=False).apply(compute_group)

    return df


# ============================================================
# SAFE binning (no qcut!)
# ============================================================
def build_expected_return_map(train_df, n_bins=5):
    df = train_df.dropna(subset=["score", TARGET_COL]).copy()

    if len(df) < 1000:
        return None, None

    # Percentile-based bins (robust)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(df["score"], percentiles)

    # Ensure unique bins (VERY important)
    bins = np.unique(bins)

    if len(bins) < 3:
        return None, None

    df["bin"] = pd.cut(df["score"], bins=bins, include_lowest=True)

    exp_map = df.groupby("bin")[TARGET_COL].mean().to_dict()

    return exp_map, bins


def apply_expected_return(df, exp_map, bins):
    df = df.copy()

    if exp_map is None or bins is None:
        df["expected_ret"] = 0.0
        return df

    try:
        df["bin"] = pd.cut(df["score"], bins=bins, include_lowest=True)
        df["expected_ret"] = df["bin"].map(exp_map)
    except Exception as e:
        logging.warning(f"apply_expected_return failed: {e}")
        df["expected_ret"] = 0.0
        return df

    df["expected_ret"] = pd.to_numeric(df["expected_ret"], errors="coerce").fillna(0.0)

    return df


# ============================================================
# Walk-forward
# ============================================================
def walk_forward(df, top_frac=0.2):
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

        train = fold_df.loc[fold_df["year"] < year].copy()
        test = fold_df.loc[fold_df["year"] == year].copy()

        logging.debug(
            f"[{year}] score NaN ratio: {test['score'].isna().mean():.3f}"
        )

        exp_map, bins = build_expected_return_map(train)

        test = apply_expected_return(test, exp_map, bins)

        test = test.dropna(subset=[TARGET_COL, "expected_ret"])

        if len(test) == 0:
            logging.debug(f"No valid test rows for year={year}")
            continue

        test = test.sort_values("expected_ret", ascending=False)

        n = max(1, int(len(test) * top_frac))
        test = test.head(n)

        test["position"] = np.sign(test["expected_ret"])

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
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    df = read_csv(args.data)
    df = parse_timestamps(df, "time", context="regime_v14")

    price_col = detect_price_column(df)
    df["price"] = df[price_col]

    fold_df = walk_forward(df, top_frac=args.top_frac)

    print("\n=== RESULTS ===")
    print(fold_df)

    if len(fold_df) > 0:
        print("\nMEAN SHARPE:", fold_df["sharpe"].mean())
    else:
        print("\nNo valid folds")


if __name__ == "__main__":
    main()