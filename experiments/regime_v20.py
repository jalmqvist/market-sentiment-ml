import argparse
import logging
import numpy as np
import pandas as pd

from utils.io import read_csv

logger = logging.getLogger(__name__)

TARGET_COL = "ret_1b"
MIN_TRAIN_ROWS = 500


# =========================
# SIGNAL DETECTION
# =========================
def detect_signal_column(df: pd.DataFrame) -> str:
    preferred = ["divergence", "signal_v2_raw"]

    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        if any(x in col.lower() for x in ["sent", "score", "signal"]):
            valid_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if valid_ratio > 0.5:
                logger.warning(f"Auto-selected signal column: {col}")
                return col

    raise ValueError("No usable signal column found")


# =========================
# LOAD DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = read_csv(path)

    if "time" not in df.columns:
        raise ValueError("Missing required column: time")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    if "symbol" not in df.columns:
        df["symbol"] = "asset"

    df = df.sort_values(["symbol", "time"])

    # -------- price selection --------
    price_candidates = ["price_end", "entry_close", "close"]

    price_col = None
    for col in price_candidates:
        if col in df.columns:
            valid_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            logger.debug(f"price candidate={col} valid_ratio={valid_ratio:.3f}")
            if valid_ratio > 0.8:
                price_col = col
                break

    if price_col is None:
        raise ValueError("No valid price column found")

    logger.info(f"Using price column: {price_col}")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # ✅ forward return (CRITICAL FIX)
    df[TARGET_COL] = (
        df.groupby("symbol")[price_col]
        .pct_change()
        .shift(-1)
    )

    # -------- signal --------
    signal_col = detect_signal_column(df)
    logger.info(f"Using signal column: {signal_col}")

    df["signal_raw"] = pd.to_numeric(df[signal_col], errors="coerce")

    return df


# =========================
# FEATURE ENGINEERING (GLOBAL, CAUSAL)
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def process_group(g):
        g = g.sort_values("time")

        # no lookahead
        g["signal"] = g["signal_raw"]

        # expanding baseline (past only)
        baseline = g["signal"].expanding(min_periods=50).mean().shift(1)
        residual = g["signal"] - baseline

        mean = residual.expanding(min_periods=50).mean().shift(1)
        std = residual.expanding(min_periods=50).std().shift(1)

        g["zscore"] = (residual - mean) / (std + 1e-6)

        return g

    return df.groupby("symbol", group_keys=False).apply(process_group)


# =========================
# POSITION (TIME-SERIES)
# =========================
def compute_positions(df: pd.DataFrame) -> pd.Series:
    return np.tanh(df["zscore"] * 0.5)


# =========================
# EVALUATION
# =========================
def evaluate(df: pd.DataFrame):
    pnl = df["position"] * df[TARGET_COL]

    if pnl.std() == 0 or np.isnan(pnl.std()):
        return 0.0, 0.0

    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    hit = (pnl > 0).mean()

    return float(sharpe), float(hit)


# =========================
# WALK FORWARD
# =========================
def walk_forward(df: pd.DataFrame, downsample: int = 1, shuffle: bool = False):
    df = df.copy()
    df["year"] = df["time"].dt.year

    results = []

    for year in sorted(df["year"].unique()):
        if year < 2020:
            logger.debug(f"Skipping year={year}")
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year].copy()

        if len(train) < MIN_TRAIN_ROWS or len(test) == 0:
            logger.debug(f"Skipping year={year} (insufficient data)")
            continue

        test = test.dropna(subset=["zscore", TARGET_COL])

        if len(test) == 0:
            continue

        # optional downsampling (diagnostic)
        if downsample > 1:
            test = test.iloc[::downsample]

        test["position"] = compute_positions(test)

        # optional shuffle test
        if shuffle:
            test["position"] = np.random.permutation(test["position"].values)

        sharpe, hit = evaluate(test)

        logger.info(f"[{year}] n={len(test)} | sharpe={sharpe:.4f} | hit={hit:.4f}")

        results.append(
            {"year": year, "n": len(test), "sharpe": sharpe, "hit_rate": hit}
        )

    return pd.DataFrame(results)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    df = load_data(args.data)

    # global features
    df = build_features(df)

    results = walk_forward(
        df,
        downsample=args.downsample,
        shuffle=args.shuffle
    )

    print("\n=== RESULTS ===")
    print(results)

    if len(results) > 0:
        print("\nMEAN SHARPE:", results["sharpe"].mean())
    else:
        print("\nNo valid folds")


if __name__ == "__main__":
    main()