# Legacy experiment — not part of current validated approach\nimport argparse
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
def load_data(path: str, signal_shift: int = 0) -> pd.DataFrame:
    df = read_csv(path)

    if "time" not in df.columns:
        raise ValueError("Missing required column: time")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    if "symbol" not in df.columns:
        df["symbol"] = "asset"

    df = df.sort_values(["symbol", "time"])

    # -------- price --------
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

    # ✅ forward returns (t → t+1)
    df[TARGET_COL] = (
        df.groupby("symbol")[price_col]
        .pct_change()
        .shift(-1)
    )

    # -------- signal --------
    signal_col = detect_signal_column(df)
    logger.info(f"Using signal column: {signal_col}")

    df["signal_raw"] = pd.to_numeric(df[signal_col], errors="coerce")

    # 🔥 optional diagnostic: shift signal
    if signal_shift != 0:
        logger.warning(f"Shifting signal by {signal_shift}")
        df["signal_raw"] = df.groupby("symbol")["signal_raw"].shift(signal_shift)

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def process_group(g):
        g = g.sort_values("time")

        signal = g["signal_raw"]

        baseline = signal.expanding(min_periods=50).mean().shift(1)
        residual = signal - baseline

        mean = residual.expanding(min_periods=50).mean().shift(1)
        std = residual.expanding(min_periods=50).std().shift(1)

        g["zscore"] = (residual - mean) / (std + 1e-6)

        # volatility estimate
        g["vol"] = g[TARGET_COL].rolling(50).std().shift(1)

        return g

    return df.groupby("symbol", group_keys=False).apply(process_group)


# =========================
# TOP-1 POSITION SELECTION
# =========================
def select_top(df: pd.DataFrame) -> pd.Series:
    z = df["zscore"]

    if z.isna().all():
        return pd.Series(0.0, index=df.index, dtype=float)

    idx = z.abs().idxmax()

    out = pd.Series(0.0, index=df.index, dtype=float)

    val = df.loc[idx, "zscore"]
    if pd.notna(val):
        out.loc[idx] = float(np.tanh(val * 0.7))

    return out


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
def walk_forward(df: pd.DataFrame):
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
            continue

        test = test.dropna(subset=["zscore", TARGET_COL])

        if len(test) == 0:
            continue

        # 🔥 TOP-1 constraint per timestamp
        test["position"] = (
            test.groupby("time", group_keys=False)
            .apply(select_top)
        )

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
    parser.add_argument("--signal-shift", type=int, default=0)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    df = load_data(args.data, signal_shift=args.signal_shift)

    df = build_features(df)

    results = walk_forward(df)

    print("\n=== RESULTS ===")
    print(results)

    if len(results) > 0:
        print("\nMEAN SHARPE:", results["sharpe"].mean())
    else:
        print("\nNo valid folds")


if __name__ == "__main__":
    main()