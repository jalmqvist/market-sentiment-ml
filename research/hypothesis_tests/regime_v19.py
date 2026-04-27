# Legacy experiment — not part of current validated approach\nimport argparse
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from utils.io import read_csv
from utils.validation import parse_timestamps

logger = logging.getLogger(__name__)

TARGET_COL = "ret_48b"


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = read_csv(path)
    df = parse_timestamps(df, "time", context="regime_v19")

    # --- choose price column ---
    price_candidates = ["price_end", "entry_close"]
    best_col = None
    best_ratio = 0

    for col in price_candidates:
        if col in df.columns:
            ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            logger.debug(f"price candidate={col} valid_ratio={ratio:.3f}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col

    if best_col is None:
        raise ValueError("No valid price column found")

    df["price"] = pd.to_numeric(df[best_col], errors="coerce")
    logger.info(f"Using price column: {best_col}")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # returns
    df["ret_1b"] = df.groupby("pair")["price"].pct_change()
    df["mom_48b"] = (
        df.groupby("pair")["ret_1b"]
        .rolling(48)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # z-scores
    df["price_mom_z"] = df.groupby("pair")["mom_48b"].transform(
        lambda x: rolling_zscore(x, 96)
    )
    df["sentiment_z"] = df.groupby("pair")["net_sentiment"].transform(
        lambda x: rolling_zscore(x, 96)
    )

    # divergence
    df["divergence"] = df["sentiment_z"] - df["price_mom_z"]

    # interactions
    df["interaction_1"] = df["divergence"] * df["sentiment_z"]
    df["interaction_2"] = df["divergence"] * df["price_mom_z"]

    return df


# ============================================================
# TIME-SERIES NORMALIZATION (FIXED CORE)
# ============================================================

def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["divergence", "interaction_1", "interaction_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        df[f"z_{col}"] = df.groupby("pair")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    return df


# ============================================================
# SCORING
# ============================================================

def normalize_weights(w: Tuple[float, float, float]) -> Tuple[float, float, float]:
    s = sum(w)
    return tuple(x / s for x in w) if s != 0 else (0, 0, 0)


def compute_score(df: pd.DataFrame, weights: Tuple[float, float, float]) -> np.ndarray:
    w1, w2, w3 = normalize_weights(weights)

    score = (
        w1 * df["z_divergence"].fillna(0)
        + w2 * df["z_interaction_1"].fillna(0)
        + w3 * df["z_interaction_2"].fillna(0)
    )

    if score.std() < 1e-6:
        return np.zeros(len(df))

    return np.tanh(score)


def evaluate(df: pd.DataFrame, weights: Tuple[float, float, float]) -> float:
    pos = compute_score(df, weights)
    pnl = pos * df[TARGET_COL]

    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()

    if len(pnl) < 50 or pnl.std() < 1e-9:
        return -999

    return pnl.mean() / pnl.std()


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    df = add_zscores(df)

    df["year"] = df["time"].dt.year
    years = sorted(df["year"].dropna().unique())

    weight_grid = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (2, 1, 0),
        (1, 2, 0),
        (1, 1, 1),
    ]

    results = []

    for year in years:
        if year <= years[0]:
            logger.debug(f"Skipping year={year}")
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year]

        if len(train) < 1000 or len(test) == 0:
            continue

        best_weights = None
        best_sharpe = -999

        for w in weight_grid:
            s = evaluate(train, w)
            if s > best_sharpe:
                best_sharpe = s
                best_weights = w

        if best_weights is None:
            best_weights = (1, 1, 1)
            logger.warning(f"[{year}] fallback weights used")

        logger.debug(
            f"[{year}] best_weights={best_weights} train_sharpe={best_sharpe:.4f}"
        )

        pos = compute_score(test, best_weights)
        pnl = pos * test[TARGET_COL]

        pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pnl) == 0 or pnl.std() < 1e-9:
            sharpe = 0.0
            hit = 0.0
        else:
            sharpe = pnl.mean() / pnl.std()
            hit = (pnl > 0).mean()

        logger.info(
            f"[{year}] n={len(pnl)} | sharpe={sharpe:.4f} | hit={hit:.4f}"
        )

        results.append(
            dict(
                year=year,
                n=len(pnl),
                sharpe=sharpe,
                hit_rate=hit,
                weights=best_weights,
            )
        )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    df = load_data(args.data)
    results = walk_forward(df)

    print("\n=== RESULTS ===")
    print(results)

    if len(results):
        print("\nMEAN SHARPE:", results["sharpe"].mean())


if __name__ == "__main__":
    main()