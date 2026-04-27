#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# =========================
# Logging (file only)
# =========================

def setup_logging():
    repo_root = Path(__file__).resolve().parent.parent
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validate_signal_raw_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

    print(f"Logging to: {log_file}")


# =========================
# Load data
# =========================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = ["entry_time", "ret_48b", "net_sentiment"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time"])

    df = df.sort_values("entry_time").reset_index(drop=True)
    df["year"] = df["entry_time"].dt.year

    return df


# =========================
# Metrics
# =========================

def compute_positions(signal: pd.Series) -> pd.Series:
    return np.sign(signal).astype(float)


def compute_metrics(df: pd.DataFrame) -> dict:
    pnl = df["position"] * df["ret_48b"]

    if len(pnl) == 0:
        return {"n": 0, "sharpe": 0.0, "hit_rate": 0.0}

    mean = pnl.mean()
    std = pnl.std()

    sharpe = mean / std if std > 1e-12 else 0.0
    hit = (pnl > 0).mean()

    return {
        "n": len(pnl),
        "sharpe": sharpe,
        "hit_rate": hit,
    }


def evaluate_by_year(df: pd.DataFrame, label: str):
    logging.info(f"\n=== {label} ===")

    results = []

    for year, g in df.groupby("year"):
        if year < 2020:
            continue

        m = compute_metrics(g)

        logging.info(
            f"[{year}] n={m['n']} | sharpe={m['sharpe']:.4f} | hit={m['hit_rate']:.4f}"
        )

        results.append(m["sharpe"])

    if len(results) > 0:
        logging.info(f"MEAN SHARPE: {np.mean(results):.6f}")
    else:
        logging.info("MEAN SHARPE: 0.0")

    density = (df["signal"] != 0).mean()
    logging.info(f"Signal density: {density:.4f}")


# =========================
# Signal definitions
# =========================

def compute_v22_signal(df, lookback=48, q=0.95):
    df = df.copy()

    df["trend"] = df["ret_48b"].rolling(lookback).mean().shift(1)

    upper = df["net_sentiment"].expanding(200).quantile(q).shift(1)
    lower = df["net_sentiment"].expanding(200).quantile(1 - q).shift(1)

    signal = np.zeros(len(df))

    signal[(df["net_sentiment"] > upper) & (df["trend"] < 0)] = -1
    signal[(df["net_sentiment"] < lower) & (df["trend"] > 0)] = +1

    df["signal"] = signal
    return df


def compute_v23_signal(df, delta_window=5):
    df = df.copy()

    df["dS"] = df["net_sentiment"].diff(delta_window)

    mean = df["dS"].expanding(200).mean().shift(1)
    std = df["dS"].expanding(200).std().shift(1)

    z = (df["dS"] - mean) / (std + 1e-6)

    signal = np.zeros(len(df))
    signal[z > 2] = -1
    signal[z < -2] = +1

    df["signal"] = signal
    return df


def compute_v24_signal(df, window=20):
    df = df.copy()

    df["S_mean"] = df["net_sentiment"].rolling(window).mean().shift(1)

    mean = df["S_mean"].expanding(200).mean().shift(1)
    std = df["S_mean"].expanding(200).std().shift(1)

    z = (df["S_mean"] - mean) / (std + 1e-6)

    signal = np.zeros(len(df))
    signal[z > 1.5] = -1
    signal[z < -1.5] = +1

    df["signal"] = signal
    return df

def compute_v25_signal(df, vol_lookback=48, q=0.95):
    df = df.copy()

    # =========================
    # 1. Proxy past returns (causal)
    # =========================
    # We cannot use ret_48b directly as a feature,
    # so we approximate past returns via shifting
    df["past_ret"] = df["ret_48b"].shift(48)

    # =========================
    # 2. Volatility (rolling std of past returns)
    # =========================
    df["vol"] = df["past_ret"].rolling(vol_lookback).std().shift(1)

    # =========================
    # 3. Volatility regime (expanding z-score)
    # =========================
    vol_mean = df["vol"].expanding(200).mean().shift(1)
    vol_std = df["vol"].expanding(200).std().shift(1)

    df["vol_z"] = (df["vol"] - vol_mean) / (vol_std + 1e-6)

    # Define "high / rising vol"
    high_vol = df["vol_z"] > 1.0

    # =========================
    # 4. Extreme sentiment (causal)
    # =========================
    upper = df["net_sentiment"].expanding(200).quantile(q).shift(1)
    lower = df["net_sentiment"].expanding(200).quantile(1 - q).shift(1)

    extreme_long = df["net_sentiment"] > upper
    extreme_short = df["net_sentiment"] < lower

    # =========================
    # 5. Final signal
    # =========================
    signal = np.zeros(len(df))

    # Only act when volatility is high
    signal[(high_vol) & (extreme_long)] = -1
    signal[(high_vol) & (extreme_short)] = +1

    df["signal"] = signal

    return df

def compute_v26_signal(df, price_window=24, sentiment_window=24):
    df = df.copy()

    # =========================
    # 1. Proxy past price movement (causal)
    # =========================
    df["past_ret"] = df["ret_48b"].shift(48)

    # cumulative price move
    df["price_move"] = df["past_ret"].rolling(price_window).sum().shift(1)

    # =========================
    # 2. Sentiment change
    # =========================
    df["sent_change"] = df["net_sentiment"].diff(sentiment_window)

    # =========================
    # 3. Normalize both (causal)
    # =========================
    def zscore(series):
        mean = series.expanding(200).mean().shift(1)
        std = series.expanding(200).std().shift(1)
        return (series - mean) / (std + 1e-6)

    price_z = zscore(df["price_move"])
    sent_z = zscore(df["sent_change"])

    # =========================
    # 4. Disagreement signal
    # =========================
    # price up but sentiment not increasing → bullish continuation
    # price down but sentiment not decreasing → bearish continuation

    signal = np.zeros(len(df))

    # strong up move, weak sentiment response
    signal[(price_z > 1.0) & (sent_z < 0.5)] = +1

    # strong down move, weak sentiment response
    signal[(price_z < -1.0) & (sent_z > -0.5)] = -1

    df["signal"] = signal

    return df
# =========================
# Signal factory
# =========================

def compute_base_signal(df, variant):
    if variant == "baseline":
        df["signal"] = df["net_sentiment"]

    elif variant == "v22":
        df = compute_v22_signal(df)

    elif variant == "v23":
        df = compute_v23_signal(df)

    elif variant == "v24":
        df = compute_v24_signal(df)

    elif variant == "v25":
        df = compute_v25_signal(df)

    elif variant == "v26":
        df = compute_v26_signal(df)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return df


def apply_mode(df, mode, shift):
    df = df.copy()

    if mode == "baseline":
        pass

    elif mode == "shifted":
        df["signal"] = df["signal"].shift(shift)

    elif mode == "shuffled":
        df["signal"] = np.random.permutation(df["signal"].values)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--modes", default="baseline,shifted,shuffled")
    parser.add_argument("--shifts", default="1,5")

    args = parser.parse_args()

    setup_logging()

    df = load_data(args.data)

    variants = args.variants.split(",")
    modes = args.modes.split(",")
    shifts = [int(s) for s in args.shifts.split(",")]

    for variant in variants:
        base_df = compute_base_signal(df.copy(), variant)

        for mode in modes:
            if mode == "shifted":
                for shift in shifts:
                    d = apply_mode(base_df.copy(), mode, shift)
                    d = d.dropna(subset=["signal"])
                    d["position"] = compute_positions(d["signal"])

                    label = f"{variant} | shifted({shift})"
                    evaluate_by_year(d, label)

            else:
                d = apply_mode(base_df.copy(), mode, 0)
                d = d.dropna(subset=["signal"])
                d["position"] = compute_positions(d["signal"])

                label = f"{variant} | {mode}"
                evaluate_by_year(d, label)


if __name__ == "__main__":
    main()
