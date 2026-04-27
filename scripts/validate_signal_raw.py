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
