#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# =========================
# Logging
# =========================

def setup_logging():
    repo_root = Path(__file__).resolve().parent
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validate_signal_dual_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.FileHandler(log_file)]
    )

    print(f"Logging to: {log_file}")


# =========================
# Load
# =========================

def load_data(path):
    df = pd.read_csv(path)

    required = ["entry_time", "ret_48b", "net_sentiment"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time"])

    df = df.sort_values("entry_time").reset_index(drop=True)
    df["year"] = df["entry_time"].dt.year

    return df


# =========================
# Metrics
# =========================

def compute_positions(signal):
    return np.sign(signal).astype(float)


def compute_metrics(df):
    pnl = df["position"] * df["ret_48b"]

    if len(pnl) == 0:
        return {"n": 0, "sharpe": 0.0, "hit": 0.0}

    mean = pnl.mean()
    std = pnl.std()

    sharpe = mean / std if std > 1e-12 else 0.0
    hit = (pnl > 0).mean()

    return {"n": len(pnl), "sharpe": sharpe, "hit": hit}


def evaluate(df):
    sharpes = []

    for year, g in df.groupby("year"):
        if year < 2020:
            continue
        m = compute_metrics(g)
        sharpes.append(m["sharpe"])

    return np.mean(sharpes) if len(sharpes) else 0.0


# =========================
# Signals (minimal set)
# =========================

def compute_baseline(df):
    df = df.copy()
    df["signal"] = df["net_sentiment"]
    return df


def compute_price(df):
    df = df.copy()
    df["past_ret"] = df["ret_48b"].shift(48)
    mom = df["past_ret"].rolling(48).mean().shift(1)
    df["signal"] = np.sign(mom)
    return df


def compute_v27(df):
    df = df.copy()

    df["past_ret"] = df["ret_48b"].shift(48)
    mom = df["past_ret"].rolling(48).mean().shift(1)
    price_signal = np.sign(mom)

    upper = df["net_sentiment"].expanding(200).quantile(0.9).shift(1)
    lower = df["net_sentiment"].expanding(200).quantile(0.1).shift(1)

    signal = np.zeros(len(df))
    signal[(price_signal > 0) & (df["net_sentiment"] < lower)] = 1
    signal[(price_signal < 0) & (df["net_sentiment"] > upper)] = -1

    df["signal"] = signal
    return df


def compute_v28(df, beta=0.5):
    df = df.copy()

    # price
    df["past_ret"] = df["ret_48b"].shift(48)
    mom = df["past_ret"].rolling(48).mean().shift(1)
    price = np.sign(mom)

    # sentiment residual (simple version)
    sent_mean = df["net_sentiment"].rolling(500).mean().shift(1)
    sent_resid = df["net_sentiment"] - sent_mean

    # combine
    signal = price + beta * sent_resid

    df["signal"] = signal
    return df


def compute_signal(df, variant):
    if variant == "baseline":
        return compute_baseline(df)
    elif variant == "price":
        return compute_price(df)
    elif variant == "v27":
        return compute_v27(df)
    elif variant == "v28":
        return compute_v28(df)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# =========================
# Modes
# =========================

def apply_mode(df, mode, shift):
    df = df.copy()

    if mode == "baseline":
        pass
    elif mode == "shifted":
        df["signal"] = df["signal"].shift(shift)
    elif mode == "shuffled":
        df["signal"] = np.random.permutation(df["signal"].values)
    else:
        raise ValueError(mode)

    return df


# =========================
# Core comparison logic
# =========================

def run_one(df, variant, mode, shift):
    d = compute_signal(df, variant)
    d = apply_mode(d, mode, shift)

    d = d.dropna(subset=["signal"])
    d["position"] = compute_positions(d["signal"])

    return evaluate(d)


def compare(full_df, core_df, variant, mode, shift):
    s_full = run_one(full_df, variant, mode, shift)
    s_core = run_one(core_df, variant, mode, shift)

    diff = s_core - s_full

    print(
        f"{variant:8s} | {mode:8s} | shift={shift:<2d} | "
        f"FULL={s_full:+.4f} | CORE={s_core:+.4f} | Δ={diff:+.4f}"
    )

    # simple interpretation
    if abs(diff) > 0.05:
        print("   ⚠️ LARGE divergence → dataset sensitivity")

    if s_core > s_full + 0.02:
        print("   🟢 Signal cleaner in CORE")

    if s_full > s_core + 0.02:
        print("   🔴 Likely contamination in FULL")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--core-data", required=True)

    parser.add_argument("--variants", default="baseline,price,v27,v28")
    parser.add_argument("--modes", default="baseline,shifted,shuffled")
    parser.add_argument("--shifts", default="1,5")

    args = parser.parse_args()

    setup_logging()

    full_df = load_data(args.data)
    core_df = load_data(args.core_data)

    variants = args.variants.split(",")
    modes = args.modes.split(",")
    shifts = [int(s) for s in args.shifts.split(",")]

    print("\n=== DUAL DATASET COMPARISON ===\n")

    for v in variants:
        for m in modes:
            if m == "shifted":
                for s in shifts:
                    compare(full_df, core_df, v, m, s)
            else:
                compare(full_df, core_df, v, m, 0)


if __name__ == "__main__":
    main()
