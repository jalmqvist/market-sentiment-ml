#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd


# =========================
# Logging
# =========================
def setup_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


# =========================
# Load
# =========================
def load_data(path):
    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["time"])

    df = df.sort_values("time").reset_index(drop=True)

    df["ret"] = pd.to_numeric(df["ret_48b"], errors="coerce")
    df["signal"] = pd.to_numeric(df["net_sentiment"], errors="coerce")

    df = df.dropna(subset=["signal", "ret"])

    return df


# =========================
# Metrics
# =========================
def evaluate(df, label):
    pnl = df["position"] * df["ret"]

    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    hit = (pnl > 0).mean()

    print(f"\n--- {label} ---")
    print(f"Sharpe: {sharpe:.4f} | Hit: {hit:.4f} | N: {len(df)}")

    return sharpe


# =========================
# Expanding zscore (causal)
# =========================
def expanding_zscore(series):
    mean = series.expanding(min_periods=50).mean().shift(1)
    std = series.expanding(min_periods=50).std().shift(1)
    return (series - mean) / (std + 1e-6)


# =========================
# Cross-sectional rank
# =========================
def cs_rank(df, col):
    return df.groupby("time")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )


# =========================
# Top-1 selection (SAFE)
# =========================
def select_top(group):
    z = group["z"]

    # all NaN → no position
    if z.isna().all():
        return pd.Series(0.0, index=group.index, dtype=float)

    idx = z.abs().idxmax()

    out = pd.Series(0.0, index=group.index, dtype=float)

    val = z.loc[idx]
    if pd.notna(val):
        out.loc[idx] = float(np.tanh(val))

    return out


# =========================
# Tests
# =========================
def run_tests(df):

    base = df.copy()

    # ---------------------
    # T0: raw sign
    # ---------------------
    t0 = base.copy()
    t0["position"] = np.sign(t0["signal"])
    evaluate(t0, "T0: raw sign")

    # ---------------------
    # T1: tanh(signal)
    # ---------------------
    t1 = base.copy()
    t1["position"] = np.tanh(t1["signal"])
    evaluate(t1, "T1: tanh(signal)")

    # ---------------------
    # T2: expanding zscore only
    # ---------------------
    t2 = base.copy()
    t2["z"] = expanding_zscore(t2["signal"])
    t2 = t2.dropna(subset=["z"])

    t2["position"] = np.tanh(t2["z"])
    evaluate(t2, "T2: expanding zscore only")

    # ---------------------
    # T3: cross-sectional only
    # ---------------------
    t3 = base.copy()
    t3["z"] = cs_rank(t3, "signal")
    t3["position"] = np.tanh(t3["z"])
    evaluate(t3, "T3: cross-sectional only")

    # ---------------------
    # T4: full pipeline
    # ---------------------
    t4 = base.copy()

    t4["z"] = expanding_zscore(t4["signal"])
    t4 = t4.dropna(subset=["z"])

    t4["z"] = cs_rank(t4, "z")

    # IMPORTANT: avoid pandas misalignment
    positions = []
    for _, group in t4.groupby("time"):
        positions.append(select_top(group))

    t4["position"] = pd.concat(positions).sort_index()

    evaluate(t4, "T4: full pipeline (top-1)")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    setup_logging(args.log_level)

    df = load_data(args.data)

    run_tests(df)


if __name__ == "__main__":
    main()
