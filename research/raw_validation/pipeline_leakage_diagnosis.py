# Legacy experiment — not part of current validated approach\n#!/usr/bin/env python3

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

    df["signal"] = pd.to_numeric(df["net_sentiment"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret_48b"], errors="coerce")

    df = df.dropna(subset=["signal", "ret"])

    return df


# =========================
# Expanding zscore (causal)
# =========================
def expanding_zscore(series):
    mean = series.expanding(min_periods=50).mean().shift(1)
    std = series.expanding(min_periods=50).std().shift(1)
    return (series - mean) / (std + 1e-6)


# =========================
# Evaluation
# =========================
def evaluate(df):
    pnl = df["position"] * df["ret"]
    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    hit = (pnl > 0).mean()
    return sharpe, hit


# =========================
# Core pipeline (SAFE version)
# =========================
def run_pipeline(df):
    d = df.copy()

    # --- Step 1: causal feature ---
    d["z"] = expanding_zscore(d["signal"])
    d = d.dropna(subset=["z"])

    # --- Step 2: cross-sectional normalization ---
    d["z"] = d.groupby("time")["z"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # --- Step 3: selection (TOP-1 per time) ---
    positions = []

    for _, g in d.groupby("time"):
        g = g.copy()

        # track order inside group
        g["time_rank"] = np.arange(len(g))

        if g["z"].isna().all():
            g["position"] = 0.0
            positions.append(g)
            continue

        idx = g["z"].abs().idxmax()

        g["position"] = 0.0
        g.loc[idx, "position"] = float(np.tanh(g.loc[idx, "z"]))

        positions.append(g)

    d = pd.concat(positions).sort_index()

    return d


# =========================
# Diagnostics
# =========================
def diagnostics(df):
    selected = df[df["position"] != 0]

    print("\n=== DIAGNOSTICS ===")

    if len(selected) == 0:
        print("No selections made.")
        return

    # --- Lookahead proxy: position in group ---
    avg_rank = selected["time_rank"].mean()
    max_rank = df.groupby("time")["time_rank"].max().mean()

    print(f"Avg selected rank: {avg_rank:.2f}")
    print(f"Avg max rank:      {max_rank:.2f}")

    if avg_rank > 0.8 * max_rank:
        print("⚠️  Suspicious: selecting late rows in group (possible lookahead)")

    # --- Future return bias ---
    df["ret_rank"] = df.groupby("time")["ret"].rank(pct=True)

    selected = df[df["position"] != 0]
    avg_ret_rank = selected["ret_rank"].mean()

    print(f"Avg future return percentile: {avg_ret_rank:.3f}")

    if avg_ret_rank > 0.6:
        print("🚨 Strong leakage: selecting high future returns")


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

    result = run_pipeline(df)

    sharpe, hit = evaluate(result)

    print("\n=== PIPELINE RESULT ===")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Hit:    {hit:.4f}")
    print(f"N:      {len(result)}")

    diagnostics(result)


if __name__ == "__main__":
    main()
