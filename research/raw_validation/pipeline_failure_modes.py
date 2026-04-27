# Legacy experiment — not part of current validated approach\n#!/usr/bin/env python3

import numpy as np
import pandas as pd


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
# Helpers
# =========================
def zscore(series):
    mean = series.expanding(min_periods=50).mean().shift(1)
    std = series.expanding(min_periods=50).std().shift(1)
    return (series - mean) / (std + 1e-6)


def evaluate(df, label):
    pnl = df["position"] * df["ret"]
    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    print(f"{label:<20} Sharpe={sharpe:.4f}")
    return sharpe


# =========================
# CLEAN PIPELINE
# =========================
def clean_pipeline(df):
    d = df.copy()

    d["z"] = zscore(d["signal"])
    d = d.dropna(subset=["z"])

    d["position"] = np.tanh(d["z"])

    return d


# =========================
# FAILURE MODE 1:
# Index misalignment
# =========================
def misaligned_apply(df):
    d = df.copy()

    d["z"] = zscore(d["signal"])
    d = d.dropna(subset=["z"])

    # BAD: apply without preserving index order
    pos = (
        d.groupby("time")
        .apply(lambda g: np.tanh(g["z"]))
        .reset_index(drop=True)   # <-- breaks alignment
    )

    d["position"] = pos  # misaligned assignment

    return d


# =========================
# FAILURE MODE 2:
# Lookahead via future selection
# =========================
def lookahead_selection(df):
    d = df.copy()

    d["z"] = zscore(d["signal"])
    d = d.dropna(subset=["z"])

    positions = []

    for _, g in d.groupby("time"):
        # BAD: use future return to pick best
        idx = g["ret"].idxmax()   # <-- illegal lookahead

        out = pd.Series(0.0, index=g.index)
        out.loc[idx] = 1.0

        positions.append(out)

    d["position"] = pd.concat(positions).sort_index()

    return d


# =========================
# FAILURE MODE 3:
# Time shuffle leakage
# =========================
def shuffled_time(df):
    d = df.copy()

    # shuffle time → destroys causality
    d["time"] = np.random.permutation(d["time"].values)

    d["z"] = zscore(d["signal"])
    d = d.dropna(subset=["z"])

    d["position"] = np.tanh(d["z"])

    return d


# =========================
# MAIN
# =========================
def main(path):
    df = load_data(path)

    print("\n=== PIPELINE FAILURE MODE TEST ===\n")

    evaluate(clean_pipeline(df), "CLEAN")

    evaluate(misaligned_apply(df), "MISALIGNED")

    evaluate(lookahead_selection(df), "LOOKAHEAD")

    evaluate(shuffled_time(df), "TIME_SHUFFLE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)

    args = parser.parse_args()

    main(args.data)
