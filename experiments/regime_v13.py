"""
EVENT-BASED signal pipeline — Version 13

Upgrade over V12:
→ Learns DIRECTION per context (contrarian vs continuation)

Core idea:
Each context decides whether to:
    - fade sentiment (contrarian)
    - follow sentiment (continuation)
based on TRAIN data only.

Everything else (event definition, context selection, ranking) unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

import config as cfg
from utils.io import read_csv
from utils.validation import parse_timestamps, require_columns

logger = logging.getLogger(__name__)

TARGET_COL = "ret_48b"

_REQUIRED_COLS = [
    TARGET_COL,
    "net_sentiment",
    "abs_sentiment",
    "extreme_streak_70",
    "trend_strength_48b",
]

_EVENT_ABS_SENTIMENT_MIN = 70
_EVENT_STREAK_MIN = 2

_DEFAULT_TOP_FRAC = 0.2
_MIN_CONTEXT_EVENTS = 30
_MIN_CONTEXT_SHARPE = 0.02

_ROLLING_VOL_WINDOW = 48

# =========================
# Logging
# =========================

def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

# =========================
# Data
# =========================

def load_data(path):
    df = read_csv(path, required_columns=["time"] + _REQUIRED_COLS)
    df = parse_timestamps(df, "time")
    df["year"] = df["time"].dt.year
    return df

# =========================
# Core features
# =========================

def compute_event_flag(df):
    return (
        (df["abs_sentiment"] >= _EVENT_ABS_SENTIMENT_MIN)
        & (df["extreme_streak_70"] >= _EVENT_STREAK_MIN)
    ).astype(int)

def compute_raw_score(df):
    raw = (
        (df["abs_sentiment"] / 100.0) ** 2
        - 0.5 * (df["net_sentiment"] * df["trend_strength_48b"]).abs()
        + 0.3 * np.log1p(df["extreme_streak_70"])
    )
    return raw.replace([np.inf, -np.inf], np.nan)

def compute_rolling_vol(df):
    if "pair" in df.columns:
        return df.groupby("pair")["net_sentiment"].transform(
            lambda x: x.rolling(_ROLLING_VOL_WINDOW, min_periods=24).std()
        )
    return df["net_sentiment"].rolling(_ROLLING_VOL_WINDOW, min_periods=24).std()

# =========================
# Context helpers
# =========================

def _tertile(series):
    s = series.dropna()
    q33 = s.quantile(1/3)
    q67 = s.quantile(2/3)
    if q67 <= q33:
        q67 = q33 + 1e-10
    return q33, q67

def _bucket(series, q33, q67, labels):
    out = pd.Series(labels[1], index=series.index)
    out[series < q33] = labels[0]
    out[series >= q67] = labels[2]
    out[series.isna()] = "unknown"
    return out

def assign_context(df, vq33, vq67, tq33, tq67, sq33, sq67):
    v = _bucket(df["rolling_vol"], vq33, vq67, ("low","mid","high"))
    t = _bucket(df["trend_strength_48b"], tq33, tq67, ("down","flat","up"))
    s = _bucket(df["abs_sentiment"], sq33, sq67, ("low","mid","high"))
    return v + "_" + t + "_" + s

# =========================
# NEW: Context stats with direction
# =========================

def compute_context_stats(train_events):

    stats = {}

    for ctx, g in train_events.groupby("context_key"):

        n = len(g)
        if n < 2:
            continue

        score = g["score_norm"].values
        sent = g["net_sentiment"].values
        ret = g[TARGET_COL].values

        contrarian_pnl = (-np.sign(sent) * score * ret)
        continuation_pnl = (+np.sign(sent) * score * ret)

        def sharpe(x):
            s = np.std(x)
            return np.mean(x) / s if s > 1e-10 else np.nan

        sharpe_c = sharpe(contrarian_pnl)
        sharpe_f = sharpe(continuation_pnl)

        if np.isnan(sharpe_c) and np.isnan(sharpe_f):
            continue

        if sharpe_c >= sharpe_f:
            direction = "contrarian"
            best_sharpe = sharpe_c
        else:
            direction = "continuation"
            best_sharpe = sharpe_f

        stats[ctx] = {
            "n": n,
            "sharpe": best_sharpe,
            "direction": direction,
        }

    return stats

def select_contexts(stats, min_n, min_sharpe):
    selected = {}
    rejected = {}

    for k, v in stats.items():
        if v["n"] >= min_n and v["sharpe"] >= min_sharpe:
            selected[k] = v
        else:
            rejected[k] = v

    return selected, rejected

# =========================
# Walk-forward
# =========================

def walk_forward(df, top_frac, min_n, min_sharpe):

    df = df.copy().sort_values("time")

    df["rolling_vol"] = compute_rolling_vol(df)
    df["score_raw"] = compute_raw_score(df)
    df["is_event"] = compute_event_flag(df)

    years = sorted(df["year"].unique())
    rows = []

    for i in range(2, len(years)):
        test_year = years[i]

        train = df[df["year"] < test_year].dropna()
        test = df[df["year"] == test_year].dropna()

        if train.empty or test.empty:
            continue

        # normalize
        mu = train["score_raw"].mean()
        sd = train["score_raw"].std() or 1.0

        train["score_norm"] = (train["score_raw"] - mu) / sd
        test["score_norm"] = (test["score_raw"] - mu) / sd

        # context thresholds
        vq33, vq67 = _tertile(train["rolling_vol"])
        tq33, tq67 = _tertile(train["trend_strength_48b"])
        sq33, sq67 = _tertile(train["abs_sentiment"])

        train["context_key"] = assign_context(train, vq33,vq67,tq33,tq67,sq33,sq67)
        test["context_key"] = assign_context(test, vq33,vq67,tq33,tq67,sq33,sq67)

        train_events = train[train["is_event"] == 1]

        stats = compute_context_stats(train_events)
        selected, rejected = select_contexts(stats, min_n, min_sharpe)

        if not selected:
            continue

        # diagnostics
        n_contra = sum(1 for v in selected.values() if v["direction"]=="contrarian")
        n_follow = sum(1 for v in selected.values() if v["direction"]=="continuation")

        logger.info(
            f"[{test_year}] contexts={len(selected)} | contrarian={n_contra} | continuation={n_follow}"
        )

        test_events = test[test["is_event"] == 1]
        test_events = test_events[test_events["context_key"].isin(selected)]

        if test_events.empty:
            continue

        selected_idx = []

        for ctx, g in test_events.groupby("context_key"):
            n = len(g)
            k = max(1, int(np.ceil(top_frac * n)))
            idx = g["score_norm"].nlargest(k).index
            selected_idx.extend(idx)

        sel = test_events.loc[selected_idx]

        dirs = sel["context_key"].map(lambda x: selected[x]["direction"])

        base_dir = np.where(
            dirs == "contrarian",
            -np.sign(sel["net_sentiment"]),
            +np.sign(sel["net_sentiment"]),
        )

        pnl = base_dir * sel["score_norm"].values * sel[TARGET_COL].values

        sharpe = np.mean(pnl) / np.std(pnl) if len(pnl)>1 else np.nan
        hit = np.mean(pnl > 0)

        rows.append({
            "year": test_year,
            "n": len(sel),
            "sharpe": sharpe,
            "hit_rate": hit,
        })

    return pd.DataFrame(rows)

# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--min-sharpe", type=float, default=0.02)
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()

    _setup_logging(args.log_level)

    df = load_data(args.data)
    require_columns(df, _REQUIRED_COLS)

    res = walk_forward(
        df,
        args.top_frac,
        args.min_n,
        args.min_sharpe,
    )

    print("\n=== RESULTS ===")
    print(res)
    print("\nMEAN SHARPE:", res["sharpe"].mean())

if __name__ == "__main__":
    main()
