#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _setup_logging():
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dl_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
    print(f"Logging to: {log_file}", flush=True)


_setup_logging()
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------


def _compute_metrics(df):
    pnl = df["position"] * df["ret_48b"]

    if len(pnl) == 0:
        return 0.0, 0.0

    sharpe = pnl.mean() / pnl.std() if pnl.std() > 1e-12 else 0.0
    hit = (pnl > 0).mean()
    return sharpe, hit


def _run(df, label):
    sharpe, hit = _compute_metrics(df)
    density = (df["signal"] != 0).mean()

    logger.info(
        "[%s] n=%d sharpe=%.4f hit=%.4f density=%.4f",
        label,
        len(df),
        sharpe,
        hit,
        density,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--shifts", default="1,5")
    args = parser.parse_args()

    shifts = [int(x) for x in args.shifts.split(",")]

    df = pd.read_csv(args.predictions)
    df["signal"] = np.sign(df["prediction"])

    # baseline
    base = df.copy()
    base["position"] = base["signal"]
    _run(base, "baseline")

    # shifts
    for s in shifts:
        shifted = df.copy()
        shifted["signal"] = shifted["signal"].shift(s)
        shifted = shifted.dropna(subset=["signal"])
        shifted["position"] = shifted["signal"]
        _run(shifted, f"shift({s})")

    # shuffled
    shuffled = df.copy()
    shuffled["signal"] = np.random.permutation(shuffled["signal"])
    shuffled["position"] = shuffled["signal"]
    _run(shuffled, "shuffled")


if __name__ == "__main__":
    main()
