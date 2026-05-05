from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _infer_ts_from_bestpath_name(path: Path) -> str | None:
    # abm_sweep_vol_bestpath_{pair}_{version}_{timestamp}.csv
    m = re.match(r"abm_sweep_vol_bestpath_.+_(\d{8}T\d{6}Z)\.csv$", path.name)
    return m.group(1) if m else None


def _load_config_for_ts(logs_dir: Path, pair: str | None, ts: str | None) -> dict | None:
    if ts is None:
        return None

    # Prefer the experiment’s config snapshot created next to the log:
    # abm_sweep-vol-{pair}_{ts}.json
    if pair is not None:
        cfg_path = logs_dir / f"abm_sweep-vol-{pair}_{ts}.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())

    # Fallback: search any JSON containing this ts (still minimal; only logs dir)
    matches = list(logs_dir.glob(f"*{ts}.json"))
    if matches:
        return json.loads(matches[0].read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs", help="Directory containing logs/ CSVs")
    ap.add_argument("--pair", default=None, help="Filter by pair (e.g. eur-usd)")
    ap.add_argument("--version", default=None, help="Filter by dataset version (e.g. 1.2.0)")
    ap.add_argument("--sat", type=float, default=90.0, help="Saturation threshold for abs(net_sentiment)")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    bestpaths = sorted(logs_dir.glob("abm_sweep_vol_bestpath_*.csv"))
    if not bestpaths:
        raise SystemExit(f"No bestpath CSVs found in {logs_dir}")

    rows = []
    for p in bestpaths:
        # Parse pair/version from filename if possible
        # abm_sweep_vol_bestpath_{pair}_{version}_{timestamp}.csv
        parts = p.stem.split("_")
        # ["abm","sweep","vol","bestpath", "{pair}", "{version}", "{timestamp}"] BUT pair has dash
        # so we recover from the right: ..._{pair}_{version}_{ts}
        ts = _infer_ts_from_bestpath_name(p)
        version = None
        pair = None
        if ts is not None:
            # remove prefix and ts
            prefix = "abm_sweep_vol_bestpath_"
            mid = p.name[len(prefix) :].rsplit(f"_{ts}.csv", 1)[0]
            # mid ends with _{version} and starts with {pair}
            # safest: split on last '_' for version
            if "_" in mid:
                pair, version = mid.rsplit("_", 1)

        if args.pair is not None and pair is not None and pair != args.pair:
            continue
        if args.version is not None and version is not None and version != args.version:
            continue

        df = pd.read_csv(p)
        if "net_sentiment" not in df.columns or "abs_sentiment" not in df.columns:
            # skip unexpected format
            continue

        s = df["net_sentiment"].to_numpy(dtype=float)
        abs_s = df["abs_sentiment"].to_numpy(dtype=float)

        hit_idx = np.where(abs_s >= args.sat)[0]
        first_hit_step = None
        if len(hit_idx) > 0 and "step" in df.columns:
            first_hit_step = int(df.loc[int(hit_idx[0]), "step"])

        sign = np.sign(s)
        sign_flips = int(((sign[1:] * sign[:-1]) < 0).sum()) if len(sign) > 1 else 0

        cfg = _load_config_for_ts(logs_dir, pair, ts)

        rows.append(
            {
                "ts": ts,
                "pair": pair,
                "version": version,
                "steps": int(cfg.get("steps")) if isinstance(cfg, dict) and "steps" in cfg else len(df),
                "volatility_scale": float(cfg.get("volatility_scale"))
                if isinstance(cfg, dict) and "volatility_scale" in cfg
                else float("nan"),
                "decay_base": float(cfg.get("decay_base")) if isinstance(cfg, dict) and "decay_base" in cfg else float("nan"),
                "decay_volatility_scale": float(cfg.get("decay_volatility_scale"))
                if isinstance(cfg, dict) and "decay_volatility_scale" in cfg
                else float("nan"),
                "decay_clip_max": float(cfg.get("decay_clip_max"))
                if isinstance(cfg, dict) and "decay_clip_max" in cfg
                else float("nan"),
                "pct_time_saturated": float((abs_s >= args.sat).mean()),
                "first_hit_step": first_hit_step,
                "sign_flips": sign_flips,
                "autocorr_lag1": _autocorr_lag1(s),
                "mean_sent": float(s.mean()),
                "std_sent": float(s.std()),
                "bestpath_csv": str(p),
            }
        )

    if not rows:
        raise SystemExit("No matching bestpaths after filtering (or unexpected CSV format).")

    out = pd.DataFrame(rows)

    # sort: by decay scale then timestamp
    sort_cols = [c for c in ["decay_volatility_scale", "volatility_scale", "ts"] if c in out.columns]
    out = out.sort_values(sort_cols)

    # Print in a compact, copy/paste-friendly way
    with pd.option_context("display.max_rows", 500, "display.max_columns", 500, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
