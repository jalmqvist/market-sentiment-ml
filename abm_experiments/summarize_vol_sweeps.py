#!/usr/bin/env python3
"""
Summarize ABM volatility sweep runs from logs/ artifacts.

For each abm_sweep_vol_{pair}_{version}_{timestamp}.csv in logs/:
  1) Extract 'rolling_vol diagnostics: ...' line from the matching log file
  2) Extract 'Best: ...' line from the matching log file
  3) Print the top N rows of the CSV

Matching strategy:
- Parse timestamp from the CSV filename (final underscore-separated token)
- Find a log file in logs/ whose filename ends with that timestamp and contains 'sweep-vol'
- Config JSON is expected to be the same stem as the log file with .json suffix
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


CSV_RE = re.compile(r"^abm_sweep_vol_(?P<pair>[^_]+)_(?P<version>[^_]+)_(?P<ts>[^.]+)\.csv$")


@dataclass
class RunArtifacts:
    csv_path: Path
    timestamp: str
    pair: str
    version: str
    log_path: Optional[Path] = None
    json_path: Optional[Path] = None


def _find_log_for_timestamp(log_dir: Path, timestamp: str) -> Optional[Path]:
    # Heuristic: any .log file containing 'sweep-vol' and ending with the timestamp token
    candidates = sorted(log_dir.glob(f"*{timestamp}.log"))
    sweep_vol = [p for p in candidates if "sweep-vol" in p.name]
    if sweep_vol:
        return sweep_vol[0]
    return candidates[0] if candidates else None


def _extract_lines(log_path: Path) -> tuple[Optional[str], Optional[str]]:
    diag_line = None
    best_line = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if diag_line is None and "rolling_vol diagnostics:" in line:
                diag_line = line.rstrip("\n")
            if best_line is None and "Best:" in line:
                best_line = line.rstrip("\n")
            if diag_line is not None and best_line is not None:
                break
    return diag_line, best_line


def _collect_runs(log_dir: Path, pair: Optional[str], version: Optional[str]) -> list[RunArtifacts]:
    runs: list[RunArtifacts] = []
    for csv_path in sorted(log_dir.glob("abm_sweep_vol_*.csv")):
        m = CSV_RE.match(csv_path.name)
        if not m:
            continue
        p = m.group("pair")
        v = m.group("version")
        ts = m.group("ts")
        if pair is not None and p != pair:
            continue
        if version is not None and v != version:
            continue
        runs.append(RunArtifacts(csv_path=csv_path, timestamp=ts, pair=p, version=v))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs", help="Directory containing ABM artifacts (default: logs)")
    ap.add_argument("--pair", default=None, help="Filter by pair (e.g. eur-usd)")
    ap.add_argument("--version", default=None, help="Filter by dataset version (e.g. 1.2.0)")
    ap.add_argument("--head", type=int, default=5, help="Number of CSV rows to print (default: 5)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir.resolve()}")

    runs = _collect_runs(log_dir, pair=args.pair, version=args.version)
    if not runs:
        raise SystemExit(f"No abm_sweep_vol CSVs found in {log_dir.resolve()}")

    for run in runs:
        run.log_path = _find_log_for_timestamp(log_dir, run.timestamp)
        if run.log_path is not None:
            run.json_path = run.log_path.with_suffix(".json")

        print("=" * 100)
        print(f"CSV : {run.csv_path}")
        print(f"TS  : {run.timestamp}")
        print(f"PAIR: {run.pair}   VERSION: {run.version}")

        if run.log_path is None or not run.log_path.exists():
            print("LOG : (not found)")
        else:
            print(f"LOG : {run.log_path}")
            diag_line, best_line = _extract_lines(run.log_path)
            print("1) rolling_vol diagnostics:")
            print(f"   {diag_line if diag_line else '(not found in log)'}")
            print("2) Best:")
            print(f"   {best_line if best_line else '(not found in log)'}")

        # Optional: show alpha from config JSON if present
        if run.json_path is not None and run.json_path.exists():
            try:
                import json
                payload = json.loads(run.json_path.read_text(encoding="utf-8"))
                alpha = payload.get("volatility_scale", None)
                print(f"   volatility_scale (from config): {alpha}")
            except Exception:
                pass

        print(f"3) CSV head({args.head}):")
        df = pd.read_csv(run.csv_path)
        print(df.head(args.head).to_string(index=False))

    print("=" * 100)


if __name__ == "__main__":
    main()
