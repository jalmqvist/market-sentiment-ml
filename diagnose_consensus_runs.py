#!/usr/bin/env python
"""
Diagnostic script: compare calibration-style episode extraction with the
Behavioral Surface engine output on the same dataset.

Usage
-----
    python diagnose_consensus_runs.py \
        --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
        --calibration-artifact bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
        [--pairs USDJPY EURJPY GBPJPY] \
        [--max-gap 1h]

The script prints:
    1. Calibration-replay episode statistics (ground truth).
    2. Behavioral Surface engine statistics.
    3. A per-observation trace of every episode boundary, annotated with
       the boundary reason.

Exit code 0 always — this is a diagnostic, not a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Attempt imports; fail gracefully so the script is self-describing.
# ---------------------------------------------------------------------------

try:
    from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter
    from bsve.calibration.calibration_contract import load_calibration_artifact
    from bsve.calibration.jpy_maturity_calibration import (
        compute_extreme_threshold,
        extract_consensus_lifecycles,
    )
    from bsve.state_machine.engine import (
        BehavioralSurfaceEngine,
        _normalize_crowd_side,
    )
    from bsve.state_machine.plugins.reactive_jpy import ReactiveJPYPlugin
    _IMPORTS_OK = True
except ImportError as _import_exc:
    _IMPORTS_OK = False
    _import_exc_msg = str(_import_exc)


# ---------------------------------------------------------------------------
# Calibration-replay helpers
# ---------------------------------------------------------------------------


def _run_calibration_replay(
    dataset: pd.DataFrame,
    *,
    pairs: list[str],
    pair_col: str,
    timestamp_col: str,
    extreme_threshold: float,
) -> dict[str, Any]:
    """Run the calibration's extract_consensus_lifecycles on each pair."""
    all_lengths: list[int] = []
    total_episodes = 0

    for pair in pairs:
        pair_data = dataset[dataset[pair_col] == pair].copy()
        pair_data = pair_data.sort_values(timestamp_col).rename(
            columns={"net_sentiment": "sentiment_net", timestamp_col: "entry_time"}
        )
        lifecycles = extract_consensus_lifecycles(
            pair_data, pair, extreme_threshold, min_episode_bars=1
        )
        for lc in lifecycles:
            all_lengths.append(lc.duration_bars)
        total_episodes += len(lifecycles)

    if not all_lengths:
        return {
            "total_episodes": 0,
            "max_episode_length": 0,
            "episodes_ge_8": 0,
            "episodes_ge_24": 0,
            "episodes_ge_48": 0,
        }

    return {
        "total_episodes": total_episodes,
        "max_episode_length": max(all_lengths),
        "episodes_ge_8": sum(1 for l in all_lengths if l >= 8),
        "episodes_ge_24": sum(1 for l in all_lengths if l >= 24),
        "episodes_ge_48": sum(1 for l in all_lengths if l >= 48),
    }


# ---------------------------------------------------------------------------
# Instrumented engine trace
# ---------------------------------------------------------------------------


def _boundary_reason(
    *,
    prior: Any,
    timestamp: pd.Timestamp,
    consensus_active: bool,
    max_gap: pd.Timedelta,
) -> str:
    """Return a human-readable reason for an episode boundary."""
    if prior is None:
        return "first_observation"
    gap = timestamp - prior["last_timestamp"]
    if gap > max_gap:
        return f"timestamp_gap ({gap})"
    if consensus_active != prior["last_consensus_active"]:
        if consensus_active:
            return "consensus_entered_extreme_region"
        return "consensus_left_extreme_region"
    return "unknown"


def _run_instrumented_surface(
    dataset: pd.DataFrame,
    *,
    pairs: list[str],
    pair_col: str,
    timestamp_col: str,
    extreme_threshold: float,
    max_gap: pd.Timedelta,
    print_boundaries: bool = True,
    max_boundary_rows: int = 50,
) -> dict[str, Any]:
    """
    Replay the Behavioral Surface engine with boundary-reason annotation.

    Returns aggregate statistics and prints per-boundary trace rows.
    """
    # Sort exactly as the engine does.
    working = dataset[[pair_col, timestamp_col, "net_sentiment", "crowd_side"]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working = working.sort_values(
        [pair_col, timestamp_col], kind="mergesort"
    ).reset_index(drop=True)

    pair_state: dict[str, dict] = {}
    episode_counter = 0
    boundary_rows: list[dict] = []
    episode_lengths: dict[str, int] = {}
    max_maturity = 0

    for _, row in working.iterrows():
        pair = str(row[pair_col]).strip()
        timestamp = pd.Timestamp(row[timestamp_col])
        crowd_side = _normalize_crowd_side(row.get("crowd_side"))
        net_sentiment = float(row["net_sentiment"])
        consensus_active = abs(net_sentiment) >= extreme_threshold

        prior = pair_state.get(pair)

        if prior is None:
            reason = "first_observation"
            boundary = True
        else:
            gap = timestamp - prior["last_timestamp"]
            gap_detected = gap > max_gap
            extreme_changed = consensus_active != prior["last_consensus_active"]
            boundary = gap_detected or extreme_changed
            reason = _boundary_reason(
                prior=prior,
                timestamp=timestamp,
                consensus_active=consensus_active,
                max_gap=max_gap,
            )

        if boundary:
            episode_counter += 1
            episode_id = f"{pair}:{episode_counter:08d}"
            maturity = 1 if consensus_active else 0
        else:
            episode_id = prior["episode_id"]
            maturity = prior["last_maturity"] + 1 if consensus_active else 0

        max_maturity = max(max_maturity, maturity)
        episode_lengths[episode_id] = episode_lengths.get(episode_id, 0) + 1

        if boundary:
            boundary_rows.append(
                {
                    "timestamp": timestamp,
                    "pair": pair,
                    "net_sentiment": round(net_sentiment, 3),
                    "crowd_side": crowd_side,
                    "consensus_active": consensus_active,
                    "previous_consensus_active": prior["last_consensus_active"] if prior else None,
                    "previous_episode_id": prior["episode_id"] if prior else None,
                    "boundary_reason": reason,
                }
            )

        pair_state[pair] = {
            "last_timestamp": timestamp,
            "last_consensus_active": consensus_active,
            "last_maturity": maturity,
            "episode_id": episode_id,
        }

    lengths = list(episode_lengths.values())
    stats = {
        "total_observations": len(working),
        "total_episodes": episode_counter,
        "max_episode_length": max(lengths) if lengths else 0,
        "max_maturity": max_maturity,
        "episodes_ge_8": sum(1 for l in lengths if l >= 8),
        "episodes_ge_24": sum(1 for l in lengths if l >= 24),
        "episodes_ge_48": sum(1 for l in lengths if l >= 48),
        "boundary_reason_counts": _count_reasons(boundary_rows),
    }

    if print_boundaries and boundary_rows:
        print(f"\n  First {min(max_boundary_rows, len(boundary_rows))} boundary events:\n")
        _print_boundary_table(boundary_rows[:max_boundary_rows])

    return stats


def _count_reasons(boundary_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in boundary_rows:
        r = row["boundary_reason"]
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _print_boundary_table(rows: list[dict]) -> None:
    headers = [
        "timestamp", "pair", "net_sentiment", "crowd_side",
        "consensus_active", "prev_consensus_active", "boundary_reason",
    ]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            key = "previous_consensus_active" if h == "prev_consensus_active" else h
            col_widths[h] = max(col_widths[h], len(str(row.get(key, ""))))

    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "  ".join("-" * col_widths[h] for h in headers)
    print("  " + header_line)
    print("  " + sep_line)
    for row in rows:
        cells = []
        for h in headers:
            key = "previous_consensus_active" if h == "prev_consensus_active" else h
            cells.append(str(row.get(key, "")).ljust(col_widths[h]))
        print("  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose BSVE episode construction vs calibration replay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default="data/output/1.5.1/master_research_dataset_core.csv",
        help="Path to the master research dataset (CSV or parquet)",
    )
    parser.add_argument(
        "--calibration-artifact",
        default="bsve/calibration_artifacts/reactive_jpy_calibration_v1.json",
        help="Path to the calibration artifact JSON",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["USDJPY", "EURJPY", "GBPJPY"],
        help="Pairs to analyse",
    )
    parser.add_argument(
        "--max-gap",
        default="1h",
        help="Maximum gap between consecutive observations before a new episode starts",
    )
    parser.add_argument(
        "--max-boundary-rows",
        type=int,
        default=50,
        help="Maximum number of boundary trace rows to print",
    )
    return parser.parse_args()


def main() -> None:
    if not _IMPORTS_OK:
        print(f"[ERROR] Import failed: {_import_exc_msg}", file=sys.stderr)
        print("        Install dependencies with: pip install -e .", file=sys.stderr)
        sys.exit(1)

    args = _parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        print(
            "        Verify --dataset-path points to the master research dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load dataset and calibration artifact
    # ------------------------------------------------------------------
    print(f"\n[BSVE Diagnosis] Loading dataset: {dataset_path}")
    adapter = MasterResearchDatasetAdapter.from_artifact(dataset_path)
    pair_col = adapter.config.pair_col
    timestamp_col = adapter.config.timestamp_col
    max_gap = pd.Timedelta(args.max_gap)

    artifact = load_calibration_artifact(
        args.calibration_artifact, strict=True
    )
    thresholds = artifact.get("thresholds", {})
    extreme_threshold = float(thresholds.get("extreme_threshold_net_pct", 70.0))

    print(f"[BSVE Diagnosis] Extreme threshold (from artifact): {extreme_threshold}")
    print(f"[BSVE Diagnosis] Max gap: {max_gap}")

    # ------------------------------------------------------------------
    # Resolve pairs
    # ------------------------------------------------------------------
    normalized_pairs = [adapter.normalize_pair(p) for p in args.pairs]
    normalized_pairs = sorted(set(normalized_pairs))
    print(f"[BSVE Diagnosis] Pairs: {normalized_pairs}")

    # ------------------------------------------------------------------
    # Load sentiment data
    # ------------------------------------------------------------------
    ds = adapter.get_sentiment_observations(
        pairs=normalized_pairs,
        columns=["net_sentiment", "crowd_side"],
    )
    if ds.empty:
        print("[ERROR] No observations found for the given pairs.", file=sys.stderr)
        sys.exit(1)

    print(f"[BSVE Diagnosis] Total observations: {len(ds)}\n")

    # ------------------------------------------------------------------
    # Calibration replay
    # ------------------------------------------------------------------
    print("=" * 70)
    print("CALIBRATION REPLAY (ground truth — abs(net_sentiment) >= threshold)")
    print("=" * 70)
    cal_stats = _run_calibration_replay(
        ds,
        pairs=normalized_pairs,
        pair_col=pair_col,
        timestamp_col=timestamp_col,
        extreme_threshold=extreme_threshold,
    )
    for k, v in cal_stats.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Instrumented behavioral surface engine
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BEHAVIORAL SURFACE ENGINE (instrumented)")
    print("=" * 70)
    eng_stats = _run_instrumented_surface(
        ds,
        pairs=normalized_pairs,
        pair_col=pair_col,
        timestamp_col=timestamp_col,
        extreme_threshold=extreme_threshold,
        max_gap=max_gap,
        print_boundaries=True,
        max_boundary_rows=args.max_boundary_rows,
    )
    print("\n  Aggregate statistics:")
    for k, v in eng_stats.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    metrics = [
        ("max_episode_length", "max_episode_length"),
        ("episodes_ge_8", "episodes_ge_8"),
        ("episodes_ge_24", "episodes_ge_24"),
        ("episodes_ge_48", "episodes_ge_48"),
    ]
    for cal_key, eng_key in metrics:
        cal_val = cal_stats.get(cal_key, "N/A")
        eng_val = eng_stats.get(eng_key, "N/A")
        match = "✓" if cal_val == eng_val else "✗"
        print(f"  {match}  {cal_key}: calibration={cal_val}  engine={eng_val}")

    print()


if __name__ == "__main__":
    main()
