"""abm_experiments/validate_episode_utils.py
================================================
Stage 1.2 — Ground-truth validation of episode_utils.py against the
frozen Reactive-JPY BSVE calibration artifact.

Runs extract_consensus_episodes() on real JPY sentiment data and
compares the resulting diagnostics against the values stored in the
calibration artifact. If this passes, episode_utils.py can be used
as a reliable ABM calibration target.

Usage:
    python abm_experiments/validate_episode_utils.py \
        --artifact bsve/calibration_artifacts/<artifact_name>.json \
        --version 1.6.1

Expected result: all per-pair checks PASSED within 10% tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm_experiments.episode_utils import (
    load_calibration_artifact,
    validate_against_artifact,
)
from research.abm.run_abm import _load_real_data


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate episode_utils.py against BSVE calibration artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--artifact",
        required=True,
        help="Path to frozen Reactive-JPY BSVE calibration artifact JSON.",
    )
    p.add_argument(
        "--version",
        default="1.6.1",
        help="Dataset version to load real sentiment data from.",
    )
    p.add_argument(
        "--variant",
        default="core",
        choices=["full", "core", "extended"],
    )
    p.add_argument(
        "--pairs",
        nargs="+",
        default=["usd-jpy", "eur-jpy", "gbp-jpy"],
        help="JPY pairs to validate (use dataset slug format, e.g. usd-jpy).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance for numeric comparisons (default 10%%).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Load calibration artifact
    print(f"[Stage 1.2] Loading calibration artifact: {args.artifact}")
    artifact = load_calibration_artifact(args.artifact)

    ontology = artifact.get("ontology_id", "unknown")
    cal_version = artifact.get("ontology_version", "unknown")
    dataset_version = artifact.get("dataset_version", "unknown")
    outcome = artifact.get("outcome", "unknown")

    print(f"  ontology_id      : {ontology}")
    print(f"  ontology_version : {cal_version}")
    print(f"  dataset_version  : {dataset_version}")
    print(f"  outcome          : {outcome}")

    if outcome != "success":
        print(f"[Stage 1.2] ERROR: artifact outcome is '{outcome}', expected 'success'.")
        sys.exit(1)

    thresholds = artifact.get("thresholds", {})
    diagnostics = artifact.get("diagnostics", {})
    extreme_threshold = thresholds.get("extreme_threshold_net_pct")
    young_boundary = thresholds.get("young_boundary_bars")
    mature_boundary = thresholds.get("mature_boundary_bars")

    print(f"\n  extreme_threshold : {extreme_threshold}")
    print(f"  young_boundary    : {young_boundary} bars")
    print(f"  mature_boundary   : {mature_boundary} bars")

    # Load real sentiment data
    print(f"\n[Stage 1.2] Loading dataset v{args.version} ({args.variant})...")
    df, dataset_path = _load_real_data(args.version, args.variant)
    print(f"  dataset path : {dataset_path}")
    print(f"  total rows   : {len(df)}")

    # Validate per-pair episode counts individually (artifact stores these)
    print("\n[Stage 1.2] Per-pair episode count validation:")
    artifact_count_per_pair = diagnostics.get("episode_count_per_pair", {})
    per_pair_failures = []

    all_episodes = []
    n_total_steps = 0

    cal_start = artifact.get("calibration_window_start")
    cal_end = artifact.get("calibration_window_end")

    for pair in args.pairs:
        pair_data = (
            df[df["pair"] == pair]
            .copy()
            .sort_values("entry_time")
            .reset_index(drop=True)
        )
        if pair_data.empty:
            print(f"  WARNING: no data for pair={pair}, skipping.")
            continue

        if cal_start and cal_end:
            mask = (
                (pair_data["entry_time"] >= pd.Timestamp(cal_start))
                & (pair_data["entry_time"] <= pd.Timestamp(cal_end))
            )
            pair_data = pair_data[mask].reset_index(drop=True)

        net_sentiment = pair_data["net_sentiment"].to_numpy()
        n_total_steps += len(net_sentiment)

        from abm_experiments.episode_utils import extract_consensus_episodes
        episodes = extract_consensus_episodes(
            net_sentiment,
            extreme_threshold=extreme_threshold,
            min_episode_steps=2,
            pair=pair,
        )
        all_episodes.extend(episodes)

        emp_count = artifact_count_per_pair.get(pair)
        sim_count = len(episodes)
        match = "✓" if emp_count is None or sim_count == emp_count else "✗"
        print(f"  {match} {pair}: extracted={sim_count}  artifact={emp_count}")
        if emp_count is not None and sim_count != emp_count:
            per_pair_failures.append(
                f"{pair}: extracted={sim_count}, artifact={emp_count}"
            )

    # Pooled validation against artifact pooled diagnostics
    print("\n[Stage 1.2] Pooled validation (all pairs combined):")
    from abm_experiments.episode_utils import (
        episode_summary,
        compute_hazard_by_maturity,
    )

    summary = episode_summary(
        all_episodes,
        n_total_steps=n_total_steps,
        young_boundary=young_boundary,
        mature_boundary=mature_boundary,
    )

    pooled_failures = []
    tolerance = args.tolerance

    def _check(label, actual, expected):
        if expected is None or actual is None:
            print(f"  - {label}: actual={actual}  expected={expected}  SKIP")
            return
        rel_err = abs(actual - expected) / (abs(expected) + 1e-12)
        ok = rel_err <= tolerance
        status = "✓" if ok else "✗"
        print(
            f"  {status} {label}: actual={actual}  expected={expected}"
            f"  rel_err={rel_err:.3f}"
        )
        if not ok:
            pooled_failures.append(
                f"{label}: actual={actual}, expected={expected}, "
                f"rel_err={rel_err:.3f}"
            )

    emp_total = diagnostics.get("episode_count")
    _check("episode_count_total", len(all_episodes), emp_total)
    _check("censoring_rate", summary["censoring_rate"], diagnostics.get("censoring_rate"))
    _check(
        "median_episode_duration",
        summary["median_episode_duration_steps"],
        diagnostics.get("median_episode_duration_bars"),
    )
    _check(
        "reversal_rate_young",
        summary["reversal_rate_young"],
        diagnostics.get("reversal_rate_young"),
    )
    _check(
        "reversal_rate_mature",
        summary["reversal_rate_mature"],
        diagnostics.get("reversal_rate_mature"),
    )

    # Survival counts: artifact stores pooled totals — compare pooled
    artifact_survival = diagnostics.get("survival_counts", {})
    for t in (8, 16, 24, 32, 48):
        _check(
            f"survival_count_{t}",
            float(summary["survival_counts"].get(str(t), 0)),
            float(artifact_survival.get(str(t), 0))
            if str(t) in artifact_survival else None,
        )

    # Final summary
    all_failures = per_pair_failures + pooled_failures
    all_passed = len(all_failures) == 0

    print("\n" + "=" * 60)
    print("[Stage 1.2] Validation Summary")
    print("=" * 60)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED")
        print("  episode_utils.py is validated against the BSVE artifact.")
        print("  Safe to proceed to Stage 2.")
    else:
        print("  ✗ FAILURES:")
        for f in all_failures:
            print(f"    {f}")
        print("\n  Note: median_duration discrepancy of 1 bar between")
        print("  artifact dataset version and current version is acceptable.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
