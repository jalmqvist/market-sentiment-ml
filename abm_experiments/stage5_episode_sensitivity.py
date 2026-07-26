"""
stage5_episode_sensitivity.py — Stage 5.2: Episode Extractor Sensitivity
=========================================================================
Determines whether the H3 freq/duration gap identified in Stages 3-4
(simulated freq/1k=58-70 vs target 45-56, median_dur~3 vs target~4,
reversal gradient=0 on all configs) is:

  A. DEFINITION SENSITIVITY: gap closes when the episode extractor is
     recalibrated to the ABM output sentiment distribution rather than
     the empirical one (different amplitude distributions => different
     operative thresholds).

  B. STRUCTURAL: gap persists regardless of extractor parameterisation,
     indicating a hard property of the persistence+decay+shock mechanism
     at the current calibrated point.

Method
------
  1. Run the ABM at the confirmed anchor config (vol_t80_f30_cd10) for
     20 seeds x 3 pairs = 60 runs. Cache all sentiment series.

  2. For each run, apply 9 extractor configurations (post-hoc, no ABM
     re-run required):
       - extreme_threshold_pct in {65, 70, 75}  (applied to the ABM
         output distribution of that run, NOT to empirical data)
       - young_boundary_bars in {6, 8, 10}
       - mature_boundary = young_boundary x 3  (preserves empirical ratio)

  3. Aggregate episode statistics across seeds per pair per extractor
     config. Report freq/1k, median_dur, rev_young, rev_mature, grad%.

  4. Auto-classify:
       DEFINITION_SENSITIVITY if any combo achieves:
         freq in [40,60] AND median_dur >= 3.5 AND rev_young > rev_mature
         in >= 50% of runs.
       STRUCTURAL otherwise.

Key design:
  - ABM runs exactly 60 times (cached). Extractor varies post-hoc.
  - extreme_threshold is computed per-run from ABM output distribution,
    not from empirical data.
  - mature_boundary = young_boundary x 3 preserves the empirical ratio
    (young=8, mature=24 in calibration artifact).

Anchor config (confirmed Stage 4):
  shock_trigger=volatility, thresh=0.80, frac=0.30, cooldown=10
  anchor=0.25, beta=0.02, steps=1500, seeds 1-20

Output:
  - Per-config summary printed to stdout
  - JSON: stage5_episode_sensitivity_summary.json
  - Verdict: DEFINITION_SENSITIVITY or STRUCTURAL

Constraints:
  - Single file, no modifications to research/abm/
  - Imports from sweep_with_shocks and episode_utils only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from abm_experiments.sweep_with_shocks import (
        CALIBRATED_PARAMS,
        run_abm_series_with_shocks,
    )
except ImportError as exc:
    raise ImportError(
        f"Cannot import from abm_experiments/sweep_with_shocks.py.\n"
        f"Ensure sweep_with_shocks.py is in abm_experiments/ and "
        f"REPO_ROOT={REPO_ROOT}\n{exc}"
    ) from exc

try:
    from abm_experiments.episode_utils import (
        extract_consensus_episodes,
        episode_summary,
    )
except ImportError as exc:
    raise ImportError(
        f"Cannot import from abm_experiments/episode_utils.py.\n{exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANCHOR_SHOCK_PARAMS: Dict = {
    "shock_enable":        True,
    "shock_trigger":       "volatility",
    "shock_vol_threshold": 0.80,
    "shock_fraction":      0.30,
    "shock_direction":     "price",
    "shock_cooldown":      10,
    "shock_period":        50,
}

THRESHOLD_PCTS:    List[int] = [65, 70, 75]
YOUNG_BOUNDARIES:  List[int] = [6, 8, 10]
MATURE_RATIO:      float     = 3.0

# H3 auto-classification thresholds
TARGET_FREQ_LO:    float = 40.0
TARGET_FREQ_HI:    float = 60.0
TARGET_MEDIAN_DUR: float = 3.5    # conservative floor (empirical ~4)
GRAD_FRAC_MIN:     float = 0.50   # fraction of runs needing young > mature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".3f") -> str:
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(f):
        return "—"
    return format(f, fmt)


def _safe_mean(vals: List) -> float:
    clean = [float(v) for v in vals
             if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else math.nan


def _compute_threshold(sentiment: np.ndarray, pct: int) -> float:
    """
    Compute extreme_threshold from this run's ABM output distribution.
    Recalibrates extractor to the ABM's own amplitude, not empirical data.
    """
    return float(np.percentile(np.abs(sentiment), pct))


# ---------------------------------------------------------------------------
# Step 1: Run ABM and cache sentiment series
# ---------------------------------------------------------------------------

def collect_sentiment_series(
        pairs:        List[str],
        params:       Dict,
        shock_params: Dict,
        runs:         int,
        steps:        int,
        seed_base:    int,
        verbose:      bool,
) -> Dict[str, List[np.ndarray]]:
    """
    Run ABM for all pairs x seeds. Return {pair: [sentiment_array, ...]}.

    ABM is run exactly once per seed/pair. The cached series are reused
    across all 9 extractor configurations — no redundant ABM computation.
    """
    cache: Dict[str, List[np.ndarray]] = {p: [] for p in pairs}

    for pair in pairs:
        print(f"\n[ABM] Caching sentiment for pair={pair}  ({runs} seeds) ...")
        for run_idx in range(runs):
            seed = seed_base + run_idx
            try:
                sentiment, n_shocks = run_abm_series_with_shocks(
                    steps=steps,
                    seed=seed,
                    params=params,
                    shock_params=shock_params,
                    pair=pair,
                    verbose=False,
                )
                cache[pair].append(np.asarray(sentiment, dtype=float))
                if verbose:
                    print(f"  seed={seed}  shocks={n_shocks}  len={len(sentiment)}")
            except Exception as exc:
                warnings.warn(
                    f"pair={pair} seed={seed} FAILED: {exc}", stacklevel=2
                )

        n_ok = len(cache[pair])
        print(f"  Cached {n_ok}/{runs} runs.")

    return cache


# ---------------------------------------------------------------------------
# Step 2: Apply one extractor config to cached series for one pair
# ---------------------------------------------------------------------------

def apply_extractor_config(
        sentiment_list:  List[np.ndarray],
        threshold_pct:   int,
        young_boundary:  int,
        mature_boundary: int,
        steps:           int,
        pair:            str,
) -> Dict:
    """
    Apply one (threshold_pct, young_boundary) extractor configuration to all
    cached sentiment series for one pair.

    Returns aggregated episode statistics across runs.
    """
    freq_list:      List[float] = []
    dur_list:       List[float] = []
    rev_y_list:     List[float] = []
    rev_m_list:     List[float] = []
    censor_list:    List[float] = []
    thresh_list:    List[float] = []
    n_ep_list:      List[int]   = []
    grad_list:      List[bool]  = []

    for sentiment in sentiment_list:
        threshold = _compute_threshold(sentiment, threshold_pct)
        thresh_list.append(threshold)

        episodes = extract_consensus_episodes(
            sentiment,
            extreme_threshold=threshold,
            min_episode_steps=2,
            pair=pair,
        )

        summary = episode_summary(
            episodes,
            n_total_steps=steps,
            young_boundary=young_boundary,
            mature_boundary=mature_boundary,
        )

        freq_list.append(summary["episode_frequency_per_1000_steps"])
        n_ep_list.append(summary["episode_count"])
        censor_list.append(summary["censoring_rate"])

        med = summary["median_episode_duration_steps"]
        dur_list.append(med if med is not None else math.nan)

        ry = summary["reversal_rate_young"]
        rm = summary["reversal_rate_mature"]

        if ry is not None:
            rev_y_list.append(ry)
        if rm is not None:
            rev_m_list.append(rm)
        if ry is not None and rm is not None:
            grad_list.append(ry > rm)

    mean_freq    = _safe_mean(freq_list)
    mean_dur     = _safe_mean(dur_list)
    mean_rev_y   = _safe_mean(rev_y_list)
    mean_rev_m   = _safe_mean(rev_m_list)
    mean_censor  = _safe_mean(censor_list)
    mean_thresh  = _safe_mean(thresh_list)
    mean_n_ep    = _safe_mean([float(x) for x in n_ep_list])
    grad_frac    = float(sum(grad_list) / len(grad_list)) if grad_list else math.nan

    return {
        "pair":            pair,
        "threshold_pct":   threshold_pct,
        "young_boundary":  young_boundary,
        "mature_boundary": mature_boundary,
        "mean_threshold":  mean_thresh,
        "mean_n_episodes": mean_n_ep,
        "mean_freq_per_1k": mean_freq,
        "mean_median_dur": mean_dur,
        "mean_rev_young":  mean_rev_y,
        "mean_rev_mature": mean_rev_m,
        "mean_censoring":  mean_censor,
        "grad_frac":       grad_frac,
        "n_runs":          len(sentiment_list),
    }


# ---------------------------------------------------------------------------
# Step 3: Auto-classify verdict
# ---------------------------------------------------------------------------

def classify_result(rows: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Return DEFINITION_SENSITIVITY or STRUCTURAL, plus list of hit rows.

    A hit requires all three criteria simultaneously:
      - freq/1k in [TARGET_FREQ_LO, TARGET_FREQ_HI]
      - mean_median_dur >= TARGET_MEDIAN_DUR
      - grad_frac >= GRAD_FRAC_MIN  (majority of runs: young > mature)
    """
    hits = []
    for row in rows:
        freq_ok = TARGET_FREQ_LO <= row["mean_freq_per_1k"] <= TARGET_FREQ_HI
        dur_ok  = (
            not math.isnan(row["mean_median_dur"])
            and row["mean_median_dur"] >= TARGET_MEDIAN_DUR
        )
        grad_ok = (
            not math.isnan(row["grad_frac"])
            and row["grad_frac"] >= GRAD_FRAC_MIN
        )
        if freq_ok and dur_ok and grad_ok:
            hits.append(row)

    verdict = "DEFINITION_SENSITIVITY" if hits else "STRUCTURAL"
    return verdict, hits


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
        all_rows: List[Dict],
        pairs:    List[str],
        verdict:  str,
        hits:     List[Dict],
        runs:     int,
        steps:    int,
) -> None:
    sep  = "=" * 110
    sep2 = "-" * 110

    print(f"\n{sep}")
    print(f"  STAGE 5.2 — EPISODE EXTRACTOR SENSITIVITY")
    print(
        f"  Anchor: vol_t80_f30_cd10  |  Runs/pair: {runs}  |  Steps: {steps}  "
        f"|  anchor=0.25  beta=0.02"
    )
    print(
        f"  Grid: pct ∈ {THRESHOLD_PCTS}  yng ∈ {YOUNG_BOUNDARIES}  "
        f"mtr = yng×{MATURE_RATIO:.0f}"
    )
    print(
        f"  H3 targets: freq ∈ [{TARGET_FREQ_LO:.0f},{TARGET_FREQ_HI:.0f}]  "
        f"dur ≥ {TARGET_MEDIAN_DUR}  grad_frac ≥ {GRAD_FRAC_MIN:.0f}  "
        f"(* = hit)"
    )
    print(sep)

    hdr = (
        f"  {'Pair':<10} {'pct':>4} {'yng':>4} {'mtr':>4}  "
        f"{'thresh':>7}  {'n_ep':>6}  {'freq/1k':>7}  {'med_dur':>7}  "
        f"{'rev_y':>6}  {'rev_m':>6}  {'grad%':>6}  {'censor':>7}  {'':>4}"
    )
    div = (
        f"  {'-'*10} {'-'*4} {'-'*4} {'-'*4}  "
        f"{'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  "
        f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}"
    )
    print(hdr)
    print(div)

    hit_keys = {
        (r["pair"], r["threshold_pct"], r["young_boundary"])
        for r in hits
    }

    for pair in pairs:
        pair_rows = [r for r in all_rows if r["pair"] == pair]
        # Sort by threshold_pct then young_boundary
        pair_rows.sort(key=lambda r: (r["threshold_pct"], r["young_boundary"]))

        for row in pair_rows:
            is_hit = (
                row["pair"],
                row["threshold_pct"],
                row["young_boundary"],
            ) in hit_keys

            print(
                f"  {row['pair']:<10} {row['threshold_pct']:>4} "
                f"{row['young_boundary']:>4} {row['mature_boundary']:>4}  "
                f"{_fmt(row['mean_threshold'], '.1f'):>7}  "
                f"{_fmt(row['mean_n_episodes'], '.1f'):>6}  "
                f"{_fmt(row['mean_freq_per_1k'], '.1f'):>7}  "
                f"{_fmt(row['mean_median_dur'], '.2f'):>7}  "
                f"{_fmt(row['mean_rev_young'],  '.3f'):>6}  "
                f"{_fmt(row['mean_rev_mature'],  '.3f'):>6}  "
                f"{_fmt(row['grad_frac'],        '.2f'):>6}  "
                f"{_fmt(row['mean_censoring'],   '.3f'):>7}  "
                f"{'*' if is_hit else '':>4}"
            )

        print(div)

    # --- Verdict block ---
    print(f"\n  VERDICT: {verdict}")
    print(f"  H3 targets: freq ∈ [{TARGET_FREQ_LO:.0f},{TARGET_FREQ_HI:.0f}]  "
          f"med_dur >= {TARGET_MEDIAN_DUR}  grad_frac >= {GRAD_FRAC_MIN:.0f}")

    if verdict == "DEFINITION_SENSITIVITY":
        print(f"\n  Gap closes under {len(hits)} extractor config(s):")
        for h in hits:
            print(
                f"    + {h['pair']}  pct={h['threshold_pct']}  "
                f"yng={h['young_boundary']}  mtr={h['mature_boundary']}  "
                f"freq={_fmt(h['mean_freq_per_1k'],'.1f')}  "
                f"dur={_fmt(h['mean_median_dur'],'.2f')}  "
                f"grad_frac={_fmt(h['grad_frac'],'.2f')}"
            )
        print(
            "\n  Interpretation: H3 freq/duration gap is an extractor "
            "definition artefact.\n"
            "  The ABM generates episode structure consistent with the "
            "empirical BSVE targets\n"
            "  when thresholds are recalibrated to the ABM output "
            "distribution.\n"
            "  Recommended next step: lock recalibrated threshold values "
            "and re-run H3\n"
            "  validation with updated extractor parameters."
        )
    else:
        print(
            "\n  Interpretation: H3 freq/duration gap is STRUCTURAL.\n"
            "  The persistence+decay+shock mechanism at anchor=0.25, "
            "beta=0.02 generates\n"
            "  short-duration threshold-exit episodes regardless of "
            "extractor parameterisation.\n"
            "  Most episodes exit before reaching young_boundary, "
            "preventing the reversal\n"
            "  gradient from operating. This is a permanent known "
            "limitation of the current\n"
            "  calibrated point. The programme is closed at L3 "
            "(predictive structure confirmed).\n"
            "  L1/L2 (episode statistics, hazard structure) remain "
            "open for future work."
        )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
        all_rows:   List[Dict],
        verdict:    str,
        hits:       List[Dict],
        pairs:      List[str],
        runs:       int,
        steps:      int,
        params:     Dict,
        output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "mode":           "stage5_episode_sensitivity",
        "verdict":        verdict,
        "runs_per_pair":  runs,
        "steps":          steps,
        "anchor_strength": params["anchor_strength"],
        "beta":           params["beta"],
        "shock_params":   ANCHOR_SHOCK_PARAMS,
        "extractor_grid": {
            "threshold_pcts":   THRESHOLD_PCTS,
            "young_boundaries": YOUNG_BOUNDARIES,
            "mature_ratio":     MATURE_RATIO,
        },
        "classification_criteria": {
            "target_freq_lo":    TARGET_FREQ_LO,
            "target_freq_hi":    TARGET_FREQ_HI,
            "target_median_dur": TARGET_MEDIAN_DUR,
            "grad_frac_min":     GRAD_FRAC_MIN,
        },
        "hits":  hits,
        "pairs": pairs,
        "rows":  all_rows,
    }

    out_file = output_dir / "stage5_episode_sensitivity_summary.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[output] {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage 5.2: Determine whether the H3 freq/duration gap is "
            "extractor definition sensitivity or a structural property."
        )
    )
    p.add_argument(
        "--bsve-states-path", type=str, required=True,
        help="Path to BSVE augmented dataset CSV (for price series)",
    )
    p.add_argument(
        "--pairs", nargs="+",
        default=["usd-jpy", "eur-jpy", "gbp-jpy"],
        help="FX pairs to run (default: all three JPY pairs)",
    )
    p.add_argument(
        "--runs", type=int, default=20,
        help="Seeds per pair (default: 20, matches Stages 3-4)",
    )
    p.add_argument(
        "--steps", type=int, default=1500,
        help="ABM steps per run (default: 1500, matches Stages 3-4)",
    )
    p.add_argument(
        "--seed", type=int, default=1,
        help="Base seed; run i uses seed+i (default: 1)",
    )
    p.add_argument(
        "--anchor-strength", type=float,
        default=CALIBRATED_PARAMS["anchor_strength"],
        help="anchor_strength override (default: calibrated 0.25)",
    )
    p.add_argument(
        "--beta", type=float,
        default=CALIBRATED_PARAMS["beta"],
        help="beta/decay_volatility_scale override (default: calibrated 0.02)",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="abm_experiments/results/stage5",
        help="Directory for JSON output",
    )
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    params = CALIBRATED_PARAMS.copy()
    params["anchor_strength"] = args.anchor_strength
    params["beta"]            = args.beta

    output_dir = Path(args.output_dir)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Stage 5.2 — Episode Extractor Sensitivity")
    print(f"  Pairs: {args.pairs}  |  Runs/pair: {args.runs}  "
          f"|  Steps: {args.steps}")
    print(f"  Extractor grid: "
          f"{len(THRESHOLD_PCTS) * len(YOUNG_BOUNDARIES)} configs")
    print(f"  ABM runs (total): "
          f"{len(args.pairs) * args.runs}  (cached, not repeated)")
    print(sep)

    # Step 1: collect ABM sentiment series (60 runs total)
    cache = collect_sentiment_series(
        pairs=args.pairs,
        params=params,
        shock_params=ANCHOR_SHOCK_PARAMS,
        runs=args.runs,
        steps=args.steps,
        seed_base=args.seed,
        verbose=args.verbose,
    )

    # Step 2: apply extractor grid post-hoc
    all_rows: List[Dict] = []

    print(f"\n[extractor] Applying {len(THRESHOLD_PCTS) * len(YOUNG_BOUNDARIES)}"
          f" extractor configs across {len(args.pairs)} pairs ...")

    for threshold_pct in THRESHOLD_PCTS:
        for young_boundary in YOUNG_BOUNDARIES:
            mature_boundary = int(round(young_boundary * MATURE_RATIO))
            for pair in args.pairs:
                series = cache.get(pair, [])
                if not series:
                    warnings.warn(
                        f"No cached series for pair={pair}, skipping.",
                        stacklevel=2,
                    )
                    continue

                row = apply_extractor_config(
                    sentiment_list=series,
                    threshold_pct=threshold_pct,
                    young_boundary=young_boundary,
                    mature_boundary=mature_boundary,
                    steps=args.steps,
                    pair=pair,
                )
                all_rows.append(row)

                if args.verbose:
                    print(
                        f"  pct={threshold_pct}  yng={young_boundary}  "
                        f"mtr={mature_boundary}  pair={pair}  "
                        f"freq={_fmt(row['mean_freq_per_1k'],'.1f')}  "
                        f"dur={_fmt(row['mean_median_dur'],'.2f')}  "
                        f"grad={_fmt(row['grad_frac'],'.2f')}"
                    )

    # Step 3: classify
    verdict, hits = classify_result(all_rows)

    # Step 4: report
    print_summary(
        all_rows=all_rows,
        pairs=args.pairs,
        verdict=verdict,
        hits=hits,
        runs=args.runs,
        steps=args.steps,
    )

    # Step 5: save
    save_results(
        all_rows=all_rows,
        verdict=verdict,
        hits=hits,
        pairs=args.pairs,
        runs=args.runs,
        steps=args.steps,
        params=params,
        output_dir=output_dir,
    )

    print(f"[done] Stage 5.2 complete.  Verdict: {verdict}")
    print(f"       Output: {output_dir / 'stage5_episode_sensitivity_summary.json'}\n")


if __name__ == "__main__":
    main()
