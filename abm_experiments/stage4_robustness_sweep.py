"""
stage4_robustness_sweep.py — Stage 4: Robustness Sweep at vol_t80_f30_cd10
===========================================================================
Tests H4 robustness by bracketing the best Stage 3 config along two
perturbation dimensions:

  1. Cooldown sweep (thresh=0.80, frac=0.30):
       cd5   — higher shock density, probes over-shocking degradation
       cd10  — Stage 3 anchor (vol_t80_f30_cd10), included for continuity
       cd20  — Stage 3 lower bound (vol_t80_f30), included as reference

  2. Fraction sweep at cooldown=10 (thresh=0.80, cooldown=10):
       f30   — anchor (overlaps with cooldown sweep, deduplicated)
       f40   — fraction sensitivity probe
       f50   — at cd=10 (Stage 3 had f50 only at cd=20)

Total: 5 unique configs x 3 pairs x 20 seeds = 300 runs.

All parameters held at the calibrated point (anchor=0.25, beta=0.02)
except the two perturbation axes.

Output:
  - Per-config per-pair JSON: stage4_<config>_<pair>.json
  - Summary JSON:             stage4_robustness_summary.json
  - Summary printed to stdout in Stage 3 format for direct comparison

Constraints:
  - Single file, no modifications to research/abm/
  - Imports helpers from sweep_with_shocks (single-source-of-truth)
  - Calibrated parameter point: anchor=0.25, beta=0.02
  - Dataset version: 1.6.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List

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
        BSVE_STATES,
        H4_ORDER,
        STATE_ID_MAP,
        run_abm_series_with_shocks,
        load_bsve_dataset,
        align_abm_to_bsve,
        compute_state_correlations,
        test_h4_hypothesis,
        aggregate_runs,
        compute_episode_metrics,
        print_run_report,
    )
except ImportError as exc:
    raise ImportError(
        f"Cannot import from abm_experiments/sweep_with_shocks.py.\n"
        f"Ensure sweep_with_shocks.py is in abm_experiments/ and "
        f"REPO_ROOT={REPO_ROOT}\n{exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Stage 4 config matrix
# ---------------------------------------------------------------------------
# cd10_f30 is the Stage 3 anchor — included here for a self-contained,
# directly comparable run. cd20_f30 matches Stage 3 vol_t80_f30 exactly
# and serves as the lower-density reference point.

STAGE4_CONFIGS: List[Dict] = [
    {
        "name":                "cd5_f30",
        "label":               "cooldown=5  frac=0.30  (densification probe)",
        "shock_enable":        True,
        "shock_trigger":       "volatility",
        "shock_vol_threshold": 0.80,
        "shock_fraction":      0.30,
        "shock_direction":     "price",
        "shock_cooldown":      5,
        "shock_period":        50,
    },
    {
        "name":                "cd10_f30",
        "label":               "cooldown=10 frac=0.30  *** Stage 3 anchor ***",
        "shock_enable":        True,
        "shock_trigger":       "volatility",
        "shock_vol_threshold": 0.80,
        "shock_fraction":      0.30,
        "shock_direction":     "price",
        "shock_cooldown":      10,
        "shock_period":        50,
    },
    {
        "name":                "cd20_f30",
        "label":               "cooldown=20 frac=0.30  (Stage 3 lower bound ref)",
        "shock_enable":        True,
        "shock_trigger":       "volatility",
        "shock_vol_threshold": 0.80,
        "shock_fraction":      0.30,
        "shock_direction":     "price",
        "shock_cooldown":      20,
        "shock_period":        50,
    },
    {
        "name":                "cd10_f40",
        "label":               "cooldown=10 frac=0.40  (fraction probe)",
        "shock_enable":        True,
        "shock_trigger":       "volatility",
        "shock_vol_threshold": 0.80,
        "shock_fraction":      0.40,
        "shock_direction":     "price",
        "shock_cooldown":      10,
        "shock_period":        50,
    },
    {
        "name":                "cd10_f50",
        "label":               "cooldown=10 frac=0.50  (fraction upper bound)",
        "shock_enable":        True,
        "shock_trigger":       "volatility",
        "shock_vol_threshold": 0.80,
        "shock_fraction":      0.50,
        "shock_direction":     "price",
        "shock_cooldown":      10,
        "shock_period":        50,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_pair(pair: str) -> str:
    return pair.lower().replace("-", "").replace("/", "")


def _h4_short(verdict: Dict) -> str:
    supported = verdict.get("h4_supported")
    partial   = verdict.get("h4_partial_supported")
    if supported is None:
        return "INC"
    if supported:
        return "FULL"
    if partial:
        return "PARTIAL"
    return "NO"


def _fmt(val, fmt=".4f") -> str:
    """Format a float, returning '—' for nan/None."""
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(f):
        return "—"
    return format(f, fmt)


def _average_episode_metrics(metrics_list: List[Dict]) -> Dict:
    """Average episode metrics across runs; majority vote for booleans."""
    if not metrics_list:
        return {}
    result = {}
    for key in metrics_list[0].keys():
        vals = [
            m[key] for m in metrics_list
            if m[key] is not None
            and not (isinstance(m[key], float) and math.isnan(m[key]))
        ]
        if not vals:
            result[key] = math.nan
        elif isinstance(vals[0], bool):
            result[key] = sum(vals) > len(vals) / 2
        else:
            result[key] = float(np.mean(vals))
    return result


def _build_bsve_multi_pair(bsve_path: str, pairs: List[str]):
    """
    Load and concatenate BSVE data for all pairs.
    Attaches a normalised 'pair_norm' column for fast per-pair filtering.
    """
    import pandas as pd

    dfs = []
    for pair in pairs:
        try:
            df_pair = load_bsve_dataset(bsve_path, pair)
            df_pair["pair_norm"] = _norm_pair(pair)
            dfs.append(df_pair)
        except Exception as exc:
            warnings.warn(
                f"Could not load BSVE data for pair={pair}: {exc}",
                stacklevel=2,
            )

    if not dfs:
        raise ValueError("No BSVE data loaded for any pair.")

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def run_one_config(
        config:      Dict,
        bsve_all,               # multi-pair BSVE DataFrame
        params:      Dict,
        forward_col: str,
        pairs:       List[str],
        runs:        int,
        steps:       int,
        seed_base:   int,
        verbose:     bool,
) -> Dict:
    """
    Run one shock configuration across all pairs and all seeds.

    Returns a dict keyed by pair, each containing:
        aggregated_corrs, h4_verdict, episode_metrics_mean,
        episode_metrics_runs, n_shocks_mean, n_shocks_list
    """
    config_name = config["name"]

    # Strip display-only keys before passing to ABM runner
    shock_params = {
        k: v for k, v in config.items()
        if k not in ("name", "label")
    }

    result: Dict = {}

    for pair in pairs:
        pair_norm = _norm_pair(pair)
        pair_bsve = bsve_all[bsve_all["pair_norm"] == pair_norm].copy()

        if pair_bsve.empty:
            warnings.warn(
                f"[{config_name}] No BSVE rows for pair={pair}, skipping.",
                stacklevel=2,
            )
            continue

        if verbose:
            print(f"\n  config={config_name}  pair={pair}  "
                  f"shock=True  trigger={config['shock_trigger']}  "
                  f"frac={config['shock_fraction']}  "
                  f"cooldown={config['shock_cooldown']}")

        run_results:          List[Dict] = []
        episode_metrics_list: List[Dict] = []
        n_shocks_list:        List[int]  = []

        for run_idx in range(runs):
            seed = seed_base + run_idx
            if verbose:
                print(f"    [Run {run_idx + 1}/{runs}] seed={seed}")

            try:
                sentiment, n_shocks = run_abm_series_with_shocks(
                    steps=steps,
                    seed=seed,
                    params=params,
                    shock_params=shock_params,
                    pair=pair,
                    verbose=verbose,
                )
            except Exception as exc:
                warnings.warn(
                    f"[{config_name}] pair={pair} seed={seed} FAILED: {exc}",
                    stacklevel=2,
                )
                continue

            n_shocks_list.append(n_shocks)

            # Episode structure (H3 metrics)
            ep_meta = compute_episode_metrics(
                sentiment=sentiment,
                extreme_threshold_pct=70.0,
                young_boundary=8,
                mature_boundary=24,
                steps=len(sentiment),
            )
            episode_metrics_list.append(ep_meta)

            # H4 correlations
            aligned_df = align_abm_to_bsve(
                sentiment, pair_bsve, verbose=False
            )
            run_corrs = compute_state_correlations(
                aligned_df,
                forward_col=forward_col,
                sentiment_col="abm_net_sentiment",
                verbose=verbose,
            )
            run_results.append(run_corrs)

        if not run_results:
            warnings.warn(
                f"[{config_name}] pair={pair}: all runs failed, skipping.",
                stacklevel=2,
            )
            continue

        # Aggregate across seeds
        aggregated_corrs = aggregate_runs(run_results, states=BSVE_STATES)

        mean_corrs = {
            state: {
                "spearman_r": aggregated_corrs[state]["spearman_r_mean"],
                "pearson_r":  aggregated_corrs[state]["pearson_r_mean"],
                "n":          int(aggregated_corrs[state]["n_mean"]),
                "pearson_p":  math.nan,
                "spearman_p": math.nan,
            }
            for state in BSVE_STATES
        }
        h4_verdict = test_h4_hypothesis(mean_corrs, metric="spearman_r")

        episode_metrics_mean = _average_episode_metrics(episode_metrics_list)
        n_shocks_mean = float(np.mean(n_shocks_list)) if n_shocks_list else 0.0

        # Print per-pair report in Stage 3 format
        print_run_report(
            pair=pair,
            shock_params=shock_params,
            aggregated_corrs=aggregated_corrs,
            h4_verdict=h4_verdict,
            episode_metrics=episode_metrics_mean,
            runs=len(run_results),
            steps=steps,
            n_shocks_mean=n_shocks_mean,
        )

        result[pair] = {
            "aggregated_corrs":     aggregated_corrs,
            "h4_verdict":           h4_verdict,
            "episode_metrics_mean": episode_metrics_mean,
            "episode_metrics_runs": episode_metrics_list,
            "n_shocks_mean":        n_shocks_mean,
            "n_shocks_list":        n_shocks_list,
        }

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
        all_results: Dict,
        configs:     List[Dict],
        pairs:       List[str],
        runs:        int,
        steps:       int,
) -> None:
    sep = "=" * 108

    print(f"\n{sep}")
    print(f"  STAGE 4 ROBUSTNESS SWEEP SUMMARY")
    print(f"  Anchor: cd10_f30  |  Runs/config: {runs}  |  Steps: {steps}  "
          f"|  anchor=0.25  beta=0.02  thresh=0.80  trigger=volatility")
    print(sep)

    hdr = (
        f"  {'Config':<14} {'Pair':<10} {'Shocks/run':>10}  "
        f"{'freq/1k':>7}  {'med_dur':>7}  {'rev_grad':>8}  "
        f"{'|r|MATURING':>11}  {'|r|ENTRY':>8}  {'|r|MATURE':>9}  {'H4':>8}"
    )
    div = (
        f"  {'-' * 14} {'-' * 10} {'-' * 10}  "
        f"{'-' * 7}  {'-' * 7}  {'-' * 8}  "
        f"{'-' * 11}  {'-' * 8}  {'-' * 9}  {'-' * 8}"
    )
    print(hdr)
    print(div)

    # Track which configs produce FULL H4 for the conclusion block
    full_h4_hits: List[str] = []

    for cfg_dict in configs:
        cname = cfg_dict["name"]
        cfg_results = all_results.get(cname, {})

        for pair in pairs:
            pr = cfg_results.get(pair)
            anchor_marker = " *" if cname == "cd10_f30" else "  "

            if pr is None:
                print(
                    f"  {cname + anchor_marker:<14} {pair:<10} {'—':>10}  "
                    f"{'—':>7}  {'—':>7}  {'—':>8}  "
                    f"{'—':>11}  {'—':>8}  {'—':>9}  {'—':>8}"
                )
                continue

            ep = pr["episode_metrics_mean"]
            h4 = pr["h4_verdict"]
            ac = pr["aggregated_corrs"]
            nsm = pr["n_shocks_mean"]

            freq = ep.get("ep_freq_per_1000", math.nan)
            med_dur = ep.get("median_duration_bars", math.nan)
            rev_grad = ep.get("rev_gradient_correct", None)
            h4_str = _h4_short(h4)

            abs_mat = h4.get("abs_MATURING", math.nan)
            abs_ent = h4.get("abs_ENTRY", math.nan)
            abs_mat2 = h4.get("abs_MATURE", math.nan)

            rev_str = (
                "✓" if rev_grad is True
                else "✗" if rev_grad is False
                else "—"
            )

            if h4_str == "FULL":
                full_h4_hits.append(f"{cname}/{pair}")

            print(
                f"  {cname + anchor_marker:<14} {pair:<10} {nsm:>10.1f}  "
                f"{_fmt(freq, '.1f'):>7}  {_fmt(med_dur, '.2f'):>7}  "
                f"{rev_str:>8}  "
                f"{_fmt(abs_mat, '.4f'):>11}  {_fmt(abs_ent, '.4f'):>8}  "
                f"{_fmt(abs_mat2, '.4f'):>9}  {h4_str:>8}"
            )

    print(sep)

    # --- Conclusion block ---
    print(f"\n  (* = Stage 3 anchor config, included for continuity)")
    print(f"\n  FULL H4 results: {len(full_h4_hits)}")
    for hit in full_h4_hits:
        print(f"    + {hit}")

    if not full_h4_hits:
        print("    (none)")

    # Cooldown gradient: summarise |r|MATURING by cooldown across pairs
    print(f"\n  Cooldown gradient (|r|MATURING mean across pairs):")
    for cfg_dict in configs:
        cname = cfg_dict["name"]
        cfg_results = all_results.get(cname, {})
        mat_vals = []
        for pair in pairs:
            pr = cfg_results.get(pair)
            if pr is not None:
                v = pr["h4_verdict"].get("abs_MATURING", math.nan)
                if not math.isnan(v):
                    mat_vals.append(v)
        mean_mat = float(np.mean(mat_vals)) if mat_vals else math.nan
        anchor_str = " <- Stage 3 anchor" if cname == "cd10_f30" else ""
        print(f"    {cname:<14}  mean |r|MATURING = {_fmt(mean_mat, '.4f')}{anchor_str}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_results(
        all_results: Dict,
        configs: List[Dict],
        pairs: List[str],
        params: Dict,
        runs: int,
        steps: int,
        forward_col: str,
        output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cfg_dict in configs:
        cname = cfg_dict["name"]
        cfg_results = all_results.get(cname, {})

        for pair in pairs:
            pr = cfg_results.get(pair)
            if pr is None:
                continue

            # Per-config per-pair JSON (mirrors Stage 3 output schema)
            payload = {
                "mode": "stage4_robustness",
                "config": cname,
                "config_label": cfg_dict["label"],
                "pair": pair,
                "runs": runs,
                "steps": steps,
                "forward_col": forward_col,
                "anchor_strength": params["anchor_strength"],
                "beta": params["beta"],
                "shock_params": {
                    k: v for k, v in cfg_dict.items()
                    if k not in ("name", "label")
                },
                "n_shocks_mean": pr["n_shocks_mean"],
                "n_shocks_list": pr["n_shocks_list"],
                "aggregated_corrs": pr["aggregated_corrs"],
                "h4_verdict": pr["h4_verdict"],
                "episode_metrics_mean": pr["episode_metrics_mean"],
                "episode_metrics_runs": pr["episode_metrics_runs"],
            }

            out_file = output_dir / f"stage4_{cname}_{pair}.json"
            with open(out_file, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"[output] {out_file}")

            # Row for summary JSON
            h4 = pr["h4_verdict"]
            ep = pr["episode_metrics_mean"]
            summary_rows.append({
                "config": cname,
                "config_label": cfg_dict["label"],
                "pair": pair,
                "shock_cooldown": cfg_dict["shock_cooldown"],
                "shock_fraction": cfg_dict["shock_fraction"],
                "n_shocks_mean": pr["n_shocks_mean"],
                "freq_per_1000": ep.get("ep_freq_per_1000", math.nan),
                "median_duration": ep.get("median_duration_bars", math.nan),
                "rev_grad": ep.get("rev_gradient_correct", None),
                "abs_MATURING": h4.get("abs_MATURING", math.nan),
                "abs_ENTRY": h4.get("abs_ENTRY", math.nan),
                "abs_MATURE": h4.get("abs_MATURE", math.nan),
                "h4_supported": h4.get("h4_supported"),
                "h4_partial": h4.get("h4_partial_supported"),
                "h4_short": _h4_short(h4),
                "cautious": h4.get("cautious", False),
                "low_n_states": h4.get("low_n_states", []),
            })

    summary_payload = {
        "mode": "stage4_robustness_summary",
        "runs": runs,
        "steps": steps,
        "forward_col": forward_col,
        "anchor_strength": params["anchor_strength"],
        "beta": params["beta"],
        "configs": [c["name"] for c in configs],
        "pairs": pairs,
        "rows": summary_rows,
    }

    summary_file = output_dir / "stage4_robustness_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    print(f"[summary] {summary_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4: Robustness sweep bracketing the vol_t80_f30_cd10 "
            "anchor config along cooldown and fraction dimensions."
        )
    )
    p.add_argument(
        "--bsve-states-path", type=str, required=True,
        help="Path to BSVE augmented dataset CSV",
    )
    p.add_argument(
        "--pairs", nargs="+", default=["usd-jpy", "eur-jpy", "gbp-jpy"],
        help="FX pairs to run (default: all three JPY pairs)",
    )
    p.add_argument(
        "--runs", type=int, default=20,
        help="Number of seeds per config/pair (default: 20, matches Stage 3)",
    )
    p.add_argument(
        "--steps", type=int, default=1500,
        help="ABM steps per run (default: 1500, matches Stage 3)",
    )
    p.add_argument(
        "--seed", type=int, default=1,
        help="Base seed; run i uses seed + i (default: 1, matches Stage 3)",
    )
    p.add_argument(
        "--forward-horizon", type=int, default=24,
        help="Forward return horizon in bars (default: 24 → ret_24b)",
    )
    p.add_argument(
        "--anchor-strength", type=float,
        default=CALIBRATED_PARAMS["anchor_strength"],
        help="Override anchor_strength (default: calibrated 0.25)",
    )
    p.add_argument(
        "--beta", type=float,
        default=CALIBRATED_PARAMS["beta"],
        help="Override beta / decay_volatility_scale (default: calibrated 0.02)",
    )
    p.add_argument(
        "--configs", nargs="+",
        default=None,
        help=(
            "Subset of configs to run by name "
            "(default: all five). "
            "Choices: cd5_f30 cd10_f30 cd20_f30 cd10_f40 cd10_f50"
        ),
    )
    p.add_argument(
        "--output-dir", type=str,
        default="abm_experiments/results/stage4/robustness",
        help="Directory for JSON outputs",
    )
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = CALIBRATED_PARAMS.copy()
    params["anchor_strength"] = args.anchor_strength
    params["beta"] = args.beta

    forward_col = f"ret_{args.forward_horizon}b"
    output_dir = Path(args.output_dir)

    # Filter configs if --configs was supplied
    configs = STAGE4_CONFIGS
    if args.configs:
        valid_names = {c["name"] for c in STAGE4_CONFIGS}
        unknown = set(args.configs) - valid_names
        if unknown:
            raise ValueError(
                f"Unknown config name(s): {unknown}. "
                f"Valid: {sorted(valid_names)}"
            )
        configs = [c for c in STAGE4_CONFIGS if c["name"] in args.configs]

    # Header
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Stage 4 Robustness Sweep")
    print(f"  Configs: {len(configs)}  |  Pairs: {len(args.pairs)}  "
          f"|  Runs/config: {args.runs}")
    print(f"  Steps: {args.steps}  |  anchor={params['anchor_strength']}  "
          f"beta={params['beta']}")
    print(f"  Forward col: {forward_col}")
    print(f"  Output dir:  {output_dir}")
    print(sep)

    # Load BSVE data once for all pairs
    print("\n[setup] Loading BSVE dataset...")
    bsve_all = _build_bsve_multi_pair(args.bsve_states_path, args.pairs)

    # Validate forward column exists
    if forward_col not in bsve_all.columns:
        available = [c for c in bsve_all.columns if c.startswith("ret_")]
        raise ValueError(
            f"Column '{forward_col}' not in BSVE dataset. "
            f"Available return columns: {available}"
        )

    if args.verbose:
        counts = bsve_all["state_id"].value_counts().to_dict()
        print(f"[setup] BSVE rows loaded: {len(bsve_all)}  states={counts}")

    # --- Main sweep loop ---
    all_results: Dict = {}

    for cfg_dict in configs:
        cname = cfg_dict["name"]
        print(f"\n{'─' * 60}")
        print(f"  Config: {cname}  ({cfg_dict['label']})")
        print(f"{'─' * 60}")

        cfg_result = run_one_config(
            config=cfg_dict,
            bsve_all=bsve_all,
            params=params,
            forward_col=forward_col,
            pairs=args.pairs,
            runs=args.runs,
            steps=args.steps,
            seed_base=args.seed,
            verbose=args.verbose,
        )
        all_results[cname] = cfg_result

    # --- Summary table ---
    print_summary(
        all_results=all_results,
        configs=configs,
        pairs=args.pairs,
        runs=args.runs,
        steps=args.steps,
    )

    # --- Save outputs ---
    save_results(
        all_results=all_results,
        configs=configs,
        pairs=args.pairs,
        params=params,
        runs=args.runs,
        steps=args.steps,
        forward_col=forward_col,
        output_dir=output_dir,
    )

    print(f"\n[done] Stage 4 robustness sweep complete.")
    print(f"       Results in: {output_dir}")
    print(f"       Summary:    {output_dir / 'stage4_robustness_summary.json'}\n")


if __name__ == "__main__":
    main()