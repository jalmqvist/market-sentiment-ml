"""
run_regime_v3.py
================
Entry-point script for the Regime V3 regime-conditioned signal pipeline.

Extends ``experiments.regime_v3`` with logging to *both* stdout and an
optional log file.  All walk-forward and regime-metrics logic lives in
``experiments/regime_v3.py``; this script is a thin launcher that
configures logging and delegates to ``experiments.regime_v3.main``.

Pipeline steps
--------------
1. Load and prepare the research dataset.
2. Compute causal volatility and interaction features (``build_features``).
3. Discretise vol/trend features into vol×trend regimes (``build_regimes``).
4. Discretise sentiment features into behavioural regimes
   (``build_behavioural_regimes``), producing ``crowding_regime``.
5. Full-dataset regime discovery baselines (regime, behavioural, crowding).
6. Walk-forward regime validation (regime, behavioural, crowding).
7. Secondary vol conditioning on top crowding regimes.
8. **Regime filter** — apply ``TOP_REGIMES`` filter; add ``is_active`` column.
9. **Full dataset performance** — aggregate metrics without filter (baseline).
10. **Filtered performance** — aggregate metrics on regime-filtered signals.
11. **Walk-forward filtered performance** — per-year OOS metrics on filtered
    signals (no refitting of the regime list).
12. **Coverage summary** — fraction of signals retained after the filter.
13. **Filter + Direction** — apply ``CONTRARIAN_REGIMES`` / ``TREND_REGIMES``
    mapping to assign a directional signal per row; compute aggregate and
    per-year OOS metrics on the active subset (``signal != 0``).
14. **Filter + Direction + Weighting** — extend step 13 with continuous regime
    weights derived from per-regime Sharpe on training data only (leakage-free
    in the walk-forward).  Configurable via ``--weight-threshold`` and
    ``--normalize-weights``.
15. **Final signal summary** — consolidated printout of (A) discovery outputs
    and (B) signal outputs (filtered, direction, weighted walk-forward).  This
    is the authoritative reported summary and replaces the baseline model
    walk-forward.

Usage::

    # Log to auto-created timestamped file (default)
    python run_regime_v3.py --data data/output/master_research_dataset.csv

    # Log to a specific file
    python run_regime_v3.py --data data/output/master_research_dataset.csv \\
                            --log-file logs/regime_v3.log

    # Verbose mode
    python run_regime_v3.py --data data/output/master_research_dataset.csv \\
                            --log-level DEBUG --log-file logs/regime_v3_debug.log

    # Custom weight threshold and normalization
    python run_regime_v3.py --data data/output/master_research_dataset.csv \\
                            --weight-threshold 0.10 --normalize-weights
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import config as cfg

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logger with a file handler only (no stdout).

    If *log_file* is None, a timestamped file is created automatically in
    ``logs/``.  The parent directory is created if it does not exist.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional explicit path to a log file.  When omitted, a
            file named ``regime_v3_YYYYMMDD_HHMMSS.log`` is created inside
            ``logs/``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"regime_v3_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logging.getLogger(__name__).info("File logging enabled: %s", log_path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Regime V3: Ridge regression walk-forward with conditional "
            "performance evaluation by vol × trend regime."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(cfg.DATA_PATH),
        help="Path to master research dataset CSV.",
    )
    p.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Optional explicit log file path.  When omitted, a timestamped "
            "file is created automatically in logs/."
        ),
    )
    p.add_argument(
        "--weight-threshold",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Minimum absolute regime weight required for a signal to be active "
            "in the FILTER + DIRECTION + WEIGHTING step.  Regimes with "
            "|weight| < threshold are set to signal = 0.  Defaults to the "
            "module-level WEIGHT_THRESHOLD constant (0.0)."
        ),
    )
    p.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help=(
            "Normalize regime weights by max_abs_sharpe instead of the default "
            "tanh(sharpe/std_sharpe) scaling.  When set, weight = sharpe / max_abs_sharpe."
        ),
    )
    p.add_argument(
        "--top-n-regimes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of top regimes to select automatically via select_top_regimes. "
            "Defaults to the module-level TOP_N_REGIMES constant (5)."
        ),
    )
    p.add_argument(
        "--min-regime-sharpe",
        type=float,
        default=None,
        metavar="SHARPE",
        help=(
            "Minimum full-dataset Sharpe for a regime to pass the auto-selection "
            "filter.  Defaults to the module-level MIN_REGIME_SHARPE constant (0.02)."
        ),
    )
    p.add_argument(
        "--min-stability",
        type=float,
        default=None,
        metavar="RATIO",
        help=(
            "Minimum positive-year ratio required for a regime to pass the "
            "stability filter in auto-selection and direction classification. "
            "Defaults to the module-level MIN_STABILITY_RATIO constant (0.55)."
        ),
    )
    args = p.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    # Import after logging is configured so that module-level loggers pick up
    # the handlers set above.
    from experiments.regime_v3 import (  # noqa: PLC0415
        TARGET_COL,
        TOP_N_REGIMES,
        MIN_REGIME_SHARPE,
        MIN_STABILITY_RATIO,
        WEIGHT_THRESHOLD,
        apply_regime_filter,
        apply_regime_direction_signal,
        build_behavioural_regimes,
        build_features,
        build_regimes,
        behavioural_regime_baseline,
        behavioural_regime_walk_forward,
        classify_regime_direction,
        compute_coverage_summary,
        compute_regime_sharpe_map,
        compute_regime_stability_summary,
        convert_sharpe_to_weight,
        crowding_regime_baseline,
        crowding_regime_walk_forward,
        filtered_regime_baseline,
        filtered_regime_walk_forward,
        full_dataset_performance,
        load_data,
        log_behavioural_regime_summary,
        log_behavioural_regime_wf,
        log_coverage_summary,
        log_crowding_regime_summary,
        log_crowding_regime_wf,
        log_filtered_performance,
        log_filtered_wf,
        log_full_dataset_performance,
        log_regime_baseline,
        log_regime_direction_performance,
        log_regime_direction_wf,
        log_regime_stability_summary,
        log_regime_weight_diagnostics,
        log_regime_weighted_performance,
        log_regime_weighted_wf,
        log_regime_wf,
        log_secondary_vol_filter,
        make_regime_weights_df,
        print_final_signal_summary,
        regime_baseline,
        regime_direction_performance,
        regime_direction_walk_forward,
        regime_walk_forward,
        regime_weighted_performance,
        regime_weighted_walk_forward,
        secondary_vol_filter,
        select_top_regimes,
    )

    # Resolve CLI overrides against module-level defaults.
    top_n_regimes = args.top_n_regimes if args.top_n_regimes is not None else TOP_N_REGIMES
    min_regime_sharpe = args.min_regime_sharpe if args.min_regime_sharpe is not None else MIN_REGIME_SHARPE
    min_stability = args.min_stability if args.min_stability is not None else MIN_STABILITY_RATIO
    weight_threshold = args.weight_threshold if args.weight_threshold is not None else WEIGHT_THRESHOLD
    normalize_weights = args.normalize_weights

    df = load_data(args.data)

    # Step 1: Compute causal volatility feature (vol_24b) and interaction features.
    df = build_features(df)

    # Step 2: Discretise features into vol/trend regimes BEFORE any modeling.
    df = build_regimes(df)

    # Step 2b: Discretise sentiment features into behavioural regimes.
    df = build_behavioural_regimes(df)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: REGIME BASELINE (NO MODEL) — full-dataset regime discovery
    # ------------------------------------------------------------------
    regime_summary = regime_baseline(df)
    log_regime_baseline(regime_summary)

    # ------------------------------------------------------------------
    # Step 3b: BEHAVIOURAL REGIME SUMMARY — full-dataset behavioural
    #          discovery (no model)
    # ------------------------------------------------------------------
    behavioural_summary = behavioural_regime_baseline(df)
    log_behavioural_regime_summary(behavioural_summary)

    # ------------------------------------------------------------------
    # Step 3c: CROWDING REGIME SUMMARY — simplified 2-axis regime discovery
    # ------------------------------------------------------------------
    crowding_summary = crowding_regime_baseline(df)
    log_crowding_regime_summary(crowding_summary)

    # ------------------------------------------------------------------
    # Step 4: REGIME WALK-FORWARD — out-of-sample regime validation
    # ------------------------------------------------------------------
    regime_wf = regime_walk_forward(df)
    log_regime_wf(regime_wf)

    # ------------------------------------------------------------------
    # Step 4b: WALK-FORWARD REGIME PERFORMANCE — out-of-sample
    #          behavioural regime validation
    # ------------------------------------------------------------------
    behavioural_wf = behavioural_regime_walk_forward(df)
    log_behavioural_regime_wf(behavioural_wf)

    # ------------------------------------------------------------------
    # Step 4c: CROWDING REGIME WALK-FORWARD — out-of-sample crowding
    #          regime validation (3-axis: streak × sentiment × vol)
    # ------------------------------------------------------------------
    crowding_wf = crowding_regime_walk_forward(df)
    log_crowding_regime_wf(crowding_wf)

    # ------------------------------------------------------------------
    # Step 4c-ii: REGIME STABILITY SUMMARY — per-regime sign consistency
    #             and Sharpe stability across walk-forward years
    # ------------------------------------------------------------------
    regime_stability = compute_regime_stability_summary(crowding_wf)
    log_regime_stability_summary(regime_stability)
    if not regime_stability.empty:
        logging.getLogger(__name__).info(
            "REGIME STABILITY SUMMARY (top rows):\n%s",
            regime_stability.head(10).to_string(index=False),
        )

    # ------------------------------------------------------------------
    # Step 4c-iii: AUTO-SELECT top regimes and direction from discovery
    #              outputs (FULL-DATASET; for diagnostics and reporting only).
    #              Walk-forward functions compute these from training data
    #              per fold — no forward bias in OOS evaluation.
    # ------------------------------------------------------------------
    top_regimes = select_top_regimes(
        crowding_summary,
        regime_stability,
        top_n=top_n_regimes,
        min_sharpe=min_regime_sharpe,
        min_stability=min_stability,
    )
    contrarian_regimes, trend_regimes = classify_regime_direction(
        regime_stability, min_stability=min_stability
    )
    logging.getLogger(__name__).info(
        "AUTO-SELECTION COMPLETE | top_regimes=%s", top_regimes
    )
    logging.getLogger(__name__).info(
        "AUTO-SELECTION COMPLETE | contrarian_regimes=%s", contrarian_regimes
    )
    logging.getLogger(__name__).info(
        "AUTO-SELECTION COMPLETE | trend_regimes=%s", trend_regimes
    )

    # ------------------------------------------------------------------
    # Step 4d: SECONDARY VOL FILTER — secondary conditioning on top
    #          crowding regimes (no combinatorial regime expansion)
    # ------------------------------------------------------------------
    vol_filter_results = secondary_vol_filter(df, crowding_summary)
    log_secondary_vol_filter(vol_filter_results)

    # ------------------------------------------------------------------
    # Step 4e: REGIME FILTER — apply top_regimes filter to dataset
    #          (deterministic, leakage-free: computed before any target use)
    # ------------------------------------------------------------------
    # n_before_filter = all non-null target rows (baseline for coverage reporting).
    n_before_filter = int(df[TARGET_COL].notna().sum())
    logging.getLogger(__name__).info(
        "=== REGIME FILTER (top_regimes=%s) === coverage before: %d signals",
        top_regimes, n_before_filter,
    )
    df = apply_regime_filter(df, top_regimes=top_regimes)
    # n_after_filter = active (filtered) rows with non-null target; strict subset
    # of n_before_filter, giving the fraction of signals retained by the filter.
    n_after_filter = int(df.loc[df["is_active"], TARGET_COL].notna().sum())
    if n_before_filter > 0:
        logging.getLogger(__name__).info(
            "REGIME FILTER: coverage after = %d signals (%.1f%% of %d)",
            n_after_filter,
            100.0 * n_after_filter / n_before_filter,
            n_before_filter,
        )
    else:
        logging.getLogger(__name__).warning(
            "REGIME FILTER: n_before_filter = 0 — no signals available before filter"
        )

    # ------------------------------------------------------------------
    # Step 4f: FULL DATASET PERFORMANCE — baseline metrics (unfiltered)
    # ------------------------------------------------------------------
    full_perf = full_dataset_performance(df)
    log_full_dataset_performance(full_perf)

    # ------------------------------------------------------------------
    # Step 4g: FILTERED PERFORMANCE — metrics on regime-filtered signals
    # ------------------------------------------------------------------
    filtered_perf = filtered_regime_baseline(df, top_regimes)
    log_filtered_performance(filtered_perf)

    # ------------------------------------------------------------------
    # Step 4h: WALK-FORWARD FILTERED PERFORMANCE — per-year OOS metrics;
    #          regime selection computed from training data per fold
    #          (strictly causal, no forward bias).
    # ------------------------------------------------------------------
    filtered_wf_results = filtered_regime_walk_forward(
        df,
        top_n=top_n_regimes,
        min_sharpe=min_regime_sharpe,
        min_stability=min_stability,
    )
    log_filtered_wf(filtered_wf_results)

    # ------------------------------------------------------------------
    # Step 4i: COVERAGE SUMMARY — fraction of signals retained by filter
    # ------------------------------------------------------------------
    coverage = compute_coverage_summary(df, top_regimes)
    log_coverage_summary(coverage)

    # ------------------------------------------------------------------
    # Step 4j: FILTER + DIRECTION — regime-specific signal direction
    #          (BASELINE (GLOBAL CONTRARIAN) and FILTER ONLY already
    #          logged above; this is the FILTER + DIRECTION (FINAL) step)
    # ------------------------------------------------------------------
    df_direction = apply_regime_direction_signal(
        df, contrarian_regimes, trend_regimes
    )
    n_active_signals = int((df_direction["signal"] != 0.0).sum())
    logging.getLogger(__name__).info(
        "FILTER + DIRECTION: %d active signals "
        "(contrarian=%d, trend=%d, total=%d)",
        n_active_signals,
        int((df_direction["regime_direction"] == "contrarian").sum()),
        int((df_direction["regime_direction"] == "trend").sum()),
        len(df_direction),
    )
    dir_perf = regime_direction_performance(df_direction)
    log_regime_direction_performance(dir_perf)

    # ------------------------------------------------------------------
    # Step 4k: WALK-FORWARD FILTER + DIRECTION — per-year OOS metrics;
    #          regime selection and direction computed from training data
    #          per fold (strictly causal, no forward bias).
    # ------------------------------------------------------------------
    dir_wf = regime_direction_walk_forward(
        df,
        top_n=top_n_regimes,
        min_sharpe=min_regime_sharpe,
        min_stability=min_stability,
    )
    log_regime_direction_wf(dir_wf)

    # ------------------------------------------------------------------
    # Step 4l: FILTER + DIRECTION + WEIGHTING — continuous regime weights
    #          derived from training-only regime Sharpe.
    # ------------------------------------------------------------------
    # Full-dataset weighted performance (uses all data for Sharpe map;
    # the leakage-free version is evaluated in the walk-forward below).
    sharpe_map_full = compute_regime_sharpe_map(
        df, contrarian_regimes, trend_regimes
    )
    weight_map_full = convert_sharpe_to_weight(
        sharpe_map_full, normalize=normalize_weights
    )
    log_regime_weight_diagnostics(
        weight_map_full, sharpe_map_full, weight_threshold=weight_threshold
    )
    regime_weights_df = make_regime_weights_df(sharpe_map_full, weight_map_full)
    if not regime_weights_df.empty:
        logging.getLogger(__name__).info(
            "REGIME WEIGHTS MAP (top rows):\n%s",
            regime_weights_df.head(10).to_string(index=False),
        )
    weighted_perf = regime_weighted_performance(
        df, weight_map_full, contrarian_regimes, trend_regimes,
        weight_threshold=weight_threshold,
    )
    log_regime_weighted_performance(weighted_perf)

    # ------------------------------------------------------------------
    # Step 4m: WALK-FORWARD FILTER + DIRECTION + WEIGHTING — per-year
    #          OOS metrics; regime selection, direction, and Sharpe map
    #          all computed from training data per fold (strictly causal,
    #          no forward bias).
    # ------------------------------------------------------------------
    weighted_wf = regime_weighted_walk_forward(
        df,
        weight_threshold=weight_threshold,
        normalize_weights=normalize_weights,
        top_n=top_n_regimes,
        min_sharpe=min_regime_sharpe,
        min_stability=min_stability,
    )
    log_regime_weighted_wf(weighted_wf)

    # ------------------------------------------------------------------
    # Final summary — regime-conditioned signal pipeline results.
    # Clearly separates (A) discovery outputs and (B) signal outputs.
    # This replaces the baseline model walk-forward as the authoritative
    # reported summary.
    # ------------------------------------------------------------------
    print_final_signal_summary(
        full_perf, filtered_perf, filtered_wf_results, dir_wf, weighted_wf, coverage
    )


if __name__ == "__main__":
    main()
