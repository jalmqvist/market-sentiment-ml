# bsve/calibration/chf_vol_calibration.py
"""
CHF volatility regime boundary calibration.

Estimates empirical ATR-based volatility regime boundaries from the
DL-active window. The goal is to find thresholds where crowd-state
persistence behavior changes meaningfully, not merely where volatility
changes statistically.

Two-stage approach:
    Stage 1 — Find candidate vol boundaries from ATR distribution
               (percentile-based + Jenks natural breaks).
    Stage 2 — Validate that candidate boundaries correspond to
               meaningful differences in crowd persistence duration.

Output: bsve/calibrations/reactive_chf_calibration_v1.json

Usage:
    python -m bsve.calibration.chf_vol_calibration \
        --dataset-version 1.3.2 \
        --output-dir bsve/calibrations \
        --pairs USDCHF EURCHF \
        --start 2019-01-01 \
        --end 2026-12-31
"""

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VolRegimeBoundary:
    """
    A single candidate volatility regime boundary.
    """
    atr_pct_threshold: float
    low_vol_label: str
    high_vol_label: str
    persistence_duration_low: float     # median crowd persistence below threshold
    persistence_duration_high: float    # median crowd persistence above threshold
    ks_statistic: float                 # KS test statistic
    ks_pvalue: float                    # KS test p-value
    effect_size: float                  # Cohen's d on persistence distributions
    behaviorally_significant: bool      # persistence_duration_low > high * min_ratio


@dataclass
class CHFVolCalibrationResult:
    """
    Output of CHF volatility regime calibration.
    """
    environment_id: str
    calibration_version: str
    dataset_version: str
    pairs: list[str]
    calibration_window_start: str
    calibration_window_end: str

    # Primary outputs
    low_vol_threshold_atr_pct: float    # ATR% below this → low vol regime
    high_vol_threshold_atr_pct: float   # ATR% above this → high vol regime
    # medium vol is the band between the two thresholds

    # Internal coherence
    usdchf_low_vol_threshold: float     # pair-specific for cross-check
    eurchf_low_vol_threshold: float
    usdchf_high_vol_threshold: float
    eurchf_high_vol_threshold: float
    pair_threshold_agreement: bool      # True if pair-specific thresholds are close

    # Diagnostics
    n_low_vol_bars: int
    n_medium_vol_bars: int
    n_high_vol_bars: int
    median_persistence_low_vol: float
    median_persistence_high_vol: float
    ks_pvalue_low_vs_high: float
    effect_size_low_vs_high: float
    snb_era_flag: bool                  # True if pre-2015 data included

    # Reproducibility
    calibration_hash: str


# ---------------------------------------------------------------------------
# ATR regime boundary candidates
# ---------------------------------------------------------------------------

def compute_atr_pct_distribution(
    df: pd.DataFrame,
    rolling_window: int = 24,
) -> pd.Series:
    """
    Compute rolling ATR% for volatility regime classification.

    Uses a rolling median rather than rolling mean to reduce
    sensitivity to spike contamination.

    Args:
        df: H1 DataFrame with 'atr_pct' column.
        rolling_window: H1 bars for rolling median (default 24 = 1 day).

    Returns:
        Series of smoothed ATR% values.
    """
    return df["atr_pct"].rolling(rolling_window, min_periods=rolling_window // 2).median()


def find_candidate_boundaries_percentile(
    atr_series: pd.Series,
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
) -> tuple[float, float]:
    """
    Simple percentile-based candidate boundaries.
    Used as a baseline and sanity check against Jenks breaks.
    """
    clean = atr_series.dropna()
    low = float(np.percentile(clean, low_percentile))
    high = float(np.percentile(clean, high_percentile))
    return low, high


def find_candidate_boundaries_jenks(
    atr_series: pd.Series,
    n_classes: int = 3,
    sample_size: int = 5000,
) -> tuple[float, float]:
    """
    Jenks natural breaks for ATR% regime boundaries.

    Finds boundaries that minimize within-class variance, which
    tends to align better with natural distributional structure
    than fixed percentiles.

    Falls back to percentile method if jenkspy is not available.

    Args:
        atr_series: Smoothed ATR% series.
        n_classes: Number of volatility regimes (default 3).
        sample_size: Subsample size for performance (Jenks is O(n^2)).

    Returns:
        (low_boundary, high_boundary) ATR% thresholds.
    """
    try:
        import jenkspy
    except ImportError:
        warnings.warn(
            "jenkspy not available. Falling back to percentile boundaries. "
            "Install with: pip install jenkspy"
        )
        return find_candidate_boundaries_percentile(atr_series)

    clean = atr_series.dropna().values
    if len(clean) > sample_size:
        rng = np.random.default_rng(seed=42)
        clean = rng.choice(clean, size=sample_size, replace=False)

    breaks = jenkspy.jenks_breaks(clean.tolist(), n_classes=n_classes)
    # breaks has n_classes + 1 values: [min, break1, break2, ..., max]
    # For 3 classes: [min, low_high_boundary, medium_high_boundary, max]
    if len(breaks) != n_classes + 1:
        raise ValueError(
            f"jenkspy.jenks_breaks returned {len(breaks)} break values for "
            f"n_classes={n_classes}; expected {n_classes + 1}. "
            "This indicates an incompatible jenkspy version or degenerate input."
        )
    low_boundary = float(breaks[1])
    high_boundary = float(breaks[2])
    return low_boundary, high_boundary


# ---------------------------------------------------------------------------
# Crowd persistence measurement
# ---------------------------------------------------------------------------

def measure_crowd_persistence(
    df: pd.DataFrame,
    extreme_threshold: float,
    vol_threshold_low: float,
    vol_threshold_high: float,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """
    Measure crowd-state persistence duration stratified by vol regime.

    For each bar where sentiment enters an extreme state, records
    how long that state persists before sentiment falls below threshold.
    Tags each episode with the dominant vol regime during the episode.

    Args:
        df: H1 DataFrame with columns:
            entry_time, sentiment_net, atr_pct
        extreme_threshold: |sentiment_net| >= this → extreme state.
        vol_threshold_low: ATR% below this → low vol regime.
        vol_threshold_high: ATR% above this → high vol regime.
        rolling_window: Smoothing window for ATR%.

    Returns:
        DataFrame with columns:
            entry_bar, duration_bars, vol_regime, mean_atr_pct,
            exit_type, pair
    """
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Smooth ATR for regime classification
    df["atr_smooth"] = compute_atr_pct_distribution(df, rolling_window)

    # Classify vol regime per bar
    df["vol_regime"] = "medium"
    df.loc[df["atr_smooth"] < vol_threshold_low, "vol_regime"] = "low"
    df.loc[df["atr_smooth"] >= vol_threshold_high, "vol_regime"] = "high"

    # Extreme sentiment flag
    df["is_extreme"] = df["sentiment_net"].abs() >= extreme_threshold

    episodes = []
    in_episode = False
    episode_start = None

    for idx, row in df.iterrows():
        if not in_episode and row["is_extreme"]:
            in_episode = True
            episode_start = idx

        elif in_episode and not row["is_extreme"]:
            episode_slice = df.loc[episode_start:idx - 1]
            duration = len(episode_slice)

            # Dominant vol regime = modal regime during episode
            dominant_regime = (
                episode_slice["vol_regime"].mode().iloc
                if not episode_slice.empty else "unknown"
            )
            mean_atr = float(episode_slice["atr_smooth"].mean())

            episodes.append({
                "entry_bar": df.loc[episode_start, "entry_time"],
                "duration_bars": duration,
                "vol_regime": dominant_regime,
                "mean_atr_pct": mean_atr,
                "exit_type": "sentiment_reset",
            })
            in_episode = False
            episode_start = None

    # Right-censored
    if in_episode and episode_start is not None:
        episode_slice = df.loc[episode_start:]
        dominant_regime = (
            episode_slice["vol_regime"].mode().iloc
            if not episode_slice.empty else "unknown"
        )
        episodes.append({
            "entry_bar": df.loc[episode_start, "entry_time"],
            "duration_bars": len(episode_slice),
            "vol_regime": dominant_regime,
            "mean_atr_pct": float(episode_slice["atr_smooth"].mean()),
            "exit_type": "censored",
        })

    return pd.DataFrame(episodes)


# ---------------------------------------------------------------------------
# Behavioral significance test
# ---------------------------------------------------------------------------

def test_boundary_behavioral_significance(
    persistence_df: pd.DataFrame,
    vol_threshold_low: float,
    vol_threshold_high: float,
    min_observations: int = 30,
    min_effect_size: float = 0.3,
    min_persistence_ratio: float = 1.25,
) -> VolRegimeBoundary:
    """
    Test whether a candidate vol boundary corresponds to meaningful
    differences in crowd persistence behavior.

    A boundary is considered behaviorally significant if:
        1. KS test rejects equal distributions (p < 0.05)
        2. Cohen's d effect size >= min_effect_size
        3. Median persistence in low vol >= min_persistence_ratio
           times median persistence in high vol

    All three conditions must hold. Statistical significance alone
    is insufficient — the effect must be behaviorally meaningful.

    Args:
        persistence_df: Output of measure_crowd_persistence().
        vol_threshold_low: Lower ATR% boundary being tested.
        vol_threshold_high: Upper ATR% boundary being tested.
        min_observations: Minimum episodes per regime for valid test.
        min_effect_size: Minimum Cohen's d for behavioral significance.
        min_persistence_ratio: Minimum ratio of low/high persistence.

    Returns:
        VolRegimeBoundary with test results.
    """
    low_durations = persistence_df.loc[
        persistence_df["vol_regime"] == "low", "duration_bars"
    ].values
    high_durations = persistence_df.loc[
        persistence_df["vol_regime"] == "high", "duration_bars"
    ].values

    if len(low_durations) < min_observations:
        warnings.warn(
            f"Only {len(low_durations)} low-vol episodes. "
            f"Minimum is {min_observations}. Results unreliable."
        )
    if len(high_durations) < min_observations:
        warnings.warn(
            f"Only {len(high_durations)} high-vol episodes. "
            f"Minimum is {min_observations}. Results unreliable."
        )

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(low_durations, high_durations)

    # Cohen's d
    pooled_std = np.sqrt(
        (np.std(low_durations, ddof=1) ** 2
         + np.std(high_durations, ddof=1) ** 2) / 2
    )
    cohens_d = (
        (np.mean(low_durations) - np.mean(high_durations)) / pooled_std
        if pooled_std > 0 else 0.0
    )

    median_low = float(np.median(low_durations)) if len(low_durations) > 0 else 0.0
    median_high = float(np.median(high_durations)) if len(high_durations) > 0 else 0.0
    persistence_ratio = (
        median_low / median_high
        if median_high > 0 else 0.0
    )

    behaviorally_significant = (
        ks_pvalue < 0.05
        and abs(cohens_d) >= min_effect_size
        and persistence_ratio >= min_persistence_ratio
    )

    return VolRegimeBoundary(
        atr_pct_threshold=vol_threshold_low,
        low_vol_label="low",
        high_vol_label="high",
        persistence_duration_low=median_low,
        persistence_duration_high=median_high,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        effect_size=float(cohens_d),
        behaviorally_significant=behaviorally_significant,
    )


# ---------------------------------------------------------------------------
# Per-pair threshold derivation for internal coherence check
# ---------------------------------------------------------------------------

def derive_pair_threshold(
    df: pd.DataFrame,
    extreme_threshold: float,
    method: str = "jenks",
) -> tuple[float, float]:
    """
    Derive vol regime boundaries for a single pair.
    Used to verify USDCHF and EURCHF produce consistent thresholds.
    """
    atr_smooth = compute_atr_pct_distribution(df)
    if method == "jenks":
        return find_candidate_boundaries_jenks(atr_smooth)
    return find_candidate_boundaries_percentile(atr_smooth)


# ---------------------------------------------------------------------------
# SNB era detection
# ---------------------------------------------------------------------------

def check_snb_era(
    start: str,
    snb_floor_removal: str = "2015-01-15",
) -> bool:
    """
    Flag whether the calibration window includes the pre-SNB-floor-
    removal era (before January 2015).

    EURCHF exhibited structurally different volatility before the
    SNB removed the 1.20 floor on 2015-01-15. Including this period
    may distort vol regime boundaries for EURCHF specifically.

    Returns True if pre-SNB data is included, as a warning flag.
    """
    return pd.Timestamp(start) < pd.Timestamp(snb_floor_removal)


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_chf_calibration(
    data: dict[str, pd.DataFrame],
    dataset_version: str,
    output_dir: Path,
    calibration_version: str = "v1",
    start: str = "2019-01-01",
    end: str = "2026-12-31",
    boundary_method: str = "jenks",
) -> CHFVolCalibrationResult:
    """
    Full CHF volatility regime calibration pipeline.

    Steps:
        1. Pool ATR data across CHF pairs
        2. Derive extreme sentiment threshold
        3. Find candidate vol boundaries (Jenks + percentile cross-check)
        4. Measure crowd persistence per vol regime
        5. Test behavioral significance of boundaries
        6. Derive per-pair thresholds for coherence check
        7. Write versioned calibration artifact

    Args:
        data: Dict of pair → H1 DataFrame with columns:
              entry_time, sentiment_net, atr_pct
        dataset_version: MSML dataset version string.
        output_dir: Directory to write calibration artifact.
        calibration_version: Version tag for this calibration run.
        start: DL-active window start.
        end: DL-active window end.
        boundary_method: 'jenks' or 'percentile'.

    Returns:
        CHFVolCalibrationResult (also written to disk).
    """
    pairs = list(data.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snb_era_flag = check_snb_era(start)
    if snb_era_flag:
        warnings.warn(
            "Calibration window includes pre-SNB-floor-removal era "
            "(before 2015-01-15). EURCHF vol regime boundaries may be "
            "distorted. Consider using start='2015-02-01' for CHF "
            "calibration or treating pre/post-SNB as separate eras."
        )

    # Step 1 — Pool ATR and sentiment for threshold derivation
    all_atr = pd.concat(
        [compute_atr_pct_distribution(df) for df in data.values()],
        ignore_index=True,
    ).dropna()

    all_sentiment = pd.concat(
        [df["sentiment_net"] for df in data.values()],
        ignore_index=True,
    ).dropna()

    # Step 2 — Extreme sentiment threshold (pooled)
    # Import from JPY calibration module — shared utility
    from bsve.calibration.jpy_maturity_calibration import (
        compute_extreme_threshold,
    )
    extreme_threshold = compute_extreme_threshold(
        all_sentiment,
        method="percentile",
        percentile=70.0,
    )

    # Step 3 — Candidate vol boundaries (pooled)
    if boundary_method == "jenks":
        low_threshold, high_threshold = find_candidate_boundaries_jenks(
            all_atr
        )
        # Cross-check with percentile method
        low_pct, high_pct = find_candidate_boundaries_percentile(all_atr)
        threshold_divergence = max(
            abs(low_threshold - low_pct),
            abs(high_threshold - high_pct),
        )
        if threshold_divergence > 0.001:
            warnings.warn(
                f"Jenks and percentile boundaries diverge by "
                f"{threshold_divergence:.4f} ATR%. "
                "Review both before committing thresholds. "
                f"Jenks: ({low_threshold:.4f}, {high_threshold:.4f}) "
                f"Percentile: ({low_pct:.4f}, {high_pct:.4f})"
            )
    else:
        low_threshold, high_threshold = find_candidate_boundaries_percentile(
            all_atr
        )

    # Step 4 — Measure persistence per vol regime (pooled)
    all_persistence = pd.concat(
        [
            measure_crowd_persistence(
                df[
                    (df["entry_time"] >= start)
                    & (df["entry_time"] <= end)
                ].copy(),
                extreme_threshold=extreme_threshold,
                vol_threshold_low=low_threshold,
                vol_threshold_high=high_threshold,
            )
            for df in data.values()
        ],
        ignore_index=True,
    )

    if len(all_persistence) < 30:
        raise ValueError(
            f"Only {len(all_persistence)} persistence episodes found. "
            "Insufficient data for reliable calibration."
        )

    # Step 5 — Behavioral significance test
    boundary_result = test_boundary_behavioral_significance(
        all_persistence,
        vol_threshold_low=low_threshold,
        vol_threshold_high=high_threshold,
    )

    if not boundary_result.behaviorally_significant:
        warnings.warn(
            "Candidate vol boundaries are NOT behaviorally significant. "
            "Crowd persistence does not differ meaningfully across regimes. "
            "Do not commit these thresholds. "
            f"KS p-value: {boundary_result.ks_pvalue:.4f}, "
            f"Cohen's d: {boundary_result.effect_size:.3f}, "
            f"Persistence ratio: "
            f"{boundary_result.persistence_duration_low:.1f}h / "
            f"{boundary_result.persistence_duration_high:.1f}h"
        )

    # Step 6 — Per-pair thresholds for internal coherence
    pair_thresholds = {
        pair: derive_pair_threshold(
            df[
                (df["entry_time"] >= start)
                & (df["entry_time"] <= end)
            ].copy(),
            extreme_threshold=extreme_threshold,
            method=boundary_method,
        )
        for pair, df in data.items()
    }

    usdchf_low = pair_thresholds.get("USDCHF", (None, None))
    usdchf_high = pair_thresholds.get("USDCHF", (None, None))
    eurchf_low = pair_thresholds.get("EURCHF", (None, None))
    eurchf_high = pair_thresholds.get("EURCHF", (None, None))

    # Coherence check — pair-specific thresholds should be close
    pair_agreement = True
    if all(v is not None for v in [usdchf_low, eurchf_low]):
        low_divergence = abs(usdchf_low - eurchf_low)
        high_divergence = abs(usdchf_high - eurchf_high)
        if low_divergence > 0.002 or high_divergence > 0.002:
            warnings.warn(
                f"USDCHF and EURCHF vol thresholds diverge materially. "
                f"Low: {low_divergence:.4f}, High: {high_divergence:.4f}. "
                "Consider pair-specific thresholds rather than pooled."
            )
            pair_agreement = False

    # Vol regime bar counts
    atr_pooled = pd.concat(
        [compute_atr_pct_distribution(df) for df in data.values()]
    ).dropna()
    n_low = int((atr_pooled < low_threshold).sum())
    n_high = int((atr_pooled >= high_threshold).sum())
    n_medium = int(len(atr_pooled) - n_low - n_high)

    # Step 7 — Build result
    result_dict = {
        "environment_id": "reactive_chf",
        "calibration_version": calibration_version,
        "dataset_version": dataset_version,
        "pairs": pairs,
        "calibration_window_start": start,
        "calibration_window_end": end,
        "low_vol_threshold_atr_pct": round(low_threshold, 6),
        "high_vol_threshold_atr_pct": round(high_threshold, 6),
        "usdchf_low_vol_threshold": round(usdchf_low, 6) if usdchf_low else None,
        "eurchf_low_vol_threshold": round(eurchf_low, 6) if eurchf_low else None,
        "usdchf_high_vol_threshold": round(usdchf_high, 6) if usdchf_high else None,
        "eurchf_high_vol_threshold": round(eurchf_high, 6) if eurchf_high else None,
        "pair_threshold_agreement": pair_agreement,
        "n_low_vol_bars": n_low,
        "n_medium_vol_bars": n_medium,
        "n_high_vol_bars": n_high,
        "median_persistence_low_vol": round(
            boundary_result.persistence_duration_low, 2
        ),
        "median_persistence_high_vol": round(
            boundary_result.persistence_duration_high, 2
        ),
        "ks_pvalue_low_vs_high": round(boundary_result.ks_pvalue, 6),
        "effect_size_low_vs_high": round(boundary_result.effect_size, 4),
        "snb_era_flag": snb_era_flag,
    }

    # Deterministic hash over inputs + outputs
    hash_payload = json.dumps(result_dict, sort_keys=True).encode()
    result_dict["calibration_hash"] = hashlib.sha256(hash_payload).hexdigest()

    result = CHFVolCalibrationResult(**result_dict)

    # Write artifact
    artifact_path = (
        output_dir
        / f"reactive_chf_calibration_{calibration_version}.json"
    )
    with open(artifact_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"[BSVE] CHF calibration complete → {artifact_path}")
    print(f"  low_vol_threshold  : {result.low_vol_threshold_atr_pct:.6f}")
    print(f"  high_vol_threshold : {result.high_vol_threshold_atr_pct:.6f}")
    print(f"  pair_agreement     : {result.pair_threshold_agreement}")
    print(f"  n_low / med / high : "
          f"{result.n_low_vol_bars} / "
          f"{result.n_medium_vol_bars} / "
          f"{result.n_high_vol_bars}")
    print(f"  persistence low    : {result.median_persistence_low_vol:.1f}h")
    print(f"  persistence high   : {result.median_persistence_high_vol:.1f}h")
    print(f"  KS p-value         : {result.ks_pvalue_low_vs_high:.4f}")
    print(f"  Cohen's d          : {result.effect_size_low_vs_high:.3f}")
    print(f"  behaviorally sig.  : {boundary_result.behaviorally_significant}")
    if result.snb_era_flag:
        print("  ⚠ SNB era flag: pre-2015 data included")

    print(
        "\n[BSVE] ⚠ Review diagnostics before committing thresholds to "
        "reactive_chf_v1.yaml.\n"
        "  Expected: median_persistence_low >> median_persistence_high\n"
        "  Expected: KS p-value < 0.05 and Cohen's d >= 0.3\n"
        "  Expected: pair_threshold_agreement = True\n"
        "  If any of these fail, do not commit thresholds."
    )

    return result


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_vol_regime_persistence(
    persistence_df: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot crowd persistence duration distributions stratified by
    vol regime. Used for visual sign-off during calibration review.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[BSVE] matplotlib not available. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    regimes = ["low", "medium", "high"]
    colors = ["steelblue", "goldenrod", "tomato"]

    for ax, regime, color in zip(axes, regimes, colors):
        durations = persistence_df.loc[
            persistence_df["vol_regime"] == regime, "duration_bars"
        ]
        if len(durations) == 0:
            ax.set_title(f"{regime.upper()} VOL\n(no data)")
            continue

        ax.hist(
            durations,
            bins=30,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            durations.median(),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {durations.median():.0f}h",
        )
        ax.set_title(
            f"{regime.upper()} VOL\n"
            f"n={len(durations)}, "
            f"median={durations.median():.0f}h"
        )
        ax.set_xlabel("Persistence duration (H1 bars)")
        ax.set_ylabel("Episode count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CHF Crowd Persistence by Vol Regime\n"
        f"Low < {low_threshold:.4f} ATR% | "
        f"High ≥ {high_threshold:.4f} ATR%",
        fontsize=12,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[BSVE] Persistence plot saved → {output_path}")
    else:
        plt.show()


def plot_atr_distribution(
    atr_series: pd.Series,
    low_threshold: float,
    high_threshold: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot ATR% distribution with regime boundaries overlaid.
    Useful for verifying Jenks breaks align with natural distribution
    structure rather than arbitrary quantile cuts.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[BSVE] matplotlib not available. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    clean = atr_series.dropna()

    ax.hist(
        clean,
        bins=80,
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(
        low_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Low/Medium boundary ({low_threshold:.4f})",
    )
    ax.axvline(
        high_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Medium/High boundary ({high_threshold:.4f})",
    )

    # Shade regimes
    ax.axvspan(clean.min(), low_threshold, alpha=0.08, color="green",
               label="Low vol region")
    ax.axvspan(low_threshold, high_threshold, alpha=0.08, color="gold")
    ax.axvspan(high_threshold, clean.max(), alpha=0.08, color="red",
               label="High vol region")

    ax.set_xlabel("Smoothed ATR% (24h rolling median)")
    ax.set_ylabel("Bar count")
    ax.set_title("CHF ATR% Distribution with Regime Boundaries")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[BSVE] ATR distribution plot saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CHF volatility regime boundary calibration"
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", default="bsve/calibrations")
    parser.add_argument(
        "--pairs", nargs="+",
        default=["USDCHF", "EURCHF"]
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument(
        "--boundary-method",
        choices=["jenks", "percentile"],
        default="jenks",
    )
    parser.add_argument("--plot", action="store_true",
                        help="Emit diagnostic plots for sign-off review")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from research.data.loader import load_h1_sentiment_data
    data = {
        pair: load_h1_sentiment_data(
            pair=pair,
            dataset_version=args.dataset_version,
            start=args.start,
            end=args.end,
        )
        for pair in args.pairs
    }

    result = run_chf_calibration(
        data=data,
        dataset_version=args.dataset_version,
        output_dir=Path(args.output_dir),
        start=args.start,
        end=args.end,
        boundary_method=args.boundary_method,
    )

    if args.plot:
        all_dfs = list(data.values())
        atr_pooled = pd.concat(
            [compute_atr_pct_distribution(df) for df in all_dfs]
        ).dropna()

        # Recompute persistence for plotting
        all_persistence = pd.concat(
            [
                measure_crowd_persistence(
                    df,
                    extreme_threshold=0.60,   # use result threshold in practice
                    vol_threshold_low=result.low_vol_threshold_atr_pct,
                    vol_threshold_high=result.high_vol_threshold_atr_pct,
                )
                for df in all_dfs
            ],
            ignore_index=True,
        )

        output_dir = Path(args.output_dir)
        plot_atr_distribution(
            atr_pooled,
            result.low_vol_threshold_atr_pct,
            result.high_vol_threshold_atr_pct,
            output_path=output_dir / "chf_atr_distribution.png",
        )
        plot_vol_regime_persistence(
            all_persistence,
            result.low_vol_threshold_atr_pct,
            result.high_vol_threshold_atr_pct,
            output_path=output_dir / "chf_persistence_by_regime.png",
        )
