"""abm_experiments/episode_utils.py
================================================
Episode extraction and scoring utilities for ABM/BSVE reconciliation.

Ports consensus lifecycle extraction logic from BSVE calibration to enable
ABM evaluation against empirically calibrated episode structure targets.

Constraints:
- Single file, no dependencies beyond numpy/pandas
- Reproduces BSVE jpy_maturity_calibration.py results on real data
- Self-contained: no imports from bsve/ (reads JSON artifacts only)
"""


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd




# ---------------------------------------------------------------------------
# Data structures (mirror BSVE ConsensusLifecycle)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusEpisode:
    """
    A single consensus state episode: from formation to exit.
    Mirrors bsve.calibration.jpy_maturity_calibration.ConsensusLifecycle.
    """
    pair: str
    entry_step: int              # Simulation step index (not timestamp)
    exit_step: Optional[int]     # None if right-censored (still active)
    duration_steps: int          # Steps from entry to exit/censor
    exit_type: str               # 'reversal' | 'censored'
    max_net_sentiment: float     # peak crowd positioning during episode
    entry_net_sentiment: float




# ---------------------------------------------------------------------------
# Episode extraction (port of extract_consensus_lifecycles)
# ---------------------------------------------------------------------------


def extract_consensus_episodes(
    net_sentiment: np.ndarray | pd.Series,
    extreme_threshold: float,
    min_episode_steps: int = 2,
    pair: str = "unknown",
) -> list[ConsensusEpisode]:
    """
    Extract discrete consensus state episodes from net sentiment timeseries.

    An episode begins when |net_sentiment| crosses extreme_threshold
    and ends when it falls back below threshold (reversal) or when the
    series ends (censored).

    Args:
        net_sentiment: 1D array of net sentiment values (long% - short%).
        extreme_threshold: |net_sentiment| >= this → extreme state.
        min_episode_steps: Discard episodes shorter than this.
        pair: Currency pair name (for metadata).

    Returns:
        List of ConsensusEpisode instances.
    """
    s = np.asarray(net_sentiment, dtype=float)
    if s.ndim != 1:
        raise ValueError("net_sentiment must be 1D")

    is_extreme = np.abs(s) >= extreme_threshold

    episodes: list[ConsensusEpisode] = []
    in_episode = False
    episode_start_idx: Optional[int] = None

    for idx, extreme_flag in enumerate(is_extreme):
        if not in_episode and extreme_flag:
            # Episode entry
            in_episode = True
            episode_start_idx = idx

        elif in_episode and not extreme_flag:
            # Episode exit via sentiment reset (reversal)
            assert episode_start_idx is not None
            duration = idx - episode_start_idx
            if duration >= min_episode_steps:
                episode_slice = s[episode_start_idx:idx]
                episodes.append(ConsensusEpisode(
                    pair=pair,
                    entry_step=episode_start_idx,
                    exit_step=idx,
                    duration_steps=duration,
                    exit_type="reversal",
                    max_net_sentiment=float(np.max(np.abs(episode_slice))),
                    entry_net_sentiment=float(s[episode_start_idx]),
                ))
            in_episode = False
            episode_start_idx = None

    # Handle right-censored episodes (still active at series end)
    if in_episode and episode_start_idx is not None:
        duration = len(s) - episode_start_idx
        if duration >= min_episode_steps:
            episode_slice = s[episode_start_idx:]
            episodes.append(ConsensusEpisode(
                pair=pair,
                entry_step=episode_start_idx,
                exit_step=None,
                duration_steps=duration,
                exit_type="censored",
                max_net_sentiment=float(np.max(np.abs(episode_slice))),
                entry_net_sentiment=float(s[episode_start_idx]),
            ))

    return episodes




# ---------------------------------------------------------------------------
# Hazard analysis (port of compute_hazard_by_maturity)
# ---------------------------------------------------------------------------


def compute_hazard_by_maturity(
    episodes: list[ConsensusEpisode],
    max_steps: int = 200,
    min_at_risk: int = 10,
) -> pd.DataFrame:
    """
    Compute empirical reversal hazard rate as a function of maturity.

    Uses Kaplan-Meier style discrete hazard estimator:
        h(t) = n_reversals_at_t / n_at_risk_at_t

    Args:
        episodes: Extracted consensus episodes.
        max_steps: Maximum maturity step to compute hazard for.
        min_at_risk: Skip steps with fewer than this many episodes at risk.

    Returns:
        DataFrame with columns:
            maturity_step, n_at_risk, n_reversals, hazard_rate,
            cumulative_survival
    """
    # Build event table (duration, event flag)
    # event=1 for reversal, event=0 for censored or other
    records = []
    for ep in episodes:
        if ep.exit_type == "censored":
            records.append({"duration": ep.duration_steps, "event": 0})
        elif ep.exit_type == "reversal":
            records.append({"duration": ep.duration_steps, "event": 1})
        else:
            # Treat unknown exit types as censored for safety
            records.append({"duration": ep.duration_steps, "event": 0})

    event_df = pd.DataFrame(records)

    rows = []
    survival = 1.0

    for t in range(1, max_steps + 1):
        n_at_risk = (event_df["duration"] >= t).sum()
        if n_at_risk < min_at_risk:
            break
        n_events = (
            (event_df["duration"] == t) & (event_df["event"] == 1)
        ).sum()
        hazard = n_events / n_at_risk if n_at_risk > 0 else 0.0
        survival *= (1 - hazard)
        rows.append({
            "maturity_step": t,
            "n_at_risk": int(n_at_risk),
            "n_reversals": int(n_events),
            "hazard_rate": hazard,
            "cumulative_survival": survival,
        })

    return pd.DataFrame(rows)




# ---------------------------------------------------------------------------
# Calibration artifact loader
# ---------------------------------------------------------------------------


def load_calibration_artifact(path: str | Path) -> dict:
    """
    Load a BSVE calibration artifact JSON file.

    Returns the raw artifact dict. Caller extracts thresholds/diagnostics
    as needed.

    Args:
        path: Path to the calibration artifact JSON.

    Returns:
        Artifact dict with schema per bsve.calibration.calibration_contract.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {p}")

    try:
        with open(p, encoding="utf-8") as f:
            artifact: dict = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in calibration artifact {p}: {exc}")

    return artifact




# ---------------------------------------------------------------------------
# Episode structure scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStructureScore:
    """
    Scalar and component scores measuring ABM episode structure vs empirical.
    Lower is better (0 = perfect match).
    """
    total_score: float           # Weighted composite (minimise this)

    # Component scores (unweighted, for diagnostics)
    duration_ratio_error: float  # |log(sim_median / emp_median)|
    reversal_gradient_error: float  # penalty if rev_young <= rev_mature
    hazard_crossover_error: float   # |sim_crossover - emp_crossover| / emp_crossover
    frequency_ratio_error: float    # |log(sim_freq / emp_freq)|

    # Raw values for inspection
    sim_median_duration: float
    emp_median_duration: float
    sim_reversal_young: float
    sim_reversal_mature: float
    emp_reversal_young: float
    emp_reversal_mature: float
    sim_hazard_crossover: float
    emp_hazard_crossover: float
    sim_episode_frequency: float    # episodes per 1000 steps
    emp_episode_frequency: float




def score_episode_structure(
    episodes: list[ConsensusEpisode],
    hazard_df: pd.DataFrame,
    calibration_artifact: dict,
    n_total_steps: int,
    young_fraction: float = 0.4,
    mature_fraction: float = 1.6,
) -> EpisodeStructureScore:
    """
    Compare simulated episode structure against BSVE calibration targets.

    Computes a composite score measuring how closely the simulated episode
    population matches the empirical structure encoded in the calibration
    artifact. Lower scores indicate better correspondence.

    Args:
        episodes: Simulated episodes from extract_consensus_episodes.
        hazard_df: Hazard curve from compute_hazard_by_maturity.
        calibration_artifact: Loaded BSVE calibration artifact dict.
        n_total_steps: Total simulation steps (for frequency calculation).
        young_fraction: Multiplier for young boundary (default 0.4).
        mature_fraction: Multiplier for mature boundary (default 1.6).

    Returns:
        EpisodeStructureScore with component and total scores.
    """
    thresholds = calibration_artifact.get("thresholds", {})
    diagnostics = calibration_artifact.get("diagnostics", {})

    emp_young_boundary = thresholds.get("young_boundary_bars", 24)
    emp_mature_boundary = thresholds.get("mature_boundary_bars", 96)
    emp_crossover = diagnostics.get("hazard_crossover_bar", 60.0)
    emp_median_dur = diagnostics.get("median_episode_duration_bars", 48.0)
    emp_rev_young = diagnostics.get("reversal_rate_young", 0.25)
    emp_rev_mature = diagnostics.get("reversal_rate_mature", 0.10)
    emp_n_episodes = diagnostics.get("episode_count", 100)

    # Handle degenerate case: no episodes extracted
    if len(episodes) == 0:
        return EpisodeStructureScore(
            total_score=float("inf"),
            duration_ratio_error=float("inf"),
            reversal_gradient_error=float("inf"),
            hazard_crossover_error=float("inf"),
            frequency_ratio_error=float("inf"),
            sim_median_duration=0.0,
            emp_median_duration=emp_median_dur,
            sim_reversal_young=0.0,
            sim_reversal_mature=0.0,
            emp_reversal_young=emp_rev_young,
            emp_reversal_mature=emp_rev_mature,
            sim_hazard_crossover=0.0,
            emp_hazard_crossover=emp_crossover,
            sim_episode_frequency=0.0,
            emp_episode_frequency=emp_n_episodes / 2000.0 * 1000.0,
        )

    # Simulated duration distribution (completed episodes only)
    completed = [ep for ep in episodes if ep.exit_type != "censored"]
    completed_durations = [ep.duration_steps for ep in completed]
    sim_median_dur = float(np.median(completed_durations)) if completed_durations else 0.0

    # Simulated reversal rates by maturity zone
    young_episodes = [ep for ep in episodes if ep.duration_steps < emp_young_boundary]
    mature_episodes = [ep for ep in episodes if ep.duration_steps >= emp_mature_boundary]

    def _reversal_rate(ep_list: list) -> float:
        if not ep_list:
            return 0.0
        return sum(1 for ep in ep_list if ep.exit_type == "reversal") / len(ep_list)

    sim_rev_young = _reversal_rate(young_episodes)
    sim_rev_mature = _reversal_rate(mature_episodes)

    # Simulated hazard crossover
    if len(hazard_df) >= 12:
        smoothed = hazard_df["hazard_rate"].rolling(12, center=True).mean()
        diff = smoothed.diff().abs()
        stable_threshold = diff.quantile(0.25)
        stable_idx = diff[diff < stable_threshold].index
        if len(stable_idx) > 0:
            sim_crossover = float(hazard_df.loc[stable_idx[0], "maturity_step"])
        else:
            sim_crossover = float(hazard_df["maturity_step"].median())
    else:
        sim_crossover = float(hazard_df["maturity_step"].median()) if len(hazard_df) > 0 else 0.0

    # Simulated episode frequency (episodes per 1000 steps)
    sim_freq = (len(episodes) / n_total_steps) * 1000.0
    emp_freq = (emp_n_episodes / 2000.0) * 1000.0

    # ------------------------------------------------------------------
    # Component errors
    # ------------------------------------------------------------------

    # 1. Duration: symmetric log-ratio error
    if sim_median_dur > 0 and emp_median_dur > 0:
        dur_error = abs(np.log(sim_median_dur / emp_median_dur))
    else:
        dur_error = float("inf")

    # 2. Reversal rates: absolute deviation from empirical targets.
    #
    #    Previous implementation used a ratio comparison that broke when
    #    emp_rev_young == emp_rev_mature (both 1.0 in this dataset).
    #    The condition "emp_rev_young > emp_rev_mature" evaluates False
    #    when both are equal, causing rev_error = 0.0 regardless of sim values.
    #
    #    Fix: use absolute deviations from empirical absolute values,
    #    plus a structural penalty for missing mature episodes entirely.
    #
    #    Three components:
    #      a. |sim_rev_y - emp_rev_y|  — young zone rate accuracy
    #      b. |sim_rev_m - emp_rev_m|  — mature zone rate accuracy
    #      c. No-mature-episodes penalty (if sim has zero mature episodes,
    #         the mature rate of 0.0 is structurally degenerate regardless
    #         of the absolute error calculation)
    #
    #    Direction check is retained: if young dissolves FASTER than mature
    #    that is a structural failure deserving a heavy penalty.

    # Direction check with sampling-noise tolerance.
    # A difference of <= 0.05 between sim_rev_young and sim_rev_mature is
    # treated as statistical noise, not a structural failure.
    # Heavy penalty is reserved for clear structural inversions (e.g. young
    # episodes dissolving at materially lower rate than mature episodes).
    _DIRECTION_TOLERANCE = 0.05

    if sim_rev_young < sim_rev_mature - _DIRECTION_TOLERANCE:
        # Structurally wrong direction
        rev_error = 10.0 + (sim_rev_mature - sim_rev_young)
    else:
        young_rate_err = abs(sim_rev_young - emp_rev_young)
        mature_rate_err = abs(sim_rev_mature - emp_rev_mature)

        if len(mature_episodes) == 0:
            no_mature_penalty = 2.0
        else:
            no_mature_penalty = 0.0

        rev_error = young_rate_err + mature_rate_err + no_mature_penalty

    # 3. Hazard crossover: relative error
    if sim_crossover > 0 and emp_crossover > 0:
        cross_error = abs(sim_crossover - emp_crossover) / emp_crossover
    else:
        cross_error = 1.0  # Capped at 1.0 when crossover is uncomputable

    # 4. Frequency: symmetric log-ratio error
    if sim_freq > 0 and emp_freq > 0:
        freq_error = abs(np.log(sim_freq / emp_freq))
    else:
        freq_error = float("inf")

    # ------------------------------------------------------------------
    # Weighted composite score
    # ------------------------------------------------------------------
    # Weights reflect scientific priority:
    #   reversal structure > duration > frequency > crossover
    # Crossover is down-weighted because it requires a dense hazard curve
    # (many long-lived episodes) that may not be achievable at all parameter
    # combinations — it should not dominate the score.
    weights = {
        "duration":  0.30,
        "reversal":  0.40,
        "frequency": 0.20,
        "crossover": 0.10,
    }
    total = (
        weights["duration"]  * dur_error +
        weights["reversal"]  * rev_error +
        weights["frequency"] * freq_error +
        weights["crossover"] * cross_error
    )

    return EpisodeStructureScore(
        total_score=total,
        duration_ratio_error=dur_error,
        reversal_gradient_error=rev_error,
        hazard_crossover_error=cross_error,
        frequency_ratio_error=freq_error,
        sim_median_duration=sim_median_dur,
        emp_median_duration=emp_median_dur,
        sim_reversal_young=sim_rev_young,
        sim_reversal_mature=sim_rev_mature,
        emp_reversal_young=emp_rev_young,
        emp_reversal_mature=emp_rev_mature,
        sim_hazard_crossover=sim_crossover,
        emp_hazard_crossover=emp_crossover,
        sim_episode_frequency=sim_freq,
        emp_episode_frequency=emp_freq,
    )


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------


def episode_summary(
        episodes: list[ConsensusEpisode],
        n_total_steps: int,
        young_boundary: int,
        mature_boundary: int,
) -> dict:
    """
    Compute summary statistics for a list of ConsensusEpisodes.

    Mirrors the diagnostic block in the BSVE calibration artifact so
    results are directly comparable.

    Args:
        episodes: Extracted consensus episodes.
        n_total_steps: Total simulation steps (for frequency calculation).
        young_boundary: young_boundary_bars from calibration artifact.
        mature_boundary: mature_boundary_bars from calibration artifact.

    Returns:
        Dict matching the structure of calibration_artifact["diagnostics"].
    """
    n = len(episodes)
    if n == 0:
        return {
            "episode_count": 0,
            "episode_frequency_per_1000_steps": 0.0,
            "censoring_rate": 0.0,
            "median_episode_duration_steps": None,
            "reversal_rate_young": None,
            "reversal_rate_mature": None,
            "survival_counts": {str(k): 0 for k in (8, 16, 24, 32, 48)},
        }

    censored = [ep for ep in episodes if ep.exit_type == "censored"]
    completed = [ep for ep in episodes if ep.exit_type != "censored"]

    young_all = [ep for ep in episodes if ep.duration_steps < young_boundary]
    mature_all = [ep for ep in episodes if ep.duration_steps >= mature_boundary]

    def _rr(ep_list: list[ConsensusEpisode]) -> Optional[float]:
        if not ep_list:
            return None
        return round(
            sum(1 for ep in ep_list if ep.exit_type == "reversal") / len(ep_list), 4
        )

    completed_durations = [ep.duration_steps for ep in completed]
    median_dur = (
        round(float(np.median(completed_durations)), 2)
        if completed_durations else None
    )

    survival_counts = {
        str(t): int(sum(1 for ep in episodes if ep.duration_steps >= t))
        for t in (8, 16, 24, 32, 48)
    }

    return {
        "episode_count": n,
        "episode_frequency_per_1000_steps": round((n / n_total_steps) * 1000.0, 4),
        "censoring_rate": round(len(censored) / n, 4),
        "median_episode_duration_steps": median_dur,
        "reversal_rate_young": _rr(young_all),
        "reversal_rate_mature": _rr(mature_all),
        "survival_counts": survival_counts,
    }


# ---------------------------------------------------------------------------
# Validation helper (Stage 1.2 ground-truth check)
# ---------------------------------------------------------------------------


def validate_against_artifact(
        net_sentiment: np.ndarray | pd.Series,
        calibration_artifact: dict,
        pair: str = "unknown",
        tolerance: float = 0.10,
        verbose: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate that episode extraction on real data reproduces the BSVE
    calibration artifact diagnostics within tolerance.

    This is the Stage 1.2 ground-truth check. If this function does not
    pass on real JPY sentiment data, episode_utils.py cannot be used as
    an ABM calibration target.

    Args:
        net_sentiment: Real H1 net sentiment series for one pair.
        calibration_artifact: Loaded BSVE calibration artifact dict.
        pair: Pair name (for messages).
        tolerance: Relative tolerance for numeric comparisons (default 10%).
        verbose: Print comparison table to stdout.

    Returns:
        (passed, list_of_failures)
    """
    thresholds = calibration_artifact.get("thresholds", {})
    diagnostics = calibration_artifact.get("diagnostics", {})

    extreme_threshold = thresholds.get("extreme_threshold_net_pct")
    young_boundary = thresholds.get("young_boundary_bars")
    mature_boundary = thresholds.get("mature_boundary_bars")

    if any(v is None for v in (extreme_threshold, young_boundary, mature_boundary)):
        return False, ["Calibration artifact missing required threshold fields"]

    n_total = len(np.asarray(net_sentiment))
    episodes = extract_consensus_episodes(
        net_sentiment,
        extreme_threshold=extreme_threshold,
        min_episode_steps=2,
        pair=pair,
    )

    summary = episode_summary(
        episodes,
        n_total_steps=n_total,
        young_boundary=young_boundary,
        mature_boundary=mature_boundary,
    )

    failures = []

    def _check(label: str, actual: Optional[float], expected: Optional[float]) -> None:
        if expected is None or actual is None:
            if verbose:
                print(f"  {label}: actual={actual}  expected={expected}  SKIP (None)")
            return
        rel_err = abs(actual - expected) / (abs(expected) + 1e-12)
        ok = rel_err <= tolerance
        if verbose:
            status = "OK" if ok else "FAIL"
            print(
                f"  {label}: actual={actual:.4f}  expected={expected:.4f}"
                f"  rel_err={rel_err:.3f}  [{status}]"
            )
        if not ok:
            failures.append(
                f"{label}: actual={actual:.4f}, expected={expected:.4f}, "
                f"rel_err={rel_err:.3f} > tolerance={tolerance}"
            )

    # Artifact stores per-pair episode counts; total may differ from
    # running extraction on a single pair
    artifact_count_per_pair = diagnostics.get("episode_count_per_pair", {})
    emp_count = artifact_count_per_pair.get(pair)

    if verbose:
        print(f"\n[episode_utils] Validation for pair={pair}")
        print(f"  extreme_threshold : {extreme_threshold}")
        print(f"  young_boundary    : {young_boundary}")
        print(f"  mature_boundary   : {mature_boundary}")
        print(f"  n_sim_steps       : {n_total}")
        print(f"  n_episodes        : {summary['episode_count']} (artifact per-pair: {emp_count})")

    _check(
        "episode_count",
        float(summary["episode_count"]),
        float(emp_count) if emp_count is not None else None,
    )
    _check(
        "censoring_rate",
        summary["censoring_rate"],
        diagnostics.get("censoring_rate"),
    )
    _check(
        "median_episode_duration_steps",
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

    # Survival counts
    artifact_survival = diagnostics.get("survival_counts", {})
    for t in (8, 16, 24, 32, 48):
        _check(
            f"survival_count_{t}",
            float(summary["survival_counts"].get(str(t), 0)),
            float(artifact_survival.get(str(t), 0))
            if str(t) in artifact_survival else None,
        )

    passed = len(failures) == 0
    if verbose:
        print(f"\n  Result: {'PASSED' if passed else 'FAILED'}")
        if failures:
            for f in failures:
                print(f"    ✗ {f}")

    return passed, failures