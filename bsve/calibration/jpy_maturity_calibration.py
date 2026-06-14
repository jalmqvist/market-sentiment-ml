# bsve/calibration/jpy_maturity_calibration.py
"""
JPY consensus-state geometry calibration.

BSVE v1 validates the maturity dimension of the
Consensus-State Geometry ontology.

Future ontology versions may introduce additional
dimensions such as saturation, velocity, transition
structure, or hidden-state representations.

This calibration derives maturity boundaries only.

Estimates empirical maturity thresholds from the DL-active window
using hazard analysis on consensus state lifecycles.

Output: bsve/calibrations/reactive_jpy_calibration_v1.json

Usage:
    python -m bsve.calibration.jpy_maturity_calibration \
        --dataset-version 1.3.2 \
        --output-dir bsve/calibrations \
        --pairs USDJPY EURJPY GBPJPY \
        --start 2019-01-01 \
        --end 2026-12-31
"""

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConsensusLifecycle:
    """
    A single consensus state episode: from formation to exit.
    """
    pair: str
    entry_bar: pd.Timestamp
    exit_bar: Optional[pd.Timestamp]   # None if right-censored (still active)
    duration_bars: int                 # H1 bars from entry to exit/censor
    exit_type: str                     # 'reversal' | 'threshold' | 'censored'
    max_net_sentiment: float           # peak crowd positioning during episode
    entry_sentiment_net: float


@dataclass
class MaturityCalibrationResult:
    """
    Output of the calibration procedure.
    Committed as a versioned artifact before validation runs.
    """
    environment_id: str
    calibration_version: str
    dataset_version: str
    pairs: list[str]
    calibration_window_start: str
    calibration_window_end: str

    # Primary outputs for the maturity dimension of the
    # Consensus-State Geometry ontology.
    #
    # Additional ontology dimensions may be added in
    # future calibration versions without changing the
    # underlying environment definition.
    extreme_threshold_net_pct: float
    young_boundary_bars: int           # entry → maturing transition
    mature_boundary_bars: int          # maturing → mature transition

    # Diagnostic outputs — not used as thresholds, used for sign-off review
    n_episodes_total: int
    n_episodes_per_pair: dict
    reversal_rate_young: float         # reversal rate below young_boundary
    reversal_rate_mature: float        # reversal rate above mature_boundary
    hazard_crossover_bar: float        # bar where P(reversal) = P(threshold)
    median_episode_duration: float
    censoring_rate: float              # fraction of episodes still active at window end

    # Reproducibility
    calibration_hash: str              # hash of inputs + outputs for artifact integrity


# ---------------------------------------------------------------------------
# Sentiment extreme detection
# ---------------------------------------------------------------------------

def compute_extreme_threshold(
    sentiment_net: pd.Series,
    method: str = "percentile",
    percentile: float = 70.0,
) -> float:
    """
    Derive the extreme sentiment threshold from the empirical distribution.

    Args:
        sentiment_net: Series of net sentiment values (long% - short%).
        method: 'percentile' or 'fixed'.
        percentile: Percentile to use for threshold (default 70th).

    Returns:
        Threshold value. Bars where |sentiment_net| >= threshold
        are considered extreme.
    """
    if method == "fixed":
        return 60.0
    abs_net = sentiment_net.abs().dropna()
    threshold = float(np.percentile(abs_net, percentile))
    return threshold


# ---------------------------------------------------------------------------
# Lifecycle extraction
# ---------------------------------------------------------------------------

def extract_consensus_lifecycles(
    df: pd.DataFrame,
    pair: str,
    extreme_threshold: float,
    min_episode_bars: int = 2,
) -> list[ConsensusLifecycle]:
    """
    Extract discrete consensus state episodes from H1 sentiment data.

    An episode begins when |sentiment_net| crosses extreme_threshold
    and ends when it falls back below threshold (reversal) or when
    a price threshold exit is detected.

    Args:
        df: H1 dataframe with columns:
            entry_time, sentiment_net, exit_type (optional pre-labeled)
        pair: Currency pair name.
        extreme_threshold: |sentiment_net| >= this → extreme state.
        min_episode_bars: Discard episodes shorter than this.

    Returns:
        List of ConsensusLifecycle instances.
    """
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["is_extreme"] = df["sentiment_net"].abs() >= extreme_threshold

    lifecycles = []
    in_episode = False
    episode_start_idx = None

    for idx, row in df.iterrows():
        if not in_episode and row["is_extreme"]:
            # Episode entry
            in_episode = True
            episode_start_idx = idx

        elif in_episode and not row["is_extreme"]:
            # Episode exit via sentiment reset (reversal)
            duration = idx - episode_start_idx
            if duration >= min_episode_bars:
                start_row = df.loc[episode_start_idx]
                lifecycles.append(ConsensusLifecycle(
                    pair=pair,
                    entry_bar=start_row["entry_time"],
                    exit_bar=row["entry_time"],
                    duration_bars=duration,
                    exit_type="reversal",
                    max_net_sentiment=df.loc[
                        episode_start_idx:idx, "sentiment_net"
                    ].abs().max(),
                    entry_sentiment_net=float(start_row["sentiment_net"]),
                ))
            in_episode = False
            episode_start_idx = None

    # Handle right-censored episodes (still active at window end)
    if in_episode and episode_start_idx is not None:
        duration = len(df) - episode_start_idx
        if duration >= min_episode_bars:
            start_row = df.loc[episode_start_idx]
            lifecycles.append(ConsensusLifecycle(
                pair=pair,
                entry_bar=start_row["entry_time"],
                exit_bar=None,
                duration_bars=duration,
                exit_type="censored",
                max_net_sentiment=df.loc[
                    episode_start_idx:, "sentiment_net"
                ].abs().max(),
                entry_sentiment_net=float(start_row["sentiment_net"]),
            ))

    return lifecycles


# ---------------------------------------------------------------------------
# Hazard analysis
# ---------------------------------------------------------------------------

def compute_hazard_by_maturity(
    lifecycles: list[ConsensusLifecycle],
    max_bars: int = 200,
    min_at_risk: int = 10,
) -> pd.DataFrame:
    """
    Compute empirical reversal hazard rate as a function of maturity.

    Uses the Kaplan-Meier style discrete hazard estimator:
        h(t) = n_reversals_at_t / n_at_risk_at_t

    Args:
        lifecycles: Extracted consensus episodes.
        max_bars: Maximum maturity bar to compute hazard for.
        min_at_risk: Skip bars with fewer than this many episodes at risk.

    Returns:
        DataFrame with columns:
            maturity_bar, n_at_risk, n_reversals, hazard_rate,
            cumulative_survival
    """
    # Build event table
    records = []
    for lc in lifecycles:
        if lc.exit_type == "censored":
            records.append({"duration": lc.duration_bars, "event": 0})
        elif lc.exit_type == "reversal":
            records.append({"duration": lc.duration_bars, "event": 1})
        # threshold exits treated as censored for reversal hazard
        else:
            records.append({"duration": lc.duration_bars, "event": 0})

    event_df = pd.DataFrame(records)

    rows = []
    survival = 1.0

    for t in range(1, max_bars + 1):
        n_at_risk = (event_df["duration"] >= t).sum()
        if n_at_risk < min_at_risk:
            break
        n_events = (
            (event_df["duration"] == t) & (event_df["event"] == 1)
        ).sum()
        hazard = n_events / n_at_risk if n_at_risk > 0 else 0.0
        survival *= (1 - hazard)
        rows.append({
            "maturity_bar": t,
            "n_at_risk": int(n_at_risk),
            "n_reversals": int(n_events),
            "hazard_rate": hazard,
            "cumulative_survival": survival,
        })

    return pd.DataFrame(rows)


def find_hazard_crossover(
    hazard_df: pd.DataFrame,
    window: int = 12,
) -> float:
    """
    Find the maturity bar where reversal hazard rate stabilizes
    (i.e. the inflection point after the initial high-hazard zone).

    Uses a rolling mean to smooth the hazard curve, then finds
    the bar where the rate of change drops below a threshold.

    Returns:
        Estimated crossover bar (float, interpolated).
    """
    smoothed = hazard_df["hazard_rate"].rolling(window, center=True).mean()
    diff = smoothed.diff().abs()

    # Find where the hazard rate stops declining sharply
    # (second derivative approaches zero after initial drop).
    # Take the first qualifying bar — the earliest point where the curve
    # has settled, rather than an arbitrary aggregation of all stable bars.
    stable_idx = diff[diff < diff.quantile(0.25)].index
    if len(stable_idx) == 0:
        return float(hazard_df["maturity_bar"].median())

    crossover_bar = float(hazard_df.loc[stable_idx[0], "maturity_bar"])
    return crossover_bar


def derive_maturity_boundaries(
    hazard_df: pd.DataFrame,
    crossover_bar: float,
    young_fraction: float = 0.4,
    mature_fraction: float = 1.6,
) -> tuple[int, int]:
    """
    Derive young and mature boundary bars from the hazard crossover.

    young_boundary = crossover * young_fraction
        (well inside high-hazard zone)
    mature_boundary = crossover * mature_fraction
        (well past the crossover into stable zone)

    Both are rounded to the nearest H1 session boundary (8 bars).

    Args:
        hazard_df: Output of compute_hazard_by_maturity.
        crossover_bar: Estimated hazard crossover point.
        young_fraction: Multiplier for young boundary.
        mature_fraction: Multiplier for mature boundary.

    Returns:
        (young_boundary_bars, mature_boundary_bars)
    """
    def round_to_session(x: float, session_bars: int = 8) -> int:
        return max(session_bars, int(round(x / session_bars) * session_bars))

    young = round_to_session(crossover_bar * young_fraction)
    mature = round_to_session(crossover_bar * mature_fraction)

    # Sanity check
    if young >= mature:
        warnings.warn(
            f"young_boundary ({young}) >= mature_boundary ({mature}). "
            "Check crossover estimation. Falling back to fixed ratio."
        )
        young = round_to_session(crossover_bar * 0.33)
        mature = round_to_session(crossover_bar * 1.5)

    return young, mature


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_jpy_calibration(
    data: dict[str, pd.DataFrame],   # pair → H1 dataframe
    dataset_version: str,
    output_dir: Path,
    calibration_version: str = "v1",
    start: str = "2019-01-01",
    end: str = "2026-12-31",
) -> MaturityCalibrationResult:
    """
    Full JPY maturity calibration pipeline.

    Steps:
        1. Pool sentiment data across JPY pairs
        2. Derive extreme threshold from empirical distribution
        3. Extract consensus lifecycles per pair
        4. Compute hazard curve across pooled lifecycles
        5. Derive maturity boundaries from hazard crossover
        6. Compute diagnostic statistics
        7. Write versioned calibration artifact

    Args:
        data: Dict of pair → H1 DataFrame (must contain entry_time,
              sentiment_net columns).
        dataset_version: MSML dataset version string.
        output_dir: Directory to write calibration artifact.
        calibration_version: Version tag for this calibration run.
        start: DL-active window start.
        end: DL-active window end.

    Returns:
        MaturityCalibrationResult (also written to disk).
    """
    pairs = list(data.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Pool sentiment for threshold derivation
    all_sentiment = pd.concat(
        [df["sentiment_net"] for df in data.values()],
        ignore_index=True,
    ).dropna()

    # Step 2 — Extreme threshold
    extreme_threshold = compute_extreme_threshold(
        all_sentiment,
        method="percentile",
        percentile=70.0,
    )

    # Step 3 — Extract lifecycles per pair
    all_lifecycles: list[ConsensusLifecycle] = []
    n_episodes_per_pair = {}

    for pair, df in data.items():
        df_window = df[
            (df["entry_time"] >= start) & (df["entry_time"] <= end)
        ].copy()
        lifecycles = extract_consensus_lifecycles(
            df_window, pair, extreme_threshold
        )
        all_lifecycles.extend(lifecycles)
        n_episodes_per_pair[pair] = len(lifecycles)

    if len(all_lifecycles) < 50:
        raise ValueError(
            f"Only {len(all_lifecycles)} episodes found. "
            "Insufficient data for reliable calibration. "
            "Check sentiment data availability and extreme threshold."
        )

    # Step 4 — Hazard curve
    hazard_df = compute_hazard_by_maturity(all_lifecycles)

    # Step 5 — Boundaries
    crossover_bar = find_hazard_crossover(hazard_df)
    young_boundary, mature_boundary = derive_maturity_boundaries(
        hazard_df, crossover_bar
    )

    # Step 6 — Diagnostics
    reversal_episodes = [lc for lc in all_lifecycles
                         if lc.exit_type != "censored"]
    young_episodes = [lc for lc in reversal_episodes
                      if lc.duration_bars < young_boundary]
    mature_episodes = [lc for lc in reversal_episodes
                       if lc.duration_bars >= mature_boundary]

    reversal_rate_young = (
        sum(1 for lc in young_episodes if lc.exit_type == "reversal")
        / len(young_episodes) if young_episodes else float("nan")
    )
    reversal_rate_mature = (
        sum(1 for lc in mature_episodes if lc.exit_type == "reversal")
        / len(mature_episodes) if mature_episodes else float("nan")
    )
    censoring_rate = (
        sum(1 for lc in all_lifecycles if lc.exit_type == "censored")
        / len(all_lifecycles)
    )

    durations = [lc.duration_bars for lc in all_lifecycles
                 if lc.exit_type != "censored"]
    median_duration = float(np.median(durations)) if durations else float("nan")

    # Step 7 — Build result
    result_dict = {
        "environment_id": "reactive_jpy",
        "calibration_version": calibration_version,
        "dataset_version": dataset_version,
        "pairs": pairs,
        "calibration_window_start": start,
        "calibration_window_end": end,
        "extreme_threshold_net_pct": round(extreme_threshold, 2),
        "young_boundary_bars": young_boundary,
        "mature_boundary_bars": mature_boundary,
        "n_episodes_total": len(all_lifecycles),
        "n_episodes_per_pair": n_episodes_per_pair,
        "reversal_rate_young": round(reversal_rate_young, 4),
        "reversal_rate_mature": round(reversal_rate_mature, 4),
        "hazard_crossover_bar": round(crossover_bar, 2),
        "median_episode_duration": round(median_duration, 2),
        "censoring_rate": round(censoring_rate, 4),
    }

    # Compute deterministic hash over inputs + outputs for artifact integrity
    hash_payload = json.dumps(result_dict, sort_keys=True).encode()
    result_dict["calibration_hash"] = hashlib.sha256(hash_payload).hexdigest()

    result = MaturityCalibrationResult(**result_dict)

    # Write artifact
    artifact_path = (
        output_dir
        / f"reactive_jpy_calibration_{calibration_version}.json"
    )
    with open(artifact_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"[BSVE] JPY calibration complete → {artifact_path}")
    print(f"  extreme_threshold : {result.extreme_threshold_net_pct:.1f}%")
    print(f"  young_boundary    : {result.young_boundary_bars} bars")
    print(f"  mature_boundary   : {result.mature_boundary_bars} bars")
    print(f"  hazard_crossover  : {result.hazard_crossover_bar:.1f} bars")
    print(f"  n_episodes        : {result.n_episodes_total}")
    print(f"  reversal_rate_young  : {result.reversal_rate_young:.1%}")
    print(f"  reversal_rate_mature : {result.reversal_rate_mature:.1%}")
    print(f"  censoring_rate    : {result.censoring_rate:.1%}")

    # Emit a sign-off reminder — thresholds must be reviewed before
    # committing to state spec
    print(
        "\n[BSVE] ⚠ Review diagnostics before committing thresholds to "
        "reactive_jpy_v1.yaml.\n"
        "  Expected: reversal_rate_young >> reversal_rate_mature\n"
        "  If this does not hold, the maturity hypothesis is not supported "
        "and thresholds should not be committed."
    )

    return result


# ---------------------------------------------------------------------------
# Diagnostic plot (optional, for sign-off review)
# ---------------------------------------------------------------------------

def plot_hazard_curve(
    hazard_df: pd.DataFrame,
    young_boundary: int,
    mature_boundary: int,
    crossover_bar: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot the empirical hazard curve with boundary annotations.
    Used for visual sign-off during calibration review.
    Not used in automated validation runs.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[BSVE] matplotlib not available. Skipping hazard plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top panel — hazard rate
    ax = axes
    ax.plot(
        hazard_df["maturity_bar"],
        hazard_df["hazard_rate"],
        color="steelblue",
        linewidth=1.5,
        label="Empirical hazard rate",
    )
    smoothed = (
        hazard_df["hazard_rate"].rolling(12, center=True).mean()
    )
    ax.plot(
        hazard_df["maturity_bar"],
        smoothed,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label="Smoothed (12-bar rolling mean)",
    )
    ax.axvline(young_boundary, color="green", linestyle=":",
               label=f"Young boundary ({young_boundary}h)")
    ax.axvline(mature_boundary, color="red", linestyle=":",
               label=f"Mature boundary ({mature_boundary}h)")
    ax.axvline(crossover_bar, color="purple", linestyle="-.",
               alpha=0.6, label=f"Crossover ({crossover_bar:.0f}h)")
    ax.set_xlabel("Consensus maturity (H1 bars)")
    ax.set_ylabel("Reversal hazard rate")
    ax.set_title("JPY Consensus State — Empirical Reversal Hazard")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom panel — survival curve
    ax2 = axes
    ax2.plot(
        hazard_df["maturity_bar"],
        hazard_df["cumulative_survival"],
        color="steelblue",
        linewidth=1.5,
    )
    ax2.axvline(young_boundary, color="green", linestyle=":")
    ax2.axvline(mature_boundary, color="red", linestyle=":")
    ax2.set_xlabel("Consensus maturity (H1 bars)")
    ax2.set_ylabel("Cumulative survival (no reversal)")
    ax2.set_title("JPY Consensus State — Kaplan-Meier Survival")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[BSVE] Hazard plot saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CalibrationPlugin implementation
# ---------------------------------------------------------------------------

class JPYMaturityCalibrationPlugin:
    """
    Calibration plugin for the ``reactive_jpy`` ontology (v1).

    Implements :class:`~bsve.calibration.registry.CalibrationPlugin`.

    Produces a versioned :data:`~bsve.calibration.calibration_contract.CalibrationArtifact`
    containing:

    * ``thresholds`` — empirically derived maturity boundaries.
    * ``diagnostics`` — informational statistics (not validation outcomes).

    Null artifacts are emitted (rather than exceptions) when quality gates
    fail due to insufficient data or unstable estimates.

    This plugin calibrates thresholds only.  It does NOT assign states.
    """

    # Default calibration parameters — all can be overridden via
    # ``calibration_params`` at call time, which in turn can be loaded
    # from ``bsve_config_v1.yaml``.
    _DEFAULTS: dict = {
        "calibration_method": "hazard_analysis",
        "extreme_percentile": 70.0,
        "min_episode_count": 50,
        "min_sample_count": 500,
        "hazard_smoothing_window": 12,
        "young_fraction": 0.4,
        "mature_fraction": 1.6,
        "calibration_window_start": "2019-01-01",
        "calibration_window_end": "2026-12-31",
        "pairs": ["USDJPY", "EURJPY", "GBPJPY"],
        "diagnostic_percentiles": [10, 25, 50, 75, 90],
    }

    def calibrate(
        self,
        dataset_adapter: Any,
        state_spec: dict,
        calibration_params: dict,
    ) -> dict:
        """
        Run JPY maturity calibration end-to-end.

        Steps:
            1. Resolve effective parameters (params override defaults).
            2. Load sentiment observations through ``dataset_adapter``.
            3. Apply calibration window filter.
            4. Quality-gate: sample count.
            5. Derive extreme-consensus threshold from empirical distribution.
            6. Extract consensus lifecycles per pair.
            7. Quality-gate: episode count.
            8. Compute hazard curve and derive maturity boundaries.
            9. Compute calibration diagnostics.
            10. Emit a success or null artifact.

        Args:
            dataset_adapter: A
                :class:`~bsve.adapters.dataset_adapter.MasterResearchDatasetAdapter`
                instance (or any compatible object).
            state_spec: Parsed state-spec YAML for ``reactive_jpy``.
            calibration_params: Run-time parameters; merged on top of
                :attr:`_DEFAULTS`.

        Returns:
            A :data:`~bsve.calibration.calibration_contract.CalibrationArtifact`
            with ``outcome`` = ``"success"`` or ``"null"``.
        """
        from bsve.calibration.calibration_contract import build_calibration_artifact
        from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter

        # ------------------------------------------------------------------
        # Step 1 — Resolve effective parameters
        # ------------------------------------------------------------------
        p = {**self._DEFAULTS, **calibration_params}

        calibration_id: str = p["calibration_id"]
        ontology_id: str = p.get("ontology_id", "reactive_jpy")
        ontology_version: str = p.get("ontology_version", "1.0.0")
        window_start: str = p["calibration_window_start"]
        window_end: str = p["calibration_window_end"]
        dataset_version: str = p.get("dataset_version", "unknown")
        calibration_method: str = p["calibration_method"]
        extreme_percentile: float = float(p["extreme_percentile"])
        min_episode_count: int = int(p["min_episode_count"])
        min_sample_count: int = int(p["min_sample_count"])
        hazard_window: int = int(p["hazard_smoothing_window"])
        young_fraction: float = float(p["young_fraction"])
        mature_fraction: float = float(p["mature_fraction"])
        raw_pairs: list = list(p["pairs"])
        diagnostic_percentiles: list = list(p["diagnostic_percentiles"])

        # Normalize pair names to match adapter internals.
        normalized_pairs = [
            MasterResearchDatasetAdapter.normalize_pair(pair)
            for pair in raw_pairs
        ]

        def _null(reason: str) -> dict:
            """Emit a null artifact with a diagnostic reason."""
            return build_calibration_artifact(
                calibration_id=calibration_id,
                ontology_id=ontology_id,
                ontology_version=ontology_version,
                calibration_window_start=window_start,
                calibration_window_end=window_end,
                dataset_version=dataset_version,
                calibration_method=calibration_method,
                outcome="null",
                null_reason=reason,
            )

        # ------------------------------------------------------------------
        # Step 2 — Load sentiment observations
        # ------------------------------------------------------------------
        pair_col = dataset_adapter.config.pair_col
        ts_col = dataset_adapter.config.timestamp_col

        try:
            obs = dataset_adapter.get_sentiment_observations(
                pairs=normalized_pairs,
                columns=["net_sentiment"],
            )
        except Exception as exc:  # pragma: no cover — adapter contract error
            return _null(f"Failed to load sentiment observations: {exc}")

        # ------------------------------------------------------------------
        # Step 3 — Apply calibration window filter
        # ------------------------------------------------------------------
        ts = obs[ts_col]
        mask = (ts >= pd.Timestamp(window_start)) & (ts <= pd.Timestamp(window_end))
        obs = obs[mask].copy()

        # ------------------------------------------------------------------
        # Step 4 — Quality gate: sample count
        # ------------------------------------------------------------------
        n_samples = len(obs)
        if n_samples < min_sample_count:
            return _null(
                f"Insufficient observations in calibration window: "
                f"{n_samples} < {min_sample_count} (min_sample_count)"
            )

        # ------------------------------------------------------------------
        # Step 5 — Derive extreme-consensus threshold
        # ------------------------------------------------------------------
        # The adapter exposes the column as ``net_sentiment``; the lifecycle
        # extraction helpers expect ``sentiment_net``.  We rename after
        # fetching to keep the adapter interface clean.
        sentiment_series = obs["net_sentiment"].dropna()
        extreme_threshold = compute_extreme_threshold(
            sentiment_series,
            method="percentile",
            percentile=extreme_percentile,
        )

        # ------------------------------------------------------------------
        # Step 6 — Extract consensus lifecycles per pair
        # ------------------------------------------------------------------
        all_lifecycles: list[ConsensusLifecycle] = []
        n_episodes_per_pair: dict = {}

        for pair in normalized_pairs:
            pair_data = obs[obs[pair_col] == pair].copy()
            # Rename to match lifecycle extraction conventions.
            pair_data = pair_data.rename(
                columns={"net_sentiment": "sentiment_net", ts_col: "entry_time"}
            )
            try:
                lifecycles = extract_consensus_lifecycles(
                    pair_data, pair, extreme_threshold
                )
            except Exception as exc:  # pragma: no cover
                lifecycles = []
                warnings.warn(
                    f"[JPYCalibration] lifecycle extraction failed for {pair}: {exc}"
                )
            all_lifecycles.extend(lifecycles)
            n_episodes_per_pair[pair] = len(lifecycles)

        # ------------------------------------------------------------------
        # Step 7 — Quality gate: episode count
        # ------------------------------------------------------------------
        n_episodes = len(all_lifecycles)
        if n_episodes < min_episode_count:
            return _null(
                f"Insufficient consensus episodes: "
                f"{n_episodes} < {min_episode_count} (min_episode_count). "
                f"Check extreme_percentile ({extreme_percentile}) and data availability."
            )

        # ------------------------------------------------------------------
        # Step 8 — Hazard analysis and boundary derivation
        # ------------------------------------------------------------------
        hazard_df = compute_hazard_by_maturity(all_lifecycles)

        if hazard_df.empty:
            return _null(
                "Hazard analysis produced an empty result — too few episodes "
                "reached the minimum at-risk count per maturity bar."
            )

        crossover_bar = find_hazard_crossover(hazard_df, window=hazard_window)

        try:
            young_boundary, mature_boundary = derive_maturity_boundaries(
                hazard_df,
                crossover_bar,
                young_fraction=young_fraction,
                mature_fraction=mature_fraction,
            )
        except Exception as exc:  # pragma: no cover
            return _null(f"Boundary derivation failed: {exc}")

        # Stability check: boundaries must be positive and ordered.
        if not (0 < young_boundary < mature_boundary):
            return _null(
                f"Derived boundaries are unstable: "
                f"young_boundary={young_boundary}, mature_boundary={mature_boundary}. "
                "Hazard crossover estimate may be unreliable."
            )

        # ------------------------------------------------------------------
        # Step 9 — Compute calibration diagnostics
        # ------------------------------------------------------------------
        completed = [lc for lc in all_lifecycles if lc.exit_type != "censored"]
        censored = [lc for lc in all_lifecycles if lc.exit_type == "censored"]

        young_eps = [lc for lc in completed if lc.duration_bars < young_boundary]
        mature_eps = [lc for lc in completed if lc.duration_bars >= mature_boundary]

        reversal_rate_young = (
            sum(1 for lc in young_eps if lc.exit_type == "reversal") / len(young_eps)
            if young_eps else float("nan")
        )
        reversal_rate_mature = (
            sum(1 for lc in mature_eps if lc.exit_type == "reversal") / len(mature_eps)
            if mature_eps else float("nan")
        )
        censoring_rate = len(censored) / n_episodes if n_episodes else 0.0

        durations = [lc.duration_bars for lc in all_lifecycles]
        completed_durations = [lc.duration_bars for lc in completed]
        median_duration = (
            float(np.median(completed_durations)) if completed_durations else float("nan")
        )

        # Maturity distribution percentiles (informational).
        maturity_pct: dict = {}
        if durations:
            for pct in diagnostic_percentiles:
                maturity_pct[f"p{pct}"] = float(np.percentile(durations, pct))

        # Calibration window coverage in calendar days.
        try:
            coverage_days = (
                pd.Timestamp(window_end) - pd.Timestamp(window_start)
            ).days
        except Exception:  # pragma: no cover
            coverage_days = None

        diagnostics = {
            "sample_count": n_samples,
            "episode_count": n_episodes,
            "episode_count_per_pair": n_episodes_per_pair,
            "censoring_rate": round(censoring_rate, 4),
            "median_episode_duration_bars": round(median_duration, 2),
            "reversal_rate_young": (
                round(reversal_rate_young, 4) if not np.isnan(reversal_rate_young) else None
            ),
            "reversal_rate_mature": (
                round(reversal_rate_mature, 4) if not np.isnan(reversal_rate_mature) else None
            ),
            "hazard_crossover_bar": round(crossover_bar, 2),
            "extreme_threshold_used": round(extreme_threshold, 4),
            "maturity_distribution_percentiles": maturity_pct,
            "calibration_window_coverage_days": coverage_days,
        }

        # ------------------------------------------------------------------
        # Step 10 — Build and return calibration artifact
        # ------------------------------------------------------------------
        calibration_mode: str = p.get("calibration_mode", "research")

        threshold_provenance = {
            "extreme_threshold_net_pct": {
                "method": "percentile",
                "parameter": extreme_percentile,
            },
            "young_boundary_bars": {
                "method": "hazard_crossover_fraction",
                "fraction": young_fraction,
            },
            "mature_boundary_bars": {
                "method": "hazard_crossover_fraction",
                "fraction": mature_fraction,
            },
        }

        return build_calibration_artifact(
            calibration_id=calibration_id,
            ontology_id=ontology_id,
            ontology_version=ontology_version,
            calibration_window_start=window_start,
            calibration_window_end=window_end,
            dataset_version=dataset_version,
            calibration_method=calibration_method,
            outcome="success",
            thresholds={
                "extreme_threshold_net_pct": round(extreme_threshold, 2),
                "young_boundary_bars": young_boundary,
                "mature_boundary_bars": mature_boundary,
            },
            diagnostics=diagnostics,
            calibration_mode=calibration_mode,
            threshold_provenance=threshold_provenance,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="JPY consensus maturity boundary calibration"
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", default="bsve/calibrations")
    parser.add_argument(
        "--pairs", nargs="+",
        default=["USDJPY", "EURJPY", "GBPJPY"]
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--plot", action="store_true",
                        help="Emit hazard curve plot for sign-off review")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Data loading — adapt to your existing dataset loader
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

    result = run_jpy_calibration(
        data=data,
        dataset_version=args.dataset_version,
        output_dir=Path(args.output_dir),
        start=args.start,
        end=args.end,
    )

    if args.plot:
        # Recompute hazard for plotting — kept separate from
        # calibration logic to avoid side effects
        from bsve.calibration.jpy_maturity_calibration import (
            extract_consensus_lifecycles,
            compute_hazard_by_maturity,
        )
        all_lc = []
        for pair, df in data.items():
            all_lc.extend(extract_consensus_lifecycles(
                df, pair, result.extreme_threshold_net_pct
            ))
        hazard_df = compute_hazard_by_maturity(all_lc)
        plot_hazard_curve(
            hazard_df,
            result.young_boundary_bars,
            result.mature_boundary_bars,
            result.hazard_crossover_bar,
            output_path=Path(args.output_dir) / "jpy_hazard_curve.png",
        )
