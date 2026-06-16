"""
BSVE Rule-Based State Assignment Engine.

Phase 3 of the BSVE roadmap: deterministic state assignment using committed
calibration artifacts.

Responsibilities
----------------
1. Load environment specification (YAML).
2. Load and validate committed calibration artifact (fail-fast).
3. Inject calibrated thresholds.
4. Assign behavioral states deterministically, per-pair.
5. Track state transitions.
6. Produce versioned BSVE state surface parquet artifacts.
7. Generate a run manifest.
8. Emit dry-run diagnostics.

Non-responsibilities
--------------------
* No calibration, threshold fitting, or optimization logic.
* No ontology discovery.
* No HMM or learned-boundary logic.

CLI usage
---------
    python -m bsve.state_machine.rule_based \\
        --dataset-path data/output/1.5.0/master_research_dataset_core.csv \\
        --environment reactive_jpy \\
        --output-dir bsve_outputs/
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter
from bsve.artifacts.io import write_bsve_artifact
from bsve.calibration.calibration_contract import load_calibration_artifact
from bsve.calibration.calibration_runner import load_state_spec
from bsve.features.consensus import compute_consensus_maturity
from schemas.bsve_artifact_schema import (
    BSVE_SCHEMA_VERSION,
    BSVE_MATURITY_CLASS_VALUES,
    BSVE_TRANSITION_EVENT_VALUES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_MACHINE_VERSION = "1.0.0"

# Calibration artifact directory (committed, immutable artifacts).
_DEFAULT_CALIBRATION_ARTIFACTS_DIR = Path("bsve/calibration_artifacts")
_DEFAULT_STATE_SPECS_DIR = Path("bsve/state_specs")


# ---------------------------------------------------------------------------
# Calibration loading / validation gate
# ---------------------------------------------------------------------------


def assert_calibrations_valid(artifact_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a committed calibration artifact.

    Fails immediately if:
    * artifact file is missing
    * artifact JSON is malformed
    * artifact hash does not verify
    * artifact outcome is "null"
    * artifact thresholds are missing or empty

    Args:
        artifact_path: Path to the calibration artifact JSON file.

    Returns:
        Validated artifact dict.

    Raises:
        FileNotFoundError: Artifact file not found.
        ValueError: Artifact is invalid, has a hash mismatch, or is a
            null-calibration artifact.
    """
    artifact = load_calibration_artifact(artifact_path, strict=True)

    if artifact.get("outcome") == "null":
        raise ValueError(
            f"Calibration artifact at {artifact_path!r} is a null-calibration "
            "artifact. Null-calibration artifacts indicate that no stable "
            "behavioral ontology was found for this environment. State "
            "assignment cannot proceed."
        )

    thresholds = artifact.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError(
            f"Calibration artifact at {artifact_path!r} contains no thresholds. "
            "State assignment cannot proceed."
        )

    return artifact


# ---------------------------------------------------------------------------
# Maturity classification
# ---------------------------------------------------------------------------


def classify_maturity(
    maturity_bars: int,
    *,
    young_boundary: int,
    mature_boundary: int,
) -> str:
    """
    Assign a maturity class from a bar count.

    Classes (for extreme-consensus bars only):
        young    — maturity_bars < young_boundary
        maturing — young_boundary <= maturity_bars < mature_boundary
        mature   — maturity_bars >= mature_boundary

    For non-extreme bars, callers should use "n_a".

    Args:
        maturity_bars: Number of consecutive H1 bars since episode start.
        young_boundary: Bars threshold separating young from maturing.
        mature_boundary: Bars threshold separating maturing from mature.

    Returns:
        One of "young", "maturing", "mature".
    """
    if maturity_bars < young_boundary:
        return "young"
    if maturity_bars < mature_boundary:
        return "maturing"
    return "mature"


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------


def classify_transition(
    prev_state_id: str | None,
    curr_state_id: str,
    prev_is_extreme: bool,
    curr_is_extreme: bool,
) -> str:
    """
    Classify the transition event for the current bar.

    Transition definitions
    ----------------------
    entry
        This bar is the first bar of a new extreme-consensus episode.
        Recorded on:
        * The first bar in the dataset for a pair (prev_state_id is None).
        * The first extreme bar following a non-extreme bar (new episode start).

    continuation
        State is unchanged from the previous bar.

    exit_reversal
        Sentiment leaves the extreme condition (was extreme, now is not).

    exit_threshold
        **This bar entered via a maturity-boundary crossing.**
        Both the previous and current bars are extreme, but the state_id
        changed because a maturity threshold was crossed.  The event is
        recorded on the *receiving* row — the first row of the new maturity
        class — not on the last row of the prior class.

        Examples:
        * Young → Maturing: ``exit_threshold`` is recorded on the first
          Maturing row.
        * Maturing → Mature: ``exit_threshold`` is recorded on the first
          Mature row.

    exit_unknown
        State changed by an unknown mechanism (fallback).

    This function is generic and contains no environment-specific branching.

    Args:
        prev_state_id: State ID of the immediately preceding bar, or None
            if this is the first bar in the dataset for this pair.
        curr_state_id: State ID assigned to the current bar.
        prev_is_extreme: Whether the previous bar was in an extreme state.
        curr_is_extreme: Whether the current bar is in an extreme state.

    Returns:
        One of the :data:`BSVE_TRANSITION_EVENT_VALUES`.
    """
    if prev_state_id is None:
        return "entry"

    if curr_state_id == prev_state_id:
        return "continuation"

    # State changed — classify the reason.
    if prev_is_extreme and not curr_is_extreme:
        return "exit_reversal"

    if prev_is_extreme and curr_is_extreme:
        # Both bars are extreme but the state changed — maturity-boundary
        # crossing.  Record exit_threshold on this (receiving) row.
        return "exit_threshold"

    if not prev_is_extreme and curr_is_extreme:
        # First extreme bar after a non-extreme bar — new episode starting.
        return "entry"

    return "exit_unknown"


# ---------------------------------------------------------------------------
# Episode-level outcome classification
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    """Deterministic outcome label for a single consensus episode.

    Attributes
    ----------
    max_maturity_bars:
        The peak maturity (bar count) reached during the episode.
    outcome_type:
        One of ``exit_reversal``, ``exit_threshold``,
        ``exit_late_reversal``, or ``exit_unknown``.
    """

    max_maturity_bars: int
    outcome_type: str


def classify_episode_outcome(
    max_maturity_bars: int,
    *,
    mature_boundary: int,
) -> EpisodeOutcome:
    """Classify the outcome of a single consensus episode.

    Classification logic
    --------------------
    exit_reversal
        Episode ended before reaching the mature boundary.
        ``max_maturity_bars < mature_boundary``.

    exit_threshold
        Episode reached the mature state and terminated within a normal
        mature lifecycle duration.
        ``mature_boundary <= max_maturity_bars < 2 * mature_boundary``.

    exit_late_reversal
        Episode survived well beyond the mature boundary before collapsing.
        ``max_maturity_bars >= 2 * mature_boundary``.

    exit_unknown
        Fallback — should remain rare (only for non-positive bar counts).

    Args:
        max_maturity_bars: Peak maturity bar count for the episode.
        mature_boundary: Calibrated mature-boundary bar count.

    Returns:
        :class:`EpisodeOutcome` with ``outcome_type`` set.
    """
    if max_maturity_bars <= 0:
        return EpisodeOutcome(max_maturity_bars=max_maturity_bars, outcome_type="exit_unknown")
    if max_maturity_bars < mature_boundary:
        outcome_type = "exit_reversal"
    elif max_maturity_bars < 2 * mature_boundary:
        outcome_type = "exit_threshold"
    else:
        outcome_type = "exit_late_reversal"
    return EpisodeOutcome(max_maturity_bars=max_maturity_bars, outcome_type=outcome_type)


def apply_episode_outcomes(
    df: pd.DataFrame,
    *,
    mature_boundary: int,
) -> pd.DataFrame:
    """Annotate the terminal bar of each consensus episode with an outcome label.

    For every contiguous run of extreme-consensus bars per pair, the last bar's
    ``transition_event`` is overwritten with the deterministic episode outcome
    (``exit_reversal``, ``exit_threshold``, or ``exit_late_reversal``).

    Non-extreme bars and non-terminal episode bars are not modified.

    This function is intentionally separate from state assignment so that
    outcome labeling can be audited independently of the bar-level state logic.

    Args:
        df: State assignment artifact DataFrame (with ``pair``, ``entry_time``,
            ``state_id``, ``maturity_bars``, and ``transition_event`` columns).
        mature_boundary: Calibrated mature-boundary bar count.

    Returns:
        A copy of *df* with terminal-bar ``transition_event`` values updated.
    """
    out = df.copy()

    for pair, grp in out.groupby("pair", sort=False):
        grp = grp.sort_values("entry_time")
        is_extreme = grp["state_id"] != "JPY_NON_EXTREME"
        run_ids = (is_extreme != is_extreme.shift()).cumsum()

        for _, run in grp.groupby(run_ids, sort=True):
            if run["state_id"].iloc[0] == "JPY_NON_EXTREME":
                continue

            max_maturity = int(run["maturity_bars"].max())
            outcome = classify_episode_outcome(max_maturity, mature_boundary=mature_boundary)
            last_idx = run.index[-1]
            out.at[last_idx, "transition_event"] = outcome.outcome_type

    return out


# ---------------------------------------------------------------------------
# State assignment — reactive-JPY ontology
# ---------------------------------------------------------------------------


def assign_states_reactive_jpy(
    df: pd.DataFrame,
    *,
    calibration_artifact: dict[str, Any],
    spec_id: str,
    pair_col: str = "pair",
    timestamp_col: str = "entry_time",
    sentiment_col: str = "net_sentiment",
    environment_id: str = "reactive_jpy",
    state_version: str = "1.0.0",
) -> pd.DataFrame:
    """
    Assign Reactive-JPY behavioral states to a sentiment timeseries.

    Thresholds are loaded exclusively from the calibration artifact.
    No threshold values are hardcoded here.

    State definitions
    -----------------
    JPY_NON_EXTREME
        Sentiment is not extreme (|net_sentiment| < extreme_threshold).

    JPY_CONSENSUS_YOUNG
        Extreme sentiment, episode maturity < young_boundary.

    JPY_CONSENSUS_MATURING
        Extreme sentiment, young_boundary <= maturity < mature_boundary.

    JPY_CONSENSUS_MATURE
        Extreme sentiment, maturity >= mature_boundary.

    Episode independence
    --------------------
    Maturity resets on every new extreme episode.  State identity does NOT
    survive a sentiment reset.

    Args:
        df: Input dataframe with at minimum ``pair_col``, ``timestamp_col``,
            and ``sentiment_col`` columns.  Must be sorted by
            (pair, timestamp) ascending before calling this function.
        calibration_artifact: Validated calibration artifact dict.
        spec_id: Spec identifier to record in the output artifact.
        pair_col: Column name for pair identifier.
        timestamp_col: Column name for bar entry timestamp.
        sentiment_col: Column name for net sentiment (signed, percentage).
        environment_id: Environment ID to record in the output artifact.
        state_version: State version string to record in the output artifact.

    Returns:
        DataFrame matching the BSVE artifact schema, one row per input row.

    Raises:
        KeyError: If required columns are missing from *df*.
        ValueError: If calibration thresholds are missing.
    """
    thresholds = calibration_artifact.get("thresholds", {})
    extreme_threshold = thresholds.get("extreme_threshold_net_pct")
    young_boundary = thresholds.get("young_boundary_bars")
    mature_boundary = thresholds.get("mature_boundary_bars")

    if extreme_threshold is None or young_boundary is None or mature_boundary is None:
        missing = [
            k for k, v in {
                "extreme_threshold_net_pct": extreme_threshold,
                "young_boundary_bars": young_boundary,
                "mature_boundary_bars": mature_boundary,
            }.items() if v is None
        ]
        raise ValueError(
            f"Calibration artifact is missing required thresholds: {missing}"
        )

    calibration_id = calibration_artifact.get("calibration_id", "")

    working = df[[pair_col, timestamp_col, sentiment_col]].copy()
    working = working.sort_values([pair_col, timestamp_col]).reset_index(drop=True)

    # Derive extreme flag.
    working["_is_extreme"] = working[sentiment_col].abs() >= float(extreme_threshold)

    # Compute episode-local maturity using the centralized feature registry.
    working["_maturity_bars"] = compute_consensus_maturity(
        working,
        pair_col=pair_col,
        extreme_flag_col="_is_extreme",
    )

    # Assign state and maturity class.
    state_ids = []
    maturity_classes = []
    transition_events = []

    prev_state_id: dict[str, str | None] = {}
    prev_is_extreme: dict[str, bool] = {}

    for _, row in working.iterrows():
        pair = row[pair_col]
        is_extreme = bool(row["_is_extreme"])
        maturity_bars = int(row["_maturity_bars"])

        if is_extreme:
            mc = classify_maturity(
                maturity_bars,
                young_boundary=int(young_boundary),
                mature_boundary=int(mature_boundary),
            )
            if mc == "young":
                sid = "JPY_CONSENSUS_YOUNG"
            elif mc == "maturing":
                sid = "JPY_CONSENSUS_MATURING"
            else:
                sid = "JPY_CONSENSUS_MATURE"
        else:
            mc = "n_a"
            sid = "JPY_NON_EXTREME"

        te = classify_transition(
            prev_state_id.get(pair),
            sid,
            prev_is_extreme.get(pair, False),
            is_extreme,
        )

        state_ids.append(sid)
        maturity_classes.append(mc)
        transition_events.append(te)

        prev_state_id[pair] = sid
        prev_is_extreme[pair] = is_extreme

    working["state_id"] = state_ids
    working["maturity_class"] = maturity_classes
    working["transition_event"] = transition_events

    # Build the BSVE artifact DataFrame.
    # prediction_available_timestamp is entry_time + 1 H1 bar (causal boundary).
    entry_times = pd.to_datetime(working[timestamp_col]).dt.tz_localize(None)
    available_times = entry_times + pd.Timedelta(hours=1)

    artifact_df = pd.DataFrame(
        {
            "entry_time": entry_times,
            "prediction_available_timestamp": available_times,
            "pair": working[pair_col].values,
            "environment_id": environment_id,
            "state_id": working["state_id"].values,
            "state_version": state_version,
            "maturity_bars": working["_maturity_bars"].values,
            "maturity_class": working["maturity_class"].values,
            "state_confidence": 1.0,
            "transition_event": working["transition_event"].values,
            "spec_id": spec_id,
            "calibration_id": calibration_id,
        }
    )

    # Apply episode-level outcome labels to terminal bars of each consensus
    # episode.  This overwrites the bar-level transition_event on the last bar
    # of each extreme-consensus run with the deterministic episode outcome
    # (exit_reversal / exit_threshold / exit_late_reversal).
    artifact_df = apply_episode_outcomes(
        artifact_df,
        mature_boundary=int(mature_boundary),
    )

    return artifact_df


# ---------------------------------------------------------------------------
# Artifact validation (fail-fast, pre-write)
# ---------------------------------------------------------------------------


def validate_state_artifact(df: pd.DataFrame) -> None:
    """
    Fail-fast validation of a state assignment artifact before writing.

    Checks:
    * No null state_id values.
    * maturity_bars >= 0 for all rows.
    * entry_time < prediction_available_timestamp for all rows.
    * calibration_id present (non-empty) for all rows.
    * spec_id present (non-empty) for all rows.

    Args:
        df: State assignment artifact DataFrame.

    Raises:
        ValueError: If any validation check fails.  Does NOT silently repair.
    """
    null_states = df["state_id"].isna().sum()
    if null_states > 0:
        raise ValueError(
            f"Artifact validation failed: {null_states} row(s) have null state_id."
        )

    bad_maturity = (df["maturity_bars"] < 0).sum()
    if bad_maturity > 0:
        raise ValueError(
            f"Artifact validation failed: {bad_maturity} row(s) have "
            "maturity_bars < 0."
        )

    entry = pd.to_datetime(df["entry_time"], errors="coerce")
    available = pd.to_datetime(df["prediction_available_timestamp"], errors="coerce")
    causal_violations = (entry >= available).sum()
    if causal_violations > 0:
        raise ValueError(
            f"Artifact validation failed: {causal_violations} row(s) have "
            "entry_time >= prediction_available_timestamp (causal ordering violation)."
        )

    missing_calibration_id = df["calibration_id"].isna() | (
        df["calibration_id"].astype(str).str.strip() == ""
    )
    if missing_calibration_id.any():
        raise ValueError(
            "Artifact validation failed: one or more rows have missing "
            "or empty calibration_id."
        )

    missing_spec_id = df["spec_id"].isna() | (
        df["spec_id"].astype(str).str.strip() == ""
    )
    if missing_spec_id.any():
        raise ValueError(
            "Artifact validation failed: one or more rows have missing "
            "or empty spec_id."
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _compute_episode_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute episode-level statistics from a state assignment artifact.

    An **episode** is a run of consecutive extreme bars within a pair.
    Maturity-class changes (Young → Maturing → Mature) within the same
    continuous extreme run do **not** split the episode — they all belong to
    the same episode.  Only a non-extreme bar breaks an episode.

    Non-extreme rows are excluded from episode accounting entirely.
    """
    durations_by_pair: dict[str, list[int]] = {}
    for pair, group in df.groupby("pair", sort=False):
        group = group.reset_index(drop=True)
        # Identify extreme bars (any state other than JPY_NON_EXTREME).
        is_extreme = group["state_id"] != "JPY_NON_EXTREME"
        # Assign a run-ID to each contiguous block of rows with the same
        # extreme flag.  Each distinct run gets a unique integer.
        extreme_run_group = (is_extreme != is_extreme.shift()).cumsum()
        pair_durations = []
        for _, run_group in group.groupby(extreme_run_group, sort=True):
            # Only count runs that are extreme-consensus episodes.
            if run_group["state_id"].iloc[0] != "JPY_NON_EXTREME":
                pair_durations.append(len(run_group))
        durations_by_pair[str(pair)] = pair_durations

    all_durations = [d for durs in durations_by_pair.values() for d in durs]

    survival_counts: dict[str, int] = {}
    for threshold in [8, 16, 24, 32, 48]:
        survival_counts[str(threshold)] = sum(d >= threshold for d in all_durations)

    return {
        "episodes_per_pair": {p: len(d) for p, d in durations_by_pair.items()},
        "episode_duration_distribution": {
            "min": int(min(all_durations)) if all_durations else 0,
            "median": float(
                sorted(all_durations)[len(all_durations) // 2]
            ) if all_durations else 0.0,
            "max": int(max(all_durations)) if all_durations else 0,
            "total_episodes": len(all_durations),
        },
        "survival_counts": survival_counts,
    }


def generate_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate dry-run diagnostics for a state assignment artifact.

    Reports:
    * State counts and frequencies.
    * Episodes per pair.
    * Episode duration distribution.
    * Survival counts at 8 and 24 bars.
    * JPY_CONSENSUS_MATURE sparsity flag (< 50 observations per pair).

    Args:
        df: Validated state assignment artifact DataFrame.

    Returns:
        Diagnostics dict suitable for JSON serialisation.
    """
    total_rows = len(df)

    # State counts and frequencies.
    state_counts = df["state_id"].value_counts().to_dict()
    state_counts = {str(k): int(v) for k, v in state_counts.items()}
    state_frequencies = {
        k: round(v / total_rows, 6) if total_rows > 0 else 0.0
        for k, v in state_counts.items()
    }

    # Episode statistics.
    episode_stats = _compute_episode_stats(df)

    # Mature sparsity — informational only, does not fail the run.
    mature_df = df[df["state_id"] == "JPY_CONSENSUS_MATURE"]
    mature_per_pair = mature_df.groupby("pair").size().to_dict()
    mature_per_pair = {str(k): int(v) for k, v in mature_per_pair.items()}
    mature_sparsity_flags = {
        pair: count < 50 for pair, count in mature_per_pair.items()
    }

    # Outcome distribution — counts of each terminal exit event across all
    # consensus episodes, per pair and globally.
    _OUTCOME_TYPES = ["exit_reversal", "exit_threshold", "exit_late_reversal", "exit_unknown"]
    outcome_distribution_global: dict[str, int] = {t: 0 for t in _OUTCOME_TYPES}
    outcome_distribution_per_pair: dict[str, dict[str, int]] = {}

    for pair, grp in df.groupby("pair", sort=False):
        grp = grp.sort_values("entry_time")
        is_extreme = grp["state_id"] != "JPY_NON_EXTREME"
        run_ids = (is_extreme != is_extreme.shift()).cumsum()
        pair_counts: dict[str, int] = {t: 0 for t in _OUTCOME_TYPES}
        for _, run in grp.groupby(run_ids, sort=True):
            if run["state_id"].iloc[0] == "JPY_NON_EXTREME":
                continue
            terminal_event = str(run["transition_event"].iloc[-1])
            if terminal_event in pair_counts:
                pair_counts[terminal_event] += 1
            else:
                pair_counts["exit_unknown"] += 1
        outcome_distribution_per_pair[str(pair)] = pair_counts
        for t in _OUTCOME_TYPES:
            outcome_distribution_global[t] += pair_counts[t]

    return {
        "state_counts": state_counts,
        "state_frequencies": state_frequencies,
        "episodes_per_pair": {
            str(k): int(v)
            for k, v in episode_stats["episodes_per_pair"].items()
        },
        "episode_duration_distribution": episode_stats["episode_duration_distribution"],
        "survival_counts": episode_stats["survival_counts"],
        "mature_observations_per_pair": mature_per_pair,
        "mature_sparsity_flags": mature_sparsity_flags,
        "outcome_distribution": outcome_distribution_global,
        "outcome_distribution_per_pair": outcome_distribution_per_pair,
    }


def print_diagnostics(diagnostics: dict[str, Any], *, pair: str | None = None) -> None:
    """Print a human-readable diagnostics summary to stdout."""
    sep = "-" * 60
    header = f"[BSVE] State Assignment Diagnostics"
    if pair:
        header += f" — {pair}"
    print(f"\n{header}")
    print(sep)

    print("\nState Counts")
    for sid, count in sorted(diagnostics["state_counts"].items()):
        freq = diagnostics["state_frequencies"].get(sid, 0.0)
        print(f"  {sid:<30s} {count:>8d}  ({freq:.2%})")

    print("\nEpisodes per pair")
    for p, n in sorted(diagnostics["episodes_per_pair"].items()):
        print(f"  {p:<20s} {n:>6d} episodes")

    dist = diagnostics["episode_duration_distribution"]
    print("\nEpisode Duration Distribution")
    print(f"  total episodes : {dist['total_episodes']}")
    print(f"  min            : {dist['min']}")
    print(f"  median         : {dist['median']}")
    print(f"  max            : {dist['max']}")

    print("\nSurvival Counts")
    for threshold, count in sorted(
        diagnostics["survival_counts"].items(), key=lambda x: int(x[0])
    ):
        print(f"  >= {threshold:>3s} bars  : {count}")

    print("\nMature Sparsity")
    for p, count in sorted(diagnostics["mature_observations_per_pair"].items()):
        flag = diagnostics["mature_sparsity_flags"].get(p, False)
        tag = "  ⚠ SPARSE (<50)" if flag else ""
        print(f"  {p:<20s} {count:>6d} observations{tag}")

    if "outcome_distribution" in diagnostics:
        print("\nOutcome Distribution")
        od = diagnostics["outcome_distribution"]
        for otype in ["exit_reversal", "exit_threshold", "exit_late_reversal", "exit_unknown"]:
            print(f"  {otype:<24s} {od.get(otype, 0):>6d}")

    print()


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def generate_run_manifest(
    *,
    run_id: str,
    environment_id: str,
    spec_id: str,
    calibration_id: str,
    dataset_version: str,
    artifact_schema_version: str = BSVE_SCHEMA_VERSION,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Generate a run manifest dict.

    Args:
        run_id: Unique identifier for this state assignment run.
        environment_id: BSVE environment ID (e.g. ``"reactive_jpy"``).
        spec_id: State spec identifier (e.g. ``"reactive_jpy_v1"``).
        calibration_id: Calibration artifact identifier.
        dataset_version: Version of the master research dataset consumed.
        artifact_schema_version: BSVE artifact schema version.
        timestamp: ISO-8601 timestamp.  Defaults to current UTC time.

    Returns:
        Run manifest dict suitable for JSON serialisation.
    """
    return {
        "run_id": run_id,
        "environment_id": environment_id,
        "spec_id": spec_id,
        "calibration_id": calibration_id,
        "dataset_version": dataset_version,
        "artifact_schema_version": artifact_schema_version,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def write_run_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write a run manifest JSON file to *output_dir/run_manifest.json*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_state_assignment(
    *,
    dataset_path: str | Path,
    environment: str,
    output_dir: str | Path,
    calibration_artifact_path: str | Path | None = None,
    state_spec_path: str | Path | None = None,
    dataset_version: str = "unknown",
    pairs: list[str] | None = None,
) -> Path:
    """
    Execute a full state assignment run end-to-end.

    Steps:
        1. Resolve paths for spec and calibration artifact.
        2. Load spec.
        3. Load and validate calibration artifact (fail-fast).
        4. Load dataset.
        5. Assign states.
        6. Validate artifact (fail-fast pre-write checks).
        7. Write parquet artifact.
        8. Generate and write diagnostics.
        9. Write run manifest.

    Args:
        dataset_path: Path to the master research dataset (CSV or parquet).
        environment: Environment ID (e.g. ``"reactive_jpy"``).
        output_dir: Directory where artifacts are written.
        calibration_artifact_path: Explicit path to calibration artifact JSON.
            Defaults to ``bsve/calibration_artifacts/<environment>_v1_<date>.json``
            (auto-discovery of latest artifact).
        state_spec_path: Explicit path to environment spec YAML.
            Defaults to ``bsve/state_specs/<environment>_v1.yaml``.
        dataset_version: Dataset version string for the run manifest.
        pairs: Explicit list of pairs to process.  Defaults to pairs listed
            in the environment spec.

    Returns:
        Path to the written parquet artifact.

    Raises:
        FileNotFoundError: If any required file is not found.
        ValueError: If calibration artifact is invalid or artifact fails
            pre-write validation.
    """
    output_dir = Path(output_dir)

    # --- Resolve spec path ---
    if state_spec_path is None:
        state_spec_path = _DEFAULT_STATE_SPECS_DIR / f"{environment}_v1.yaml"
    state_spec_path = Path(state_spec_path)

    # --- Resolve calibration artifact path ---
    if calibration_artifact_path is None:
        calibration_artifact_path = _find_calibration_artifact(environment)
    calibration_artifact_path = Path(calibration_artifact_path)

    print(f"[BSVE] Loading environment spec: {state_spec_path}")
    spec = load_state_spec(state_spec_path)

    print(f"[BSVE] Loading calibration artifact: {calibration_artifact_path}")
    artifact = assert_calibrations_valid(calibration_artifact_path)

    calibration_id = artifact["calibration_id"]
    ontology_version = artifact.get("ontology_version", "1.0.0")

    # Derive spec_id as <environment>_v<major_version> (e.g. "reactive_jpy_v1").
    major_version = ontology_version.split(".")[0]
    spec_id = f"{environment}_v{major_version}"

    # Resolve pairs from spec if not explicitly provided.
    if pairs is None:
        pairs = spec.get("environment", {}).get("pairs", [])
    if not pairs:
        raise ValueError(
            f"No pairs configured for environment '{environment}'. "
            "Pass --pairs or add pairs to the state spec."
        )

    print(f"[BSVE] Processing pairs: {pairs}")
    print(f"[BSVE] Loading dataset: {dataset_path}")

    adapter = MasterResearchDatasetAdapter.from_artifact(dataset_path)

    # Filter to configured pairs.
    normalized_pairs = [
        MasterResearchDatasetAdapter.normalize_pair(p) for p in pairs
    ]
    ds = adapter.dataset
    ds = ds[ds[adapter.config.pair_col].isin(normalized_pairs)].copy()

    if ds.empty:
        raise ValueError(
            f"Dataset contains no rows for environment pairs {pairs}. "
            "Check dataset path and pair configuration."
        )

    print(f"[BSVE] Assigning states to {len(ds):,} rows…")

    if environment == "reactive_jpy":
        artifact_df = assign_states_reactive_jpy(
            ds,
            calibration_artifact=artifact,
            spec_id=spec_id,
            pair_col=adapter.config.pair_col,
            timestamp_col=adapter.config.timestamp_col,
            environment_id=environment,
            state_version=ontology_version,
        )
    else:
        raise NotImplementedError(
            f"Environment '{environment}' is not implemented. "
            "Supported environments: reactive_jpy."
        )

    print("[BSVE] Validating artifact (pre-write)…")
    validate_state_artifact(artifact_df)

    # Diagnostics.
    diagnostics = generate_diagnostics(artifact_df)
    print_diagnostics(diagnostics)

    # Write parquet artifact.
    run_id = str(uuid.uuid4())
    artifact_filename = f"bsve_states_{environment}_{ontology_version}.parquet"
    artifact_out_path = output_dir / artifact_filename

    print(f"[BSVE] Writing parquet artifact: {artifact_out_path}")
    write_bsve_artifact(
        artifact_df,
        artifact_out_path,
        metadata={"schema_version": BSVE_SCHEMA_VERSION},
        artifact_metadata={
            "run_id": run_id,
            "environment_id": environment,
            "calibration_id": calibration_id,
            "spec_id": spec_id,
        },
    )

    # Write diagnostics JSON.
    diag_path = output_dir / "diagnostics.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[BSVE] Diagnostics written: {diag_path}")

    # Write run manifest.
    manifest = generate_run_manifest(
        run_id=run_id,
        environment_id=environment,
        spec_id=spec_id,
        calibration_id=calibration_id,
        dataset_version=dataset_version,
    )
    manifest_path = write_run_manifest(manifest, output_dir)
    print(f"[BSVE] Run manifest written: {manifest_path}")

    print(f"[BSVE] State assignment complete. Run ID: {run_id}")
    return artifact_out_path


def _find_calibration_artifact(environment: str) -> Path:
    """
    Auto-discover the most recent committed calibration artifact for *environment*.

    Searches ``bsve/calibration_artifacts/`` for files matching
    ``<environment>_*.json`` and returns the most recently modified one.

    Raises:
        FileNotFoundError: No matching artifact found.
    """
    candidates = sorted(
        _DEFAULT_CALIBRATION_ARTIFACTS_DIR.glob(f"{environment}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No calibration artifact found for environment '{environment}' "
            f"in {_DEFAULT_CALIBRATION_ARTIFACTS_DIR}. "
            "Run calibration first or pass --calibration-artifact explicitly."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BSVE rule-based state assignment engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the master research dataset (CSV or parquet).",
    )
    parser.add_argument(
        "--environment",
        required=True,
        help="BSVE environment ID (e.g. reactive_jpy).",
    )
    parser.add_argument(
        "--output-dir",
        default="bsve_outputs/",
        help="Directory where output artifacts are written.",
    )
    parser.add_argument(
        "--calibration-artifact",
        default=None,
        help=(
            "Path to a specific calibration artifact JSON. "
            "Defaults to auto-discovery of the latest committed artifact."
        ),
    )
    parser.add_argument(
        "--state-spec",
        default=None,
        help=(
            "Path to the environment state spec YAML. "
            "Defaults to bsve/state_specs/<environment>_v1.yaml."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Pairs to process. Defaults to pairs listed in the state spec.",
    )
    parser.add_argument(
        "--dataset-version",
        default="unknown",
        help="Dataset version string for run manifest provenance.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_state_assignment(
        dataset_path=args.dataset_path,
        environment=args.environment,
        output_dir=args.output_dir,
        calibration_artifact_path=args.calibration_artifact,
        state_spec_path=args.state_spec,
        dataset_version=args.dataset_version,
        pairs=args.pairs,
    )
