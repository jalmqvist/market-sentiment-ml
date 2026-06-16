"""BSVE behavioral outcome analysis for Criterion 1 evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

_CONSENSUS_STATES = [
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
]

_BEHAVIORAL_COMPARISONS = [
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURING"),
    ("JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_MATURE"),
    ("JPY_CONSENSUS_MATURING", "JPY_CONSENSUS_MATURE"),
]

_OUTCOME_TYPES = ["exit_reversal", "exit_threshold", "exit_late_reversal", "exit_unknown"]

_ALPHA = 0.05
_MIN_EPISODE_COUNT = 5


def _compute_episode_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-episode terminal transition events from a state surface.

    Each episode is a consecutive run of the same state_id within a pair.
    The terminal event is taken from the last bar of the run and used to
    classify the episode outcome.
    """
    ordered = df.sort_values(["pair", "entry_time"]).copy()
    shifted_pair = ordered["pair"].ne(ordered["pair"].shift())
    shifted_state = ordered["state_id"].ne(ordered["state_id"].shift())
    ordered["episode_id"] = (shifted_pair | shifted_state).cumsum()

    episodes = (
        ordered.groupby(["pair", "state_id", "episode_id"], as_index=False)
        .agg(
            terminal_event=("transition_event", "last"),
            duration_bars=("entry_time", "size"),
        )
    )
    episodes["is_reversal"] = (episodes["terminal_event"] == "exit_reversal").astype(int)
    episodes["is_threshold"] = (episodes["terminal_event"] == "exit_threshold").astype(int)
    episodes["is_late_reversal"] = (
        episodes["terminal_event"] == "exit_late_reversal"
    ).astype(int)
    return episodes


def _cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions."""
    return float(abs(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))))


def analyze_behavioral_outcomes(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze behavioral outcomes from a state surface.

    Computes per-state outcome distributions for consensus states and tests
    whether those states exhibit statistically distinct reversal behavior using
    Fisher's exact test.  Cohen's h is used as the effect size metric.

    The reversal rate comparison (Fisher's exact) is the primary behavioral
    differentiation test.  Outcome type distributions (exit_reversal,
    exit_threshold, exit_late_reversal) are reported for all consensus states
    to provide Criterion 1 with independent behavioral labels derived from
    episode-level outcome classification.

    Parameters
    ----------
    df:
        State-surface DataFrame with columns ``pair``, ``entry_time``,
        ``state_id``, ``maturity_bars``, and ``transition_event``.

    Returns
    -------
    dict with keys:
        ``behavioral_evidence_available`` (bool),
        ``behavioral_effect_size`` (float | None),
        ``behavioral_tests`` (list[dict]),
        ``behavioral_outcomes`` (list[dict]).
    """
    episodes = _compute_episode_outcomes(df)

    # Per-state outcome summary
    behavioral_outcomes: list[dict[str, Any]] = []
    for state in _CONSENSUS_STATES:
        state_eps = episodes[episodes["state_id"] == state]
        n = int(len(state_eps))
        n_reversal = int(state_eps["is_reversal"].sum())
        n_threshold = int(state_eps["is_threshold"].sum())
        n_late_reversal = int(state_eps["is_late_reversal"].sum())
        n_unknown = n - n_reversal - n_threshold - n_late_reversal
        reversal_rate = float(n_reversal / n) if n > 0 else 0.0
        behavioral_outcomes.append(
            {
                "state_id": state,
                "episode_count": n,
                "reversal_count": n_reversal,
                "threshold_count": n_threshold,
                "late_reversal_count": n_late_reversal,
                "unknown_count": max(n_unknown, 0),
                "reversal_rate": round(reversal_rate, 6),
                "outcome_distribution": {
                    "exit_reversal": n_reversal,
                    "exit_threshold": n_threshold,
                    "exit_late_reversal": n_late_reversal,
                    "exit_unknown": max(n_unknown, 0),
                },
            }
        )

    # Pairwise Fisher's exact test on reversal rates
    behavioral_tests: list[dict[str, Any]] = []
    significant_effect_sizes: list[float] = []

    for state_a, state_b in _BEHAVIORAL_COMPARISONS:
        eps_a = episodes[episodes["state_id"] == state_a]
        eps_b = episodes[episodes["state_id"] == state_b]

        if len(eps_a) < _MIN_EPISODE_COUNT or len(eps_b) < _MIN_EPISODE_COUNT:
            behavioral_tests.append(
                {
                    "comparison": f"{state_a} vs {state_b}",
                    "state_a": state_a,
                    "state_b": state_b,
                    "test": "fisher_exact",
                    "p_value": None,
                    "significant": False,
                    "effect_size": None,
                    "effect_size_metric": "cohens_h",
                    "skipped": True,
                    "skip_reason": (
                        f"insufficient episodes: {len(eps_a)} ({state_a}), "
                        f"{len(eps_b)} ({state_b}); minimum is {_MIN_EPISODE_COUNT}"
                    ),
                }
            )
            continue

        n_a = int(len(eps_a))
        r_a = int(eps_a["is_reversal"].sum())
        n_b = int(len(eps_b))
        r_b = int(eps_b["is_reversal"].sum())

        p_a = r_a / n_a
        p_b = r_b / n_b
        _, p_value = fisher_exact([[r_a, n_a - r_a], [r_b, n_b - r_b]])
        effect_size = _cohens_h(p_a, p_b)

        is_significant = bool(p_value < _ALPHA)
        if is_significant:
            significant_effect_sizes.append(effect_size)

        behavioral_tests.append(
            {
                "comparison": f"{state_a} vs {state_b}",
                "state_a": state_a,
                "state_b": state_b,
                "test": "fisher_exact",
                "p_value": float(p_value),
                "significant": is_significant,
                "effect_size": float(effect_size),
                "effect_size_metric": "cohens_h",
                "skipped": False,
                "skip_reason": None,
            }
        )

    behavioral_evidence_available = len(significant_effect_sizes) > 0
    behavioral_effect_size = (
        max(significant_effect_sizes) if significant_effect_sizes else None
    )

    return {
        "behavioral_evidence_available": behavioral_evidence_available,
        "behavioral_effect_size": behavioral_effect_size,
        "behavioral_tests": behavioral_tests,
        "behavioral_outcomes": behavioral_outcomes,
    }
