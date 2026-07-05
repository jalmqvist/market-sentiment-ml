"""Rule-based scientific interpretation for behavioral experiment reports.

This module generates a concise Key Observations section that:
- describes what was observed
- explains why it matters
- suggests possible follow-up investigations

Interpretations identify observations that are potentially interesting without
drawing unsupported scientific conclusions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# Thresholds used by interpretation rules
_LOW_COVERAGE_THRESHOLD = 0.10      # <10% behavioral coverage is notable
_HIGH_IMBALANCE_RATIO = 3.0         # largest / smallest state > 3x is imbalanced
_LOW_AGREEMENT_THRESHOLD = 0.60     # MLP/LSTM agreement < 60% is notable
_HIGH_AGREEMENT_THRESHOLD = 0.85    # > 85% is strong agreement
_LOW_EFFECTIVE_COVERAGE = 0.40      # <40% effective prediction coverage is notable
_HIGH_ENTROPY_THRESHOLD = 0.90      # prediction entropy mean > 0.9 bits (of 1.0 max)
_LOW_SHARPNESS_THRESHOLD = 0.10     # sharpness < 0.1 suggests near-uniform predictions


def _pct(fraction: float | None) -> str:
    if fraction is None or (isinstance(fraction, float) and math.isnan(fraction)):
        return "N/A"
    return f"{fraction * 100:.1f}%"


def _fmt(value: float | None, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def _safe_float(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


class Observation:
    """A single rule-triggered observation."""

    def __init__(self, observed: str, why_it_matters: str, follow_up: str) -> None:
        self.observed = observed
        self.why_it_matters = why_it_matters
        self.follow_up = follow_up

    def as_dict(self) -> dict[str, str]:
        return {
            "observed": self.observed,
            "why_it_matters": self.why_it_matters,
            "follow_up": self.follow_up,
        }


def _rule_behavioral_coverage(coverage_df: pd.DataFrame) -> list[Observation]:
    """Comment on the fraction of the canonical dataset covered by the behavioral partition."""
    obs: list[Observation] = []
    if coverage_df.empty or "scope" not in coverage_df.columns:
        return obs
    total_row = coverage_df[coverage_df["scope"] == "full_dataset"]
    beh_row = coverage_df[coverage_df["scope"] == "behavioral_coverage"]
    if total_row.empty or beh_row.empty:
        return obs
    frac = _safe_float(beh_row.iloc[0].get("coverage_fraction"))
    beh_n = int(beh_row.iloc[0]["row_count"])
    total_n = int(total_row.iloc[0]["row_count"])
    if frac is None:
        return obs
    if frac < _LOW_COVERAGE_THRESHOLD:
        obs.append(Observation(
            observed=f"Behavioral coverage represents {_pct(frac)} of the canonical dataset ({beh_n:,} of {total_n:,} rows).",
            why_it_matters=(
                "A small behavioral fraction means the models train on a limited data subset. "
                "This can inflate variance in reported metrics and make comparisons with "
                "the full-dataset baseline misleading."
            ),
            follow_up=(
                "Consider comparing per-state metrics against random size-matched controls "
                "to separate coverage effects from behavioral effects."
            ),
        ))
    else:
        obs.append(Observation(
            observed=f"Behavioral coverage is {_pct(frac)} of the canonical dataset ({beh_n:,} of {total_n:,} rows).",
            why_it_matters=(
                "Substantial behavioral coverage reduces the risk that per-state metrics are "
                "dominated by sampling variance."
            ),
            follow_up="Verify that temporal coverage is similarly proportional (not concentrated in a short window).",
        ))
    return obs


def _rule_state_imbalance(coverage_df: pd.DataFrame) -> list[Observation]:
    """Detect strong state occupancy imbalance."""
    obs: list[Observation] = []
    if coverage_df.empty or "scope" not in coverage_df.columns:
        return obs
    state_rows = coverage_df[coverage_df["scope"].str.startswith("state:")]
    if len(state_rows) < 2:
        return obs
    counts = state_rows["row_count"].astype(int)
    max_count = int(counts.max())
    min_count = int(counts.min())
    if min_count == 0:
        return obs
    ratio = max_count / min_count
    if ratio > _HIGH_IMBALANCE_RATIO:
        largest_state = state_rows.loc[counts.idxmax(), "scope"]
        smallest_state = state_rows.loc[counts.idxmin(), "scope"]
        obs.append(Observation(
            observed=(
                f"State occupancy is strongly imbalanced (ratio {ratio:.1f}×): "
                f"largest state '{largest_state}' ({max_count:,} rows) vs "
                f"smallest '{smallest_state}' ({min_count:,} rows)."
            ),
            why_it_matters=(
                "Imbalanced state occupancy means metrics for smaller states have higher variance "
                "and are harder to interpret. Model quality may differ substantially across states."
            ),
            follow_up=(
                "Inspect per-state effective prediction coverage and confidence separately. "
                "Consider whether the smaller state has sufficient data for reliable training."
            ),
        ))
    return obs


def _rule_mlp_lstm_agreement(compare_df: pd.DataFrame) -> list[Observation]:
    """Flag low or high MLP/LSTM directional agreement per state."""
    obs: list[Observation] = []
    if compare_df.empty or "agreement_rate" not in compare_df.columns:
        return obs
    for _, row in compare_df.iterrows():
        rate = _safe_float(row.get("agreement_rate"))
        state_id = row.get("state_id", "unknown")
        if rate is None:
            continue
        if rate < _LOW_AGREEMENT_THRESHOLD:
            obs.append(Observation(
                observed=f"MLP/LSTM directional agreement is low ({_pct(rate)}) for state '{state_id}'.",
                why_it_matters=(
                    "Low agreement between architectures trained on the same data suggests that "
                    "the state partition does not produce a consistently learnable signal. "
                    "The two models may be capturing different noise structures."
                ),
                follow_up=(
                    "Inspect prediction entropy for each model separately. "
                    "Check whether the temporal window is sufficient for stable training."
                ),
            ))
        elif rate > _HIGH_AGREEMENT_THRESHOLD:
            obs.append(Observation(
                observed=f"MLP/LSTM directional agreement is high ({_pct(rate)}) for state '{state_id}'.",
                why_it_matters=(
                    "High cross-architecture agreement is consistent with a learnable and stable signal, "
                    "but does not confirm predictive value without ground-truth evaluation."
                ),
                follow_up=(
                    "Verify whether the high agreement is driven by a consistently predicted direction "
                    "(directional bias) rather than genuine mutual information with future prices."
                ),
            ))
    return obs


def _rule_prediction_entropy(metrics_df: pd.DataFrame) -> list[Observation]:
    """Detect near-maximum or unusually low prediction entropy."""
    obs: list[Observation] = []
    if metrics_df.empty or "prediction_entropy_mean" not in metrics_df.columns:
        return obs
    for _, row in metrics_df.iterrows():
        entropy = _safe_float(row.get("prediction_entropy_mean"))
        state_id = row.get("state_id") or row.get("artifact_file") or "unknown"
        if entropy is None:
            continue
        if entropy > _HIGH_ENTROPY_THRESHOLD:
            obs.append(Observation(
                observed=f"Mean prediction entropy is high ({_fmt(entropy)} bits) for '{state_id}'.",
                why_it_matters=(
                    "High entropy indicates that predicted probabilities are concentrated near 0.5, "
                    "meaning the model is largely uncertain about direction."
                ),
                follow_up=(
                    "Check whether training converged and whether the feature set contains any "
                    "discriminative signal for this state."
                ),
            ))
    return obs


def _rule_effective_coverage(metrics_df: pd.DataFrame) -> list[Observation]:
    """Detect low effective prediction coverage (few confident predictions)."""
    obs: list[Observation] = []
    if metrics_df.empty or "effective_prediction_coverage" not in metrics_df.columns:
        return obs
    for _, row in metrics_df.iterrows():
        eff = _safe_float(row.get("effective_prediction_coverage"))
        state_id = row.get("state_id") or row.get("artifact_file") or "unknown"
        if eff is None:
            continue
        if eff < _LOW_EFFECTIVE_COVERAGE:
            obs.append(Observation(
                observed=(
                    f"Effective prediction coverage is low ({_pct(eff)}) for '{state_id}': "
                    "fewer than half of predictions are materially informative."
                ),
                why_it_matters=(
                    "A low effective coverage means the model produces near-uniform predictions "
                    "for most observations, providing little actionable signal even where it has been trained."
                ),
                follow_up=(
                    "Compare against random-matched controls to determine whether the low coverage "
                    "is specific to the behavioral state or a general property of the training window."
                ),
            ))
    return obs


def _rule_pair_balance(metrics_df: pd.DataFrame) -> list[Observation]:
    """Flag strongly imbalanced pair distribution in predictions."""
    obs: list[Observation] = []
    if metrics_df.empty or "pair_balance" not in metrics_df.columns:
        return obs
    for _, row in metrics_df.iterrows():
        balance = _safe_float(row.get("pair_balance"))
        state_id = row.get("state_id") or row.get("artifact_file") or "unknown"
        if balance is None:
            continue
        if balance < 0.5:
            obs.append(Observation(
                observed=f"Pair distribution is strongly imbalanced (normalised entropy {_fmt(balance)}) for '{state_id}'.",
                why_it_matters=(
                    "A skewed pair distribution means metrics are dominated by one or a few currency pairs. "
                    "Aggregate statistics may not represent the full breadth of pairs."
                ),
                follow_up="Report per-pair metrics separately and consider whether predictions generalise across pairs.",
            ))
    return obs


def _rule_overlap_percentage(compare_df: pd.DataFrame) -> list[Observation]:
    """Flag low timestamp overlap between MLP and LSTM prediction sets."""
    obs: list[Observation] = []
    if compare_df.empty:
        return obs
    for col in ["overlap_pct_of_mlp", "overlap_pct_of_lstm"]:
        if col not in compare_df.columns:
            return obs
    for _, row in compare_df.iterrows():
        state_id = row.get("state_id", "unknown")
        pct_mlp = _safe_float(row.get("overlap_pct_of_mlp"))
        pct_lstm = _safe_float(row.get("overlap_pct_of_lstm"))
        if pct_mlp is not None and pct_mlp < 80.0:
            obs.append(Observation(
                observed=(
                    f"MLP/LSTM timestamp overlap is {_fmt(pct_mlp, 1)}% (of MLP) for state '{state_id}'."
                ),
                why_it_matters=(
                    "Low temporal overlap means MLP and LSTM predictions are evaluated on different "
                    "observation windows, limiting the validity of direct comparisons."
                ),
                follow_up="Investigate whether the export-split configuration differs between training runs.",
            ))
            break
        if pct_lstm is not None and pct_lstm < 80.0:
            obs.append(Observation(
                observed=(
                    f"MLP/LSTM timestamp overlap is {_fmt(pct_lstm, 1)}% (of LSTM) for state '{state_id}'."
                ),
                why_it_matters=(
                    "Low temporal overlap means MLP and LSTM predictions are evaluated on different "
                    "observation windows, limiting the validity of direct comparisons."
                ),
                follow_up="Investigate whether the export-split configuration differs between training runs.",
            ))
            break
    return obs


def generate_key_observations(
    coverage_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    controls_df: pd.DataFrame | None = None,
) -> list[Observation]:
    """Apply all interpretation rules and return a deduplicated list of observations.

    Parameters
    ----------
    coverage_df:
        Output of :func:`analysis.behavioral.coverage.build_coverage_table`.
    compare_df:
        Output of :func:`analysis.behavioral.compare_predictions.compare_mlp_lstm_predictions`.
    metrics_df:
        Output of :func:`analysis.behavioral.metrics.compute_prediction_metrics`.
    controls_df:
        Output of :func:`analysis.behavioral.controls.generate_controls` (optional).
    """
    observations: list[Observation] = []
    observations.extend(_rule_behavioral_coverage(coverage_df))
    observations.extend(_rule_state_imbalance(coverage_df))
    observations.extend(_rule_mlp_lstm_agreement(compare_df))
    observations.extend(_rule_prediction_entropy(metrics_df))
    observations.extend(_rule_effective_coverage(metrics_df))
    observations.extend(_rule_pair_balance(metrics_df))
    observations.extend(_rule_overlap_percentage(compare_df))
    return observations


def format_key_observations(observations: list[Observation]) -> str:
    """Render a list of observations as a markdown **Key Observations** section."""
    if not observations:
        return "## Key Observations\n\nNo notable observations generated.\n"

    lines = ["## Key Observations", ""]
    for i, obs in enumerate(observations, start=1):
        lines.append(f"### Observation {i}")
        lines.append(f"**Observed:** {obs.observed}")
        lines.append(f"**Why it matters:** {obs.why_it_matters}")
        lines.append(f"**Follow-up:** {obs.follow_up}")
        lines.append("")
    return "\n".join(lines)
