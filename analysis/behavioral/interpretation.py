"""Scientific interpretation for Behavioral Characterization Framework reports.

This module synthesizes experimental evidence into concise scientific findings.

Public API (Behavioral Characterization Framework — PR5.1):
    Finding                     — aggregated scientific finding with Interest/Confidence
    generate_findings()         — synthesize metrics/coverage/comparison into ≤5 findings
    format_executive_summary()  — one-page executive summary block
    format_findings()           — markdown rendering of Finding objects
    format_research_recommendation() — recommended next experimental step

Legacy API (preserved for backwards compatibility):
    Observation                 — per-artifact rule-triggered observation
    generate_key_observations() — apply all per-artifact rules
    format_key_observations()   — markdown rendering of Observation objects
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# PR5.1 — Behavioral Characterization Framework: Finding
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """An aggregated scientific finding synthesized from experimental evidence.

    Attributes
    ----------
    title:
        Short headline for the finding (one phrase).
    description:
        Full synthesized statement of what was found.
    evidence:
        List of supporting evidence lines (one per model/state/artifact).
    interest:
        Scientific Interest — how important or novel would this finding be if
        confirmed?  One of ``"low"``, ``"medium"``, ``"high"``.
    confidence:
        Scientific Confidence — how strongly is the finding supported by
        available evidence?  One of ``"low"``, ``"medium"``, ``"high"``.
    follow_up:
        Recommended investigation to increase confidence or exploit the finding.
    """

    title: str
    description: str
    evidence: list[str] = field(default_factory=list)
    interest: str = "medium"     # "low" | "medium" | "high"
    confidence: str = "medium"   # "low" | "medium" | "high"
    follow_up: str = ""


# ---------------------------------------------------------------------------
# Aggregated finding rules (PR5.1)
# ---------------------------------------------------------------------------

def _finding_prediction_entropy(metrics_df: pd.DataFrame) -> Finding | None:
    """Aggregate prediction entropy across all states into one finding."""
    if metrics_df.empty or "prediction_entropy_mean" not in metrics_df.columns:
        return None
    rows_with_entropy = metrics_df.dropna(subset=["prediction_entropy_mean"])
    if rows_with_entropy.empty:
        return None

    high_rows = rows_with_entropy[
        rows_with_entropy["prediction_entropy_mean"] > _HIGH_ENTROPY_THRESHOLD
    ]
    if high_rows.empty:
        return None

    evidence: list[str] = []
    for _, row in rows_with_entropy.iterrows():
        entropy = _safe_float(row.get("prediction_entropy_mean"))
        state = row.get("state_id") or row.get("artifact_file") or "unknown"
        model = str(row.get("model", "")).upper() or None
        label = f"{model} / {state}" if model else str(state)
        evidence.append(f"- {label}: entropy {_fmt(entropy)} bits")

    all_high = len(high_rows) == len(rows_with_entropy)
    frac = len(high_rows) / len(rows_with_entropy)

    if all_high:
        description = (
            "Prediction entropy is consistently high across all Behavioral States, "
            "indicating that predicted probabilities are concentrated near 0.5."
        )
        confidence = "high"
    elif frac >= 0.5:
        description = (
            f"Prediction entropy is above threshold in {len(high_rows)} of "
            f"{len(rows_with_entropy)} state/model combinations."
        )
        confidence = "medium"
    else:
        return None

    return Finding(
        title="High prediction entropy across states",
        description=description,
        evidence=evidence,
        interest="medium",
        confidence=confidence,
        follow_up=(
            "Verify that training converged (inspect loss curves). "
            "Consider increasing epochs or enriching the feature set with more "
            "discriminative signals for these states."
        ),
    )


def _finding_effective_coverage(metrics_df: pd.DataFrame) -> Finding | None:
    """Aggregate effective prediction coverage across all states."""
    if metrics_df.empty or "effective_prediction_coverage" not in metrics_df.columns:
        return None
    rows = metrics_df.dropna(subset=["effective_prediction_coverage"])
    if rows.empty:
        return None

    low_rows = rows[rows["effective_prediction_coverage"] < _LOW_EFFECTIVE_COVERAGE]
    if low_rows.empty:
        return None

    evidence: list[str] = []
    for _, row in rows.iterrows():
        eff = _safe_float(row.get("effective_prediction_coverage"))
        state = row.get("state_id") or row.get("artifact_file") or "unknown"
        model = str(row.get("model", "")).upper() or None
        label = f"{model} / {state}" if model else str(state)
        evidence.append(f"- {label}: effective coverage {_pct(eff)}")

    all_low = len(low_rows) == len(rows)
    frac = len(low_rows) / len(rows)

    if all_low:
        description = (
            "Effective prediction coverage is low across all Behavioral States: "
            "fewer than half of predictions are materially informative."
        )
        confidence = "high"
    elif frac >= 0.5:
        description = (
            f"Effective prediction coverage is low in {len(low_rows)} of "
            f"{len(rows)} state/model combinations."
        )
        confidence = "medium"
    else:
        return None

    return Finding(
        title="Low effective prediction coverage",
        description=description,
        evidence=evidence,
        interest="high",
        confidence=confidence,
        follow_up=(
            "Compare against random-matched controls to determine whether low "
            "coverage is state-specific or a general property of the training "
            "window. If universal, more training epochs or a richer feature set "
            "may be required."
        ),
    )


def _finding_mlp_lstm_agreement(compare_df: pd.DataFrame) -> Finding | None:
    """Aggregate MLP/LSTM directional agreement across all states."""
    if compare_df.empty or "agreement_rate" not in compare_df.columns:
        return None
    rows = compare_df.dropna(subset=["agreement_rate"])
    if rows.empty:
        return None

    low_rows = rows[rows["agreement_rate"] < _LOW_AGREEMENT_THRESHOLD]
    high_rows = rows[rows["agreement_rate"] > _HIGH_AGREEMENT_THRESHOLD]

    if low_rows.empty and high_rows.empty:
        return None

    evidence: list[str] = []
    for _, row in rows.iterrows():
        rate = _safe_float(row.get("agreement_rate"))
        state = row.get("state_id", "unknown")
        evidence.append(f"- {state}: agreement {_pct(rate)}")

    if not low_rows.empty:
        all_low = len(low_rows) == len(rows)
        if all_low:
            description = (
                "MLP/LSTM directional agreement is consistently low across all states, "
                "suggesting the behavioral partitions do not produce a robustly learnable signal."
            )
        else:
            states = ", ".join(str(r.get("state_id", "?")) for _, r in low_rows.iterrows())
            description = f"MLP/LSTM directional agreement is low for states: {states}."
        return Finding(
            title="Low MLP/LSTM directional agreement",
            description=description,
            evidence=evidence,
            interest="high",
            confidence="medium" if not all_low else "high",  # type: ignore[possibly-undefined]
            follow_up=(
                "Inspect prediction entropy per model separately. "
                "Verify that the temporal training window is sufficient for stable convergence."
            ),
        )

    if not high_rows.empty:
        all_high = len(high_rows) == len(rows)
        description = (
            "MLP/LSTM directional agreement is high across all states."
            if all_high else
            f"MLP/LSTM directional agreement is high in {len(high_rows)} of {len(rows)} states."
        )
        return Finding(
            title="High MLP/LSTM directional agreement",
            description=description,
            evidence=evidence,
            interest="medium",
            confidence="high" if all_high else "medium",
            follow_up=(
                "Verify that high agreement reflects genuine mutual information with future "
                "prices rather than directional bias. Consider walk-forward evaluation."
            ),
        )

    return None


def _finding_behavioral_coverage(coverage_df: pd.DataFrame) -> Finding | None:
    """Synthesize behavioral coverage into a single finding."""
    if coverage_df.empty or "scope" not in coverage_df.columns:
        return None
    total_row = coverage_df[coverage_df["scope"] == "full_dataset"]
    beh_row = coverage_df[coverage_df["scope"] == "behavioral_coverage"]
    if total_row.empty or beh_row.empty:
        return None

    frac = _safe_float(beh_row.iloc[0].get("coverage_fraction"))
    beh_n = int(beh_row.iloc[0]["row_count"])
    total_n = int(total_row.iloc[0]["row_count"])
    if frac is None:
        return None

    state_rows = coverage_df[coverage_df["scope"].str.startswith("state:")]
    evidence: list[str] = [
        f"- Full dataset: {total_n:,} rows",
        f"- Behavioral coverage: {beh_n:,} rows ({_pct(frac)})",
    ]
    for _, r in state_rows.iterrows():
        count = int(r.get("row_count", 0))
        sfrac = _safe_float(r.get("state_fraction_of_behavioral"))
        evidence.append(f"- {r['scope']}: {count:,} rows ({_pct(sfrac)} of behavioral)")

    if frac < _LOW_COVERAGE_THRESHOLD:
        description = (
            f"Behavioral coverage represents only {_pct(frac)} of the canonical dataset "
            f"({beh_n:,} of {total_n:,} rows). "
            "This limits the statistical reliability of per-state metrics."
        )
        interest = "medium"
        confidence = "high"
        follow_up = (
            "Compare per-state metrics against random size-matched controls. "
            "Consider whether the behavioral ontology covers a wider range of market conditions."
        )
    else:
        description = (
            f"Behavioral coverage is {_pct(frac)} of the canonical dataset "
            f"({beh_n:,} of {total_n:,} rows), providing adequate data volume for initial characterization."
        )
        interest = "low"
        confidence = "high"
        follow_up = "Verify that temporal coverage spans multiple market regimes."

    return Finding(
        title="Behavioral Surface coverage",
        description=description,
        evidence=evidence,
        interest=interest,
        confidence=confidence,
        follow_up=follow_up,
    )


def _finding_state_imbalance(coverage_df: pd.DataFrame) -> Finding | None:
    """Detect strongly imbalanced state occupancy."""
    if coverage_df.empty or "scope" not in coverage_df.columns:
        return None
    state_rows = coverage_df[coverage_df["scope"].str.startswith("state:")]
    if len(state_rows) < 2:
        return None

    counts = state_rows["row_count"].astype(int)
    max_count = int(counts.max())
    min_count = int(counts.min())
    if min_count == 0:
        return None
    ratio = max_count / min_count
    if ratio <= _HIGH_IMBALANCE_RATIO:
        return None

    largest = state_rows.loc[counts.idxmax(), "scope"]
    smallest = state_rows.loc[counts.idxmin(), "scope"]
    evidence = [
        f"- Largest state '{largest}': {max_count:,} rows",
        f"- Smallest state '{smallest}': {min_count:,} rows",
        f"- Ratio: {ratio:.1f}×",
    ]

    return Finding(
        title="Strongly imbalanced state occupancy",
        description=(
            f"State occupancy is strongly imbalanced ({ratio:.1f}× ratio between largest and "
            "smallest states). Metrics for smaller states carry higher variance."
        ),
        evidence=evidence,
        interest="medium",
        confidence="high",
        follow_up=(
            "Inspect per-state effective prediction coverage and confidence separately. "
            "Consider whether the smaller state has sufficient data for reliable training."
        ),
    )


def generate_findings(
    coverage_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    controls_df: pd.DataFrame | None = None,
    *,
    max_findings: int = 5,
) -> list[Finding]:
    """Synthesize experimental evidence into a ranked list of scientific findings.

    Each finding aggregates evidence from multiple states/models into a single
    statement with Scientific Interest and Scientific Confidence ratings.
    Repeated per-artifact observations are collapsed into one finding with
    supporting evidence lines (noise suppression).

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
    max_findings:
        Cap on the number of findings returned (default 5).
    """
    candidates: list[Finding] = []

    f = _finding_behavioral_coverage(coverage_df)
    if f is not None:
        candidates.append(f)

    f = _finding_state_imbalance(coverage_df)
    if f is not None:
        candidates.append(f)

    f = _finding_mlp_lstm_agreement(compare_df)
    if f is not None:
        candidates.append(f)

    f = _finding_prediction_entropy(metrics_df)
    if f is not None:
        candidates.append(f)

    f = _finding_effective_coverage(metrics_df)
    if f is not None:
        candidates.append(f)

    # Rank: high interest first, then high confidence, then preserve order
    _rank = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda fi: (_rank.get(fi.interest, 1), _rank.get(fi.confidence, 1)))

    return candidates[:max_findings]


# ---------------------------------------------------------------------------
# PR5.1 — Report sections
# ---------------------------------------------------------------------------

_INTEREST_ICON = {"high": "⬆", "medium": "●", "low": "⬇"}
_CONFIDENCE_ICON = {"high": "★★★", "medium": "★★☆", "low": "★☆☆"}


def format_executive_summary(
    *,
    experiment_id: str,
    run_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    discovered_states: list[dict[str, str]],
    findings: list[Finding],
    recommendation: str,
) -> str:
    """Render the one-page executive summary block."""
    total = len(run_df)
    successful = int((run_df["status"] == "success").sum()) if not run_df.empty else 0
    failed = total - successful
    status_line = (
        f"✓ {successful}/{total} runs succeeded"
        if failed == 0
        else f"⚠ {successful}/{total} runs succeeded — {failed} failed"
    )

    # Behavioral Surface names
    surfaces: list[str] = []
    seen: set[str] = set()
    for s in discovered_states:
        key = f"{s['surface_id']} / {s['state_id']}"
        if key not in seen:
            surfaces.append(f"- {key}")
            seen.add(key)

    # Coverage fraction
    cov_line = "Coverage data unavailable."
    if not coverage_df.empty and "scope" in coverage_df.columns:
        beh_row = coverage_df[coverage_df["scope"] == "behavioral_coverage"]
        if not beh_row.empty:
            frac = _safe_float(beh_row.iloc[0].get("coverage_fraction"))
            n = int(beh_row.iloc[0]["row_count"])
            cov_line = f"Behavioral states cover {_pct(frac)} of the canonical dataset ({n:,} rows)."

    # Key findings bullets
    if findings:
        bullets = "\n".join(f"- **{f.title}** — {f.description}" for f in findings)
    else:
        bullets = "- No significant findings generated."

    lines = [
        "## Executive Summary",
        "",
        f"**Experiment:** `{experiment_id}`",
        "",
        f"**Experiment status:** {status_line}",
        "",
        "**Behavioral Surface:**",
        *surfaces,
        "",
        f"**Coverage:** {cov_line}",
        "",
        "**Key Findings:**",
        bullets,
        "",
        f"**Research Recommendation:** {recommendation}",
    ]
    return "\n".join(lines)


def format_findings(findings: list[Finding]) -> str:
    """Render a list of Finding objects as a markdown **Scientific Findings** section."""
    if not findings:
        return "## Scientific Findings\n\nNo significant findings generated.\n"

    lines = ["## Scientific Findings", ""]
    for i, f in enumerate(findings, start=1):
        interest_icon = _INTEREST_ICON.get(f.interest, "●")
        confidence_icon = _CONFIDENCE_ICON.get(f.confidence, "★★☆")
        lines.append(f"### Finding {i}: {f.title}")
        lines.append("")
        lines.append(f.description)
        lines.append("")
        if f.evidence:
            lines.append("**Supporting evidence:**")
            lines.extend(f.evidence)
            lines.append("")
        lines.append(
            f"**Scientific Interest:** {interest_icon} {f.interest.capitalize()}  "
            f"**Scientific Confidence:** {confidence_icon} {f.confidence.capitalize()}"
        )
        lines.append("")
        if f.follow_up:
            lines.append(f"**Recommended follow-up:** {f.follow_up}")
            lines.append("")
    return "\n".join(lines)


def derive_research_recommendation(
    *,
    run_df: pd.DataFrame,
    findings: list[Finding],
    coverage_df: pd.DataFrame,
) -> str:
    """Return a single recommended next experimental step.

    The recommendation is derived from experiment status and synthesized
    findings.  It follows the pattern:

        Continue | Repeat with more epochs | Proceed to walk-forward |
        Compare with Reactive CHF | Insufficient evidence | ...
    """
    # Failures take priority
    if not run_df.empty and (run_df["status"] != "success").any():
        n_failed = int((run_df["status"] != "success").sum())
        return (
            f"**Diagnose and repeat** — {n_failed} training run(s) failed. "
            "Resolve failures before drawing conclusions from partial results."
        )

    if run_df.empty:
        return "**Insufficient evidence** — no runs completed."

    # Check findings for signals
    low_confidence_signal = any(
        f.title in ("High prediction entropy across states", "Low effective prediction coverage")
        for f in findings
        if f.confidence in ("medium", "high")
    )

    low_coverage = False
    if not coverage_df.empty and "scope" in coverage_df.columns:
        beh_row = coverage_df[coverage_df["scope"] == "behavioral_coverage"]
        if not beh_row.empty:
            frac = _safe_float(beh_row.iloc[0].get("coverage_fraction"))
            if frac is not None and frac < _LOW_COVERAGE_THRESHOLD:
                low_coverage = True

    high_agreement = any(
        f.title == "High MLP/LSTM directional agreement" and f.confidence in ("medium", "high")
        for f in findings
    )

    if low_coverage and low_confidence_signal:
        return (
            "**Insufficient evidence** — behavioral coverage is low and prediction confidence "
            "is weak. Consider a dataset variant with broader behavioral coverage, or increase "
            "training epochs before re-evaluating."
        )

    if low_confidence_signal:
        return (
            "**Repeat with more epochs** — prediction entropy is high and/or effective coverage "
            "is low, suggesting training has not yet converged to a discriminative solution. "
            "Increase epochs (e.g. `--profile standard` or `--profile publication`) and re-run."
        )

    if high_agreement:
        return (
            "**Proceed to walk-forward evaluation (PR7)** — cross-architecture agreement is "
            "high, suggesting a potentially learnable signal. Walk-forward evaluation will "
            "assess whether the agreement reflects genuine predictive value."
        )

    return (
        "**Continue characterization** — initial experiment completed without critical issues. "
        "Consider comparing against Reactive CHF or a Persistent surface to evaluate "
        "family-specific behavioral differences."
    )


# ---------------------------------------------------------------------------
# Legacy API — per-artifact Observation rules (preserved for backwards compat)
# ---------------------------------------------------------------------------

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
