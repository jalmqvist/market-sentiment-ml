# Reactive-JPY Findings

## Purpose

This document records empirical findings, calibration results, validation outcomes, and research observations related to the Reactive-JPY ontology.

Unlike `CONCEPT_DRAFT.md`, this document is expected to evolve as new evidence is gathered.

---

# Ontology Summary

Reactive-JPY models the lifecycle of extreme crowd consensus states.

The current ontology contains:

* `JPY_CONSENSUS_YOUNG`
* `JPY_CONSENSUS_MATURING`
* `JPY_CONSENSUS_MATURE`
* `JPY_NON_EXTREME`

State transitions are driven by consensus persistence duration calibrated from historical sentiment behavior.

---

# Calibration Results

## Dataset

Calibration performed on:

* Dataset version: 1.5.1
* Window: 2019–2026
* Pairs:

  * USDJPY
  * EURJPY
  * GBPJPY

## Derived Thresholds

| Parameter         | Value   |
| ----------------- | ------- |
| Extreme threshold | 70      |
| Young boundary    | 8 bars  |
| Mature boundary   | 24 bars |
| Hazard crossover  | 13 bars |

## Episode Statistics

| Metric              | Value  |
| ------------------- | ------ |
| Total episodes      | 441    |
| Median duration     | 4 bars |
| Survival to 8 bars  | 113    |
| Survival to 24 bars | 21     |
| Survival to 48 bars | 3      |

Observations:

* Reversal hazard is strongly front-loaded.
* Most consensus episodes terminate quickly.
* Mature consensus states are rare.
* Hazard structure supports a young/non-young distinction.

---

# State Assignment Results

State assignment executed successfully on dataset v1.5.1.

Observed state distribution:

| State                  | Observations |
| ---------------------- | ------------ |
| JPY_NON_EXTREME        | 6616         |
| JPY_CONSENSUS_YOUNG    | 2198         |
| JPY_CONSENSUS_MATURING | 798          |
| JPY_CONSENSUS_MATURE   | 209          |

Observations:

* Mature states are sparse but present across all JPY pairs.
* State assignment reproduces calibration statistics.
* Survival counts match calibration expectations.

---

# Criterion 1 Investigation

## Duration-Based Validation

Duration distributions differ significantly between maturity states.

KS tests consistently identify strong separation between:

* Young vs Maturing
* Young vs Mature

However, duration-based evidence is not considered independent validation because maturity states are themselves defined using persistence duration.

Result:

**INCONCLUSIVE**

---

## Independent Outcome Labeling (v1)

Outcome definition:

```text
SUCCESS if abs(forward_return_24b) >= vol_48b
FAILURE otherwise
```

Properties:

* Independent of maturity duration
* Independent of calibration thresholds
* Independent of state assignment rules
* Uses only post-episode price behavior

Results:

| Metric             | Value |
| ------------------ | ----- |
| Evaluable episodes | 791   |
| SUCCESS            | 756   |
| FAILURE            | 35    |
| Success rate       | 95.6% |

Observations:

* Outcome labels are extremely imbalanced.
* Most episodes are classified as SUCCESS.
* Statistical power for behavioral differentiation is weak.

Criterion 1 result:

**INCONCLUSIVE**

No statistically significant differentiation was observed between maturity states under this outcome definition.

---

# Current Interpretation

The Reactive-JPY ontology successfully produces:

* Stable calibration artifacts
* Reproducible state surfaces
* Distinct maturity regimes

The remaining scientific question is whether these maturity regimes correspond to meaningful differences in independently measured future behavior.

Current evidence is insufficient to answer that question.

---

# Active Research Questions

## Outcome Discovery

The current outcome definition appears too permissive.

Candidate directions:

* Higher volatility thresholds (2×, 3×, 4× vol_48b)
* Directional outcomes
* Continuation vs reversal outcomes
* Maximum adverse excursion
* Maximum favorable excursion
* Trend continuation measures
* Alternative fixed-horizon outcome definitions

## Ontology Structure

Open questions:

* Is `JPY_CONSENSUS_MATURING` independently meaningful?
* Is the primary distinction simply young vs non-young?
* Does mature consensus contain unique behavioral information?

These questions will be revisited after additional outcome studies.

---

# Current Status

Framework status:

* Calibration: Complete
* State assignment: Complete
* Independent outcome labeling: Complete
* Criterion 1 validation: Operational

Scientific status:

**Reactive-JPY remains under validation.**

Criterion 1 has not yet been demonstrated using independent behavioral evidence.

