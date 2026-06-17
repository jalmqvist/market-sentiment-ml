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

# Outcome Discovery Studies (June 2026)

Following the initial Criterion 1 investigation, a series of exploratory outcome-discovery studies were performed to identify candidate independent behavioral outcomes for Reactive-JPY.

These studies are descriptive only and must not be interpreted as validation because they were conducted on the same DL-active window used during ontology development.

## Magnitude-Based Outcomes

Multiple volatility-adjusted outcome definitions were evaluated:

```text
SUCCESS if |return_24b| >= k × vol_48b
```

for:

- k = 1
- k = 2
- k = 3
- k = 4
- k = 5

Observations:

- Outcome distributions were highly similar across maturity states.
- State separation remained weak regardless of threshold.
- Magnitude-based outcomes do not currently appear to capture the primary behavioral distinction represented by the ontology.

Current assessment:

**Low priority.**

------

## Directional Continuation Outcomes

Continuation and reversal outcomes were evaluated across horizons:

- 1 bar
- 2 bars
- 4 bars
- 6 bars
- 12 bars
- 24 bars
- 48 bars

A notable separation was observed at 24–48 bars:

| State    | Continuation (24b) |
| -------- | ------------------ |
| Young    | 45.8%              |
| Maturing | 29.3%              |
| Mature   | 61.9%              |

However:

- The effect was not consistently present across shorter horizons.
- Mature observations remain sparse.
- The temporal pattern does not yet support a clear lifecycle interpretation.

Current assessment:

**Interesting but inconclusive.**

------

## Return Distribution Analysis

Future return distributions were examined directly rather than through binary outcome labels.

At both 24-bar and 48-bar horizons:

- Young episodes were approximately neutral.
- Maturing episodes exhibited consistently more positive future returns.
- The effect was observed across EURJPY, GBPJPY, and USDJPY.

Examples (24-bar horizon):

| Pair   | Young Mean | Maturing Mean |
| ------ | ---------- | ------------- |
| EURJPY | -0.05%     | +0.14%        |
| GBPJPY | +0.03%     | +0.22%        |
| USDJPY | +0.04%     | +0.16%        |

The same directional relationship persisted at 48 bars.

Additional observations:

- Median returns also increased for Maturing episodes.
- The improvement was not confined to the upper tail.
- The entire return distribution appears shifted upward relative to Young episodes.

Current assessment:

**Most promising outcome family identified so far.**

------

## Episode Lifecycle Findings

Consensus episode reconstruction produced:

| Terminal State | Episodes |
| -------------- | -------- |
| Young          | 554      |
| Maturing       | 92       |
| Mature         | 21       |

Observations:

- Most consensus episodes terminate in the Young state.
- Only a minority survive long enough to become Maturing.
- Mature episodes are rare.
- The lifecycle funnel observed during calibration is reproducible in state-surface artifacts.

Current exit-event labels are derived from ontology logic and therefore cannot be used as independent behavioral evidence.

Future exit-mechanism studies should focus on raw episode characteristics rather than BSVE-generated outcome labels.

------

# Updated Interpretation

Reactive-JPY currently shows little evidence that maturity states differentiate future volatility magnitude.

However, exploratory evidence suggests that Maturing episodes may be associated with systematically different future directional return behavior.

This effect:

- survives pair decomposition,
- appears at both 24-bar and 48-bar horizons,
- is visible in both means and medians,
- is observed across EURJPY, GBPJPY, and USDJPY.

These findings are exploratory and require independent validation.

------

# Current Research Priority

Highest-priority follow-up:

## Outcome Discovery Study #2

Investigate crowd-relative returns rather than raw market returns.

Candidate formulation:

```text
crowd_relative_return =
    (-crowd_side) × future_return
```

This directly measures crowd success versus crowd failure and may provide a more behaviorally meaningful independent outcome definition for Reactive-JPY validation.

---

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

