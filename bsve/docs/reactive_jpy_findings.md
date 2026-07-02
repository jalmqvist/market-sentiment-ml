# Reactive-JPY Findings

Research status: Behavioral validation complete
Dataset window: 2019–2026
Ontology status: Frozen
Independent behavioral validation: Complete
Formal verdict: INCONCLUSIVE (directional effect replicated, effect size criterion missed by 0.58 pp)

---

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

![jpy_hazard_curve](../calibration_artifacts/plots/jpy_hazard_curve.png)

**Figure: Empirical reversal hazard and consensus-state calibration (Reactive-JPY).**

The upper panel shows the empirical probability that a consensus episode terminates (reverses) at each maturity level, measured in hourly bars. The blue line shows the observed reversal hazard rate, while the orange dashed line shows a 12-bar rolling average used to identify broad structural trends. The green vertical line marks the Young boundary (8 bars), the red vertical line marks the Mature boundary (24 bars), and the purple line indicates the approximate crossover point (~13 bars) identified during calibration analysis.

The lower panel shows the Kaplan–Meier survival curve for consensus episodes. The y-axis represents the probability that a consensus episode remains active without reversal as maturity increases. Rapid early decay indicates that most consensus episodes terminate within the first several hours, while a small minority survive long enough to reach advanced maturity states.

Together, the hazard and survival curves provide the empirical basis for the Reactive-JPY ontology. The Young, Maturing, and Mature state boundaries were selected from observed episode-lifecycle dynamics rather than imposed *a priori*. The Young→Maturing transition occurs near the region where reversal risk begins to stabilise, while the Mature boundary identifies a small population of unusually persistent consensus episodes.

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

## Calibration Episode Statistics

The following statistics refer to the **consensus episodes used during ontology calibration**. Calibration intentionally excludes one-bar consensus episodes (`min_episode_bars = 2`) before hazard estimation in order to suppress transient threshold crossings. These reconstructed calibration episodes therefore represent a filtered subset of the Behavioral Surface episode identifiers generated during deterministic state assignment and should not be compared directly.

| Metric               | Value  |
| -------------------- | ------ |
| Calibration episodes | 441    |
| Median duration      | 4 bars |
| Survival to 8 bars   | 113    |
| Survival to 24 bars  | 21     |
| Survival to 48 bars  | 3      |

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

Behavioral Surface construction produced **1,337 deterministic behavioral segments**, of which **667** entered at least one consensus state. Unlike the calibration procedure, the Behavioral Surface records every consensus episode, including one-bar consensus events. The calibration process subsequently filters these to episodes of at least two bars (`min_episode_bars = 2`) before estimating hazard and survival functions, yielding **441 calibration episodes**. Survival statistics for longer-lived episodes (≥8, ≥24 and ≥48 bars) are identical between the two representations, confirming that they differ only by this intentional filtering step rather than by ontology behavior.

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

## Crowd-Relative Outcome Analysis

A second outcome-discovery study transformed future returns into crowd-relative outcomes.

Definition:

crowd_relative_return =
    crowd_side × future_return

Interpretation:

* Positive values indicate crowd success.
* Negative values indicate crowd failure.

An initial implementation incorrectly used:

    (-crowd_side) × future_return

which inverted outcome interpretation. The issue was identified through manual sanity checks and corrected before analysis continued.

### Crowd Success Rates

24-bar horizon:

| State    | Success | Failure |
| -------- | ------: | ------: |
| Young    |   45.8% |   54.2% |
| Maturing |   29.3% |   70.7% |
| Mature   |   61.9% |   38.1% |

48-bar horizon:

| State    | Success | Failure |
| -------- | ------: | ------: |
| Young    |   42.9% |   57.1% |
| Maturing |   31.9% |   68.1% |
| Mature   |   38.1% |   61.9% |

Observations:

* Maturing episodes exhibit substantially higher crowd-failure rates than Young episodes.
* The effect appears at both 24-bar and 48-bar horizons.
* The effect survives transformation from raw returns to crowd-relative outcomes.
* Mature-state results remain difficult to interpret because of limited sample size.

---

## Pair Decomposition

Crowd-failure outcomes were examined separately for EURJPY, GBPJPY, and USDJPY.

24-bar horizon:

| Pair   | Young Failure | Maturing Failure | Difference |
| ------ | ------------: | ---------------: | ---------: |
| EURJPY |         51.6% |            71.4% |     +19.8% |
| GBPJPY |         55.7% |            68.2% |     +12.4% |
| USDJPY |         56.3% |            71.4% |     +15.1% |

48-bar horizon:

| Pair   | Young Failure | Maturing Failure | Difference |
| ------ | ------------: | ---------------: | ---------: |
| EURJPY |         54.8% |            68.3% |     +13.5% |
| GBPJPY |         57.9% |            68.2% |     +10.3% |
| USDJPY |         59.3% |            67.9% |      +8.5% |

Observations:

* All three JPY pairs show the same directional relationship.
* Maturing failure rates exceed Young failure rates for every pair tested.
* The effect weakens somewhat at 48 bars but remains directionally consistent.
* The result is not driven by a single currency pair.

---

## Statistical Assessment

Young vs Maturing crowd-failure rates were compared using contingency-table methods.

### 24-Bar Horizon

| Metric                  |                   Value |
| ----------------------- | ----------------------: |
| Fisher p-value          |                  0.0032 |
| Chi-square p-value      |                  0.0047 |
| Odds ratio              |                    0.49 |
| Failure-rate difference | +16.4 percentage points |

### 48-Bar Horizon

| Metric                  |                   Value |
| ----------------------- | ----------------------: |
| Fisher p-value          |                   0.051 |
| Chi-square p-value      |                   0.061 |
| Odds ratio              |                    0.62 |
| Failure-rate difference | +11.0 percentage points |

Interpretation:

* The 24-bar effect is statistically significant in exploratory testing.
* The 48-bar effect is weaker but remains directionally consistent.
* Results should be interpreted as outcome discovery rather than validation because the hypothesis emerged during exploratory analysis.

---

# Current Interpretation

The strongest behavioral distinction identified so far is not volatility magnitude, but crowd-failure probability.

Reactive-JPY Maturing episodes appear substantially more likely than Young episodes to experience crowd-unfavorable outcomes over subsequent 24–48 bar horizons.

The effect:

* survives pair decomposition,
* survives horizon variation,
* survives transformation to crowd-relative outcomes,
* exhibits exploratory statistical support at 24 bars.

Current working hypothesis:

> Maturing consensus episodes represent a vulnerable consensus state in which crowd positioning is more likely to fail than during newly established consensus episodes.

This hypothesis remains exploratory and requires future independent validation.

---

![Figure_1](../calibration_artifacts/plots/Figure_1.png)

**Figure: Crowd-failure rate as a function of consensus maturity (Reactive-JPY).**

The x-axis shows the maximum maturity reached by a consensus episode before termination, measured in hourly bars. A maturity of 1 indicates that consensus failed almost immediately, while higher values indicate longer-lived consensus episodes.

The y-axis shows the proportion of episodes that subsequently produced a crowd-unfavourable outcome over a 24-bar horizon. A crowd-failure occurs when price movement over the following 24 hours moves against the majority trader positioning implied by sentiment data.

Blue points show observed failure rates at each maturity level. Vertical error bars show 95% Wilson confidence intervals. The dark blue curve shows a 5-bar rolling average used only for visualisation. Grey bars indicate the number of episodes contributing to each maturity level. The dashed vertical line marks the Young→Maturing boundary (8 bars) and the dotted vertical line marks the Maturing→Mature boundary (24 bars).

Failure rates appear to increase as maturity approaches the Young→Maturing boundary, consistent with the state-level finding that Maturing episodes exhibit higher crowd-failure rates than Young episodes. Beyond approximately 12–15 bars, sample sizes become sparse and maturity-level estimates should be interpreted cautiously.

---

### Reframing the Maturity States

The calibration defined maturity states in terms of survival duration.
The outcome discovery studies suggest a different interpretation:

- JPY_CONSENSUS_YOUNG: Episodes where crowd positioning is still being established. Crowd failure rate is only modestly elevated (~54–57%), suggesting that newly formed consensus episodes are not yet strongly associated with systematic crowd failure.
  
- JPY_CONSENSUS_MATURING: Episodes where crowd positioning has survived initial reversal pressure and consolidated. Crowd failure rate rises to ~68-71%. The crowd has committed, and that commitment appears to be associated with a substantially higher probability of subsequent crowd failure.

This is consistent with the consensus formation → maturation → decay chain described in RESEARCH_STATE.md, but gives it a specific mechanistic interpretation: maturation may represent the point at which crowd positioning becomes overextended rather than merely persistent. 

This interpretation was treated as speculative at the time of writing. The independent validation results (July 2026) are consistent with it, although the formal verdict remains INCONCLUSIVE under the pre-registered protocol.

---

# **Frozen Findings — Exploratory Phase (June 2026)**

The following findings are considered stable enough to record and carry forward:

1. Magnitude-based outcome families show little differentiation between maturity states.
2. Duration-derived outcomes are not independent of the ontology and cannot support validation.
3. Maturing episodes exhibit elevated crowd-failure rates relative to Young episodes.
4. The crowd-failure effect is present across EURJPY, GBPJPY, and USDJPY.
5. The strongest observed outcome family is crowd-failure probability at approximately 24 bars.

Reactive-JPY outcome discovery is therefore considered provisionally complete.

The independent validation study described below was conducted against these frozen findings without modification.

---

# Research Status

Engineering status

- Calibration: Complete
- Behavioral Surface generation: Complete
- Behavioral validation pipeline: Complete

Scientific status

- Outcome discovery: Complete
- Independent behavioral validation: Complete (formal verdict: INCONCLUSIVE)
- Ontology status: Frozen
- Next research stage: MSML predictive validation

---

## Outstanding Research Questions

The independent validation substantially reduced uncertainty regarding the behavioral representation, but several scientific questions remain open:

- Does the observed OOS effect size reflect the true population effect, or is the exploratory estimate inflated by in-sample selection? This question will be partially addressed by the second-broker replication and by MSML walk-forward evaluation.
  
- Does the Mature state contain information beyond the Maturing state, or is it principally an extreme-duration subset requiring larger datasets for reliable analysis?
  
- Does the Reactive-JPY behavioral representation improve predictive performance under repeated walk-forward evaluation within MSML?
  
- Do analogous behavioral structures emerge in other currency families (e.g. Reactive-CHF), or are the observed dynamics specific to JPY markets?

These questions concern downstream interpretation and generalization rather than ontology development. The Reactive-JPY ontology itself remains frozen.

---

## Independent Validation Results (July 2026)

The Reactive-JPY crowd-failure hypothesis was evaluated on an independent out-of-sample window (2023-01-01 to 2024-08-22) using the frozen ontology calibration artifact (`reactive_jpy_v1_20260615`). No ontology parameters, calibration thresholds or state definitions were modified before or after the validation.

Prior to statistical evaluation, the Behavioral Surface passed the predefined validation gate. Independent inspection confirmed plausible state frequencies, episode-duration distributions, maturity progression and one-to-one alignment with the master research dataset. No structural inconsistencies were detected, and all validation artifacts were generated deterministically from the frozen ontology without recalibration.

During validation, the apparent discrepancy between the calibration episode count (441) and the Behavioral Surface episode count was investigated and reconciled. The difference was found to arise from the calibration pipeline's intentional exclusion of one-bar consensus episodes (`min_episode_bars = 2`) prior to hazard estimation. Longer-lived survival statistics matched exactly between the calibration artifact and the Behavioral Surface, confirming that both representations are internally consistent and differ only in their intended purpose.

The pooled validation analysed 2,996 labeled observations across USDJPY, EURJPY and GBPJPY. The directional behavioral effect replicated on independent data: Maturing episodes exhibited a higher crowd-failure rate than Young episodes (59.65% vs. 55.23%, Δ = +4.42 percentage points, 95% CI [+0.43%, +8.41%], one-sided Fisher p = 0.017). The effect was present in all three currency pairs individually, with USDJPY showing the strongest individual signal (Δ = +7.93 percentage points, p = 0.019), while EURJPY and GBPJPY exhibited smaller effects in the same direction.

Applying the frozen validation protocol produced the formal outcome **INCONCLUSIVE**, because the pooled effect size of 4.42 percentage points did not satisfy the pre-specified practical-effect criterion of 5 percentage points. This criterion was missed by 0.58 percentage points, while all remaining predefined validation criteria were satisfied. The **INCONCLUSIVE** label is therefore the outcome of the pre-registered decision protocol rather than a scientific conclusion that the behavioral relationship is absent.

Relative to the exploratory study, the independent validation produced a smaller estimated effect size. This attenuation is consistent with several plausible explanations, including regression toward a smaller true effect size, limited statistical power within the OOS window, or genuine temporal variation in market behavior. The current data do not distinguish between these possibilities. Importantly, the confidence interval spans the pre-specified practical-effect threshold, indicating that the available evidence is insufficient to determine whether the true effect lies slightly below or slightly above that boundary.

The 5 percentage point practical-effect criterion was specified conservatively before the OOS analysis, based on substantially larger differences observed during exploratory ontology development. Exploratory studies commonly overestimate effect sizes because of selection and sampling variability; consequently, a smaller effect in an independent validation is methodologically expected and does not, by itself, invalidate the underlying behavioral hypothesis.

Taken together, the independent validation provides evidence consistent with the hypothesized behavioral relationship. The predicted directional effect replicated on unseen data, achieved statistical significance in the pooled analysis, and was observed consistently across all three JPY pairs. At the same time, the validation did not satisfy every requirement of the frozen decision protocol and therefore cannot be classified as a formal confirmation under the pre-registered specification.

The ontology remains frozen. No recalibration or post-hoc modification will be performed on the basis of these results, and the reserved future holdout window (2025-05-10 onward) will remain untouched in order to preserve its independence for future studies.

The behavioral phase of the research program is therefore considered complete. Subsequent work moves from behavioral validation to predictive validation, where the Behavioral Surface will be evaluated under full walk-forward validation within the MSML-MPML framework. The objective of that stage is no longer to re-test the behavioral hypothesis, but to determine whether the Reactive-JPY behavioral representation improves predictive performance relative to the existing LVTF/HVTF/LVR/HVR market-regime partition under repeated out-of-sample evaluation.

A second independent replication remains planned using the separate broker sentiment dataset, collection of which began in July 2026. Because sufficient history will require substantial time to accumulate, this replication is treated as a future confirmatory study rather than a prerequisite for advancing the current research program.
