### CHF Null Hypothesis (Pre-Study)

**H0:** Volatility context does not predict crowd-state persistence duration or crowd-failure probability in CHF pairs within the 2019–2026 sentiment-data window.

Studies 0A–0C are designed to test whether volatility contains information about persistence behavior and subsequent crowd outcomes. The Volatility-Conditioned Persistence ontology is therefore treated as a hypothesis rather than an assumption.

If H0 cannot be rejected after Studies 0A–0C, the Volatility-Conditioned Persistence ontology should be abandoned. In that case, either an alternative CHF behavioral hypothesis should be formulated or the CHF environment should be classified as having no stable behavioral ontology within the available dataset.

The purpose of these studies is hypothesis testing rather than ontology construction. Ontology development will proceed only if exploratory evidence demonstrates that volatility context is behaviorally informative.

---

## Phase 0: Exploratory Ontology Discovery

The Reactive-CHF program begins with exploratory studies rather than ontology construction.

This differs from the Reactive-JPY sequence. In Reactive-JPY, the maturity ontology was calibrated first and outcome discovery followed. The outcome studies subsequently changed the interpretation of the states and identified crowd-failure probability as the primary behavioral outcome family.

The CHF hypothesis is structurally different. Rather than focusing on episode lifecycle, it proposes that volatility context may influence how crowd states form, persist, and eventually fail. Before constructing a volatility-conditioned ontology, it is therefore necessary to establish whether volatility contains meaningful behavioral information at all.

The purpose of Studies 0A–0C is to determine whether volatility should be treated as an ontology variable, a calibration variable, a proxy for another mechanism, or not used at all.

Importantly, these studies are exploratory but falsifiable. The objective is not to confirm the existence of a CHF ontology. The objective is to determine whether the data supports one.

------

## Study -1: Volatility Measure Discovery (Completed)

### Purpose

Before evaluating volatility-conditioned persistence, it was necessary to determine whether volatility itself exhibited any meaningful behavioral relationship to the persistence features that originally caused CHF pairs to be classified as reactive.

This study therefore preceded Studies 0A–0C and served as an exploratory assessment of volatility representations and their relationship to crowd persistence.

### Data

The study used the 2019–2026 CHF subset of the master research dataset.

Pairs examined:

- EURCHF
- USDCHF

Volatility measures examined:

- vol_12b (12-bar rolling return volatility)
- vol_48b (48-bar rolling return volatility)

At this stage, volatility was treated as a continuous variable rather than a regime variable.

### Study -1A: Volatility Distribution Analysis

The first objective was to determine whether CHF volatility naturally separated into distinct regimes.

Results:

- Both volatility measures exhibited largely unimodal distributions.
- No clear evidence of naturally occurring low-, medium-, and high-volatility clusters was observed.
- The distributions were right-skewed, with elevated-volatility tail events but no obvious multimodal structure.
- EURCHF and USDCHF displayed similar distributional shapes, although USDCHF exhibited consistently higher volatility levels.

Interpretation:

The results did not support the existence of obvious volatility regimes based solely on distributional structure.

However, a unimodal distribution does not imply behavioral irrelevance. Volatility may still contain predictive information even if natural regime boundaries are not visually apparent.

![volatility_distributions](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/bsve/calibration_artifacts/plots/volatility_distributions.png)

> **Figure -1A. CHF volatility distributions (2019–2026).** Distribution of rolling volatility measures used during the CHF exploratory studies. Both the 12-bar (`vol_12b`) and 48-bar (`vol_48b`) measures exhibit largely unimodal, right-skewed distributions. Elevated-volatility observations occur as a continuous tail rather than as clearly separated clusters. This suggests that volatility environments in CHF are not naturally partitioned into discrete low-, medium-, and high-volatility regimes based on distributional structure alone. Notice the volatility range difference between the two plots.

---

**Figure -1A** summarizes the distributional structure of the volatility measures examined during Study -1. Contrary to the original intuition behind a volatility-conditioned ontology, neither measure exhibits obvious multimodal behavior. Instead, volatility appears to vary along a largely continuous spectrum with an extended high-volatility tail. This result weakens the argument for defining CHF states directly from volatility buckets alone. At the same time, the absence of discrete volatility clusters does not imply behavioral irrelevance. Subsequent analyses demonstrated that persistence behavior varies substantially across this continuous volatility spectrum, suggesting that volatility may still influence crowd dynamics even in the absence of naturally separated volatility regimes.

---

### Study -1B: Volatility vs Persistence

The second objective was to determine whether volatility was related to the persistence features that originally motivated the CHF classification.

Persistence variables examined:

- side_streak
- extreme_streak_70
- extreme_streak_80

Results:

- A strong inverse relationship was observed between volatility and persistence.
- The relationship was strongest for vol_48b.
- Low-volatility observations exhibited substantially longer side streaks than high-volatility observations.
- The effect survived pair decomposition and was observed independently in both EURCHF and USDCHF.
- The relationship appeared nonlinear and threshold-like rather than purely linear.

The most stable signal was observed in side_streak. Extreme-state persistence displayed the same broad direction but with greater noise.
![eurchf_vol48b_sidestreak_median_extremestreak70_vol48b_percentile_bin](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/bsve/calibration_artifacts/plots/eurchf_vol48b_sidestreak_median_extremestreak70_vol48b_percentile_bin.png)

> **Figure -1B. Volatility versus persistence in EURCHF.**
>
> Scatterplot showing the relationship between 48-bar rolling volatility (`vol_48b`) and crowd persistence (`side_streak`) in EURCHF observations from 2019–2026. Each point represents a sentiment observation. The x-axis measures how long the crowd has maintained the same directional positioning, while the y-axis measures recent realized market volatility.
>
> The relationship is strongly nonlinear. High-volatility environments are associated exclusively with short-lived crowd states, whereas very persistent crowd states occur only within a narrow low-volatility corridor. Importantly, low volatility does not guarantee persistence; many low-volatility observations still exhibit short streak lengths. This suggests that low volatility may be a necessary but not sufficient condition for extended crowd persistence.
>
> Several visually distinct clusters correspond to long-lived crowd episodes that remained active within relatively stable volatility environments. These observations provide preliminary support for the Volatility-Conditioned Persistence hypothesis and motivated the subsequent CHF ontology studies.

---

**Figure -1B** provides a more direct view of the volatility–persistence relationship than percentile-bin analysis. Rather than imposing volatility buckets, the scatterplot shows the raw relationship between volatility and crowd-state duration. The absence of highly persistent crowd states at elevated volatility levels suggests that volatility acts as a constraint on persistence. The figure therefore supports the interpretation that volatility may influence the conditions under which long-lived CHF crowd states can emerge.

---

### **USDCHF Mid-Volatility Persistence Anomaly**

Unlike EURCHF, which exhibits a broadly monotonic decline in persistence as volatility increases, USDCHF displays a secondary persistence peak at intermediate volatility levels (approximately volatility percentile bins 7–8 in the vol_48b analysis).

This feature was not anticipated by the original CHF hypothesis and remains unexplained at this stage. Several interpretations remain possible:

- The USDCHF volatility–persistence relationship may differ structurally from EURCHF, reflecting the dual influence of CHF safe-haven dynamics and USD macro cycles on USDCHF positioning.
- USD-driven trend environments may sustain crowd states at intermediate volatility levels through a different mechanism than the low-volatility equilibrium persistence observed in EURCHF.
- The feature may reflect concentration of a small number of unusually long-lived episodes at a specific volatility level.
- The feature may be a statistical artifact that disappears under alternative volatility definitions or finer bin resolution.

At present there is insufficient evidence to distinguish between these explanations. The anomaly therefore remains an open question rather than a confirmed finding.

If the effect survives into Studies 0B and 0C -- particularly if it appears in outcome-conditioned analyses -- it will be treated as evidence that EURCHF and USDCHF do not share a single volatility-conditioned persistence ontology and may require separate ontological treatment. This would represent a significant structural finding for the CHF research program.

---

### Preliminary Interpretation

The findings suggest that volatility context is behaviorally relevant in CHF pairs.

Importantly, the volatility distributions themselves appeared largely continuous, while persistence behavior displayed threshold-like structure. This implies that regime-like behavior may emerge from the relationship between volatility and persistence rather than from discrete volatility clusters.

At present, the strongest candidate explanatory variable is vol_48b.

This does not establish a CHF ontology. However, it provides preliminary evidence that the Volatility-Conditioned Persistence hypothesis survives initial exploratory testing and warrants continuation into Studies 0A–0C.

---

### Volatility Measure Selection

Study -1 evaluated two volatility representations: **vol_12b** and **vol_48b**.

Both measures produced broadly consistent results, but the relationship between volatility and persistence was substantially clearer for **vol_48b**. Across pair-level analyses, volatility-binned persistence statistics, and volatility-versus-persistence scatterplots, the vol_48b measure consistently produced:

- Stronger persistence relationships.
- Cleaner threshold behavior.
- Reduced noise.
- More interpretable behavioral structure.

By comparison, vol_12b exhibited greater short-term variability and weaker persistence signals, making interpretation less stable.

Accordingly, **vol_48b will be treated as the primary volatility measure for all subsequent CHF studies.** The vol_12b measure will be retained as a robustness check and sensitivity analysis but will not be used as the primary ontology variable unless later studies provide evidence that it contains additional behavioral information.

This decision reflects empirical performance rather than theoretical preference and may be revisited if future analyses identify superior volatility representations.

---

### Methodological Note

An exploratory attempt was made to introduce ATR-based and Kaufman-style volatility measures.

This effort revealed that the master sentiment dataset consists of irregularly sampled sentiment observations rather than continuous hourly price bars. ATR-derived measures therefore cannot be computed directly within the sentiment dataset and require reconstruction from the underlying continuous OHLC price series.

This limitation does not affect the validity of the existing vol_12b and vol_48b features, which were generated during dataset construction.

---

### Study 0A: Volatility Partition Discovery

**Question:** Can the continuous CHF volatility spectrum be partitioned into behaviorally distinct regions that exhibit different persistence characteristics?

**Why this study comes first**

Study -1 demonstrated that CHF volatility distributions are largely unimodal and do not exhibit strong evidence of naturally occurring volatility regimes. However, the persistence analyses revealed substantial behavioral variation across the volatility spectrum.

The question is therefore no longer whether volatility regimes exist statistically. The question is whether meaningful behavioral partitions exist within a continuous volatility variable.

Study 0A investigates whether persistence behavior changes smoothly across volatility levels or whether threshold effects emerge that justify dividing volatility into separate behavioral environments.

**What to look for**

- Persistence thresholds where crowd-state duration changes abruptly.
- Stable low-volatility regions associated with unusually persistent crowd behavior.
- Intermediate-volatility regions that exhibit distinct persistence characteristics.
- Evidence that persistence changes smoothly with volatility, implying that volatility should remain a continuous variable.
- Pair-specific partition structure that differs between EURCHF and USDCHF.

**Important caveat**

Behavioral partitions do not require multimodal volatility distributions. A continuous variable can still contain meaningful thresholds if behavior changes nonlinearly across its range.

The objective of Study 0A is therefore not to identify statistical clusters in volatility itself, but to determine whether volatility contains behavioral breakpoints that may justify ontology boundaries.

**Output**

- Volatility-versus-persistence scatterplots.
- Persistence statistics across volatility percentile bins.
- Threshold and breakpoint analysis.
- Candidate volatility partitions for ontology construction.
- Assessment of whether EURCHF and USDCHF share a common partition structure.

------

### Study 0B: Volatility vs Persistence

**Question:** Does volatility context contain information about crowd-state persistence duration?

**Why this matters**

The current CHF hypothesis proposes that volatility influences how long crowd states survive.

This study tests that assumption directly before any volatility regimes are defined.

The working mechanism can be expressed as:

```
volatility context
        ↓
persistence duration
        ↓
behavioral outcome
```

However, persistence is treated as a candidate mechanism rather than an established ontology variable.

A null result would imply only that persistence is unlikely to be the primary channel through which volatility influences behavior. It would not imply that volatility itself is behaviorally irrelevant.

**What to look for**

- Strong monotonic relationships between volatility and persistence.
- Nonlinear relationships.
- Threshold effects.
- Evidence that persistence is independent of volatility.

**Methodological note**

Persistence should initially be examined as a continuous function of volatility rather than through predefined volatility buckets.

The goal is to allow the data to suggest possible partitions rather than imposing them in advance.

**Output**

- Persistence duration versus vol_48b scatter plots.
- Binned persistence statistics across volatility percentiles.
- Survival curves conditioned on volatility.
- Preliminary assessment of whether persistence appears linked to volatility.

> **Relationship to Study 0A**
>
> Study 0A and Study 0B use many of the same exploratory visualizations, including volatility-versus-persistence scatterplots and volatility-binned persistence statistics. The distinction lies in the question being asked rather than the plots themselves.
>
> Study 0A uses these analyses to identify possible behavioral thresholds and candidate partition boundaries within the volatility spectrum.
>
> Study 0B uses the same analyses to determine whether persistence is meaningfully related to volatility and therefore whether persistence should be treated as a candidate ontology variable.
>
> Thus, Study 0A focuses on partition discovery, while Study 0B focuses on mechanism validation.

------

### Study 0C: Volatility vs Crowd Failure

**Question:** Does volatility context predict crowd-failure probability?

**Why this matters**

Study 0C investigates whether volatility has direct behavioral consequences.

This is the study most likely to determine whether volatility belongs in the ontology itself.

Several outcomes are possible.

#### Case 1: Full mediation

```
volatility
        ↓
  persistence
        ↓
  crowd failure
```

Volatility influences outcomes only because it influences persistence.

In this case, persistence is likely the primary ontology variable and volatility acts mainly as a calibration input.

#### Case 2: Direct volatility effect

```
volatility
        ↓
  crowd failure
```

Volatility predicts outcomes independently of persistence.

In this case, volatility itself likely belongs in the ontology.

#### Case 3: Mixed effect

```
volatility
        ↓
  persistence
        ↓
  crowd failure
```

and

```
volatility
        ↓
  crowd failure
```

Both channels exist simultaneously.

In this case, a volatility × persistence interaction ontology may be required.

#### Case 4: Null result

Volatility predicts neither persistence nor crowd-failure behavior.

In this case, volatility is unlikely to be an appropriate ontology variable and the CHF hypothesis should be reconsidered.

**Outcome-family discovery remains in scope.**

If crowd failure does not vary meaningfully with volatility context, this does not necessarily invalidate the CHF research program. It may instead indicate that crowd failure is not the primary behavioral outcome family for CHF and that alternative outcome families should be investigated.

------

## Current Status

The exploratory pre-study phase produced two important findings.

First, CHF volatility distributions appear largely unimodal and do not provide strong visual evidence for naturally occurring volatility regimes.

Second, volatility exhibits a robust inverse relationship with crowd-persistence measures across both EURCHF and USDCHF. The relationship is strongest for vol_48b and appears nonlinear, suggesting possible threshold behavior.

As a result, the Volatility-Conditioned Persistence hypothesis remains viable. The primary unresolved question is no longer whether volatility matters, but rather how volatility should be represented within the ontology and whether persistence mediates subsequent behavioral outcomes.

The next phase therefore shifts from volatility-measure discovery toward testing whether persistence and volatility jointly predict meaningful outcome families.

---

## Decision Logic

The purpose of Studies 0A–0C is not to prove that a volatility ontology exists.

The purpose is to determine whether one is justified.

### **Study 0A**

 Does volatility contain identifiable behavioral thresholds or partition structure?

- No threshold evidence → treat volatility as a continuous covariate in Studies 0B and 0C; no ontology partition proposed at this stage.
- Weak evidence → retain candidate partitions and proceed to Study 0B.
- Strong evidence → candidate volatility partitions become ontology candidates; proceed to Study 0B.

### Study 0B

Does volatility contain information about persistence duration?

- No → persistence is unlikely to be the primary mechanism; proceed to Study 0C.
- Weak or nonlinear → persistence may play a partial role; proceed to Study 0C.
- Yes → persistence becomes a candidate ontology variable; proceed to Study 0C.

### Study 0C

Does volatility predict crowd-failure behavior?

- Only through persistence → persistence-based ontology favored.
- Independently of persistence → volatility belongs in the ontology.
- Through both channels → interaction ontology may be required.
- Not at all → crowd failure may not be the correct outcome family.

A final possibility must remain available throughout the CHF program:

> CHF may not possess a stable behavioral ontology within the available dataset.

This is a legitimate scientific conclusion rather than a failure state. The purpose of the CHF investigation is to determine whether a meaningful ontology exists, not to guarantee that one will be found.

---

### Working Interpretation (Tentative)

EURCHF and USDCHF are structurally different instruments:

- **EURCHF** is a cross rate driven primarily by European risk appetite and SNB policy. The CHF safe-haven bid compresses volatility and creates the calm, persistent crowd states that motivated the CHF classification in the first place. The inverse vol-persistence relationship in EURCHF is almost mechanically explained by this: when European risk appetite is stable and the SNB is not intervening, volatility is low and crowd states are long-lived.
- **USDCHF** is also influenced by the CHF safe-haven dynamic, but it has a second major driver: USD macro cycles. The 2022 USD strength cycle appears in the volatility time series as a sustained elevated-volatility period. During that period, USDCHF was trending strongly on USD fundamentals, not CHF safe-haven flows. A trend-driven market can produce long-lived crowd states *at intermediate volatility* — the crowd is persistently positioned with the trend, and the trend sustains itself through fundamental momentum rather than low-volatility equilibrium.

---

