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

### Preliminary Interpretation

The findings suggest that volatility context is behaviorally relevant in CHF pairs.

Importantly, the volatility distributions themselves appeared largely continuous, while persistence behavior displayed threshold-like structure. This implies that regime-like behavior may emerge from the relationship between volatility and persistence rather than from discrete volatility clusters.

At present, the strongest candidate explanatory variable is vol_48b.

This does not establish a CHF ontology. However, it provides preliminary evidence that the Volatility-Conditioned Persistence hypothesis survives initial exploratory testing and warrants continuation into Studies 0A–0C.

### Methodological Note

An exploratory attempt was made to introduce ATR-based and Kaufman-style volatility measures.

This effort revealed that the master sentiment dataset consists of irregularly sampled sentiment observations rather than continuous hourly price bars. ATR-derived measures therefore cannot be computed directly within the sentiment dataset and require reconstruction from the underlying continuous OHLC price series.

This limitation does not affect the validity of the existing vol_12b and vol_48b features, which were generated during dataset construction.

---

### Study 0A: Volatility Distribution

**Question:** Is the volatility distribution for CHF pairs structured in a way that supports natural volatility regimes, or is it primarily unimodal?

**Why this study comes first**

The current CHF hypothesis assumes that distinct volatility environments exist. If volatility regimes do not exist empirically, any subsequent ontology based on volatility buckets would be arbitrary.

Study 0A therefore examines the shape of the volatility distribution before any regime boundaries are proposed.

**What to look for**

- Bimodal or multimodal distributions may indicate naturally occurring volatility regimes.
- Heavy-tailed distributions may support a "high-volatility tail" interpretation rather than discrete regimes.
- Smooth unimodal distributions may suggest that volatility is better treated as a continuous variable.

**Important caveat**

A unimodal distribution does not imply that volatility is behaviorally irrelevant. Many valid behavioral variables are continuous. Study 0A evaluates whether natural volatility regimes are visually supported by the data; it does not test whether volatility influences market behavior.

**Output**

- Volatility distribution plots for EURCHF and USDCHF separately.
- Pooled volatility distribution.
- Kernel density estimates.
- Notes on visible modes, shoulders, tails, and structural features.

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

- Persistence duration versus ATR% scatter plots.
- Binned persistence statistics across volatility percentiles.
- Survival curves conditioned on volatility.
- Preliminary assessment of whether persistence appears linked to volatility.

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

### Study 0A

Is volatility structurally distributed in a way that suggests natural regimes?

- Yes → proceed to Study 0B.
- No → volatility may still be behaviorally informative, but regime boundaries require stronger justification.

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

