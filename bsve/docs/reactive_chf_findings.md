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

### Study 0A: ATR% Distribution

**Question:** Is the ATR% distribution for CHF pairs structured in a way that supports natural volatility regimes, or is it primarily unimodal?

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

- ATR% distribution plots for EURCHF and USDCHF separately.
- Pooled ATR% distribution.
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

