# Persistent BSVE Candidate Surface Roadmap

**Status:** Design / research specification  
**Surface:** Persistent Commitment Lifecycle  
**Candidate version:** v0.1.0  
**Dataset:** 1.6.1  
**Last updated:** 2026-09-23

---

## 1. Purpose

This document defines the research and implementation roadmap for the first
candidate Behavioral Surface for the Persistent FX pair family.

The purpose is not to establish a final Persistent ontology.

The immediate objective is to determine whether the behavioral structure
identified during the P0C Persistent re-audit can be expressed as a deterministic,
causal, BSVE-compatible Behavioral Surface and subsequently evaluated as a
representation in MSML.

The candidate surface must therefore be treated as a **research hypothesis**,
not as an established predictive or trading model.
### Coverage-aware calibration constraint

P0C-BSVE-0 and P0C-BSVE-1 established an important data-quality constraint
for Persistent calibration.

P0C-BSVE-0 confirmed two synchronized sentiment-coverage outages across all
five Persistent pairs in dataset 1.6.1:

- 2024-08-23 to 2024-10-03;
- 2024-10-31 to 2025-05-09.

The audit detected 1,626 empirical long gaps in the Persistent sample. P0C-BSVE-1
found that 407 of the 1,421 canonical Persistent episodes cross at least one
long gap. Therefore canonical same-crowd-side continuity cannot be assumed
across a long coverage gap.

For BSVE calibration, a long gap is consequently treated as a **hard
observational break**. No sentiment state is forward-filled, interpolated, or
otherwise reconstructed across such a gap.

This does not redefine the historical P0C-0 canonical episode reconstruction.
Instead, calibration uses a coverage-aware **observed segment** representation
alongside the canonical episode representation.


---

## 2. Scientific Context

The Persistent family currently consists of:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

The P0C re-audit investigated whether Persistent episodes contain a structured
behavioral representation beyond raw sentiment and episode duration.

The analysis reduced a larger set of historical commitment descriptors to two
approximately independent dimensions:

1. **Commitment Level**
2. **Commitment Trajectory**

The selected representatives are:

| Behavioral dimension  | Current representative        | Interpretation                                 |
| --------------------- | ----------------------------- | ---------------------------------------------- |
| Commitment Level      | `prior_mean_depth`            | Historical depth/intensity of crowd commitment |
| Commitment Trajectory | `early_late_commitment_delta` | Evolution of commitment through the episode    |

These dimensions produced a coherent 3×3 termination surface across the
Persistent family.

The resulting surface is strongly associated with episode termination, but
P0C did **not** establish a direct, temporally stable relationship between the
surface and unconditional future returns.

This distinction is important.

The candidate surface is therefore being developed as a **behavioral
representation**, not as a direct return predictor.

---

## 2A. P0C-BSVE Coverage Findings

P0C-BSVE-0 established that the major 2024-2025 sentiment-collection gaps
are present across the full Persistent family rather than being specific to
CHF pairs or to episode construction.

P0C-BSVE-1 then quantified their effect on Persistent episode continuity:

| Quantity | Result |
| --- | ---: |
| Persistent observations | 16,204 |
| Canonical episodes | 1,421 |
| Empirical long gaps | 1,626 |
| Canonical episodes crossing >=1 long gap | 407 |
| Single-segment episodes | 1,014 |
| Observations with >=4 within-segment observations | 12,078 |

The last figure is based on the provisional joint-history rule used in the
study and is not a final calibration eligibility count.

These findings motivate the coverage-aware calibration rules in Section 7.
They do not imply a cause for the missing data and do not justify reconstructing
unobserved sentiment states.

---

## 3. Relationship to BSVE

The existing BSVE architecture separates:

```text
Calibration
    ↓
Behavioral Surface
    ↓
Behavioral Dataset Variant
    ↓
MSML
    ↓
Prediction Artifact
    ↓
MPML
```

The Persistent candidate must follow the same artifact and causal principles
 as the existing Reactive-JPY implementation.

In particular:

- calibration precedes state assignment;
- calibration is frozen before evaluation;
- state assignment is deterministic;
- historical observations must not depend on future observations;
- the Behavioral Surface is an explicit artifact contract;
- scientific interpretation remains separate from experiment outputs;
- promotion to the Behavioral Surface Registry is a deliberate research
   decision and is not part of initial implementation.

The candidate should therefore be compatible with BSVE without assuming that
 Persistent and Reactive-JPY have identical behavioral mechanisms.

------

# 4. Candidate Behavioral Object

The current proposed behavioral object is:

> **Persistent Commitment Lifecycle**

The object is represented by two dimensions:

```
Commitment Level
        ×
Commitment Trajectory
```

### Commitment Level

Representative:

```
prior_mean_depth
```

Interpretation:

> Historical average commitment depth accumulated before the current
>  observation.

### Commitment Trajectory

Representative:

```
early_late_commitment_delta
```

Interpretation:

> Change in commitment depth between the early and later portion of the
>  current episode.

These definitions are inherited from the P0C analysis and should not be
 silently replaced by alternative features during implementation.

------

# 5. Candidate State Space

The first candidate surface is a 3×3 discretization:

```
                         Commitment Trajectory

                    LOW        MID        HIGH
                 ┌─────────┬─────────┬─────────┐
Commitment LOW   │  LL     │  LM     │  LH     │
                 ├─────────┼─────────┼─────────┤
Commitment MID   │  ML     │  MM     │  MH     │
                 ├─────────┼─────────┼─────────┤
Commitment HIGH  │  HL     │  HM     │  HH     │
                 └─────────┴─────────┴─────────┘
```

The intended canonical state identifiers are:

```
PERSISTENT_LL
PERSISTENT_LM
PERSISTENT_LH
PERSISTENT_ML
PERSISTENT_MM
PERSISTENT_MH
PERSISTENT_HL
PERSISTENT_HM
PERSISTENT_HH
```

The nine states are a candidate representation, not nine independently
 validated behavioral mechanisms.

The 3×3 structure should therefore remain simple and interpretable.

------

# 6. Why Tertiles?

The 3×3 representation was selected because P0C found that:

- historical commitment descriptors form a low-dimensional structure;
- Level and Trajectory each contribute independently;
- a compact two-dimensional representation is sufficient relative to the
   larger descriptor set;
- the resulting termination surface has a coherent broad ordering;
- the ordering is reasonably stable across Persistent pairs and market
   regimes;
- further addition of the remaining historical descriptors did not provide
   compelling incremental information.

The tertile discretization is therefore an operational representation of the
 two-dimensional structure.

It should not be interpreted as evidence that three is a naturally occurring
 number of behavioral states.

------

# 7. Calibration: Open Design Question

The retrospective P0C analysis produced global tertile boundaries:

```
Commitment Level:
50
59
65.23434343
83.55555556

Commitment Trajectory:
-20.91666667
0.2
4.0
17.33333333
```

These values must **not automatically become the production BSVE calibration**.

They were obtained retrospectively from the full P0C population.

Using them directly for a subsequent OOS MSML experiment could allow the
 representation itself to benefit from information from the evaluation period.

The production candidate therefore requires a frozen calibration protocol.

### Calibration requirements

The final protocol must specify:

1. calibration population;
2. calibration time window;
3. calibration features;
4. quantile/binning method;
5. handling of ties;
6. handling of missing values;
7. minimum population requirements;
8. artifact versioning;
9. calibration provenance;
10. how calibration is frozen before MSML evaluation.

### Candidate principle

The semantic definition of the surface should be fixed now.

The numerical boundaries should be calibrated using only information permitted
 by the chosen evaluation protocol.
### Coverage-aware calibration eligibility

The following rules are now part of the candidate calibration protocol:

1. A long empirical observation gap is a hard observational break.
2. A canonical episode that crosses a long gap is retained for provenance and
   audit, but is not treated as one continuous calibration episode.
3. The first observation after a long gap begins a new observed segment, even
   when `crowd_side` is unchanged across the gap.
4. `Level` and `Trajectory` history must be computed only from observations
   within the current observed segment.
5. No feature may use inferred, forward-filled, interpolated, or otherwise
   reconstructed sentiment during a gap.
6. A post-gap observed segment may become calibration-eligible once it has
   accumulated sufficient within-segment history.
7. Gap adjacency and gap crossing should remain explicit provenance fields;
   they should not be silently encoded as behavioral states.

P0C-BSVE-1 used provisional minimum-history thresholds of three observations
for Level and four observations for Trajectory to quantify the available
population. Under those provisional thresholds, 12,078 of 16,204 observations
had sufficient joint history. These thresholds are **not yet frozen ontology
rules** and must be resolved before implementation.

The intended distinction is therefore:

```text
canonical episode
    = historical P0C episode representation

observed segment
    = continuous observed history eligible for calibration
```

This separation preserves the P0C research reconstruction while preventing
unobserved intervals from becoming implicit behavioral continuity.


------

# 8. Calibration Experiments

Before implementation, compare possible calibration strategies.

Candidate approaches include:

### A. Fixed development-period calibration

Estimate Level and Trajectory tertiles from a predefined historical development
 period and freeze them for subsequent MSML evaluation.

Advantages:

- simple;
- transparent;
- easy to audit;
- produces a single immutable v0.1 calibration artifact.

### B. Walk-forward calibration

Estimate calibration boundaries from the training portion of each MSML fold
 and freeze them for that fold's test period.

Advantages:

- maximally aligned with causal walk-forward evaluation;
- naturally handles distribution drift.

Disadvantages:

- produces fold-specific numerical boundaries;
- requires careful artifact/provenance handling;
- makes the surface slightly less visually uniform across folds.

### C. Hybrid approach

Freeze ontology semantics globally while estimating numerical calibration
 within each training fold.

This may provide the best separation between:

```
scientific ontology
```

and

```
numerical calibration
```

but requires explicit BSVE/MSML support.

### Decision

This remains an open design question and must be resolved before the
 implementation PR.

------

# 9. Episode Semantics

Persistent episodes are currently defined by consecutive observations with the
 same `crowd_side`:

```
crowd_side = sign(net_sentiment)
```

The candidate surface should preserve this episode structure.

A Level or Trajectory state change must **not** create a new behavioral episode.

Thus:

```
episode_id
    remains constant

while

state_id
    may change
```

during a crowd-side episode.

This distinction is important because the behavioral surface describes the
 configuration of an episode, while the episode itself represents the
 underlying commitment lifecycle.

------

# 10. `maturity_bars`

The existing BSVE artifact contract contains a `maturity_bars` field.

Persistent should not inherit the Reactive-JPY semantic meaning blindly.

For the candidate surface, the proposed interpretation is:

```
maturity_bars = current crowd-side episode age
```

This should be documented explicitly as **episode age**, rather than as a
 claim that episode age is the primary Persistent behavioral dimension.

The P0C analysis found that raw duration is important but substantially
 confounded with the Level dimension and should not replace the Level ×
 Trajectory representation.

------

# 11. `crowd_side`

`crowd_side` remains part of the Behavioral Surface artifact because it is
 already part of the behavioral episode construction and the existing BSVE
 contract.

However:

> `crowd_side` is not itself a third Persistent ontology dimension in v0.1.

The exploratory P0C directional-return work found localized crowd-side
 effects, but subsequent walk-forward testing did not establish temporal
 stability.

Therefore crowd-side × surface interactions must remain outside the candidate
 ontology at this stage.

They may be investigated later through MSML ablation experiments.

------

# 12. `transition_event`

The existing BSVE contract contains a transition-event field.

Persistent requires explicit semantics rather than copying the Reactive-JPY
 labels.

Candidate event vocabulary:

```
entry
continuation
state_transition
exit_reversal
exit_unknown
```

Proposed interpretation:

- `entry` — first observation of a crowd-side episode;
- `continuation` — episode continues and the surface state is unchanged;
- `state_transition` — episode continues but Level/Trajectory state changes;
- `exit_reversal` — crowd-side episode terminates through a crowd-side change;
- `exit_unknown` — episode termination cannot be assigned a more specific
   causal label.

This proposal requires confirmation against the BSVE schema and existing
 implementation before being frozen.

------

# 13. Required Calibration Artifact

The Persistent calibration artifact should contain enough information to
 reproduce state assignment independently of the original research code.

At minimum:

```
surface_id
surface_version
dataset_version
family
pairs
calibration_window
calibration_method
level_feature
trajectory_feature
level_boundaries
trajectory_boundaries
missing-value policy
tie/binning policy
creation timestamp
source/provenance metadata
artifact hash
```

The artifact should explicitly identify:

```
Commitment Level
    = prior_mean_depth

Commitment Trajectory
    = early_late_commitment_delta
```

The calibration artifact should be immutable once used for an MSML evaluation.

------

# 14. Behavioral Surface Artifact

The generated surface should conform to the existing BSVE artifact contract.

The public state assignment should contain the required canonical fields,
 including:

```
timestamp
pair
surface_id
surface_version
state_id
episode_id
maturity_bars
crowd_side
transition_event
```

Internal calibration variables should not be unnecessarily exposed as part of
 the public state vocabulary.

The surface must be deterministic:

```
same dataset
+
same calibration artifact
=
same surface
```

------

# 15. Causality Requirements

No Persistent surface assignment may depend on:

- future observations;
- final episode duration;
- future crowd-side changes;
- future returns;
- future market regime;
- future news;
- future calibration information.

In particular, the surface must not use the termination outcome to assign a
 historical state.

Running features such as:

```
prior_mean_depth
early_late_commitment_delta
episode_age
```

must only use information available at the assigned timestamp.

------

# 16. MSML Evaluation Objective

The first MSML experiment should test **representation value**, not direct
 return prediction.

The central question is:

> Does the Persistent Commitment Lifecycle surface simplify the predictive
>  learning problem relative to less structured representations?

The candidate surface should therefore be evaluated against controlled
 representation baselines.

Proposed ladder:

```
A. Market/environment baseline
B. Raw sentiment baseline
C. Raw commitment-history representation
D. Persistent Commitment Lifecycle surface
E. Expanded historical representation
```

The exact feature sets and model architecture will be specified in the MSML
 experiment plan after the BSVE surface is frozen.

The primary comparison should be:

```
raw / high-dimensional representation
            vs
compact Persistent surface
```

while keeping model architecture, target, training protocol and walk-forward
 schedule fixed.

------

# 17. Important Evaluation Constraint

The Persistent surface was discovered through P0C analysis on the historical
 dataset.

Therefore the eventual MSML evaluation must prevent the representation
 selection process from contaminating the evaluation period.

The following distinction must be maintained:

```
P0C discovery
    ↓
candidate ontology semantics
    ↓
calibration
    ↓
frozen surface
    ↓
OOS MSML evaluation
```

The MSML evaluation must not retrospectively modify the ontology or calibration
 after observing test-period predictive results.

------

# 18. What the Candidate Surface Is Not

The v0.1 surface must not be presented as:

- a trading strategy;
- a return predictor;
- a validated causal mechanism;
- a replacement for trend/volatility states;
- a claim that sentiment directly predicts price;
- a final Persistent ontology;
- evidence that the nine states have independently distinct mechanisms.

The purpose is representation.

Predictive and trading value must be established downstream.

------

# 19. Validation Requirements for the Implementation PR

Before merging the candidate surface implementation, the PR should include
 tests demonstrating:

### Population integrity

- correct Persistent pair set;
- expected dataset version;
- no duplicate `(pair, timestamp)` assignments;
- complete expected population;
- episode integrity.

### Causality

- state assignment does not use future observations;
- final episode duration cannot affect historical assignments;
- future returns are never read by the state engine.

### Determinism

- repeated assignment produces identical output;
- calibration artifact produces identical state boundaries;
- same input + same calibration = identical artifact.

### Calibration

- calibration boundaries are reproducible;
- calibration metadata is complete;
- frozen calibration is respected.

### State coverage

- all valid observations receive exactly one state;
- no observation receives multiple states;
- all nine candidate states can be represented when data permit.

### Artifact contract

- generated artifact conforms to the BSVE schema;
- required metadata fields are present;
- artifact is consumable by the existing MSML pipeline.

------

# 20. Research Questions Remaining Before Implementation

The following questions must be resolved before the Copilot implementation PR.

### Q1 — Calibration protocol

The calibration protocol must include the coverage-aware observed-segment rules
defined in Section 7. In particular, calibration must not allow information
from an evaluation period to affect state boundaries.

Should v0.1 use:

- fixed development calibration,
- fold-specific walk-forward calibration,
- or a hybrid ontology/fold-calibration approach?

### Q2 — Coverage and minimum-history thresholds

P0C-BSVE-1 established the observed-segment rule, but the minimum amount of
within-segment history required before Level and Trajectory become valid
calibration inputs remains open.

Questions:

- What minimum history is required for `prior_mean_depth`?
- What minimum history is required for `early_late_commitment_delta`?
- Should very young observed segments receive an explicit insufficient-history
  status, or another BSVE-compatible representation?
- Should gap-adjacent segments receive special provenance only, or any additional
  eligibility restriction?

Current provisional study thresholds:

- Level: >= 3 observations;
- Trajectory: >= 4 observations.

These are measurement-study thresholds, not final ontology decisions.

### Q3 — State granularity

Should v0.1 remain:

```
3 × 3 = 9 states
```

or should an experiment establish whether a coarser representation is
 preferable?

Current default: retain 3×3 unless evidence argues otherwise.

### Q4 — Episode age

Should `maturity_bars` be explicitly documented as episode age for Persistent?

Current proposal: yes.

### Q5 — Transition events

Should `state_transition` be a first-class Persistent transition event?

Current proposal: yes, subject to schema/implementation verification.

### Q6 — Continuous vs discrete representation

Should MSML receive:

- only discrete states;
- continuous Level/Trajectory values;
- or both?

Initial candidate surface should remain discrete for BSVE compatibility.
 Continuous variables may be retained as experimental controls in MSML rather
 than incorporated into the public surface.

------

# 21. Planned Research Sequence

### Step 1 — Resolve design questions

Finalize:

- coverage-aware observed-segment semantics;
- minimum Level/Trajectory history;
- treatment of young and gap-adjacent segments;
- calibration protocol;
- state granularity;
- maturity semantics;
- transition semantics;
- continuous/discrete treatment.

### Step 2 — Update this roadmap

Record the decisions and their rationale.

### Step 3 — Commit roadmap

The roadmap becomes the version-controlled specification for the
 implementation PR.

### Step 4 — Write Copilot implementation prompt

The prompt should explicitly instruct Copilot to:

- read this roadmap first;
- inspect existing BSVE implementation;
- reuse existing contracts where appropriate;
- avoid copying Reactive-JPY assumptions;
- implement the Persistent candidate surface;
- add tests;
- add calibration artifacts;
- add documentation;
- preserve existing surfaces and behavior.

### Step 5 — Copilot PR

Copilot implements the candidate surface according to the frozen roadmap.

### Step 6 — Local validation

Run the complete BSVE test suite and Persistent-specific validation.

### Step 7 — MSML representation benchmark

Use the frozen surface in a controlled MSML experiment.

### Step 8 — MPML

Only if MSML produces credible OOS representation value, evaluate the
 resulting prediction artifact in MPML.

### Step 9 — Registry promotion

Do not promote the surface to a stronger scientific status until the evidence
 supports doing so.

------

# 22. Success Criteria

The initial implementation is successful if it produces a:

- deterministic;
- causal;
- reproducible;
- versioned;
- BSVE-compatible

Persistent behavioral surface.

Scientific success is a separate question.

The surface will only become scientifically established if subsequent MSML
 and/or MPML experiments provide reproducible evidence that the representation
 adds value beyond appropriate controls.

------

# 23. Current Working Hypothesis

The current working hypothesis is:

> Persistent episodes contain a low-dimensional commitment lifecycle that can
>  be represented by historical commitment Level and commitment Trajectory.

The 3×3 surface is the first operational representation of that hypothesis.

It is deliberately being tested rather than assumed to be correct.

------

# 24. Decision Log

| Decision                                                  | Status            | Rationale                                   |
| --------------------------------------------------------- | ----------------- | ------------------------------------------- |
| Persistent family = 5 pairs                               | Established       | Existing Persistent family definition       |
| Behavioral object = commitment lifecycle                  | Proposed          | P0C findings                                |
| Level = `prior_mean_depth`                                | Proposed          | P0C dimensional reduction                   |
| Trajectory = `early_late_commitment_delta`                | Proposed          | P0C dimensional reduction                   |
| 3×3 state space                                           | Proposed          | Compact representation of P0C surface       |
| Nine canonical states                                     | Proposed          | Direct representation of 3×3 grid           |
| `maturity_bars` = episode age                             | Open              | Requires BSVE contract confirmation         |
| `crowd_side` retained                                     | Proposed          | Existing BSVE / episode semantics           |
| `crowd_side` as ontology dimension                        | Rejected for v0.1 | OOS directional evidence not stable         |
| `state_transition` event                                  | Open              | Requires schema/implementation confirmation |
| Retrospective global thresholds as production calibration | Rejected          | Would compromise clean OOS evaluation       |
| Long coverage gap = hard observational break              | Established       | P0C-BSVE-0/1 coverage audit                 |
| Cross-gap canonical continuity for calibration            | Rejected          | 407/1,421 canonical episodes cross gaps     |
| Observed segment used for calibration history             | Proposed          | Prevents inferred continuity across gaps     |
| No gap reconstruction / forward fill                      | Established       | Coverage audit cannot support reconstruction |
| Minimum Level history = 3                                 | Provisional       | P0C-BSVE-1 measurement threshold             |
| Minimum Trajectory history = 4                            | Provisional       | P0C-BSVE-1 measurement threshold             |
| Continuous Level/Trajectory in public surface             | Open              | Prefer discrete BSVE surface initially      |
| Registry promotion                                        | Deferred          | Requires downstream evidence                |

------

## 25. Current Status

**P0C behavioral discovery:** complete enough for candidate-surface design.

**Candidate semantics:** substantially defined.

**Coverage-aware calibration semantics:** established enough for protocol design.

**Minimum Level/Trajectory history:** provisional; unresolved.

**Calibration protocol:** unresolved.

**Artifact semantics:** partially unresolved.

**Implementation:** not started.

**MSML evaluation:** not started.

**MPML evaluation:** deferred.

The next action is to resolve the remaining design questions and update this
 document before implementation.