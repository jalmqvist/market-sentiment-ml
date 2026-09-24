# Persistent BSVE Candidate Surface Roadmap

**Status:** Design / research specification  
**Surface:** Persistent Commitment Lifecycle  
**Candidate version:** v0.1.0  
**Dataset:** 1.6.1  
**Last updated:** 2026-09-24

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

### Implementation-contract constraints

The implementation must preserve the existing BSVE public artifact and
calibration contracts. The Persistent surface is a new ontology implementation,
not a redefinition of Reactive-JPY semantics.

In particular:

- the public Behavioral Surface remains one row per `(timestamp, pair)` and
  uses the existing canonical fields;
- Level and Trajectory are internal state-assignment variables and should not
  become additional public surface columns;
- Persistent calibration must use the existing versioned and hashed calibration
  artifact mechanism rather than introduce a parallel artifact format;
- the Persistent calibration artifact must carry the four numerical boundaries
  required for state assignment;
- MSML consumes the resulting Behavioral Surface / Behavioral Dataset Variant
  and must not reproduce Persistent episode, gap, feature, or calibration
  semantics;
- existing Reactive-JPY surfaces and behavior must remain unchanged.

The implementation should make the smallest backwards-compatible extension to
the generic BSVE machinery required to support Persistent semantics. It must not
force Persistent into Reactive-JPY's consensus semantics merely for code reuse.

The Persistent state engine must explicitly support the distinction between:

```text
canonical crowd-side episode
    = historical P0C episode representation

observed segment
    = continuous observed history used for causal feature calculation
```

A long observational gap therefore resets Persistent historical feature state,
even when the canonical crowd-side episode remains continuous for provenance.

The public `maturity_bars` field is retained. For Persistent it is defined as
the current crowd-side episode age, not Reactive-JPY consensus maturity.

The public `transition_event` field is retained. Persistent candidate semantics
are:

```text
entry
continuation
state_transition
exit_reversal
exit_unknown
```

where `state_transition` means that the Persistent Level × Trajectory state
changes while the underlying crowd-side episode continues. This value must be
verified against the existing schema and implementation before it is frozen;
existing Reactive-JPY transition semantics must not be changed.

Persistent insufficient-history observations are not behavioral ontology states.
They require an explicit non-state handling compatible with the existing BSVE
artifact and MSML dataset contracts.

The implementation must preserve deterministic assignment:

```text
same dataset
+
same Persistent calibration artifact
=
same Persistent surface
```

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

P0C-BSVE-2 subsequently tested Level minima of 2–5 and Trajectory minima of
4–6 using the canonical feature definitions. The study supports freezing the
minimum history at **3 prior observations for Level and 4 prior observations
for Trajectory**. The Trajectory requirement is binding; the Level minimum is
retained as an explicit protocol parameter but is non-binding in the jointly
eligible population.

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

**Resolved:** use the hybrid approach. Ontology semantics are fixed globally,
while numerical Level/Trajectory boundaries are calibrated from the training
portion of each MSML fold and then frozen for that fold's test period.

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

The core research-design questions are now resolved. The remaining questions
are implementation/schema confirmations derived from the existing BSVE and
MSML contracts.

### Resolved research decisions

- hybrid calibration: fixed ontology semantics with fold-specific numerical
  calibration;
- pooled Persistent-family training population;
- hard observational gaps;
- no cross-gap history;
- post-gap eligibility after sufficient within-segment history;
- gap adjacency as provenance only;
- Level minimum = 3 prior observations;
- Trajectory minimum = 4 prior observations;
- training-only Q33/Q67;
- no test-period recalibration;
- deterministic half-open bins;
- insufficient-history observations are not behavioral states.

### Q1 — State granularity

Retain:

```text
3 × 3 = 9 states
```

The nine-state representation is frozen for v0.1.

### Q2 — Episode age

`maturity_bars` is defined for Persistent as current crowd-side episode age.
Final confirmation is an implementation-contract test, not a research-design
question.

### Q3 — Transition events

`state_transition` is the candidate Persistent event when the Level × Trajectory
state changes while the crowd-side episode continues. Final confirmation
against the existing schema/implementation is required before merge.

### Q4 — Continuous vs discrete representation

The public BSVE surface remains discrete for v0.1. Continuous Level/Trajectory
values may be retained as experimental MSML controls but are not part of the
public behavioral state vocabulary.
------

# 21. Planned Research Sequence

### Step 1 — Research design

Completed:

- coverage-aware observed-segment semantics;
- minimum Level/Trajectory history;
- treatment of young and gap-adjacent segments;
- hybrid calibration protocol;
- pooled family calibration;
- training-only tertiles;
- frozen test-fold calibration.

### Step 2 — Contract reconciliation

Completed against the BSVE and MSML documentation:

- public Behavioral Surface schema;
- calibration artifact contract;
- state-engine/plugin architecture;
- maturity semantics;
- transition-event contract;
- BSVE → Behavioral Dataset Variant → MSML boundary;
- backwards-compatibility requirements.

### Step 3 — Commit this roadmap

This document is the version-controlled research and implementation
specification for the Persistent candidate surface.

### Step 4 — Write Copilot implementation prompt

The prompt must explicitly instruct Copilot to:

- read this roadmap first;
- inspect the existing BSVE implementation and tests;
- inspect the MSML integration/documentation;
- reuse existing contracts and artifact formats where appropriate;
- implement Persistent as a distinct ontology rather than a Reactive-JPY fork;
- make only the smallest backwards-compatible generic-engine extension
  required by Persistent semantics;
- implement coverage-aware observed-segment history;
- implement causal Level/Trajectory features;
- implement fold-specific Persistent calibration;
- add the Persistent calibration and surface artifacts;
- add comprehensive Persistent-specific tests;
- preserve all existing surfaces and behavior;
- update documentation without silently changing scientific semantics.

### Step 5 — Copilot PR

Copilot implements the candidate surface according to the frozen roadmap.

### Step 6 — Local validation

Run the complete BSVE test suite and Persistent-specific validation, including
causality, gap handling, calibration freezing, determinism, state coverage,
artifact-schema validation, and MSML consumption.

### Step 7 — MSML representation benchmark

Use the frozen Persistent surface in a controlled MSML experiment.

### Step 8 — MPML

Only if MSML produces credible OOS representation value, evaluate the resulting
prediction artifact in MPML.

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
| Behavioral object = commitment lifecycle                  | Established       | P0C findings                                |
| Level = `prior_mean_depth`                                | Established       | P0C dimensional reduction                   |
| Trajectory = `early_late_commitment_delta`                | Established       | P0C dimensional reduction                   |
| 3×3 state space                                           | Established       | Compact representation of P0C surface       |
| Nine canonical states                                     | Established       | Direct representation of 3×3 grid           |
| `maturity_bars` = episode age                             | Established       | Persistent meaning = crowd-side episode age; contract test required |
| `crowd_side` retained                                     | Established       | Existing BSVE / episode semantics           |
| `crowd_side` as ontology dimension                        | Rejected for v0.1 | OOS directional evidence not stable         |
| `state_transition` event                                  | Proposed          | Candidate semantics; confirm against BSVE schema/implementation |
| Retrospective global thresholds as production calibration | Rejected          | Would compromise clean OOS evaluation       |
| Long coverage gap = hard observational break              | Established       | P0C-BSVE-0/1 coverage audit                 |
| Cross-gap canonical continuity for calibration            | Rejected          | 407/1,421 canonical episodes cross gaps     |
| Observed segment used for calibration history             | Established       | Prevents inferred continuity across gaps     |
| No gap reconstruction / forward fill                      | Established       | Coverage audit cannot support reconstruction |
| Minimum Level history = 3                                 | Established       | P0C-BSVE-2; non-binding under joint eligibility |
| Minimum Trajectory history = 4                            | Established       | P0C-BSVE-2; binding minimum-history constraint |
| Hybrid fold-specific numerical calibration                | Established       | Fixed semantics; training-only fold boundaries |
| Pooled family calibration                                  | Established       | Five Persistent pairs share calibration boundaries |
| Training-only Q33/Q67                                     | Established       | Boundaries frozen before each test fold      |
| Test-period recalibration                                 | Rejected          | Would violate clean OOS evaluation          |
| Young observations = insufficient-history status          | Established       | Not a behavioral ontology state             |
| Continuous Level/Trajectory in public surface             | Rejected for v0.1 | Retain discrete public surface; continuous values remain MSML controls |
| Registry promotion                                        | Deferred          | Requires downstream evidence                |

------

## 25. Current Status

**P0C behavioral discovery:** complete enough for candidate-surface design.

**Candidate semantics:** substantially defined.

**Coverage-aware calibration semantics:** established enough for protocol design.

**Minimum Level/Trajectory history:** established at 3 / 4 by P0C-BSVE-2.

**Calibration protocol:** substantially resolved: hybrid fold-specific
numerical calibration, pooled training population, training-only Q33/Q67,
frozen before test evaluation.

**Artifact semantics:** reconciled against the existing BSVE and MSML contracts;
remaining items are implementation-level confirmations/tests.

**Implementation:** not started.

**MSML evaluation:** not started.

**MPML evaluation:** deferred.

The next action is to inspect the existing BSVE artifact contracts and MSML
documentation, then resolve only the remaining implementation/schema questions
before the Copilot implementation PR.

---

## Update 2026-09-24: Final calibration and contract decisions

| Question | Decision | Status |
| --- | --- | --- |
| Calibration strategy | **Hybrid: fixed ontology semantics + fold-specific numerical calibration** | **Established** |
| Calibration population | **Pooled Persistent family; training observations only; eligible observations rather than episodes as statistical units** | **Established** |
| Gap treatment | **Hard observational break** | **Established** |
| Cross-gap continuity | **Never used for calibration history** | **Established** |
| Post-gap segment | **Eligible after sufficient within-segment history** | **Established** |
| Gap adjacency | **Provenance only; no automatic exclusion** | **Established** |
| Pair-specific boundaries | **No; pooled family calibration across the five Persistent pairs** | **Established** |
| Tertiles | **Training-only Q33/Q67** | **Established** |
| Test recalibration | **Never** | **Established** |
| Quantile tie handling | **Deterministic half-open bins; no arbitrary tie splitting** | **Established** |
| Young observations | **Insufficient-history status, not a behavioral state** | **Established** |
| Calibration failure | **Fail loudly on insufficient/degenerate calibration populations or boundaries** | **Established** |
| Level minimum | **3 prior observations** | **Established by P0C-BSVE-2; non-binding in the joint population** |
| Trajectory minimum | **4 prior observations** | **Established by P0C-BSVE-2; binding minimum** |

### P0C-BSVE-2 minimum-history sensitivity result

The minimum-history sensitivity study tested Level minima of 2, 3, 4, and 5
observations against Trajectory minima of 4, 5, and 6 observations, using the
canonical feature definitions and coverage-aware observed segments.

Decision-relevant findings:

- Level minimum 2→5 produced **no change** in the jointly eligible population,
  calibration boundaries, or nine-state representation once the Trajectory
  requirement was imposed.
- Trajectory minimum 4 and 5 produced **identical** results.
- Trajectory minimum 6 removed 895 observations from the L3/T4 reference
  population and shifted the numerical tertile boundaries modestly, but all
  five Persistent pairs and all nine candidate states remained represented.
- Nested eligibility checks passed for all tested configurations.
- No configuration failed the distinct-value or state-coverage checks.

The reference L3/T4 population contains 8,630 jointly eligible observations
(53.26% of the 16,204 Persistent observations), 1,060 observed segments, and
436 canonical episodes. Its pooled descriptive boundaries are:

```text
Level Q33 = 58.307692
Level Q67 = 65.200000

Trajectory Q33 = -1.916667
Trajectory Q67 = 0.750000
```

These full-sample values are descriptive study outputs only. They are **not**
production calibration values; production boundaries remain fold-specific and
training-only under the hybrid calibration protocol.

The minimum-history decision is therefore frozen at:

```text
minimum Level history      = 3 prior observations
minimum Trajectory history = 4 prior observations
```

Trajectory is the binding constraint. Level 3 remains an explicit protocol
parameter for clarity and reproducibility, even though tightening it above 3
does not reduce the jointly eligible population in the tested configurations.
