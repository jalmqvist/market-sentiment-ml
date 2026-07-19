# Research State

Last updated: July 2026

This document is intentionally evidence-centric. Conclusions recorded here should summarize the
current body of evidence rather than the history of how that evidence was obtained. Experimental
details, statistical analyses and implementation notes belong in the corresponding research documents.

## Purpose

This document summarizes the current scientific state of the Market Sentiment ML (MSML) research program.

Unlike the project documentation, which describes the repository architecture and implementation, this
document focuses exclusively on research progress. It records the current evidence, established findings,
open questions, and immediate research priorities.

The intended audience is researchers and contributors who wish to understand the present state of the
project without reconstructing its history from commits or experimental logs.

For implementation details, see:

- `README.md` — repository architecture and developer orientation.
- `PROJECT_DESCRIPTION.md` — scientific motivation, research methodology and long-term vision.
- `docs/` — subsystem-specific documentation.

---

## Executive Summary

The project has now completed the transition toward behavioral representation research.

Current work is centered on three scientific questions:

- Which behavioral representations can be validated independently?
- Do validated behavioral representations improve predictive learning?
- Can behavioral organization ultimately be explained mechanistically?

The first complete behavioral ontology (Reactive-JPY) has been frozen following independent validation and
integrated into the MSML data pipeline. This document summarizes the current body of scientific knowledge
and is updated as new evidence becomes available.

Background on the scientific evolution of the project is provided in `PROJECT_DESCRIPTION.md`.

---

### Behavioral Surface Predictive Learning (July 2026)

The first predictive evaluation of the frozen Reactive-JPY Behavioral Surface has now been completed using independent walk-forward validation across both MLP and LSTM architectures. Epoch-sweep experiments produced several consistent findings.

Both architectures demonstrated reproducible improvements over multiple control baselines, including permutation, random partitioning, trend/volatility partitioning and base-rate controls. While the magnitude of improvement varied between architectures, the qualitative ordering of behavioral states remained remarkably consistent.

The experiments further suggest that individual behavioral states exhibit distinct learning dynamics. Some states converge rapidly and remain largely insensitive to training duration, whereas others continue to accumulate predictive signal over extended training or exhibit substantially more complex optimization behaviour. These differences were reproduced across independent neural architectures, indicating that they are more likely to reflect intrinsic properties of the Behavioral Surface than optimizer-specific artefacts.

Taken together, these results provide the first evidence that independently validated Behavioral Surfaces not only represent meaningful behavioral organization, but also simplify downstream predictive learning. Although the current findings are limited to the Reactive-JPY ontology and require replication on additional Behavioral Surfaces, they represent an important milestone in the project's transition from behavioral representation discovery to predictive validation.

---

## Current Phase Supporting Documents

The following documents contain the primary evidence and implementation guidance supporting the current
phase of the research programme.

### Behavioral Representation

- `bsve/docs/reactive_jpy_findings.md` — Reactive-JPY ontology, validation methodology and results
- `bsve/docs/reactive_chf_findings.md` — Reactive-CHF exploratory findings and ongoing investigations
- `bsve/docs/persistent_findings.md` — Persistent behavioral family empirical evidence
- `bsve/docs/synthesis_document.md` — complete BSVE methodology from ontology construction through validation

### Integration

- `docs/BSVE_MSML_integration_architecture.md` — behavioral surface integration into predictive pipeline
- `docs/integration/dl_prediction_artifacts.md` — prediction artifact schema and downstream contracts

### Research Planning

- `docs/research/RESEARCH_STRATEGY.md` — research priorities and strategic direction

---

## Current Project Status

### Area Status

| Area                            | Status                     |
| ------------------------------- | -------------------------- |
| Master research dataset         | ✓ Stable                   |
| Dataset validation              | ✓ Complete                 |
| Leakage auditing                | ✓ Complete                 |
| Pair-family discovery           | ✓ Established              |
| Persistent-family studies       | ✓ Complete (current phase) |
| Reactive-JPY ontology           | ✓ Frozen                   |
| Reactive-CHF ontology           | ◐ Exploratory              |
| BSVE framework                  | ✓ Operational              |
| Behavioral surface export       | ✓ Operational              |
| Dataset behavioral augmentation | ✓ Operational              |
| MSML behavioral integration     | ◐ In progress              |
| MPML behavioral integration     | Planned                    |
| Deep-learning evaluation        | Ongoing                    |
| Agent-based modelling           | Ongoing                    |

### Research Infrastructure

The research infrastructure required for the current programme is considered operational. This includes:

- validated dataset construction,
- artifact versioning,
- leakage auditing,
- deterministic behavioral representation generation,
- reproducible validation workflows.

The findings summarized below therefore concern scientific conclusions rather than implementation maturity.

---

## Methodological Infrastructure

Development of the Behavioral Surface Validation Engine (BSVE) represents an important methodological
contribution of the project.

The framework now provides:

- deterministic behavioral ontology calibration,
- explicit behavioral state assignment,
- behavioral surface generation,
- reproducible artifact contracts,
- independent behavioral validation,
- deterministic dataset augmentation.

Importantly, BSVE separates behavioral validation from predictive modelling. Behavioral representations are
therefore evaluated as independent scientific hypotheses before being incorporated into machine-learning
experiments. This separation has become one of the defining methodological characteristics of the repository.

---

## Established Scientific Findings

The findings summarized below represent the current body of scientific knowledge rather than the complete
experimental history of the project.

As evidence accumulates, hypotheses are promoted from active research questions to established findings.
Conversely, hypotheses that fail independent validation are removed from the active research programme
and retained only within the archived research documentation.

This document therefore records the current state of knowledge rather than every experiment performed
during the evolution of the project.

---

### Retail Sentiment

Early research investigated retail positioning as a direct predictive signal.

Current evidence indicates that this interpretation is inadequate.

Established findings include:

- Raw retail positioning exhibits little standalone predictive power.
- Simple linear combinations of price and sentiment do not improve predictive performance.
- Sentiment contributes little additional information when treated as a conventional predictive feature.
- Predictive value appears highly conditional rather than universal.

These negative findings were scientifically important because they motivated the transition from sentiment
prediction toward behavioral representation learning. The project therefore no longer regards retail sentiment
as the primary scientific object. Instead, it is treated as a behavioral observation from which latent market
organization may be inferred.

### Behavioral Organization

One of the principal discoveries of the project is that foreign exchange markets exhibit reproducible behavioral
organization.

Multiple independent analyses indicate that currency pairs cannot be adequately described using a single
behavioral model.

Instead:

- distinct behavioral families emerge repeatedly across independent analyses,
- behavioral differences remain stable across multiple validation approaches,
- behavioral organization appears substantially more informative than raw sentiment values,
- behavioral representations provide a more useful abstraction than individual sentiment observations.

These findings motivated the development of explicit behavioral ontologies and the Behavioral Surface
Validation Engine (BSVE). Behavioral organization is therefore regarded as the central scientific object of
the current research programme.

### Persistent Behavioral Family

Research into Persistent currency pairs has produced one of the strongest bodies of evidence within the
project.

Current evidence supports the following conclusions:

- Persistent pairs exhibit long-lived behavioral organization rather than isolated predictive effects.
- Behavioral persistence appears to evolve through identifiable stages rather than remaining stationary.
- Transition structure carries substantially more information than raw positioning alone.
- Behavioral evolution is closely coupled to changes in underlying market conditions rather than instantaneous sentiment values.
- Persistent-family behavior is sufficiently reproducible to motivate ontology construction.

The Persistent family therefore represents a validated behavioral phenomenon rather than an isolated
empirical observation. Research emphasis has consequently shifted from establishing its existence toward
understanding the mechanisms responsible for its behavior.

Supporting documentation: `bsve/docs/persistent_findings.md`

### Reactive JPY

Reactive-JPY represents the first complete behavioral ontology developed within the project.

Current evidence supports:

- reproducible consensus lifecycle structure,
- deterministic identification of consensus episodes,
- empirically calibrated maturity boundaries,
- reproducible hazard-rate evolution throughout consensus lifecycles,
- deterministic Behavioral Surface generation,
- successful independent behavioral validation under the BSVE framework.

Behavioral validation produced results that were close to satisfying predefined confirmation criteria but
ultimately remained formally inconclusive under the preregistered evaluation protocol.

This distinction is important. The ontology is considered scientifically useful and operationally mature, while
acknowledging that the current evidence does not yet justify treating every behavioral interpretation as
conclusively established.

Reactive-JPY nevertheless provides the first complete demonstration of the full BSVE methodology from
behavioral hypothesis through deterministic state assignment and behavioral surface generation.

Supporting documentation: `bsve/docs/reactive_jpy_findings.md`

### Reactive CHF

Reactive-CHF remains under active investigation.

Current exploratory studies indicate that:

- volatility strongly organizes persistence behaviour, while persistence appears more closely associated with crowd outcomes than volatility itself,
- behavioral transitions appear continuous rather than discretely separated,
- volatility-derived behavioral representations remain promising,
- deterministic ontology construction appears feasible.

However, the ontology has not yet reached the validation maturity achieved for Reactive-JPY.
Consequently, Reactive-CHF should currently be regarded as an active research programme rather than an
established behavioral representation.

Supporting documentation: `bsve/docs/reactive_chf_findings.md`

---

## Current Confidence Assessment

Validated as of: July 2026

Scientific progress depends not only on identifying promising hypotheses, but also on understanding the
strength of the available evidence.

The following table summarizes the current confidence associated with the principal components of the
research programme. Confidence assessments should be interpreted as summaries of the current body of
evidence rather than permanent classifications. They are expected to evolve as additional behavioral
representations are developed and independently validated.

| Research Area                          | Current Confidence | Comments                                                     |
| -------------------------------------- | ------------------ | ------------------------------------------------------------ |
| Master Research Dataset                | Very High          | Construction pipeline validated, leakage audits complete, stable artifact contracts. |
| Dataset methodology                    | Very High          | Reproducible dataset generation and causal alignment considered mature. |
| Retail sentiment as a direct predictor | High               | Multiple independent studies consistently show limited standalone predictive value. |
| Behavioral organization                | High               | Strong evidence that behavioral structure exists beyond raw sentiment observations. |
| Currency pair families                 | High               | Reproduced across several independent analyses and experimental methodologies. |
| Persistent behavioral family           | High               | Mature empirical findings with consistent behavioral characteristics. |
| Reactive-JPY ontology                  | Moderate–High      | Deterministic ontology complete; behavioral validation narrowly inconclusive under preregistered criteria. |
| Reactive-CHF ontology                  | Moderate           | Promising exploratory evidence; ontology construction and validation ongoing. |
| BSVE methodology                       | High               | Framework operational and successfully applied to Reactive-JPY. |
| Behavioral dataset augmentation        | High               | Deterministic augmentation pipeline implemented and validated. |
| Behavioral ML integration              | Moderate           | Infrastructure largely complete; predictive evaluation now underway. |
| Dynamic behavioral representations     | Low                | Conceptual direction supported by current findings but not yet investigated experimentally. |
| Mechanistic interpretation (ABM)       | Low                | Active exploratory research with no definitive conclusions at present. |

---

## Open Scientific Questions

The project has successfully answered several important questions regarding behavioral organization in
foreign exchange markets. At the same time, each completed research programme has generated new
scientific questions.

The following questions currently define the next phase of the project.

### Behavioral Representation

Several questions concern the nature of behavioral representations themselves. Current questions include:

- How many distinct behavioral representations exist within foreign exchange markets?
- Which market observations best reveal latent behavioral organization?
- How should deterministic behavioral ontologies be constructed?
- Which behavioral phenomena are universal, and which are market-specific?
- Can behavioral representations be generalized across different financial instruments?

### Behavioral Dynamics

Current findings suggest that behavioral organization is dynamic rather than stationary. Important open
questions include:

- Are observed currency pair families permanent or transient?
- Can markets transition between different behavioral representations?
- What mechanisms govern these transitions?
- How stable are behavioral representations across changing macroeconomic environments?

Understanding behavioral evolution is expected to become one of the major themes of future research.

### Predictive Learning

Behavioral representations are ultimately valuable only if they simplify predictive learning. Current questions
include:

- Which predictive architectures benefit most from behavioral specialization?
- How should behavioral representations be encoded for machine learning?
- Do behavioral representations improve generalization more than raw predictive accuracy?
- Which behavioral representations provide the greatest downstream value?

These questions now represent the primary focus of ongoing MSML development. Related implementation
guidance is available in `docs/BSVE_MSML_integration_architecture.md` and `docs/research/RESEARCH_STRATEGY.md`.

### Adaptive Decision Making

Predictive improvements must ultimately demonstrate practical utility. Current research therefore investigates:

- Which behavioral representations produce the largest improvements in adaptive strategy selection?
- Should predictive models specialize within behavioral representations?
- How should multiple behavioral representations be combined?
- Can representation selection itself become adaptive?

These questions motivate ongoing integration with the MPML framework. See `docs/integration/dl_prediction_artifacts.md`
for the current prediction artifact contract and downstream integration schema.

### Mechanistic Understanding

Empirical observations naturally motivate questions regarding the underlying mechanisms responsible for
observed behavioral organization. Current questions include:

- Why do Persistent and Reactive behavioral families emerge?
- Which interacting-agent mechanisms reproduce observed behavioral dynamics?
- Can synthetic markets generate similar behavioral ontologies?
- Which empirical findings remain unexplained by existing behavioral models?

Addressing these questions represents the long-term objective of the Agent-Based Modeling programme.

---

## Current Research Priorities

The immediate objective of the project is no longer the discovery of additional behavioral phenomena in
isolation. Instead, the focus has shifted toward determining whether validated behavioral representations
improve predictive modelling and downstream adaptive decision-making.

Several complementary research programmes are therefore progressing in parallel.

### Priority 1 — Behavioral Predictive Validation

The highest current priority is evaluating whether validated behavioral representations simplify predictive
learning.

This work focuses on integrating Behavioral Surfaces into the MSML training pipeline and comparing
behavioral representations against traditional volatility/trend partitioning under identical walk-forward
evaluation protocols.

The principal scientific question is: **Do validated behavioral representations improve predictive learning?**

This represents the first complete evaluation of the Behavioral Surface methodology using predictive machine
learning.

### Priority 2 — Behavioral Surface Expansion

Reactive-JPY represents the first complete Behavioral Surface produced by the BSVE framework. Current
work focuses on extending the methodology to additional behavioral phenomena while preserving
deterministic ontology construction and independent validation.

Important objectives include:

- completion of the Reactive-CHF ontology,
- comparison between alternative behavioral representations,
- generalization of BSVE to support multiple concurrent behavioral surfaces,
- continued refinement of behavioral validation methodology.

The long-term objective is a library of independently validated behavioral representations rather than a
single ontology.

### Priority 3 — MPML Integration

Behavioral representations have now entered the predictive modelling pipeline. The next stage is evaluating
whether those representations improve adaptive strategy selection within the downstream MPML framework.

Current work includes:

- behavioral-state model specialization,
- behavioral routing,
- walk-forward adaptive evaluation,
- comparison against traditional market-state partitioning.

The principal scientific question is no longer whether Behavioral Surfaces can be constructed, but whether
they improve adaptive decision-making under realistic deployment conditions.

**Initial MPML integration milestone (July 2026).** The first end-to-end integration experiments using the frozen Reactive-JPY Behavioral Surface have now been completed. Behavioral Prediction Artifacts generated by MSML were successfully consumed by the downstream MPML framework without architectural modifications to the adaptive selector. Initial walk-forward experiments demonstrate that Behavioral Surface specialization influences downstream strategy selection and produces measurable changes in out-of-sample trading performance. Importantly, currency pairs outside the Reactive-JPY Behavioral Surface remained unchanged across behavioral-state experiments, providing an internal negative control confirming correct behavioral routing. Although predictive performance remains preliminary and further model optimization is expected, these experiments provide the first evidence that validated Behavioral Surfaces can contribute useful information to downstream adaptive decision-making. 

### Priority 4 — Mechanistic Understanding

Empirical behavioral organization has now been demonstrated sufficiently to motivate renewed emphasis on
mechanistic explanation.

Agent-Based Modelling (ABM) therefore enters a new phase. Rather than generating synthetic market
behaviour in isolation, future work will increasingly investigate whether observed behavioral representations
can emerge naturally from interacting market participants.

Long-term success will therefore require connecting three independent forms of evidence:

- empirical behavioral observations,
- predictive validation,
- mechanistic simulation.

Achieving agreement between these three perspectives represents one of the most ambitious objectives of
the project.

---

## Immediate Development Roadmap

The current implementation roadmap follows directly from the scientific priorities identified above.

### 1. Behavioral integration within MSML

Complete integration of Behavioral Surface dataset variants into the predictive training pipeline.

Relevant documentation:
- `docs/BSVE_MSML_integration_architecture.md`

### 2. Predictive evaluation of Reactive-JPY

Perform the first walk-forward predictive evaluation using the frozen Reactive-JPY Behavioral Surface.

Relevant documentation:
- `bsve/docs/reactive_jpy_findings.md`
- `bsve/docs/synthesis_document.md`
- `docs/BSVE_MSML_integration_architecture.md`

### 3. Behavioral routing in MPML

Evaluate Behavioral Surface specialization within the downstream adaptive framework.

Relevant documentation:
- `docs/BSVE_MSML_integration_architecture.md`
- `docs/integration/dl_prediction_artifacts.md`

### 4. Reactive-CHF

Continue ontology construction and independent validation for volatility-derived behavioral representations.

Relevant documentation:
- `bsve/docs/reactive_chf_findings.md`
- `bsve/docs/synthesis_document.md`

### 5. Multi-representation comparison

Compare multiple validated behavioral representations within a common predictive framework.

Relevant documentation:
- `docs/research/RESEARCH_STRATEGY.md`

### 6. Mechanistic investigation

Continue Agent-Based Modelling research aimed at explaining observed behavioral organization.

Relevant documentation:
- `docs/abm/`
- `docs/research/RESEARCH_STRATEGY.md`

---

## Closing Remarks

The project has completed its transition from sentiment prediction toward behavioral representation research.
For background on this evolution, see `PROJECT_DESCRIPTION.md`. This document records where that
programme currently stands.