# Project Description

## Introduction

Financial markets are often studied as prediction problems. Given a sufficiently rich collection of historical
observations, the objective is typically to estimate future price movements as accurately as possible using
increasingly sophisticated statistical or machine-learning models.

This project approaches the problem from a different perspective.

Rather than asking how to build better predictive models, it asks whether the learning problem itself can
first be simplified by discovering latent behavioral organization within financial markets.

The central idea is that market observations may contain information about underlying behavioral processes
that are not directly observable. If these processes can be identified, represented and validated, they may
provide a more informative foundation for predictive modelling than treating all market conditions as a
single heterogeneous prediction problem.

Market Sentiment ML (MSML) therefore investigates how market observations can be transformed into
behavioral representations that simplify predictive learning while remaining scientifically interpretable and
empirically testable.

Although the current research focuses primarily on foreign exchange markets and retail positioning data,
the underlying methodology is intentionally designed to remain independent of any particular asset class
or behavioral observation source.

### About this document

This document describes the scientific motivation, research methodology and long-term vision of the
project. It intentionally focuses on enduring scientific questions rather than current empirical findings or
implementation details.

- Readers interested in the present state of the research programme should consult `RESEARCH_STATE.md`.
- Repository organization and implementation are documented separately in `README.md`.

---

## Motivation

The original motivation for this project was straightforward: does retail foreign exchange sentiment
contain useful predictive information?

Retail positioning data has long attracted interest within quantitative finance because it provides a direct
observation of one class of market participants. If retail traders systematically exhibit persistent behavioral
biases, their aggregate positioning might contain information relevant to future market behavior. Early
research therefore concentrated on evaluating retail sentiment as a potential predictive signal.

Repeated experimentation gradually revealed a more nuanced picture. Raw sentiment consistently
exhibited weak and highly conditional predictive value. Apparent improvements frequently disappeared
under stricter causal validation, walk-forward evaluation or appropriate control experiments. Negative
findings proved to be just as informative as positive ones, gradually demonstrating that sentiment alone
was unlikely to provide a universal forecasting signal.

At the same time, those same experiments repeatedly suggested that sentiment was not random. Its
relationship to future market behavior appeared to depend strongly on broader market context, behavioral
persistence and structural market conditions.

This observation fundamentally changed the direction of the research. Rather than viewing retail sentiment
as a predictor, the project began treating it as a behavioral observation capable of revealing latent market
organization.

The research question therefore evolved from:

> Can sentiment predict returns?

to:

> Can market observations reveal behavioral representations that simplify predictive learning problems?

This transition ultimately motivated the development of the Behavioral Surface Validation Engine (BSVE)
and the broader behavioral representation framework that now forms the foundation of the repository.

---

## Scientific Perspective

The project views financial markets as adaptive behavioral systems rather than purely statistical
forecasting problems.

Price movements emerge from the collective actions of heterogeneous market participants whose behavior
evolves over time. Many of the processes responsible for those decisions cannot be observed directly.
Instead, they must be inferred indirectly through observable quantities such as price dynamics, volatility,
positioning data and other market-derived measurements.

Within this perspective, behavioral representations become intermediate scientific objects. Rather than
attempting to predict prices directly from raw observations, the project investigates whether those
observations can first be transformed into representations that better capture the underlying behavioral
state of the market.

Predictive machine learning then becomes a tool for evaluating those representations rather than the
primary object of study.

This distinction is central to the philosophy of the project. Machine-learning models may evolve over time,
but behavioral representations remain explicit, deterministic and independently testable scientific
hypotheses.

---

## Scientific Questions

The current research programme is organized around a small number of interconnected scientific
questions. Rather than attempting to answer a single predictive question, the project investigates an
entire behavioral modelling pipeline, beginning with market observations and ending with adaptive
decision-making.

### 1. Can sparse market observations reveal latent behavioral organization?

Financial markets are only partially observable. Although prices, volatility and positioning can be measured
directly, the underlying behavioral processes responsible for those observations remain hidden.

The first objective of the project is therefore to determine whether sufficiently informative observations
can reveal reproducible behavioral organization that is otherwise inaccessible.

### 2. Can behavioral representations be constructed deterministically?

If latent behavioral organization exists, it should be possible to describe it using explicit behavioral models
rather than opaque statistical classifiers.

The project therefore investigates whether behavioral states can be defined using deterministic ontologies
whose assumptions remain transparent, reproducible and independently testable. This question ultimately
motivated the development of the Behavioral Surface Validation Engine (BSVE).

### 3. Do behavioral representations simplify predictive learning?

Behavioral representations are valuable only if they improve the structure of the learning problem itself.
Rather than attempting to improve predictive models directly, the project asks whether partitioning market
behavior into meaningful behavioral contexts produces simpler and more stable prediction problems.

Predictive performance therefore becomes an empirical evaluation of the behavioral representation rather
than an objective in itself.

### 4. Do improved behavioral representations translate into improved decisions?

Even if behavioral representations improve predictive modelling, they must ultimately demonstrate value
within realistic downstream decision-making. Prediction alone is therefore not considered sufficient
evidence. Behavioral representations must also improve adaptive walk-forward evaluation under realistic
deployment conditions.

### 5. Can behavioral representations themselves become adaptive?

Markets evolve continuously as participants adapt to changing economic environments. The project
therefore investigates whether behavioral representations can eventually be selected dynamically
according to the current behavioral state of the market, rather than remaining permanently associated
with particular asset classes or currency families.

This question represents the long-term research direction of the project.

---

## Scope

The objective of this project is not to develop a single predictive trading model, nor to identify universally
predictive market indicators.

Instead, the project seeks to develop a general scientific methodology for:

- constructing behavioral representations,
- validating those representations independently,
- evaluating their predictive usefulness,
- understanding the mechanisms responsible for the observed behavioral organization.

Behavioral representations, observation sources and predictive architectures are therefore regarded as
evolving scientific objects rather than fixed components of the research programme. Individual behavioral
models are expected to improve over time. The methodology itself is intended to remain broadly
applicable.

---

## Central Hypothesis

The central hypothesis underlying the repository is that financial markets exhibit latent behavioral
organization that can be inferred from sufficiently informative market observations.

Behavioral representations derived from those observations are expected to partition heterogeneous
market behavior into more coherent behavioral contexts, thereby simplifying subsequent predictive
learning problems.

Importantly, the project does not assume that any particular observation source provides universal
predictive information. Instead, it assumes that different observations reveal different aspects of the
underlying behavioral system. Retail positioning currently provides the primary behavioral observation
source, but the surrounding methodology intentionally remains independent of any specific data source.
Consequently, future behavioral representations may emerge from alternative observations while
continuing to use the same validation and predictive framework.

---

## Current Behavioral Observation Sources

The behavioral representation methodology is intentionally independent of any particular market
observation source.

The current implementation investigates behavioral representations constructed primarily from:

- foreign exchange price dynamics,
- retail positioning (market sentiment),
- derived structural market features.

These observation sources represent the present empirical foundation of the project rather than a
permanent limitation of the methodology.

Descriptions of the current observation schema are available in:

- `docs/data/SENTIMENT_FEATURE_SCHEMA.md`
- `docs/behavioral/behavioral_model.md`

The complete behavioral representation methodology is documented in `bsve/docs/synthesis_document.md`.

---

## Research Methodology

The project follows a staged research methodology in which behavioral hypotheses are evaluated
independently before being exposed to predictive machine learning. Rather than combining behavioral
discovery and predictive optimization into a single process, each stage addresses a separate scientific question.

```
Market observations
        │
        ▼
Behavioral hypothesis
        │
        ▼
Deterministic ontology
        │
        ▼
Behavioral representation
        │
        ▼
Predictive evaluation
        │
        ▼
Adaptive evaluation
        │
        ▼
Mechanistic interpretation
```

Each stage provides evidence for the next while remaining independently reproducible.

This separation serves several purposes:

- Behavioral hypotheses remain explicit rather than becoming embedded within machine-learning models.
- Predictive performance evaluates behavioral representations instead of defining them.
- Independent validation reduces the risk of conflating behavioral interpretation with statistical optimization.
- New behavioral representations can be introduced without modifying downstream predictive infrastructure.
- Alternative behavioral observation sources can be incorporated while preserving the overall research methodology.

The resulting framework emphasizes scientific transparency, reproducibility and causal validation throughout
the entire research pipeline.

---

## Behavioral Representation Learning

The central methodological contribution of this project is the explicit separation between discovering
behavioral representations and evaluating them predictively.

Traditional financial machine-learning pipelines frequently combine feature engineering, representation
learning and predictive optimization into a single process. While this may maximize predictive performance for
a particular dataset, it often becomes difficult to determine why a model succeeds or whether the learned
representations correspond to meaningful market structure.

This project adopts a different philosophy.

Behavioral representations are treated as explicit scientific hypotheses rather than implicit properties of
trained machine-learning models. Consequently, representation discovery and predictive modelling become
separate research activities.

Behavioral representations are first proposed, calibrated and validated using deterministic methodology. Only
after satisfying predefined validation criteria are they exposed to predictive machine learning. Prediction
therefore evaluates the usefulness of behavioral representations rather than creating them.

This distinction allows behavioral hypotheses to remain transparent, reproducible and independently testable
throughout the research process.

---

## Scientific Methodology

The research programme deliberately separates four complementary scientific activities.

### Observation

The first stage concerns collecting and organizing market observations. Current work focuses primarily on
foreign exchange price data and retail positioning, but the methodology remains intentionally independent of
any particular observation source.

The objective is to produce causally aligned observations suitable for both behavioral analysis and predictive
modelling.

### Representation

Observed market behavior is then interpreted through explicit behavioral models. Rather than allowing
latent representations to emerge implicitly from machine-learning models, behavioral states are defined using
deterministic ontologies that describe hypothesized market organization.

Behavioral representations are treated as scientific objects whose validity can be investigated independently of
predictive performance.

### Prediction

Only after behavioral representations have been constructed are predictive models introduced. Machine
learning therefore answers a very specific scientific question: does this behavioral representation simplify the
prediction problem?

Predictive performance becomes evidence supporting—or contradicting—the behavioral representation itself.
The machine-learning model is consequently viewed as an experimental instrument rather than the central
contribution of the project.

### Explanation

Finally, empirical behavioral findings motivate mechanistic investigation. Agent-Based Modelling (ABM)
provides an environment for exploring whether interacting market participants can plausibly generate the
behavioral phenomena observed empirically.

This final stage attempts to move beyond prediction toward explanation. Rather than asking only whether a
behavioral representation works, the project also asks why such organization might emerge in real financial
markets.

---

## Architectural Consequences

Separating observation, representation, prediction and explanation has important architectural implications.

Each scientific activity can evolve independently. For example:

- new behavioral observation sources can replace or complement retail positioning,
- new behavioral ontologies can be introduced without modifying predictive models,
- predictive architectures can evolve without changing behavioral representations,
- downstream adaptive evaluation can compare competing behavioral representations using identical prediction pipelines,
- mechanistic models can investigate empirical findings without influencing predictive validation.

This modularity is intentional. The architecture is designed to encourage scientific iteration while minimizing
coupling between independent research questions.

---

## Relationship Between BSVE, MSML and MPML

The repository ecosystem reflects the scientific methodology directly. Each stage answers a different scientific
question.

```
Market observations
        │
        ▼
Master Research Dataset
        │
        ▼
BSVE
Behavioral Representation Discovery
        │
        ▼
Behavioral Dataset Variant
        │
        ▼
MSML
Predictive Evaluation
        │
        ▼
Prediction Artifact
        │
        ▼
MPML
Adaptive Decision Evaluation
        │
        ▼
ABM
Mechanistic Investigation
```


**BSVE asks:** Does this behavioral representation exist?

**MSML asks:** Is this representation predictive?

**MPML asks:** Does improved prediction improve adaptive decision-making?

**ABM asks:** Could interacting market participants plausibly generate this behavioral organization?

The distinction between these questions is fundamental to the philosophy of the project. Rather than
attempting to answer every question simultaneously using a single machine-learning model, the repository
decomposes behavioral research into a sequence of independently testable scientific problems.

---

## Research Programme

The project is organized into several complementary research programmes. Each programme addresses a
distinct scientific question while contributing to a common objective: constructing increasingly informative
behavioral representations from observable market data.

Current empirical progress within each programme is summarized separately in `RESEARCH_STATE.md`.

### Behavioral Organization

Investigates whether financial markets exhibit reproducible behavioral organization beyond conventional
statistical market descriptions. Representative questions include:

- Do stable behavioral structures exist?
- How should behavioral organization be characterized?
- Which observations reveal latent behavioral structure?

### Behavioral Representation

Investigates how behavioral organization can be represented explicitly through deterministic behavioral
ontologies. Representative questions include:

- How should behavioral states be defined?
- How should behavioral ontologies be calibrated?
- How should behavioral representations be validated independently?

The methodology supporting this programme is documented in detail in `bsve/docs/synthesis_document.md`.

### Predictive Evaluation

Investigates whether behavioral representations simplify predictive learning. Representative questions include:

- Do behavioral representations improve predictive modelling?
- Which representations provide the greatest predictive utility?
- Which predictive architectures benefit from behavioral specialization?

### Adaptive Decision Making

Investigates whether improved behavioral representations ultimately improve adaptive downstream decision-making.
Representative questions include:

- How should predictive models specialize within behavioral representations?
- How should competing behavioral representations be compared?
- Can behavioral representation selection itself become adaptive?

### Mechanistic Understanding

Investigates the mechanisms capable of generating empirically observed behavioral organization. Representative
questions include:

- Why do observed behavioral structures emerge?
- Which interacting-agent mechanisms reproduce empirical findings?
- How should empirical behavioral representations relate to mechanistic behavioral models?

Each programme evolves independently while remaining connected through explicit behavioral representations
and reproducible artifact contracts.

---

## Long-Term Vision

The current behavioral representations developed within this project should not be viewed as final
descriptions of financial markets. Instead, they represent successive approximations toward a broader
objective: understanding how latent behavioral organization emerges from observable market activity.

The project therefore views every validated behavioral representation as one point within a continuously
evolving research programme rather than as a permanent model of market behavior.

As empirical understanding improves, behavioral representations are expected to evolve, merge, split and
eventually be replaced by more informative descriptions of market dynamics. This philosophy intentionally
prioritizes scientific discovery over architectural permanence.

---

## From Static Classification to Dynamic Representation

Current research has identified reproducible behavioral differences between groups of foreign exchange pairs.
These findings have proven valuable for understanding why different market environments respond differently
to identical predictive methodologies.

However, the project does not regard these groups as fundamental properties of the corresponding currency
pairs. Instead, they are interpreted as observable manifestations of deeper behavioral processes.

One long-term objective is therefore to move beyond static classifications toward dynamic behavioral
representations. Rather than assuming that a particular financial instrument permanently belongs to a single
behavioral family, future systems should be capable of selecting among multiple behavioral representations
according to the currently observed market behavior.

Within this perspective:

- currency pair families become empirical observations rather than fixed assumptions,
- behavioral representations become reusable scientific objects,
- representation selection itself becomes a learnable problem.

The objective is therefore not to discover the correct behavioral representation, but to develop methodologies
capable of discovering, validating and selecting behavioral representations as markets evolve.

---

## Future Research Directions

The long-term objective of the project is not the development of increasingly specialized behavioral models.
Instead, it is the development of a general methodology for discovering, validating and comparing behavioral
representations across financial markets.

Several long-term directions naturally follow from this objective.

### Richer Behavioral Observation Sources

Future behavioral representations may emerge from additional observations describing market participants,
positioning dynamics or broader market structure. The methodology intentionally remains independent of
the specific observations used.

### General Behavioral Ontologies

Future ontologies are expected to become increasingly expressive while preserving deterministic construction,
interpretability and independent validation.

### Dynamic Behavioral Representations

Current behavioral representations provide alternative descriptions of market behavior. A natural long-term
extension is the ability to select among multiple validated behavioral representations according to the
observed market context rather than treating any single representation as universally applicable.

### Mechanistic Behavioral Models

Empirical behavioral representations describe observed market organization. Mechanistic behavioral models
seek to explain why that organization emerges. Bringing together empirical observation, predictive evaluation
and mechanistic explanation remains one of the long-term scientific objectives of the project.

---

## Concluding Remarks

Market Sentiment ML began as an investigation into retail foreign exchange sentiment. Through successive
empirical studies, the project gradually evolved into a broader scientific programme concerned with
behavioral representations of financial markets.

This evolution fundamentally changed the role of machine learning within the project. Predictive models are
no longer regarded as the primary scientific object. Instead, they serve as experimental instruments for
evaluating explicit behavioral representations constructed and validated independently.

The long-term ambition of the project is therefore not simply improved financial prediction, but a deeper
scientific understanding of how latent behavioral organization can be observed, represented and explained.