# BSVE → MSML Integration Architecture

## Purpose

Behavioral State Validation Engine (BSVE) has successfully demonstrated that
behavioral state ontologies can be constructed and independently validated
outside of the prediction and trading pipelines.

The next research objective is to determine whether these behavioral
representations simplify the market prediction problem.

This document describes the target architecture for integrating Behavioral
Surfaces into MSML while preserving clear ownership boundaries and backwards
compatibility.

Individual implementation steps are intentionally delivered through small,
testable pull requests.

---

# Overall Architecture

Current pipeline:

Master Research Dataset
        ↓
MSML train.py
        ↓
Prediction Artifact
        ↓
MPML

Target pipeline:

Master Research Dataset
        │
        ├────────────► BSVE
        │                 │
        │                 ▼
        │      Behavioral Surface Artifact
        │                 │
        └────────► Dataset Builder (join)
                          │
                          ▼
            Extended Master Research Dataset
                          │
                          ▼
                     MSML train.py
                          │
                          ▼
                  Prediction Artifact
                          │
                          ▼
                        MPML

The Behavioral Surface becomes the canonical interface between BSVE and the
prediction pipeline.

The exported Behavioral Surface is treated as an immutable deterministic
feature source rather than as a model output. BSVE never modifies the master
research dataset directly.

Dataset construction remains responsible for joining behavioral information
onto the research dataset. After the join, behavioral attributes become
ordinary dataset columns and participate in feature selection exactly like
price-, trend- or volatility-derived variables.

This separation preserves architectural independence between behavioral state
assignment and downstream predictive modeling.

---

## Architectural Responsibilities

The integration is intentionally partitioned into independent components.

- BSVE owns behavioral state assignment.
- The dataset builder owns dataset integration.
- MSML owns predictive modelling.
- MPML owns trading strategy selection.

No component is permitted to reproduce or reinterpret the responsibilities of
another component. Information flows only through published artifacts and
stable public contracts.

---

# Behavioral Surface Contract

The exported Behavioral Surface is the canonical downstream artifact produced
by BSVE.

The contract guarantees:

- exactly one row per `(timestamp, pair)`
- deterministic ordering
- deterministic state assignment
- stable schema
- append-only schema evolution
- ontology-independent semantics
- suitability for left-joining onto compatible research datasets

Downstream systems should consume only this artifact and should not depend on
internal BSVE implementation details.

---

## PR1 — Behavioral Surface Contract

Objective

Establish the Behavioral Surface as the canonical downstream interface
produced by BSVE.

Result

BSVE exports a deterministic, versioned Behavioral Surface together with
associated provenance metadata. Downstream systems consume this artifact
through the published Behavioral Surface contract rather than through BSVE
internals.

Implementation details are specified in the corresponding PR.

---

## PR2 — Dataset Augmentation

Objective

Integrate the Behavioral Surface into the master research dataset while
preserving complete backwards compatibility.

Result

Behavioral attributes become ordinary dataset features through deterministic
augmentation performed by the dataset builder. BSVE remains the sole owner of
behavioral state assignment.

Operational workflow:

1. Build canonical master research datasets.
2. Generate a frozen Behavioral Surface artifact in BSVE.
3. Run dataset augmentation against existing canonical datasets.

Canonical datasets are immutable during augmentation. The augmentation stage
writes behavioral variants alongside canonical outputs and never rewrites
`master_research_dataset.csv`, `master_research_dataset_core.csv`, or
`master_research_dataset_extended.csv` unless an explicit force rebuild is
requested during canonical dataset construction.

Implementation details are specified in the corresponding PR.

---

# PR3 — Behavioral Surface Training

## Objective

Teach both MSML training pipelines (`train.py` and `train_lstm.py`) to train
on Behavioral Surface partitions in addition to existing market-regime
partitions.

Behavioral partitioning is implemented purely as an alternative dataset
selection mechanism. Once the training rows have been selected, the remainder
of each training pipeline executes unchanged.

Behavioral semantics remain entirely the responsibility of BSVE. MSML consumes
only dataset columns.

---

## Dataset Selection

Behavioral dataset augmentation (PR2) introduced versioned dataset variants.

Training datasets are therefore identified by two independent dimensions:

- dataset_version
- dataset_variant

Examples:

Canonical dataset

    dataset_version = 1.5.1
    dataset_variant = core

Behavioral dataset

    dataset_version = 1.5.1
    dataset_variant = reactive_jpy_v1_core

The dataset loader is responsible for resolving these identifiers to the
appropriate dataset files.

Training pipelines must never construct dataset filenames directly.

---

## Partitioning

Exactly one partitioning mechanism is active during any training run.

Regime mode

    --regime HVTF

Behavioral mode

    --surface reactive_jpy
    --state JPY_CONSENSUS_YOUNG

Behavioral mode filters

    surface_id == reactive_jpy

and

    state_id == JPY_CONSENSUS_YOUNG

Once the dataset has been filtered, preprocessing, normalization, optimization,
evaluation and prediction export proceed identically.

---

## Dataset Provenance

Prediction artifacts shall preserve complete dataset provenance.

At minimum:

- dataset_version
- dataset_variant

Behavioral-trained artifacts additionally record

- surface_id
- state_id
- ontology_version (when available)

The dataset variant recorded in prediction artifacts shall always match the
dataset variant supplied to the training pipeline.

Hard-coded dataset variants are not permitted.

---

## Deliverables

- identical CLI support in `train.py` and `train_lstm.py`
- Behavioral Surface filtering
- dataset variant selection
- updated artifact provenance
- updated manifests
- updated logging

No changes to

- preprocessing
- feature engineering
- normalization
- train/test splitting
- optimization
- model architectures
- evaluation
- prediction export

---

## Result

MSML becomes capable of training predictive models from arbitrary Behavioral
Surface dataset variants while preserving complete backwards compatibility with
the existing regime-based workflow.

---

# PR4 — Behavioral Surface Experiment Runner

Objective

Automate predictive validation experiments.

Behavioral Surface experiment manifests should mirror the existing regime experiment manifests so that predictive validation is fully reproducible.

Deliverables

- experiment runner analogous to existing regime experiments
- iterate over Behavioral Surface states
- export artifacts
- generate manifests
- maintain reproducibility

No architectural changes.

Result

Entire Behavioral Surface can be trained in one command.

---

# PR5: Behavioral Prediction Routing

Objective

Teach MPML to consume Behavioral Surface prediction artifacts.

Deliverables

- detect Behavioral Surface artifacts
- load predictions by

    surface_id

    state_id

- preserve regime routing
- preserve backwards compatibility

Behavioral routing should be additive rather than replacing regime routing.

Result

MPML can consume Behavioral Surface prediction artifacts.

---

# PR6 — Behavioral Walk-forward Evaluation

Objective

Evaluate whether Behavioral Surface partitioning simplifies the prediction
problem.

This PR contains no architectural innovation.

It exists solely to answer the primary scientific question.

Deliverables

- full walk-forward
- predictive metrics
- comparison against trend/volatility regimes
- reproducible reports

Expected comparisons

- Behavioral Surface vs Trend/Volatility Surface
- Per-state predictive performance
- Aggregate predictive performance
- Fold stability
- State occupancy
- Coverage
- Statistical significance of predictive differences

Possible outcomes

CONFIRMED

Behavioral surfaces outperform the existing market-regime partition.

INCONCLUSIVE

Behavioral representation appears useful but requires refinement.

NOT CONFIRMED

Behavioral partitioning does not improve prediction despite successful BSVE
validation.

---

PR7 — Surface Generalization

Replace the current surface-specific interfaces with a generic SurfaceProvider abstraction capable of supporting multiple behavioral or market surfaces simultaneously.

This refactor is intentionally deferred until Behavioral Surface predictive value has been demonstrated.

---

# Future Work

The implementation above intentionally postpones architectural generalization.

If Behavioral Surface predictive validation succeeds, a later refactor may
introduce a generalized SurfaceProvider abstraction.

That abstraction would allow multiple behavioral or market surfaces to coexist
without modifying train.py or MPML.

Examples include

- Trend/Volatility Surface
- Reactive-JPY Surface
- Reactive-CHF Surface
- Persistent Surface

This refactor is intentionally deferred until predictive value has been
demonstrated.

The current implementation prioritizes answering the scientific question over
architectural elegance.

---

# Research Philosophy

This implementation intentionally separates three independent scientific questions.

Stage 1 — BSVE

Does the behavioral representation describe a genuine market phenomenon?

↓

Stage 2 — MSML

Does the behavioral representation simplify the prediction problem?

↓

Stage 3 — MPML

Does improved prediction translate into improved trading performance?

Each stage validates the previous stage before introducing additional complexity.

This separation prevents trading performance from being interpreted as evidence
for or against the underlying behavioral hypothesis.

---

## Relationship to Implementation

This document describes the intended architecture of Behavioral Surface
integration.

Concrete implementation details—including CLI syntax, exported schemas,
artifact formats, regression requirements and testing—are specified in the
individual PR design documents.

If any implementation detail described elsewhere conflicts with this document,
the architecture described here takes precedence.