# Market Sentiment ML (MSML)

Market Sentiment ML (MSML) is a research platform for constructing, validating and evaluating behavioral
representations of financial markets.

The repository provides the complete research pipeline from canonical dataset construction through
deterministic behavioral representation learning and predictive machine learning. Downstream adaptive
evaluation is performed by the sibling Market Phase ML (MPML) repository.

This document describes the repository architecture, major components and development workflow.

For complementary perspectives on the project:

| Document                 | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `PROJECT_DESCRIPTION.md` | Scientific motivation, research methodology and long-term vision. |
| `RESEARCH_STATE.md`      | Current empirical findings, confidence assessment and active research priorities. |

The remainder of this document focuses exclusively on the repository architecture and developer workflow.

---

## Repository Overview

The repository is organized as a collection of independent research components connected through
explicit, versioned artifact contracts.

Each component addresses a distinct stage of the research pipeline while remaining independently
testable and reproducible.

The principal components are:

- Master Research Dataset
- Behavioral Surface Validation Engine (BSVE)
- Predictive Modelling (MSML)
- Adaptive Evaluation (MPML)
- Agent-Based Modelling (ABM)

Together these components provide a reproducible research pipeline spanning dataset construction,
behavioral representation, predictive modelling, adaptive evaluation and mechanistic investigation.

---

## Repository Architecture

The following diagram shows the flow of artifacts through the pipeline from raw market observations to
adaptive walk-forward evaluation.

```
Market observations
        │
        ▼
Master Research Dataset
        │
        ▼
Behavioral Surface Validation Engine (BSVE)
        │
        ▼
Behavioral Dataset Variant
        │
        ▼
MSML Predictive Models
        │
        ▼
Prediction Artifacts
        │
        ▼
MPML (sibling repository)
Adaptive Walk-Forward Evaluation
        │
        ▼
Agent-Based Modelling (ABM)
Mechanistic Investigation
```

Each stage communicates through explicit artifact contracts, allowing behavioral research, predictive modeling and downstream evaluation to evolve independently.

### Primary Artifacts

Each stage communicates through explicit artifact contracts, allowing behavioral research, predictive
modeling and downstream evaluation to evolve independently.

| Artifact                   | Producer             | Consumer             |
| -------------------------- | -------------------- | -------------------- |
| Master Research Dataset    | Dataset builder      | BSVE, MSML           |
| Behavioral Surface         | BSVE                 | Dataset augmentation |
| Behavioral Dataset Variant | Dataset augmentation | MSML                 |
| Prediction Artifact        | MSML                 | MPML                 |

---

## Repository Layout

The repository is organized around a small number of long-lived research components. Experimental
outputs, logs and temporary analyses are intentionally kept separate from the core implementation.

```text
market-sentiment-ml/

├── bsve/
│   Behavioral Surface Validation Engine
│
├── data/
│   Canonical research datasets
│
├── docs/
│   Documentation and research notes
│
├── research/
│   Predictive modelling and experimentation
│
├── scripts/
│   Dataset construction and utilities
│
├── schemas/
│   Versioned artifact contracts
│
├── tests/
│   Automated validation
│
└── ...
```


The repository intentionally separates implementation, documentation, datasets and experimental
artifacts. Individual research projects may generate additional outputs (logs, artifacts and exploratory
analyses), but these are not considered part of the stable repository architecture.

---

## Documentation Guide

Documentation is organized according to responsibility rather than chronology.

### Core project documentation

| Document                 | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `README.md`              | Repository architecture and developer orientation.           |
| `PROJECT_DESCRIPTION.md` | Scientific motivation, methodology and long-term research programme. |
| `RESEARCH_STATE.md`      | Current empirical findings, confidence assessment and active research priorities. |

### Dataset documentation

The Master Research Dataset forms the foundation of the repository. Documentation is located under
`docs/data/`, including:

- dataset construction
- feature definitions
- dataset versions
- sentiment feature schema

### Behavioral Surface Validation Engine

Behavioral Surface documentation is maintained independently within `bsve/docs/`. The recommended
entry point is `bsve/docs/synthesis_document.md`, which describes the complete behavioral representation
methodology from ontology construction through validation and artifact export.

Additional key documents include:

- `behavioral_surface_schema.md` — artifact schema and field definitions
- `reactive_jpy_findings.md` — Reactive-JPY ontology and validation results
- `reactive_chf_findings.md` — Reactive-CHF exploratory findings
- `CLI.md` — command-line workflows for behavioral surface generation

### Predictive Modelling Documentation

Predictive modelling is documented under `docs/models/` and `docs/integration/`. These documents
describe predictive model architectures, artifact contracts and downstream integration.

Behavioral prediction experiments should be orchestrated through:

```bash
python analysis/behavioral/run_behavioral_suite.py \
  --dataset-version 1.5.1 \
  --dataset-variant reactive_jpy_v1_core
```

This framework discovers `(surface_id, state_id)` combinations from the selected dataset variant,
runs both `research/deep_learning/train.py` and `research/deep_learning/train_lstm.py` per state,
collects manifests/prediction artifacts, and writes a reproducible experiment bundle under
`analysis/output/<experiment_id>/`.

### Behavioral Research Documentation

Behavioral analyses are documented under `docs/behavioral/`, including:

- currency pair families
- latent behavioral structure
- transfer experiments
- sentiment ablation studies

### Integration Documentation

The interaction between BSVE, MSML and MPML is documented separately. Primary references include:

- `docs/BSVE_MSML_integration_architecture.md`
- `docs/integration/dl_prediction_artifacts.md`
- `docs/integration/dl_artifact_contract.md`

These documents describe artifact contracts, dataset augmentation, prediction artifacts and downstream
integration.

### Agent-Based Modelling Documentation

The repository contains an independent Agent-Based Modelling (ABM) framework used to investigate
candidate mechanisms capable of generating empirically observed market behavior. Implementation and
experimental documentation is maintained under `docs/abm/`. This framework is architecturally
independent of the predictive modelling pipeline.

---

## Typical Research Workflows

Although different research projects use different subsets of the repository, most work follows one of a
small number of common workflows.

### Dataset construction

The following workflow constructs the canonical research dataset used throughout the repository.

Raw observations │ ▼ Master Research Dataset


### Behavioral representation development

The following workflow evaluates a behavioral hypothesis deterministically before exposing it to predictive
machine learning.

Master Research Dataset │ ▼ Calibration │ ▼ Behavioral Surface │ ▼ Behavioral Dataset Variant


### Predictive modelling

The following workflow trains predictive models on a behavioral dataset variant and exports versioned
prediction artifacts.

Behavioral Dataset Variant │ ▼ MSML Training │ ▼ Prediction Artifacts


### Adaptive evaluation

The following workflow evaluates prediction artifacts under realistic adaptive trading conditions within the
downstream MPML framework.

Prediction Artifacts │ ▼ MPML │ ▼ Walk-Forward Evaluation


---

## Quick Start

Most development work begins with one of the following entry points. New contributors are encouraged to
start with the documentation corresponding to the subsystem they intend to modify rather than exploring
the repository structure directly.

**Working on datasets**

Start with `docs/data/` and the dataset construction scripts under `scripts/`.

**Working on behavioral representations**

Start with `bsve/docs/synthesis_document.md`. Behavioral Surfaces are constructed, validated and
exported entirely within the BSVE subsystem.

**Working on predictive models**

Start with `research/` and `docs/models/`. Behavioral dataset variants generated by BSVE can be used
directly without modifying model architectures.

**Working on downstream evaluation**

Prediction artifacts generated by MSML are consumed by the sibling MPML repository for adaptive
walk-forward evaluation.

---

## Design Principles

The repository is organized around a small number of architectural principles.

**Explicit artifact contracts.**
Wherever practical, components communicate through versioned artifacts rather than implementation
coupling, allowing each subsystem to evolve independently without breaking downstream consumers.

**Deterministic data processing.**
Dataset construction, behavioral state assignment and feature engineering are implemented as
deterministic, versioned procedures so that any artifact can be reproduced from its inputs alone.

**Causal validation.**
Every stage of the pipeline is designed to preserve temporal causality; no stage may consume information
that would not have been available at the corresponding observation time.

**Reproducible research.**
Intermediate artifacts are versioned and accompanied by provenance metadata, enabling every stage of
the pipeline to be reproduced and audited independently.

**Clear subsystem boundaries.**
Behavioral interpretation, predictive modelling and adaptive evaluation are implemented as independent
subsystems with explicitly documented interfaces, so that changes to one subsystem do not require
changes to another.

---

## Contributing

When extending the repository, contributors are encouraged to preserve the existing architectural
boundaries and artifact contracts.

The principal subsystem responsibilities are:

- **Dataset construction** — produces canonical research datasets
- **Behavioral representation (BSVE)** — constructs deterministic behavioral representations
- **Predictive modelling (MSML)** — trains models and exports prediction artifacts
- **Adaptive evaluation (MPML)** — evaluates prediction artifacts under walk-forward deployment
- **Mechanistic investigation (ABM)** — investigates mechanisms underlying observed behavioral phenomena

New functionality should communicate through documented artifact contracts wherever practical. The
`schemas/` directory contains versioned artifact schema definitions. See `docs/integration/dl_artifact_contract.md`
for a worked example of a documented artifact contract.

Subsystem-specific implementation guidance is available under `docs/` and `bsve/docs/`.

---

## License

See `LICENSE` for details.