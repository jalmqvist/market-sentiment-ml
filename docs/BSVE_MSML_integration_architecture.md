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

## Integration Roadmap

The Behavioral Surface integration is intentionally delivered as a sequence of
small architectural steps. Each PR introduces a single capability while
preserving backwards compatibility.

| PR    | Capability                                                  |
| ----- | ----------------------------------------------------------- |
| PR1   | Behavioral Surface contract                                 |
| PR2   | Dataset augmentation                                        |
| PR3   | Behavioral training                                         |
| PR3.1 | Dataset variants                                            |
| PR4   | Behavioral Experiment Framework                             |
| PR5   | Behavioral Characterization Framework                       |
| PR6   | MPML behavioral prediction routing                          |
| PR7   | Walk-forward scientific evaluation                          |
| PR8   | Generalized SurfaceProvider abstraction (only if justified) |

Each stage validates the previous stage before introducing additional
architectural complexity.

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

# PR3.1 — Dataset Variant Support

## Objective

Teach both MSML training pipelines to load arbitrary dataset variants while
preserving complete backwards compatibility.

Dataset variants become a first-class concept alongside dataset versions.

---

## CLI

Both `research/deep_learning/train.py` and `research/deep_learning/train_lstm.py`
accept a new argument:

    --dataset-variant

Default: `core`

Examples:

Canonical

    --dataset-version 1.5.1
    --dataset-variant core

Behavioral

    --dataset-version 1.5.1
    --dataset-variant reactive_jpy_v1_core

Existing commands that omit `--dataset-variant` continue loading
`master_research_dataset_core.csv` unchanged.

---

## Dataset Loader

`research/deep_learning/dataset_loader.py` resolves datasets using
`dataset_version` and `dataset_variant`.

The loader owns filename resolution.  Training scripts must never construct
filenames directly.

Filename resolution:

| variant      | filename                                            |
|--------------|-----------------------------------------------------|
| `full`       | `master_research_dataset.csv`                       |
| `core`       | `master_research_dataset_core.csv`                  |
| `extended`   | `master_research_dataset_extended.csv`              |
| `<other>`    | `master_research_dataset_<other>.csv`               |

The variant parameter accepts any string, enabling arbitrary Behavioral Surface
variants without modifying the loader.

---

## Provenance

Prediction artifacts and manifests record the actual dataset variant used for
training.  Hard-coded `dataset_variant = "core"` values have been removed from
both training pipelines.

Behavioral provenance receives the selected dataset variant rather than a
hard-coded value.

---

## Result

Both training pipelines can load any dataset variant by passing
`--dataset-variant` on the CLI.  All provenance fields reflect the actual
variant used.  Existing commands without `--dataset-variant` are fully
backwards-compatible.

---

# PR4 — Behavioral Experiment Framework

## Objective

Introduce a reproducible experiment framework that automates Behavioral Surface
training, artifact collection and comparative analysis.

The framework exists above the individual training pipelines. Rather than
requiring users to invoke `train.py` or `train_lstm.py` directly, it orchestrates
complete behavioral experiments from a single command.

This layer is responsible for experiment orchestration only. It does not modify
the underlying training pipelines.

---

## Responsibilities

The framework shall

- discover available Behavioral Surface states from the selected dataset
  using the (surface_id, state_id) pairs actually present in the dataset
  rather than from hard-coded ontologies.
- launch the required MLP and LSTM training runs
- collect prediction artifacts and manifests
- verify artifact provenance
- compare prediction coverage
- compare prediction distributions
- generate reproducible experiment reports
- archive experiment outputs under a dedicated experiment directory

---

## Experiment Inputs

At minimum

- dataset_version
- dataset_variant

Optional

- surface_id
- feature_set
- target_horizon
- train_pairs
- predict_pairs
- selected models

Behavioral states are discovered automatically from the dataset whenever
possible.

---

## Experiment Outputs

Each experiment produces

analysis/output/<experiment_id>/

    experiment_manifest.json
    
    report.md
    
    summary.csv
    
    metrics.csv
    
    manifests/
    
    prediction_artifacts/
    
    plots/
    
    logs/

where experiment_manifest.json contains

- CLI arguments
- trainer versions
- discovered states
- models executed
- git commit
- timestamps
- success/failure

No manual artifact collection should be required.

---

## Initial Analyses

The initial framework shall compare

- behavioral state occupancy
- prediction coverage
- timestamp overlap
- pair coverage
- prediction distributions
- signal-strength distributions
- MLP vs LSTM agreement
- prediction correlations

No trading evaluation is performed at this stage.

Coverage is considered a first-class experimental quantity.

Every analysis shall distinguish between

- canonical dataset coverage
- behavioral coverage
- per-state coverage

to avoid diluting behavioral effects across timestamps where no behavioral
information exists.

---

## Design Principles

The framework should expose the scientific question rather than the
implementation.

The preferred user workflow becomes

    run_behavioral_suite.py

rather than manually executing multiple training commands.

Reference entrypoint:

    python analysis/behavioral/run_behavioral_suite.py \
        --dataset-version 1.5.1 \
        --dataset-variant reactive_jpy_v1_core

The suite discovers available `(surface_id, state_id)` pairs from the selected
dataset variant, launches MLP/LSTM training runs through shared subprocess
helpers, and materializes a reproducible experiment directory:

    analysis/output/<experiment_id>/
        experiment_manifest.json
        report.md
        summary.csv
        metrics.csv
        manifests/
        prediction_artifacts/
        plots/
        logs/

Future Behavioral Surfaces (Reactive CHF, Persistent, etc.) should require no
changes to the framework beyond selecting a different dataset variant.

---

## Result

Behavioral experimentation becomes fully reproducible and largely
independent of manual scripting, allowing researchers to focus on interpreting
results rather than managing training runs and artifacts.

---

# PR5 — Behavioral Evaluation Framework

## Objective

Transform the Behavioral Experiment Framework into a Behavioral Characterization Framework capable of synthesizing evidence about Behavioral Surfaces rather than merely reporting experiment metrics.

This PR introduces no new predictive models.

Its purpose is to improve experiment robustness and the scientific value of the
 generated reports.

------

## Engineering Improvements

The framework shall

- replace filesystem-difference artifact discovery with trainer-reported artifact identities
- report percentage-based coverage metrics
- report overlap percentages in addition to raw overlap counts
- summarize discovered Behavioral Surface states in every report
- classify informational messages, warnings and errors separately
- improve experiment provenance where appropriate

These changes improve robustness without changing experimental results.

------

## Scientific Metrics

Reports shall include metrics describing prediction behavior independently of trading performance. The objective is not to maximize the number of reported metrics, but to increase the scientific interpretability of Behavioral Surface experiments.

At minimum:

- behavioral coverage fraction
- per-state occupancy fraction
- prediction probability distribution
- signal-strength distribution
- prediction entropy
- prediction confidence distribution
- effective prediction coverage
- pair balance
- timestamp coverage
- calibration metrics (e.g. probability calibration or reliability summaries, where practical)

These metrics describe model behavior rather than trading performance.

---

## Statistical Estimates

Where practical, reported metrics should include simple uncertainty estimates.

Preferred methods include

- bootstrap confidence intervals
- sampling variability
- effect size estimates

Formal hypothesis testing is intentionally deferred.

The goal is to quantify confidence in observed effects rather than to establish
statistical significance.

------

## Controls

The framework shall support comparison against baseline controls.

Initial controls include

- full dataset
- behavioral partition
- regime partition
- randomly sampled partitions matched for sample size

Controls exist to distinguish genuine behavioral effects from effects caused by
 dataset size or partitioning alone.

Random partitions shall be matched for sample size and temporal coverage
whenever possible to avoid confounding coverage effects with predictive effects.

------

## Scientific Interpretation

Generated reports should contain a concise rule-based interpretation section.

Interpretations summarize the experimental observations without drawing
 scientific conclusions.

Examples include

- Behavioral coverage represents 6.7% of the canonical dataset.
- MLP/LSTM agreement is substantially lower in JPY_CONSENSUS_YOUNG than in
   JPY_NON_EXTREME.
- Behavioral state occupancy is strongly imbalanced.

Interpretations shall explain

- what was observed
- why it matters
- what follow-up investigation it suggests

without making unsupported scientific claims.

Interpretations should explain why observations matter rather than merely
 repeating numeric values.

---

### Evidence Assessment

Scientific findings should distinguish between

**Scientific Interest**

How important or potentially novel would this finding be if confirmed?

and

**Scientific Confidence**

How strongly is the finding currently supported by available evidence?

These quantities intentionally measure different properties.

High-interest findings may initially have low confidence.

Conversely, well-established findings may eventually have relatively modest scientific interest.

This distinction helps prioritize future research without overstating current evidence.

---

### Scientific Synthesis

The primary purpose of generated reports is to support scientific decision-making rather than comprehensive metric reporting.

Reports should therefore prioritize

- concise executive summaries
- aggregated findings rather than repetitive observations
- supporting evidence for each finding
- recommended next experiments

Engineering diagnostics remain available in appendices or auxiliary output files.

Behavioral characterization reports should answer

> **What have we learned about this Behavioral Surface?**

rather than

> **What happened during this experiment?**

Characterization reports are intended to support research decisions, not publication-quality scientific conclusions.

------

## Experiment Comparison

Introduce

```
analysis/behavioral/compare_experiments.py
```

capable of comparing two or more completed experiment directories without requiring prediction regeneration or retraining.

Comparisons include

- coverage
- prediction distributions
- state occupancy
- prediction agreement
- experiment metadata
- provenance

------

## Out of Scope

No MPML integration.

No walk-forward evaluation.

No trading metrics.

Those remain the responsibility of subsequent PRs.

------

## Result

The Behavioral Experiment Framework evolves from an execution framework into a scientific characterization framework, allowing researchers to focus primarily on  interpreting behavioral experiments rather than assembling metrics manually.

### Delivered modules

| Module | Purpose |
|---|---|
| `analysis/behavioral/metrics.py` | Scientific prediction metrics (entropy, confidence, effective coverage, pair balance, etc.) |
| `analysis/behavioral/controls.py` | Baseline controls (full dataset, behavioral partition, regime partition, random matched) |
| `analysis/behavioral/interpretation.py` | Rule-based Key Observations interpreter |
| `analysis/behavioral/compare_experiments.py` | CLI for comparing two or more completed experiment directories |

### Delivered engineering improvements

- `utils.py`: `RunResult` carries `reported_parquet_path` / `reported_manifest_path`; `parse_reported_artifact_paths()` extracts trainer-reported paths from combined stdout/stderr.
- `run_behavioral_suite.py`: prefers trainer-reported artifact paths over filesystem-diff, with fallback; `artifact_discovery` provenance field in `summary.csv`; wires scientific metrics, controls, and Key Observations into every run.
- `coverage.py`: adds `coverage_fraction` and `state_fraction_of_behavioral` columns.
- `compare_predictions.py`: adds `overlap_pct_of_mlp` and `overlap_pct_of_lstm` columns.
- `analyze_manifests.py`: classifies manifest messages into `notes`, `warnings`, and `errors`; legacy `extract_manifest_warnings()` preserved for backwards compatibility.
- `reporting.py`: includes Discovered States section, Scientific Prediction Metrics, Baseline Controls, Key Observations, and severity-classified manifest issues.

---

The primary output of the Behavioral Evaluation Framework is not the experiment report itself, but an improved understanding of the evaluated Behavioral Surface. Reports summarize individual experiments; scientific knowledge accumulates across experiments.

---

# PR6 — Behavioral Prediction Routing

### Objective

Teach MPML to consume Behavioral Surface prediction artifacts while preserving complete backwards compatibility with the existing regime-based workflow.

This PR intentionally introduces **no scientific evaluation**.

Its purpose is purely architectural: to establish a causal path by which Behavioral Surface predictions can participate in downstream walk-forward evaluation.

Behavioral routing should be additive rather than replacing existing regime routing.

Scientific conclusions remain explicitly out of scope until PR7.

---

# PR7 — Behavioral Walk-forward Validation

## Objective

PR7 represents the first stage at which Behavioral Surfaces may be judged as predictive representations rather than merely behavioral representations.

Previous PRs establish

- Behavioral Surface construction
- dataset augmentation
- model training
- experiment orchestration
- experiment characterization

PR7 introduces **predictive validation**.

The central scientific question becomes

> **Does partitioning the market according to Behavioral Surface states produce more predictable price behavior than existing regime-based representations?**

This distinction is fundamental.

Behavioral characterization is not behavioral validation.

Observations regarding entropy, confidence, agreement or state occupancy may suggest interesting hypotheses, but they do not establish predictive usefulness.

Only reproducible walk-forward evaluation may support such conclusions.

### Deliverables

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

### Possible outcomes

CONFIRMED

Behavioral surfaces outperform the existing market-regime partition.

INCONCLUSIVE

Behavioral representation appears useful but requires refinement.

NOT CONFIRMED

Behavioral partitioning does not improve prediction despite successful BSVE
validation.

---

## Scientific Maturity

Behavioral Surfaces evolve through several distinct stages of scientific maturity.

### Stage 1 — Behavioral Characterization

Questions answered

- Does the Behavioral Surface appear internally consistent?
- Are the discovered states meaningful?
- Does the induced prediction problem exhibit interesting properties?

Primary outputs

- experiment reports
- prediction characterization
- coverage analysis

------

### Stage 2 — Predictive Validation

Questions answered

- Does the Behavioral Surface simplify prediction?
- Does it outperform existing market-regime representations?
- Are improvements stable under walk-forward validation?

Primary outputs

- walk-forward reports
- predictive comparisons
- robustness estimates

------

### Stage 3 — Trading Validation

Questions answered

- Does improved prediction improve trading?
- Does adaptive routing benefit from Behavioral Surface information?
- Does the effect survive realistic execution?

Primary outputs

- MPML evaluation
- trading performance
- robustness analysis

---

# PR8 — Surface Generalization

Replace the current surface-specific interfaces with a generic SurfaceProvider abstraction capable of supporting multiple behavioral or market surfaces simultaneously.

This refactor is intentionally deferred until Behavioral Surface predictive value has been demonstrated.

---

# Future Direction — Behavioral Surface Registry

Repeated Behavioral Surface experiments suggest that the enduring scientific object is the **Behavioral Surface itself**, rather than any individual experiment.

Experiment reports are transient.

Behavioral Surfaces accumulate evidence through repeated characterization, validation and comparison.

A future Behavioral Surface Registry may therefore be introduced to maintain accumulated evidence across experiments.

Possible responsibilities include

- evaluation history
- walk-forward history
- comparison history
- researcher annotations
- current research status
- recommended next steps

The precise contents of such a registry are intentionally left undefined.

The project is expected to discover which quantities prove scientifically valuable through the evaluation of multiple independent Behavioral Surfaces before committing to a permanent schema.

This registry is conceptually distinct from experiment tracking systems (such as MLflow), whose primary responsibility is experiment provenance rather than long-term scientific interpretation.

The registry is expected to evolve from accumulated evidence rather than from predefined scoring rules.

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

# Layered Architecture

The integration intentionally separates behavioral discovery from predictive
modelling and experiment execution.

BSVE
        │
        ▼
Behavioral Surface Artifact
        │
        ▼
Dataset Builder
        │
        ▼
Augmented Dataset
        │
        ▼
MSML Training Pipelines
        │
        ▼
Prediction Artifacts
        │
        ▼
Behavioral Experiment Framework
        │
        ▼
Behavioral Evaluation Framework
        │
        ▼
Experiment Comparison
        │
        ▼
MPML Evaluation

Each layer consumes only the published artifacts of the previous layer and
never depends upon internal implementation details.

---

# Research Philosophy

The Behavioral Surface integration intentionally separates three independent scientific questions.

## Stage 1 — Behavioral Discovery (BSVE)

Question

> Does the Behavioral Surface represent a genuine and reproducible behavioral phenomenon?

Primary evidence

Behavioral consistency.

------

## Stage 2 — Behavioral Characterization (MSML)

Question

> Does the Behavioral Surface induce an interesting prediction problem?

Primary evidence

Prediction characteristics, coverage, agreement, uncertainty and comparison against baseline controls.

Characterization is descriptive rather than predictive.

It exists to generate hypotheses.

------

## Stage 3 — Behavioral Validation (MSML Walk-forward)

Question

> Does the Behavioral Surface improve predictive performance under causal walk-forward evaluation?

Primary evidence

Predictive accuracy, stability and comparison against established market-regime baselines.

Validation determines whether the Behavioral Surface possesses predictive value.

------

## Stage 4 — Trading Validation (MPML)

Question

> Does improved prediction translate into improved trading performance?

Primary evidence

Walk-forward trading performance.

Trading performance is evaluated only after predictive value has been established independently.

------

This hierarchy intentionally separates

behavioral evidence

↓

predictive evidence

↓

trading evidence

preventing later stages from being interpreted as evidence for earlier scientific claims.

---

## Relationship to Implementation

This document describes the intended architecture of Behavioral Surface
integration.

Concrete implementation details—including CLI syntax, exported schemas,
artifact formats, regression requirements and testing—are specified in the
individual PR design documents.

If any implementation detail described elsewhere conflicts with this document,
the architecture described here takes precedence.