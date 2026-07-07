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
| PR5   | Behavioral Evaluation Framework                             |
| PR5.1 | Behavioral Characterization Framework                       |
| PR5.4 | Behavioral Surface Registry                                 |
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

| variant    | filename                               |
| ---------- | -------------------------------------- |
| `full`     | `master_research_dataset.csv`          |
| `core`     | `master_research_dataset_core.csv`     |
| `extended` | `master_research_dataset_extended.csv` |
| `<other>`  | `master_research_dataset_<other>.csv`  |

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

Transform the Behavioral Experiment Framework into a Behavioral Evaluation
Framework capable of producing standardized, comparable characterizations about
Behavioral Surfaces rather than merely reporting raw experiment outputs.

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

## Experiment Comparison

Introduce

`analysis/behavioral/compare_experiments.py`

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

The Behavioral Experiment Framework evolves into the foundation upon which the Behavioral Characterization Framework (PR5.1) builds, allowing researchers to focus primarily on interpreting behavioral experiments rather than assembling metrics manually.

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

# PR5.1 — Behavioral Characterization Framework

## Objective

Transform the Behavioral Evaluation Framework into a **Behavioral Characterization Framework** that synthesizes experimental evidence into concise scientific findings.

The primary objective is to improve scientific decision-making, not to increase the amount of reported information.

Engineering diagnostics remain available in report appendices, but the primary report becomes a research decision-support document.

---

## Report Structure

Every report begins with a one-page **Executive Summary**:
```

Experiment status Behavioral Surface Coverage Key Findings (≤ 5) Research Recommendation

```
The primary body contains **Scientific Findings** — aggregated, noise-suppressed statements with:
```

Finding title Observation (factual — what the data shows) Interpretation (scientific — what it may mean) Supporting evidence (one line per model/state) Scientific Interest Scientific Confidence Recommended follow-up

```
Engineering diagnostics (raw metrics, coverage tables, manifest warnings, baseline controls, Key Observations legacy section, reproducibility) are moved to a report **Appendix**.

---

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

## Comparative Characterization

Behavioral Characterization should become increasingly comparative as evidence accumulates.

Early reports necessarily interpret individual experiments in isolation.

Once multiple Behavioral Surfaces have been evaluated, reports should increasingly compare new results against accumulated historical evidence.

Preferred comparisons include

- previous ontology versions,
- related Behavioral Surfaces,
- architecture agreement trends,
- prediction uncertainty,
- coverage,
- registry status.

Characterization therefore evolves from

> "What happened during this experiment?"

towards

> "How does this Behavioral Surface compare with everything currently known?"

The Behavioral Surface Registry (PR5.4) provides the accumulated scientific context required for these comparisons.

---

## Scientific Interest and Scientific Confidence

Reports distinguish two independent quantities for each finding:

**Scientific Interest**
How important or potentially novel would this finding be if confirmed?  Interest is assigned based on the scientific importance of the phenomenon, not solely on threshold values.

| Interest | Examples |
|----------|---------|
| High     | Strong architecture disagreement; unexpected transfer behaviour; unusually stable Behavioral Surface; surprising comparison against controls |
| Medium   | Expected coverage limitations; moderate state imbalance |
| Low      | Expected engineering properties; routine diagnostics |

**Scientific Confidence**
How strongly is the finding currently supported by available evidence?

These quantities intentionally measure different properties. High-interest findings may initially carry low confidence. Well-established findings may eventually have relatively modest scientific interest. This distinction helps prioritize future research without overstating current evidence.

Interest and Confidence are rated `low` / `medium` / `high`.

---

## Scientific Restraint

Behavioral Characterization Reports intentionally distinguish between

**observations**

and

**scientific conclusions**.

Characterization experiments describe properties of the induced prediction problem.

They do **not** establish predictive usefulness.

Interpretations should therefore avoid attributing observed behaviour directly to the Behavioral Surface when alternative explanations remain plausible.

Examples include

- model capacity,
- optimization,
- feature representation,
- sample size,
- calibration,
- class imbalance.

Reports should therefore prefer wording such as

> "The current evidence is consistent with..."

rather than

> "The Behavioral Surface demonstrates..."

This preserves the distinction between hypothesis generation and predictive validation.

---

## Noise Suppression

Repeated per-artifact observations are collapsed into single aggregated findings.

Instead of:
```

Entropy high (STATE_A / MLP) Entropy high (STATE_A / LSTM) Entropy high (STATE_B / MLP) Entropy high (STATE_B / LSTM)

```
the report produces:
```

Finding: High prediction entropy across states Observation: Prediction entropy is consistently high across all Behavioral States, with predicted probabilities concentrated near 0.5. Interpretation: Behavioral characterization suggests training has not yet converged to a discriminative solution for these states. Supporting evidence:

- MLP / STATE_A: entropy 0.952 bits
- LSTM / STATE_A: entropy 0.961 bits
- MLP / STATE_B: entropy 0.948 bits
- LSTM / STATE_B: entropy 0.957 bits Scientific Interest: medium Scientific Confidence: high Recommended follow-up: Verify training convergence. Consider increasing epochs.

```
---

## Research Recommendation

Every report ends with a single recommended next experimental step, derived exclusively from synthesized ``Finding`` objects rather than independently re-evaluating raw metrics. Examples:

- **Proceed to walk-forward evaluation (PR7)** — when cross-architecture agreement is high.
- **Repeat characterization with additional training** — when entropy is high or effective coverage is low.
- **Diagnose and repeat** — when training runs failed.
- **Acquire additional Behavioral Surface evidence** — when coverage is limited and prediction confidence is weak.
- **Proceed to initial comparison** — when the experiment completed without critical issues.

---

## Named Training Profiles

`run_behavioral_suite.py` exposes named profiles that replace raw epoch counts as the primary way to express training intent:

| Profile       | Epochs | Purpose |
|---------------|--------|---------|
| `smoke`       | 2      | Pipeline smoke-test; results are not scientifically meaningful |
| `standard`    | 10     | Initial scientific characterization (default) |
| `publication` | 50     | Publication-quality results |

The default profile is `standard` (10 epochs), replacing the previous default of 1 epoch.

Individual hyperparameters (`--epochs`, `--hidden-dim`) can still be overridden via explicit CLI flags.

Example:

```bash
python analysis/behavioral/run_behavioral_suite.py \
  --dataset-version 1.5.1 \
  --dataset-variant reactive_jpy_v1_core \
  --profile standard
```

------

## Future Compatibility

The report architecture anticipates comparison against:

- Reactive CHF Behavioral Surface
- Persistent Behavioral Surface
- Additional future surfaces

No fixed state ontology is assumed. All finding rules operate on the metrics present in the experiment outputs.

------

## Delivered modules

| Module                                        | Purpose                                                      |
| --------------------------------------------- | ------------------------------------------------------------ |
| `analysis/behavioral/interpretation.py`       | `Finding` dataclass (with `observation`, `interpretation`, `evidence_strength`); `generate_findings()` (noise suppression, Finding-centric recommendation); `format_executive_summary()`; `format_findings()`; `derive_research_recommendation()` (derives from Findings only); legacy `Observation` API preserved |
| `analysis/behavioral/reporting.py`            | Restructured report: Executive Summary → Scientific Findings → Appendix (diagnostics) |
| `analysis/behavioral/run_behavioral_suite.py` | Named profiles (`smoke` / `standard` / `publication`); default epoch 10; profile recorded in `experiment_manifest.json` |

------

The primary question answered by a Behavioral Characterization Report is:

> **What have we learned about this Behavioral Surface?**

not

> **What happened during this experiment?**

------

# PR5.4 — Behavioral Surface Registry

## Objective

Introduce a durable, cross-experiment record of accumulated evidence for each Behavioral Surface, independent of any individual experiment report.

Behavioral Characterization Reports answer

> **What happened during this experiment?**

Walk-forward validation answers

> **Does this Behavioral Surface possess predictive value?**

The Behavioral Surface Registry answers

> **What do we currently believe about this Behavioral Surface, based on all accumulated evidence?**

As the number of Behavioral Surfaces grows beyond the initial Reactive JPY ontology, maintaining scientific continuity across experiments becomes an architectural responsibility rather than merely a documentation convenience.

------

## Scope

This PR introduces the registry schema and promotion workflow only.

It does **not**

- introduce new evaluation metrics,
- modify experiment reports,
- automate scientific interpretation,
- perform walk-forward evaluation,
- or perform trading evaluation.

Registry entries initially contain Stage 1 evidence.

The schema intentionally reserves sections for Stage 2 (Predictive Validation) and Stage 3 (Trading Validation) so that future evidence extends existing records rather than introducing parallel registries.

------

## Design Principles

### Scientific judgments, not measurements

The registry stores scientific judgments derived from accumulated evidence rather than experimental measurements.

Experiment reports remain the authoritative source of measurements.

The registry records the current scientific interpretation of that evidence.

------

### Human interpretation remains authoritative

Scientific Interest and Scientific Confidence are the concepts introduced in PR5.

Within the registry they are always human-authored.

Each assessment records

- author
- timestamp
- supporting experiments

These assessments are informed by experiment evidence but are never computed automatically.

------

### One Behavioral Surface, one scientific record

Each Behavioral Surface owns exactly one registry entry.

Evidence accumulates.

Interpretation evolves.

History is preserved.

------

### The registry is an index

The registry references experiment reports.

It never duplicates measurements or report content.

Reports remain responsible for experimental evidence.

The registry records the project's current working scientific interpretation.

------

### Promotion is explicit

Completed experiments append candidate evidence.

A separate promotion step updates the authoritative registry entry.

Scientific interpretation therefore remains a deliberate research decision rather than an automatic consequence of experiment completion.

------

### Retirement is informative

A Behavioral Surface that ultimately fails predictive validation is retained.

Its registry entry transitions to

```
Retired
```

together with the supporting evidence.

Negative results remain durable scientific outputs.

------

## Registry Lifecycle

Each Behavioral Surface progresses independently through the research programme.

Typical progression:

```
Characterization

↓

Predictive Validation

↓

Trading Validation

↓

Integrated
```

or

```
Characterization

↓

Predictive Validation

↓

Retired
```

Lifecycle stage records research progress rather than scientific quality.

------

## Deliverables

```
registry/surfaces/<surface>.yaml

analysis/registry/promote.py

analysis/registry/high_score.py
```

Each registry entry records

- lifecycle stage
- Scientific Interest
- Scientific Confidence
- accumulated supporting evidence
- recommended next research step
- promotion history

------

## Delivered modules (PR5.4)

| Module | Purpose |
|---|---|
| `registry/surfaces/<surface_id>.yaml` | Authoritative scientific record per surface |
| `analysis/registry/promote.py` | Explicit manual promotion workflow |
| `analysis/registry/high_score.py` | Human-readable registry summary (scientific triage) |

Schema documentation: `registry/surfaces/README.md`

Regression tests: `tests/test_behavioral_surface_registry.py`

------

## Result

The Behavioral Surface Registry becomes the project's durable scientific memory.

Experiment reports describe individual experiments.

Walk-forward reports establish predictive evidence.

MPML establishes trading evidence.

The registry records the current scientific understanding of every Behavioral Surface.

------

# PR6 — Behavioral Prediction Routing

## Objective

Teach MPML to consume Behavioral Surface prediction artifacts while preserving complete backwards compatibility with the existing regime-based workflow.

This PR intentionally introduces **no scientific evaluation**.

Its purpose is purely architectural: to establish a causal path by which Behavioral Surface predictions can participate in downstream walk-forward evaluation.

Behavioral routing should be additive rather than replacing existing regime routing.

Scientific conclusions remain explicitly out of scope until PR7.

------

## Result

Behavioral prediction artifacts become first-class routing inputs for MPML while remaining completely independent of the existing market-regime workflow.

This PR establishes the architectural connection between Behavioral Characterization and downstream predictive validation without drawing scientific conclusions about predictive value.

------

# PR7 — Behavioral Walk-forward Validation

## Objective

PR7 represents the first stage at which Behavioral Surfaces may be judged as
predictive representations rather than merely behavioral representations.

Previous PRs establish

- Behavioral Surface construction
- dataset augmentation
- model training
- experiment orchestration
- experiment characterization
- behavioral prediction routing

PR7 introduces **predictive validation**.

The central scientific question becomes

> **Does partitioning the market according to Behavioral Surface states
> produce more predictable price behavior than existing regime-based
> representations?**

This distinction is fundamental.

Behavioral characterization is not behavioral validation.

Observations regarding entropy, confidence, agreement or state occupancy may
suggest interesting hypotheses, but they do not establish predictive
usefulness.

Only reproducible walk-forward evaluation may support such conclusions.

------

## Walk-forward Protocol

Predictive walk-forward evaluation shall reuse the existing MPML walk-forward schedule.

To guarantee identical temporal partitioning between Stage 2 (Predictive Validation) and Stage 3 (Trading Validation), PR7 shall use the reference implementation located at:

```
research/utils/mpml_walkforward_reference.py
```

This module is a synchronized copy of the MPML walk-forward implementation and is treated as the authoritative definition of fold construction within MSML. Changes to fold generation should be made in MPML first and then synchronized into the reference module.

Using an identical sequence of train/test folds ensures that Stage 2
(Predictive Validation) and Stage 3 (Trading Validation) evaluate identical
market periods, allowing predictive improvements and downstream trading
performance to be compared without introducing differences caused solely by
different fold definitions.

Behavioral Walk-forward therefore evaluates **prediction** under the same
temporal conditions that MPML later uses to evaluate **trading**.

Walk-forward experiment manifests shall record the fold protocol (train years, test months, step months, protocol identifier) so that predictive and trading evaluations can be verified to use identical schedules.

------

## Metric Selection

Behavioral Surfaces are expected to produce weak, noisy signals. This is an
expected property of the research programme, not a defect to be engineered
away.

Consequently, the metrics used to evaluate walk-forward performance must be
chosen to fairly characterize weak predictive effects rather than to reward or
penalize them by construction.

Metric selection is therefore treated as a first-class scientific decision
rather than an implementation detail.

### Principles

- Metrics shall evaluate the Behavioral Surface as a predictive
  representation, not as a trading strategy. Tradeability remains the
  exclusive responsibility of MPML (Stage 3).
- Metrics shall remain robust under severe class imbalance, since imbalance
  varies materially across Behavioral Surfaces and individual Behavioral
  States.
- Metrics shall avoid implicit threshold selection wherever practical.
  Choosing an operating threshold presumes a trading cost structure and
  therefore belongs to Stage 3 rather than Stage 2.
- Metric behaviour shall be understood and documented before adoption.
  Metrics are selected because they answer appropriate scientific questions,
  not because they are commonly reported.

------

## Class Imbalance

Class imbalance is expected to vary substantially between Behavioral Surfaces
and between individual Behavioral States.

Reactive JPY already demonstrates severe imbalance, and future Behavioral
Surfaces should be expected to exhibit different imbalance characteristics.

Evaluation metrics must therefore remain informative even when minority-class
support is limited.

F1 score, while useful historically during early experimentation, exhibits
several undesirable properties under severe imbalance:

- the F1-optimal threshold may vary substantially between walk-forward folds
  despite unchanged predictive quality
- precision and recall are collapsed into a single quantity, obscuring the
  underlying failure mode
- F1 has no natural baseline; identical F1 values may represent very different
  predictive behaviour under different class balances

Accordingly,

**F1 shall be retained only as a secondary diagnostic metric for historical
continuity and shall not be interpreted as primary scientific evidence.**

Primary evaluation metrics for PR7 shall include:

| Metric                                     | Purpose                                                      |
| ------------------------------------------ | ------------------------------------------------------------ |
| **PR-AUC (Average Precision)**             | Primary threshold-free ranking metric emphasizing minority-class performance |
| **Brier Score**                            | Measures probability quality and overall probabilistic accuracy |
| **Calibration analysis**                   | Evaluates whether predicted probabilities correspond to observed event frequencies |
| **Matthews Correlation Coefficient (MCC)** | Thresholded cross-check using all four confusion-matrix cells |
| **Balanced Accuracy**                      | Simple interpretable diagnostic complement                   |

Calibration quality is considered a first-class quantity because downstream
systems (including MPML) consume calibrated prediction probabilities rather
than binary class labels.

Per-fold and per-state class balance shall always accompany reported metrics.

------

## Baseline Controls

Absolute metric values are not scientifically interpretable in isolation.

Every reported metric shall therefore be accompanied by matched reference
baselines.

Required baselines include

- **Permutation baseline**

  Labels are randomly permuted **independently within each walk-forward fold**,
  preserving temporal partitioning while destroying any genuine predictive
  relationship.

- **Base-rate baseline**

  Predictor matching the observed positive-class frequency within each fold.

- **Random matched-partition baseline**

  Random partitions matched for sample size and temporal coverage, consistent
  with the controls introduced in PR5.

Behavioral predictive effects shall be interpreted relative to these baselines
rather than against fixed numerical thresholds.

------

## Reporting Requirements

Consistent with the scientific-restraint principles established in PR5.1,
walk-forward reports shall

- report per-fold metrics rather than only aggregate values
- report per-state metrics wherever practical
- report class balance per fold and per Behavioral State
- report calibration quality
- report precision and recall as secondary diagnostic quantities
- express findings relative to baseline controls using the existing Finding
  structure
- avoid language implying tradeability

Walk-forward reports answer

> **Does this Behavioral Surface improve prediction?**

They intentionally do **not** answer

> **Should this Behavioral Surface be traded?**

------

## Deliverables

- reproducible walk-forward evaluation
- imbalance-robust predictive metrics
- probability calibration analysis
- permutation, base-rate and matched-partition controls
- comparison against existing Trend/Volatility partitioning
- per-fold and per-state reporting
- reproducible reports

Expected comparisons include

- Behavioral Surface vs Trend/Volatility Surface
- Per-state predictive performance
- Aggregate predictive performance
- Fold stability
- Coverage
- Calibration quality
- Statistical significance relative to baseline controls

------

## Possible Outcomes

### CONFIRMED

Behavioral Surface prediction consistently outperforms the existing
Trend/Volatility representation across walk-forward folds, with improvements
remaining distinguishable from permutation and matched baseline controls.

### INCONCLUSIVE

Evidence suggests predictive benefit, but uncertainty remains too large to
draw a robust conclusion. Additional Behavioral Surface refinement or further
data are warranted.

### NOT CONFIRMED

Behavioral partitioning successfully characterizes market behaviour but does
not produce reproducible predictive improvement beyond baseline controls.

Negative results remain scientifically valuable and shall be preserved within
the Behavioral Surface Registry.

------

# Scientific Evidence Hierarchy

Behavioral Surfaces evolve through several distinct stages of scientific maturity.

## Stage 1 — Behavioral Characterization

Questions answered

- Does the Behavioral Surface appear internally consistent?
- Are the discovered states meaningful?
- Does the induced prediction problem exhibit interesting properties?

Primary outputs

- characterization reports
- prediction characterization
- coverage analysis

------

## Stage 2 — Predictive Validation

Questions answered

- Does the Behavioral Surface simplify prediction?
- Does it outperform existing market-regime representations?
- Are improvements stable under walk-forward validation?

Primary outputs

- walk-forward reports
- predictive comparisons
- robustness estimates

------

## Stage 3 — Trading Validation

Questions answered

- Does improved prediction improve trading?
- Does adaptive routing benefit from Behavioral Surface information?
- Does the effect survive realistic execution?

Primary outputs

- MPML evaluation
- trading performance
- robustness analysis

These stages represent increasing levels of scientific evidence.

Progression between stages is deliberate and requires supporting experimental evidence rather than successful software implementation.

------

# PR8 — Surface Generalization

Replace the current surface-specific interfaces with a generic SurfaceProvider abstraction capable of supporting multiple behavioral or market surfaces simultaneously.

This refactor is intentionally deferred until Behavioral Surface predictive value has been demonstrated.

------

# Future Work

The implementation above intentionally postpones architectural generalization.

If Behavioral Surface predictive validation succeeds, a later refactor may introduce a generalized SurfaceProvider abstraction.

That abstraction would allow multiple behavioral or market surfaces to coexist without modifying train.py or MPML.

Examples include

- Trend/Volatility Surface
- Reactive-JPY Surface
- Reactive-CHF Surface
- Persistent Surface

This refactor is intentionally deferred until predictive value has been demonstrated.

The current implementation prioritizes answering the scientific question over architectural elegance.

------

# Layered Architecture

The integration intentionally separates behavioral discovery from predictive modelling and experiment execution.

BSVE │ ▼ Behavioral Surface Artifact │ ▼ Dataset Builder │ ▼ Augmented Dataset │ ▼ MSML Training Pipelines │ ▼ Prediction Artifacts │ ▼ Behavioral Experiment Framework │ ▼ Behavioral Evaluation Framework │ ▼ Experiment Comparison │ ▼ MPML Evaluation

Each layer consumes only the published artifacts of the previous layer and never depends upon internal implementation details.

------

# Research Philosophy

The Behavioral Surface integration intentionally separates three independent scientific questions, each answered by a distinct stage of evidence. Behavioral discovery in BSVE is the prerequisite that produces the Behavioral Surface; the three MSML/MPML stages then characterize, validate, and trade on it.

## Behavioral Discovery (BSVE) — Prerequisite

Question

> Does the Behavioral Surface represent a genuine and reproducible behavioral phenomenon?

Primary evidence

Behavioral consistency.

This stage is owned by BSVE and produces the Behavioral Surface artifact consumed by the stages below. It is a precondition for — not a substitute for — predictive or trading evidence.

------

## Stage 1 — Behavioral Characterization (MSML)

Question

> Does the Behavioral Surface induce an interesting and internally consistent prediction problem?

Primary evidence

Prediction characterization — coverage, state occupancy, prediction distributions, entropy, confidence, and cross-architecture agreement.

This stage describes properties of the induced prediction problem. It generates hypotheses about predictive usefulness but does not establish it. Owned by the Behavioral Experiment, Evaluation, and Characterization Frameworks (PR4, PR5, PR5.1).

------

## Stage 2 — Predictive Validation (MSML)

Question

> Does partitioning the market according to Behavioral Surface states produce more predictable price behavior than existing regime-based representations?

Primary evidence

Reproducible walk-forward evaluation using the shared MPML fold protocol, enabling direct comparison between predictive and downstream trading evidence — predictive comparison against trend/volatility regimes, per-state and aggregate performance, fold stability, and statistical significance of predictive differences.

This is the first stage at which a Behavioral Surface may be judged as a predictive representation rather than merely a behavioral one. Owned by PR7.

------

## Stage 3 — Trading Validation (MPML)

Question

> Does improved prediction translate into improved trading under realistic execution?

Primary evidence

MPML evaluation — adaptive routing performance, trading metrics, and robustness under realistic execution assumptions.

This is the final stage of scientific maturity. A Behavioral Surface reaches `Integrated` status only after surviving this stage. Owned by MPML.

------

## Separation of Evidence

Each stage answers an independent question with its own evidence standard:

| Stage                     | System | Question                                       | Evidence                    |
| ------------------------- | ------ | ---------------------------------------------- | --------------------------- |
| Discovery                 | BSVE   | Is the phenomenon real?                        | Behavioral consistency      |
| 1 — Characterization      | MSML   | Is the induced prediction problem interesting? | Prediction characterization |
| 2 — Predictive Validation | MSML   | Does it improve prediction?                    | Walk-forward evaluation     |
| 3 — Trading Validation    | MPML   | Does it improve trading?                       | MPML evaluation             |

Progression between stages is deliberate and requires supporting experimental evidence rather than successful software implementation.

Crucially, evidence from a later stage must never be interpreted as retroactive support for an earlier stage's claim, nor may an earlier stage's evidence be treated as anticipating a later stage's conclusion. A confirmed characterization does not imply predictive value; confirmed predictive value does not guarantee trading value. This strict separation is the central discipline of the research programme: it prevents the accumulation of software progress from being mistaken for the accumulation of scientific evidence.

---

# Summary

The BSVE → MSML → MPML integration establishes a single, reproducible research pipeline in which behavioral discovery, predictive modelling, and trading evaluation remain rigorously separated.

- **BSVE** discovers and validates behavioral phenomena, exporting them as immutable Behavioral Surface artifacts.
- **The dataset builder** joins those artifacts onto the master research dataset without ever rewriting canonical outputs.
- **MSML** trains predictive models from behavioral dataset variants and characterizes, then validates, the induced prediction problem.
- **MPML** evaluates whether validated predictive improvements translate into trading value.
- **The Behavioral Surface Registry** accumulates durable scientific interpretation across all experiments, preserving both positive and negative results.

Information flows only through published artifacts and stable public contracts. No component reproduces the responsibilities of another. Each architectural step is additive and backwards-compatible, and each scientific stage is gated by its own evidence standard.

The architecture prioritizes answering the scientific question over architectural elegance, deferring generalization (PR8) until predictive value has been demonstrated.
