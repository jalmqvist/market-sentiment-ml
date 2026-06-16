# BSVE Command-Line Interface

## Overview

BSVE (Behavioral State Validation Engine) is the calibration and validation
subsystem for the Market Sentiment ML project.  It derives empirical maturity
boundaries for consensus-state ontologies from historical H1 sentiment data,
produces versioned calibration artifacts, and provides inspection and
validation utilities for pre-deployment sign-off.

---

Current status:
✓ Calibration framework
✓ Artifact validation
✓ Rule-based state assignment
✓ Criterion validation (Reactive-JPY Criterion 1)
□ Environment validation
□ Multi-ontology support

---

## Calibration

Run the JPY maturity boundary calibration against a master research dataset.

```bash
python -m bsve.calibration.jpy_maturity_calibration \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --dataset-version 1.5.1 \
    --output-dir bsve.test/
```

**Additional options**

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | `USDJPY EURJPY GBPJPY` | Space-separated JPY pairs to include |
| `--start` | `2019-01-01` | Calibration window start (ISO date) |
| `--end` | `2026-12-31` | Calibration window end (ISO date) |
| `--state-spec` | `bsve/state_specs/reactive_jpy_v1.yaml` | State-spec YAML |
| `--calibration-id` | auto-generated | Override artifact identifier |
| `--plot` | off | Emit hazard curve PNG for sign-off review |

**Outputs**

- A versioned JSON calibration artifact written to `--output-dir`.
- Console summary of thresholds and diagnostics.
- Optional hazard curve PNG (`jpy_hazard_curve.png`) when `--plot` is set.

---

## Artifact Inspection

Inspect a calibration artifact and display its thresholds and diagnostics.

```bash
python -m bsve.calibration.inspect <artifact.json>
```

**Example**

```bash
python -m bsve.calibration.inspect bsve/calibration_artifacts/reactive_jpy_calibration_v1.json
```

**Displayed diagnostics**

- Ontology ID and version
- Calibration mode and outcome
- Thresholds: extreme threshold, young boundary, mature boundary
- Episode count, median duration, censoring rate
- Reversal rates (young and mature maturity zones)
- Hazard crossover bar
- Survival counts (when present):

  ```
  Survival Counts
  ---------------
  >= 8 bars  : XXX
  >=16 bars  : XXX
  >=24 bars  : XXX
  >=32 bars  : XXX
  >=48 bars  : XXX
  ```

- Threshold provenance metadata

Older artifacts that do not contain survival counts are handled gracefully;
the section is omitted from the output.

---

## Calibration Validation

Validate all committed calibration artifacts against the BSVE schema and
quality gates.

```bash
python -m bsve.calibration.validate_calibrations
```

**Purpose**

Ensures every artifact in `bsve/calibration_artifacts/` is schema-valid,
has a consistent hash, and meets minimum quality criteria before state
assignment begins.  Run this after each new calibration run.

---

## State Assignment

Assign behavioral states using a committed calibration artifact and
environment specification.

```bash
python -m bsve.state_machine.rule_based \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --environment reactive_jpy \
    --output-dir bsve.test/
```

**Required inputs**

| Input                | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| Dataset              | Master research dataset (e.g. `data/output/1.5.1/master_research_dataset_core.csv`) |
| Environment spec     | State ontology YAML (e.g. `reactive_jpy_v1.yaml`)                                   |
| Calibration artifact | Signed calibration artifact stored under `bsve/calibration_artifacts/`              |

**Outputs**

The command produces:

```
bsve.test/
├── bsve_states_reactive_jpy_1.0.0.parquet
├── diagnostics.json
└── run_manifest.json
```

**Artifact descriptions**

| Artifact                                 | Purpose                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `bsve_states_reactive_jpy_1.0.0.parquet` | Row-level state assignments suitable for criterion testing and downstream analysis    |
| `diagnostics.json`                       | State counts, episode statistics, survival counts, maturity sparsity diagnostics      |
| `run_manifest.json`                      | Provenance record linking dataset version, ontology, calibration artifact, and run ID |

**Validation behavior**

Before writing artifacts the state machine performs:

* calibration artifact validation
* ontology validation
* state artifact validation
* uniqueness checks on `(pair, environment_id, entry_time)`

State assignment fails fast if any contract violation is detected.

---

## Criterion Validation

Validate Criterion 1 (Behavioral Differentiation) using the Reactive-JPY
state-surface artifact generated by state assignment.

```bash
python -m bsve.validation.criterion1 \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --output-dir bsve.test/
```

**Required inputs**

| Input | Description |
| ----- | ----------- |
| `--artifact` | Path to `bsve_states_reactive_jpy_1.0.0.parquet` |
| `--output-dir` | Directory where `bsve_validation_report.json` will be written |

**Generated outputs**

```
bsve.test/
└── bsve_validation_report.json
```

`bsve_validation_report.json` includes:

- metadata
- state frequencies
- duration statistics
- duration KS diagnostics (calibration-consistency only)
- behavioral outcomes
- behavioral tests
- validation outcome

**Status interpretation**

Criterion 1 emits one of:

- `PASS`: behavioral differentiation evidence is present and quality gates pass
- `FAIL`: quality gates fail (for example insufficient observations)
- `INCONCLUSIVE`: only duration-derived diagnostics are available, so behavioral differentiation is not established

For Reactive-JPY Criterion 1, duration KS diagnostics are reported as
calibration-consistency checks and are **not** treated as evidence of behavioral
differentiation. Therefore, runs with sufficient samples but only duration-derived
evidence are marked `INCONCLUSIVE`.

Current support is limited to Reactive-JPY Criterion 1 validation.

---

## Behavioral Outcome Validation

Run independent post-entry behavioral validation on an assigned Reactive-JPY
state-surface artifact.

```bash
python -m bsve.validation.behavioral_outcomes \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --output-dir bsve.test/
```

**Required inputs**

| Input | Description |
| ----- | ----------- |
| `--artifact` | Path to `bsve_states_reactive_jpy_1.0.0.parquet` generated by state assignment |
| `--output-dir` | Directory where `behavioral_outcomes_report.json` will be written |

**Generated outputs**

```
bsve.test/
└── behavioral_outcomes_report.json
```

`behavioral_outcomes_report.json` includes:

- state transition matrix
- reversal probabilities within 4, 8, and 12 bars
- persistence probabilities at 4, 8, and 12 bars
- state progression analysis
- independent behavioral differentiation tests

**Interpretation**

- Reversal probabilities measure how often a maturity state reverts to
  `JPY_NON_EXTREME` within the stated horizon after the state is entered.
- Persistence probabilities measure whether the future trajectory remains in the
  allowed consensus-state family at the stated horizon:
  - `YOUNG → YOUNG | MATURING | MATURE`
  - `MATURING → MATURING | MATURE`
  - `MATURE → MATURE`
- Criterion 1 can only move to `PASS` when these independent behavioral tests
  have sufficient samples and at least one state comparison is significant with
  a reported effect size.

Behavioral Outcome Validation is separate from calibration-consistency
diagnostics. Duration-derived checks remain visible in Criterion Validation
reports but do not determine `PASS` or `FAIL`.

---

## Artifact Locations

Committed, immutable calibration artifacts are stored under:

```
bsve/calibration_artifacts/
```

Ad-hoc or test calibration runs write artifacts to the directory specified
via `--output-dir` (e.g. `bsve.test/`).  Test-run artifacts must not be
committed to `bsve/calibration_artifacts/` unless they have passed manual
review and validation.

---

## Current Workflow

```
Calibration
  ↓
State Assignment
  ↓
Criterion Validation
  ↓
Behavioral Outcome Validation
  ↓
Research Review
```

---

## Future Commands

The following commands are planned and will be documented here as they are
implemented.

| Command | Purpose |
|---------|---------|
| `python -m bsve.timeline.generate` | Generate behavioral state timelines from state assignments |
| `python -m bsve.ontology.validate` | Validate ontology YAML definitions against the BSVE schema |
