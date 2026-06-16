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
✓ Criterion validation reporting (Reactive-JPY Criterion 1)
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
- behavioral outcomes (descriptive reversal-rate diagnostics per consensus state)
- behavioral tests (descriptive Fisher's exact test results with Cohen's h effect sizes)
- validation outcome

**Report sections**

| Section | Description |
|---------|-------------|
| `metadata` | Criterion name, generation timestamp, thresholds, and explicit descriptive-only behavioral diagnostic metadata |
| `state_frequencies` | Observation counts per state and per pair |
| `duration_statistics` | Median, mean, P25/P75, and max episode durations per state |
| `duration_ks_diagnostics` | KS-test results for duration distributions (calibration-consistency diagnostics only) |
| `survival_analysis` | Fraction of episodes surviving past 8, 24, and 48 bars per state |
| `transition_frequencies` | Counts of entry, continuation, exit_reversal, and exit_unknown events per state |
| `behavioral_outcomes` | Per consensus state: episode count, reversal count, reversal rate, and descriptive-diagnostic classification |
| `behavioral_tests` | Pairwise Fisher's exact test results with Cohen's h effect sizes, reported as descriptive diagnostics only |
| `validation_outcome` | Final PASS / FAIL / INCONCLUSIVE verdict and supporting counts |

**Status interpretation**

Criterion 1 invokes behavioral outcome analysis internally before determining
its status. The current outcome-distribution analysis computes per-state exit
reversal rates for the three consensus states (`YOUNG`, `MATURING`, `MATURE`)
and tests whether those rates are statistically distinct using Fisher's exact
test. Cohen's h is reported as a descriptive effect size metric.

Current outcome-distribution analysis is descriptive.

Because outcome labels are derived from episode duration, they are not
considered independent behavioral evidence.

Criterion 1 emits one of:

- `FAIL`: quality gates fail (for example insufficient observations)
- `INCONCLUSIVE`: minimum sample thresholds pass, but independent behavioral evidence has not yet been implemented for Reactive-JPY

Reactive-JPY Criterion 1 therefore remains `INCONCLUSIVE` until threshold-exit
labeling or another independent behavioral outcome mechanism is implemented.

For Reactive-JPY Criterion 1, duration KS diagnostics and current outcome
distribution diagnostics are reported for ontology inspection only. They are
**not** treated as independent evidence of behavioral differentiation, so the
current workflow cannot produce `PASS`.

**Progression analysis**

Criterion 1 reports KS-test comparisons for the three consensus-state transitions:

- `YOUNG → MATURING`
- `YOUNG → MATURE`
- `MATURING → MATURE`

These results are **descriptive only**. They are included in the validation report
for ontology interpretation and future research, but are **not** used in Criterion 1
`PASS`/`FAIL` determination.

Current support is limited to Reactive-JPY Criterion 1 validation.

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
  ├─ Duration Diagnostics
  └─ Behavioral Outcome Validation
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
