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
✓ Behavioral Surface Generator
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

## Validation Window Extraction

Partition the master research dataset into the frozen validation windows defined
by the BSVE validation protocol.

```bash
python -m bsve.validation.extract_validation_windows \
    --dataset data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/validation
```

### Purpose

This utility extracts the four fixed validation windows defined by the
Behavioral Validation Specification:

- Development (2019–2022)
- OOS validation
- Excluded sentiment-gap window
- Future holdout

The extraction is purely chronological and contains no ontology-specific logic.

### Outputs

```
bsve.test/
└── validation/
    ├── development/
    │   └── master_research_dataset_core.csv
    ├── oos/
    │   └── master_research_dataset_core.csv
    ├── excluded/
    │   └── master_research_dataset_core.csv
    └── holdout/
        └── master_research_dataset_core.csv
```

### Integrity checks

The utility automatically verifies that:

- every observation belongs to exactly one validation window,
- validation windows are mutually exclusive,
- all master-dataset observations are accounted for.

These checks should pass before any Behavioral Surface generation begins.

---

## Behavioral Surface Generator

Run the deterministic, causal Behavioral Surface Generator from a frozen calibration artifact.

```bash
python -m bsve.state_machine.rule_based \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/ \
    --calibration-artifact bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --state-spec bsve/state_specs/reactive_jpy_v1.yaml \
    --dataset-version 1.5.1
```

**Required inputs**

| Input                | Description                                                                          |
| -------------------- | ------------------------------------------------------------------------------------ |
| Dataset              | Master research dataset (CSV/parquet) loaded via `MasterResearchDatasetAdapter`     |
| Calibration artifact | Validated `CalibrationArtifact` loaded through `load_calibration_artifact`          |
| Ontology spec        | Frozen ontology YAML (default: `bsve/state_specs/reactive_jpy_v1.yaml`)            |

**Outputs**

The command produces:

```
bsve.test/
├── behavioral_surface_reactive_jpy_1.0.0.parquet
└── behavioral_surface_manifest.json
```

**Behavioral Surface columns**

```
timestamp
pair
state
episode_id
maturity_bars
crowd_side
```

**Provenance fields**

```
ontology_id
ontology_version
calibration_id
calibration_hash
schema_version
dataset_version
generated_timestamp
```

**Manifest fields**

- ontology id/version
- calibration artifact id/hash
- dataset version
- row count
- pair counts
- state counts
- schema version
- generation timestamp

The orchestration layer is intentionally thin: behavioral assignment logic lives in
the Behavioral Surface Generator engine and ontology plugin.

---

## Behavioral Surface Inspection

Inspect a generated Behavioral Surface before running any validation.

```bash
python -m bsve.validation.inspect_surface \
    --surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
    --calibration bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --output-dir bsve.test/inspection
```

**Purpose**

The inspection utility performs structural sanity checks on a generated Behavioral Surface before downstream validation. It is intended to catch implementation errors (such as broken episode construction or maturity tracking) before Criterion 1 is executed.

**Generated outputs**

```
bsve.test/
└── inspection/
    ├── 01_episode_length_distribution.png
    ├── 02_state_frequencies.png
    └── 03_maturity_distribution.png
```

**Console summary**

The inspection report includes:

- total observations
- total episodes
- pair frequencies
- state frequencies
- episode length statistics
- longest detected episodes
- maximum observed maturity
- number of episodes surviving beyond the ontology boundaries (8 and 24 bars)
- calibration thresholds
- automatic warnings for suspicious Behavioral Surface properties

Typical warning conditions include:

- no MATURING observations
- no MATURE observations
- maturity never exceeds zero
- unusually short episode lengths

The inspection utility is intended as the first validation step after Behavioral Surface generation and before ontology validation or Criterion 1 analysis.

---

## Calibration Drift

Compare a development Behavioral Surface with an out-of-sample (OOS)
Behavioral Surface to assess whether the ontology continues to produce a
plausible behavioral structure before statistical validation.

```bash
python -m bsve.validation.calibration_drift \
    --development bsve/calibration_artifacts/reference_surface.parquet \
    --oos bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
    --development-calibration bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --oos-calibration bsve/calibration_artifacts/reactive_jpy_calibration_v1.json \
    --output-dir bsve.test/drift
```

### Purpose

Calibration drift is assessed descriptively rather than inferentially.

The utility compares the canonical Behavioral Surface summaries produced by
`bsve.validation.inspect_surface.summarize_surface()` and reports whether the
behavioral structure observed during development remains plausible on the
validation window.

Calibration drift provides context for interpreting statistical validation and
is **not** itself a pass/fail criterion.

### Comparison metrics

The report compares:

- state occupancy (counts and percentages),
- episode count,
- episode length statistics,
- maturity survival counts,
- pair frequencies,
- structural warnings.

### Outputs

```
bsve.test/
└── drift/
    └── calibration_drift_report.json
```

### Console summary

The utility prints:

- state occupancy comparison,
- episode statistics comparison,
- maturity survival comparison,
- pair-count comparison,
- descriptive assessment.

No statistical tests are performed. Calibration drift is interpreted together
with the Behavioral Surface inspection report and automated sentinel checks
before contingency analysis is executed.

---

## Join Validation

Validate that a generated Behavioral Surface aligns one-to-one with the originating master research dataset.

```bash
python -m bsve.validation.validate_join \
    --surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
    --dataset data/output/1.5.1/master_research_dataset_core.csv
```

### Purpose

This utility verifies that the Behavioral Surface preserves the original research observations before downstream analysis.

The validator automatically detects the ontology's participating currency pairs, filters the master dataset accordingly, and verifies:

- identical row counts,
- duplicate `(timestamp, pair)` keys,
- identical key sets,
- one-to-one join cardinality,
- pair frequencies,
- crowd-side consistency.

The script aborts immediately if any validation fails.

This validation should be run before outcome labeling.

---

## Behavioral Outcome Labeling

Attach crowd-relative outcome labels to a Behavioral Surface.

```bash
python -m bsve.validation.label_outcomes \
    --surface bsve.test/behavioral_surface_reactive_jpy_1.0.0.parquet \
    --dataset data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/labeled \
    --dataset-version 1.5.1 \
    --horizon 24
```

### Purpose

Outcome labeling combines a validated Behavioral Surface with the originating master research dataset and computes ontology-independent outcome labels for downstream statistical validation.

The utility:

- validates the one-to-one join,
- retrieves the requested forward-return horizon,
- computes

```
future_return
crowd_relative_return
crowd_failed
```

- preserves the behavioral state assignments,
- exports a canonical labeled Behavioral Surface together with a provenance manifest.

The implementation is ontology-agnostic and accepts any Behavioral Surface conforming to the standard BSVE schema.

### Outputs

```
behavioral_surface_<ontology>_labeled_<horizon>b.parquet

behavioral_surface_labels_manifest.json
```

### Console summary

The utility reports:

- state frequencies,
- crowd-failure frequency by behavioral state,
- overall crowd-failure rate,
- pair frequencies,
- confirmation that join validation succeeded.

The labeled Behavioral Surface becomes the canonical input to the statistical validation stage.

---

## Behavioral Validation

Run the predefined statistical validation on a labeled Behavioral Surface.

```bash
python -m bsve.validation.behavioral_validation \
    --labeled-surface bsve.test/labeled/behavioral_surface_reactive_jpy_1.0.0_labeled_24b.parquet \
    --output-dir bsve.test/analysis
```

### Purpose

This utility performs the primary statistical validation defined by the BSVE
validation specification.

The implementation is ontology-agnostic. It consumes a canonical labeled
Behavioral Surface and compares the predefined behavioral states using
contingency analysis.

For the Reactive-JPY ontology the default comparison is:

- `JPY_CONSENSUS_YOUNG`
- `JPY_CONSENSUS_MATURING`

Alternative state names may be supplied through:

```
--reference-state
--target-state
```

making the utility reusable for future ontologies without code changes.

### Statistical outputs

The pooled analysis reports:

- contingency table,
- Reference failure rate,
- Target failure rate,
- failure-rate difference,
- relative risk,
- odds ratio,
- one-sided Fisher exact test.

The analysis is then repeated independently for each currency pair.

### Validation criteria

The implementation evaluates the frozen BSVE validation criteria exactly as
specified in `VALIDATION_SPEC_JPY.md`.

The report records:

- Criterion 1
- Criterion 2
- Criterion 3
- Criterion 4

and emits one of:

```
CONFIRMED
INCONCLUSIVE
NOT_CONFIRMED
```

### Outputs

```
bsve.test/
└── analysis/
    ├── behavioral_validation_report.json
    ├── behavioral_validation_pooled.csv
    └── behavioral_validation_pairs.csv
```

### Console summary

The console report displays:

- pooled contingency table,
- pooled failure rates,
- effect sizes,
- Fisher p-value,
- pair decomposition,
- validation criteria,
- final validation outcome.

This utility performs the primary statistical analysis of the frozen behavioral
hypothesis and should only be run after the Behavioral Surface has passed
inspection, calibration-drift assessment, join validation and outcome
labeling.

---

## Criterion Validation

Validate Criterion 1 (Behavioral Differentiation) using the Reactive-JPY
state-surface artifact and precomputed independent outcomes.

```bash
python -m bsve.validation.criterion1 \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --output-dir bsve.test/
```

**Required inputs**

| Input | Description |
| ----- | ----------- |
| `--artifact` | Path to `bsve_states_reactive_jpy_1.0.0.parquet` |
| `--output-dir` | Directory containing `independent_outcomes.json` and where `bsve_validation_report.json` will be written |
| `--independent-outcomes` | Optional explicit path to `independent_outcomes.json` |

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
- independent outcome distribution
- outcome tests (independent behavioral evidence)
- descriptive diagnostics
- validation outcome

**Report sections**

| Section | Description |
|---------|-------------|
| `metadata` | Criterion name, generation timestamp, thresholds, and independent-evidence availability metadata |
| `state_frequencies` | Observation counts per state and per pair |
| `duration_statistics` | Median, mean, P25/P75, and max episode durations per state |
| `duration_ks_diagnostics` | KS-test results for duration distributions (calibration-consistency diagnostics only) |
| `survival_analysis` | Fraction of episodes surviving past 8, 24, and 48 bars per state |
| `transition_frequencies` | Counts of entry, continuation, exit_reversal, and exit_unknown events per state |
| `independent_outcome_distribution` | Per consensus state SUCCESS/FAILURE distribution from independent outcomes |
| `outcome_tests` | Pairwise Fisher's exact tests over independent SUCCESS rates, with Cohen's h effect size |
| `independent_behavioral_evidence` | Availability, sample sufficiency, significance, and effect-size gating status |
| `descriptive_diagnostics` | Duration-derived diagnostics reported for ontology interpretation only |
| `validation_outcome` | Final PASS / FAIL / INCONCLUSIVE verdict and supporting counts |

**Status interpretation**

Criterion 1 compares state membership against independent outcome distributions
for the consensus states (`YOUNG`, `MATURING`, `MATURE`) using Fisher's exact
tests and Cohen's h effect size.

Criterion 1 emits one of:

- `FAIL`: insufficient state observations or insufficient independent outcome samples
- `INCONCLUSIVE`: independent outcomes exist but no significant differentiation or effect size below threshold
- `PASS`: minimum samples met, independent outcomes available, significant differentiation observed, and effect size threshold satisfied

Duration KS diagnostics and legacy duration-derived outcome diagnostics remain
descriptive and are not used as Criterion 1 behavioral evidence.

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

## Independent Outcome Labeling (Legacy Research Utility)

Generate independent episode-level outcome labels for the original Criterion 1 behavioral validation workflow.

```bash
python -m bsve.validation.outcome_labeling \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/
```

### Purpose

This utility implements the original independent-outcome methodology developed during early BSVE research.

Rather than attaching observation-level crowd-relative returns to a Behavioral Surface, it computes fixed-horizon episode outcomes (`SUCCESS` / `FAILURE`) suitable for the original Criterion 1 validation framework.

It remains part of BSVE because:

- it reproduces the historical validation methodology,
- it supports comparison with earlier research results,
- it remains useful for ontology-level behavioral studies.

### Relationship to `label_outcomes`

BSVE now provides two complementary outcome-labeling utilities:

| Utility                            | Purpose                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| `bsve.validation.label_outcomes`   | Canonical observation-level labeling used for Behavioral Surface validation and contingency analysis. |
| `bsve.validation.outcome_labeling` | Legacy episode-level independent outcome labeling used by the original Criterion 1 workflow. |

For new Behavioral Surface validation work, `label_outcomes` is the recommended utility.

`outcome_labeling` remains available for backwards compatibility and historical Criterion 1 analyses.

### Criterion 1 outcome definition

Outcome labels are:

```
SUCCESS
FAILURE
```

based on:

```python
abs(forward_return) >= volatility_threshold
```

where:

```
volatility_threshold = vol_48b  (default)
```

Key properties of this definition:

- **Direction-agnostic**: either sufficiently large up or down follow-through
  is classified as SUCCESS. Direction is intentionally ignored in the current
  Reactive-JPY implementation. The objective is detecting whether maturity
  states lead to different *magnitudes* of post-episode behavior.
- **Maturity-duration-independent**: outcome labeling uses only the episode
  end time and a fixed forward window. It does not depend on maturity duration
  boundaries.
- **Ontology-independent**: outcome labeling does not depend on state
  assignment logic; it joins only on `(pair, episode_end_time)`.
- **Unit-consistent**: `forward_return` and `vol_48b` are both expressed in
  return units, so no conversion is required.

Directional outcome variants remain a possible future enhancement.

**Unit convention**

- `forward_return` is stored as a fraction: `future_close / close - 1`
- `vol_48b` is expressed in return units (e.g. `0.01` means 1% volatility)
- `success_threshold = abs(vol_48b)` — no conversion required
- Classification rule: `SUCCESS` if `abs(forward_return) >= success_threshold`,
  else `FAILURE`

**Generated outputs**

```
bsve.test/
└── independent_outcomes.json
```

`independent_outcomes.json` includes:

- metadata describing fixed-horizon labeling and the threshold column used
- summary counts (total episodes, evaluable episodes, SUCCESS/FAILURE, success_rate)
- episode-level independent outcomes for consensus states

**Console output**

```
[BSVE] Independent Outcome Labeling (Reactive-JPY)
------------------------------------------------------------
Threshold column:   vol_48b
Consensus episodes: N
Evaluable episodes: N
SUCCESS: N
FAILURE: N
Success rate:       XX.X%
Output: bsve.test/independent_outcomes.json
```

A healthy label distribution avoids extreme imbalance (e.g. ~0% or ~100%
success rate). Inspect `success_rate` immediately after generation.

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

The standard BSVE behavioral-validation workflow is:

```text
Calibration
    ↓
Behavioral Surface Generator
    ↓
Behavioral Surface Inspection
    ↓
Calibration Drift
    ↓
Join Validation
    ↓
Behavioral Outcome Labeling
    ↓
Behavioral Validation
    ↓
Research Interpretation
```

Each stage validates the previous stage before additional information is introduced.

The recommended execution order is therefore:

1. Run the calibration (or use an existing frozen calibration artifact).
2. Extract the frozen validation windows.
3. Generate the Development and OOS Behavioral Surfaces.
4. Inspect the OOS Behavioral Surface.
5. Compare Development and OOS Behavioral Surfaces using the calibration drift utility.
6. Validate one-to-one alignment with the OOS master research dataset.
7. Attach behavioral outcome labels.
8. Run the Behavioral Validation utility.
9. Interpret the statistical results in the context of the inspection report, calibration-drift report and sentinel checks.

This staged workflow ensures that engineering, data integrity and statistical validation remain methodologically independent.

---

## Future Commands

The following commands are planned and will be documented here as they are
implemented.

| Command | Purpose |
|---------|---------|
| `python -m bsve.timeline.generate` | Generate behavioral state timelines from state assignments |
| `python -m bsve.ontology.validate` | Validate ontology YAML definitions against the BSVE schema |
