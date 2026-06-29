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
✓ Behavioral Surface generation
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

## Behavioral Surface Generation

Generate the deterministic, causal Behavioral Surface from a frozen calibration artifact.

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

## Independent Outcome Labeling

Generate ontology-independent, episode-level outcome labels from fixed
post-state market behavior.

```bash
python -m bsve.validation.outcome_labeling \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/
```

**Purpose**

Independent outcomes are required for Criterion 1 PASS eligibility. Labels are
computed from realized forward returns over a fixed 24-bar H1 window after
episode termination and do not depend on maturity duration boundaries.

**Additional options**

| Flag | Default | Description |
|------|---------|-------------|
| `--outcome-window-bars` | `24` | Fixed forward horizon (H1 bars) used to label outcomes |
| `--threshold-column` | `vol_48b` | Dataset column used as the volatility threshold (must be in return units) |

**Threshold column**

The default threshold column is `vol_48b`, which is present in all BSVE-format
datasets (v1.5.1+), is strictly backward-looking, and is expressed in return
units — consistent with `forward_return`.

To use a different column from the same dataset:

```bash
python -m bsve.validation.outcome_labeling \
    --artifact bsve.test/bsve_states_reactive_jpy_1.0.0.parquet \
    --dataset-path data/output/1.5.1/master_research_dataset_core.csv \
    --output-dir bsve.test/ \
    --threshold-column vol_12b
```

The selected threshold column is validated for existence and surfaced in the
output metadata under `threshold_column`.

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
Behavioral Surface Generation
  ↓
Independent Outcome Labeling
  ↓
Criterion Validation
  ├─ Independent Behavioral Evidence
  └─ Descriptive Diagnostics
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
