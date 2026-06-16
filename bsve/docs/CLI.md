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
✓ Threshold-exit labeling and episode outcome classification
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
| `diagnostics.json`                       | State counts, episode statistics, survival counts, maturity sparsity diagnostics, and outcome distribution |
| `run_manifest.json`                      | Provenance record linking dataset version, ontology, calibration artifact, and run ID |

**Validation behavior**

Before writing artifacts the state machine performs:

* calibration artifact validation
* ontology validation
* state artifact validation
* uniqueness checks on `(pair, environment_id, entry_time)`

State assignment fails fast if any contract violation is detected.

---

## State Assignment Outputs

### `transition_event` meanings

Each row in the state-surface parquet has a `transition_event` column.
Values and their semantics:

| Event | Description |
|-------|-------------|
| `entry` | First bar of a new extreme-consensus episode, or the first bar in the dataset for a pair. |
| `continuation` | State and extreme condition are unchanged from the previous bar. |
| `exit_threshold` | Recorded on the first bar of a new maturity class (Young → Maturing or Maturing → Mature). Also used as the episode outcome label when a consensus episode reaches the mature state and terminates within a normal mature lifecycle (`mature_boundary <= max_maturity < 2 * mature_boundary`). |
| `exit_reversal` | Terminal bar of a consensus episode that ended before crossing the mature boundary (`max_maturity < mature_boundary`). Consensus failed before maturing. |
| `exit_late_reversal` | Terminal bar of a consensus episode that survived well beyond the mature boundary (`max_maturity >= 2 * mature_boundary`) and then collapsed. Mature consensus eventually failed rather than dissipating normally. |
| `exit_unknown` | Fallback; used only when the termination reason cannot be determined. Should remain rare. |

### Episode outcome labeling

After bar-level state assignment, the state machine applies a deterministic
episode outcome classifier to every consensus episode (contiguous run of
extreme-consensus bars within a pair).

The classifier computes `max_maturity_bars` for the episode and assigns an
outcome type based on calibrated boundaries:

```
max_maturity < mature_boundary             → exit_reversal
mature_boundary ≤ max_maturity < 2×mature  → exit_threshold
max_maturity ≥ 2×mature_boundary           → exit_late_reversal
```

The outcome label overwrites the `transition_event` on the **last bar** of
the episode only.  All non-terminal bars keep their bar-level events.

This labeling is performed **after** state assignment and is independent of
maturity classification.  State assignment can be audited separately.

### Diagnostics interpretation

`diagnostics.json` includes an `outcome_distribution` section:

```json
{
  "outcome_distribution": {
    "exit_reversal":      412,
    "exit_threshold":      87,
    "exit_late_reversal":  21,
    "exit_unknown":         3
  },
  "outcome_distribution_per_pair": {
    "usd-jpy": { "exit_reversal": 120, ... },
    "eur-jpy": { "exit_reversal": 140, ... },
    ...
  }
}
```

- `exit_reversal`: count of consensus episodes that terminated before reaching the mature state.
- `exit_threshold`: count of episodes that reached the mature state and completed a normal lifecycle.
- `exit_late_reversal`: count of long-lived mature episodes that eventually collapsed.
- `exit_unknown`: count of episodes that could not be classified (should be near zero).

The console also prints an Outcome Distribution table after each run:

```
Outcome Distribution
  exit_reversal            412
  exit_threshold            87
  exit_late_reversal        21
  exit_unknown               3
```

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
- behavioral outcomes (per-state outcome distributions and reversal rates)
- behavioral tests (Fisher's exact test results with Cohen's h effect sizes)
- validation outcome

**Report sections**

| Section | Description |
|---------|-------------|
| `metadata` | Criterion name, generation timestamp, thresholds, behavioral evidence flags |
| `state_frequencies` | Observation counts per state and per pair |
| `duration_statistics` | Median, mean, P25/P75, and max episode durations per state |
| `duration_ks_diagnostics` | KS-test results for duration distributions (calibration-consistency diagnostics only) |
| `survival_analysis` | Fraction of episodes surviving past 8, 24, and 48 bars per state |
| `transition_frequencies` | Counts of all transition events (`entry`, `continuation`, `exit_reversal`, `exit_threshold`, `exit_late_reversal`, `exit_unknown`) per state |
| `behavioral_outcomes` | Per consensus state: episode count, outcome type counts and rates, full `outcome_distribution` dict |
| `behavioral_tests` | Pairwise Fisher's exact test results with Cohen's h effect sizes |
| `validation_outcome` | Final PASS / FAIL / INCONCLUSIVE verdict and supporting counts |

### Criterion 1 behavioral evidence

Criterion 1 derives behavioral evidence from **independently labeled episode
outcomes** produced by the episode outcome classifier.  The behavioral analysis:

1. Groups bars into state-level episodes (consecutive bars with the same `state_id`).
2. Reads the terminal bar's `transition_event` for each episode as the outcome label.
3. Computes per-state outcome distributions:
   - `exit_reversal` count and rate (episode failed before or during maturation)
   - `exit_threshold` count (episode completed a normal mature lifecycle)
   - `exit_late_reversal` count (mature consensus eventually collapsed)
4. Runs pairwise Fisher's exact tests on reversal rates between:
   - `YOUNG` vs `MATURING`
   - `YOUNG` vs `MATURE`
   - `MATURING` vs `MATURE`
5. Uses Cohen's h as the effect size metric.

Because exit events are now placed on the **terminal bar of each consensus
episode** (not on the first non-extreme bar), the reversal rates computed for
each maturity state reflect genuine within-state behavioral differences.

**Status interpretation**

Criterion 1 emits one of:

- `PASS`: behavioral differentiation evidence is present, the effect size meets the minimum threshold (`MIN_BEHAVIORAL_EFFECT_SIZE = 0.10`), and quality gates pass
- `FAIL`: quality gates fail (for example insufficient observations)
- `INCONCLUSIVE`: behavioral differentiation is not established — either only duration-derived diagnostics are available, behavioral evidence is present but the effect size is below `0.10`, or no effect size was supplied

A statistically significant p-value alone is not sufficient for `PASS`.  The effect
size must also meet or exceed `MIN_BEHAVIORAL_EFFECT_SIZE = 0.10` to ensure that
the observed behavioral differentiation is practically meaningful, not merely a
product of a large sample.

For Reactive-JPY Criterion 1, duration KS diagnostics are reported as
calibration-consistency checks and are **not** treated as evidence of behavioral
differentiation. Therefore, runs with sufficient samples but only duration-derived
evidence are marked `INCONCLUSIVE`.

**Progression analysis**

Criterion 1 reports KS-test comparisons for the three consensus-state transitions:

- `YOUNG → MATURING`
- `YOUNG → MATURE`
- `MATURING → MATURE`

These results are **descriptive only**.  They are included in the validation report
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
  ├─ Bar-level state and transition classification
  └─ Episode outcome labeling (exit_reversal / exit_threshold / exit_late_reversal)
  ↓
Criterion Validation
  ├─ Duration Diagnostics
  ├─ Outcome Distribution
  └─ Behavioral Outcome Validation (Fisher's exact, Cohen's h)
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
