# BSVE Concept Draft
## Behavioral State Validation Engine — Design Document
### Status: Draft (pre-implementation)
### Target repo: market-sentiment-ml (MSML)
### Last updated: 2026-06-15 (revised **Implementation Strategy** and Implementation Checklist to reflect the reality of PR1-PR3, updated Ambiguous state assignments)

---

## Purpose

The Behavioral State Validation Engine (BSVE) is a subsystem of MSML that formalizes, calibrates, validates, and compares behavioral ontologies for FX pair families and subfamilies.

BSVE is not designed specifically for JPY or CHF environments. Rather, it provides a general framework for testing whether proposed behavioral state definitions:

- produce meaningful differentiation of downstream behavior,
- remain stable through time,
- exhibit family-specific organization,
- provide explanatory value beyond structural market features.

JPY and CHF behavioral ontologies are the first environments studied under this framework and serve as initial validation targets rather than special cases.

It answers a specific research question:

> Do sentiment-derived behavioral state definitions produce statistically
> significant, family-specific, and structurally non-redundant
> differentiation of downstream market behavior?

BSVE is not a trading system and not a prediction pipeline. It is a
validation framework whose outputs are versioned state surface artifacts
consumed downstream by MPML.

---

## Motivation

Previous MSML research established that:

- Pair families exhibit meaningfully different behavioral organization
- Persistent and Reactive subfamilies differ in their organizing mechanisms
- Within Reactive, CHF and JPY appear organized by distinct processes:
  - CHF: volatility context → crowd-state persistence → predictive structure
  - JPY: consensus formation → consensus maturation → consensus decay
- Consensus maturity is a core state variable for JPY environments
- These findings have not yet been validated by walk-forward runs

BSVE provides the infrastructure to validate these findings rigorously
before they are used as the basis for MPML routing decisions or
downstream trading logic.

---

## Ontology Ownership Hypothesis

An emerging hypothesis from recent MSML research is that different currency-family environments may not share a common optimal behavioral ontology.

Historically, MPML assumed a universal:

Trend × Volatility

regime structure.

This ontology has proven useful and remains the strongest currently validated regime definition.

However, subsequent investigations suggest that different pair families may organize behavioral information through different state variables.

The purpose of BSVE is therefore not merely to validate candidate ontologies, but also to determine whether specific ontologies exhibit family ownership.

In this context, ownership does not imply exclusivity.

Rather, it refers to the possibility that a particular ontology provides a more natural or information-efficient representation of a given behavioral environment.

Current candidate ownerships are:

Persistent Families
→ Trend × Volatility

Reactive-CHF Families
→ Persistence × Volatility (candidate)

Reactive-JPY Families
→ Consensus-State Geometry (candidate)

These mappings are working hypotheses rather than validated facts.

BSVE exists specifically to test whether such ownership relationships are real, transferable, stable through time, and meaningfully distinct from alternative ontologies.

The long-term objective is not to prove family-specific ownership, but to determine whether different behavioral environments are most naturally represented by different state spaces.

---

## Scope

BSVE v1 ships with two reference environments:

- Reactive-JPY subfamily (USDJPY, EURJPY, GBPJPY)
- Reactive-CHF subfamily (USDCHF, EURCHF)

The Persistent family (EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD) is
already validated through walk-forward runs in MPML and does not
require BSVE validation at this stage.

Framework components must remain environment-agnostic.

---

## Design Decisions

## Framework vs Ontology Separation

BSVE consists of two distinct layers:

### Layer 1 — Framework

The framework provides reusable infrastructure for:

- calibration
- state assignment
- artifact generation
- validation
- transfer analysis
- MPML integration

Framework components must remain environment-agnostic.

Framework code must not contain environment-specific logic such as:

```python
if environment == "reactive_jpy":
```

or

```python
if environment == "reactive_chf":
```

Environment behavior must instead be defined through versioned specifications and calibration artifacts.

### Layer 2 — Ontologies

Behavioral ontologies are environment-specific hypotheses implemented on top of the framework.

Examples:

- Reactive-JPY Consensus Maturity
- Reactive-CHF Volatility-Conditioned Persistence

BSVE v1 ships with these as reference ontologies.

The framework must assume that future ontologies may be added without modification to framework code.

The purpose of the framework is therefore not to validate a particular ontology, but to provide a reproducible process for validating any ontology that satisfies the BSVE artifact contract.

### Option A: Rule-Based State Machine

BSVE v1 uses a rule-based state machine (Option A). State definitions
are specified as versioned YAML configuration files. No learned
boundaries are used in v1.

Rationale:
- Strong prior hypotheses exist from accumulated research
- Learned boundaries (Option B, e.g. HMM) introduce an additional
  training/validation split and new leakage vectors
- Option A produces fully auditable, reproducible state assignments
- Option B is a natural extension once Option A produces validated
  artifacts and is already listed in the MSML research roadmap

### State Machine Design Principle

The state machine is not responsible for discovering behavioral structure.

Behavioral structure is discovered during calibration and encoded in
committed calibration artifacts.

The state machine is responsible only for:

- loading calibrated thresholds,
- applying state definitions deterministically,
- recording transitions,
- producing reproducible state surface artifacts.

No threshold estimation, optimization, or calibration logic may exist
inside the state assignment layer.

### Timeframe: H1

BSVE operates entirely at H1 resolution.

Rationale:
- Sentiment snapshot data is H1 (scraped at 08:00, 12:00, 18:00 UTC)
- Consensus maturity is a count of elapsed H1 bars — intrinsically H1
- Volatility regime transitions in CHF are H1-scale processes
- Aggregating to D1 before studying these dynamics would destroy
  the transition geometry that is the primary object of study
- MPML consumes BSVE artifacts via a defined D1 aggregation contract
  (see Artifact Contract section)

### Sentiment Timestamp Normalization

Sentiment snapshots are scraped at fixed times but written with
variable wall-clock delay (e.g. 08:03, 08:04, 08:05 due to server
load). A snapshot scraped near 08:00 belongs to the 08:00 H1 bar,
not the 09:00 bar.

Normalization rule (applied at ingestion, not at feature engineering):

```python
normalized_bar = scraped_at.floor("H")
```

This is a versioned pipeline step. All BSVE runs must use normalized timestamps. The causal boundary rule follows the existing DL artifact contract:

> A sentiment value belonging to bar T is available at bar T open. It may inform predictions for bar T+1 and beyond. It must NOT be used to predict the outcome of bar T itself.

# Master Research Dataset Contract

BSVE follows the same layered data philosophy established in MSML.

The Master Research Dataset remains immutable and contains only raw observations and universally applicable derived features.

The master dataset must not contain:

- behavioral state labels
- maturity classifications
- ontology-specific regime assignments
- BSVE calibration outputs
- family-specific annotations

All BSVE-derived information is generated in downstream layers.

This separation ensures:

- reproducibility
- ontology independence
- support for multiple competing behavioral definitions
- future recalibration without modifying historical datasets

BSVE outputs are therefore treated as derived feature surfaces rather than master dataset features.

## Dataset Adapter Layer

BSVE operates on the Master Research Dataset but must not modify it.

To enforce this separation, BSVE introduces a dedicated adapter layer between the dataset and ontology logic.

### Responsibilities

The adapter layer:

- loads master dataset artifacts
- exposes normalized feature access
- exposes pair-family membership information
- exposes sentiment observations
- exposes structural features
- exposes ontology feature registry functions

### Non-Responsibilities

The adapter layer must not:

- assign behavioral states
- derive ontology labels
- persist BSVE artifacts
- contain environment-specific logic

### Rationale

Calibration scripts, state machines, and validation routines should all consume the same adapter interfaces.

This avoids duplicated feature engineering logic and ensures that ontology definitions remain independent from underlying dataset storage details.

All BSVE components should consume data through the adapter layer rather than reading raw dataset structures directly.

------

## Behavioral Feature Registry

Behavioral ontologies often rely on derived quantities that are reused across multiple environments.

Examples include:

- consensus maturity
- persistence duration
- volatility regime
- consensus velocity
- saturation metrics
- transition geometry metrics

To prevent duplicated implementations, BSVE maintains a centralized feature registry.

### Repository Layout

```text
bsve/
└── features/
    ├── consensus.py
    ├── persistence.py
    ├── volatility.py
    └── registry.py
```

### Design Principle

Behavioral features are computed once and reused throughout the system.

Calibration scripts, state machines, and validation routines must consume the same feature implementations.

This creates a single source of truth for behavioral feature definitions and prevents calibration and validation pipelines from diverging over time.

### Example Functions

```python
compute_consensus_maturity(...)
compute_persistence_duration(...)
compute_volatility_regime(...)
compute_consensus_velocity(...)
```

The registry is ontology-independent and should support future environments without modification.

---

## Repository Layout (Target After PR5)

Repository Layout (Target after PR5)

```
bsve/
├── calibration/
│   ├── jpy_maturity_calibration.py   # JPY hazard analysis + boundary derivation
│   ├── chf_vol_calibration.py        # CHF vol regime boundary derivation
│   └── validate_calibrations.py      # Sign-off gate (blocks invalid runs)
├── calibration_artifacts/                      # Committed calibration artifacts
│   ├── reactive_jpy_calibration_v1.json
│   ├── reactive_chf_calibration_v1.json
│   └── plots/                         # Diagnostic plots for sign-off review
│       ├── jpy_hazard_curve.png
│       ├── chf_atr_distribution.png
│       └── chf_persistence_by_regime.png
├── state_specs/                       # Versioned environment + state definitions
│   ├── reactive_jpy_v1.yaml
│   └── reactive_chf_v1.yaml
├── state_machine/
│   └── rule_based.py                  # State machine engine, implemented (PR4)
├── validation/
│   └── criterion_tests.py             # Three-part validation criterion tests
├── artifacts/                         # Output state surface artifacts
│   └── <run_id>/
│       └── bsve_states_<pair>_<env>_<version>.parquet
└── bsve_config_v1.yaml                # Top-level BSVE configuration
```

------

## Environment Specifications

Each environment is defined in a versioned YAML file under `bsve/state_specs/`. The spec defines:

- Environment metadata (pairs, timeframe, sentiment requirements)
- Dataset windows (full range and DL-active range)
- Feature sets (structural, sentiment, state-derived)
- Named states with entry/exit conditions and behavioral hypotheses
- Validation criteria (three-part, see below)
- Open questions

### Current Environments

| Environment  | File                 | Subfamily | Pairs                  | Organizing Mechanism                                    |
| ------------ | -------------------- | --------- | ---------------------- | ------------------------------------------------------- |
| reactive_jpy | reactive_jpy_v1.yaml | JPY       | USDJPY, EURJPY, GBPJPY | Consensus-State Geometry (maturity currently validated) |
| reactive_chf | reactive_chf_v1.yaml | CHF       | USDCHF, EURCHF         | Volatility-Conditioned Persistence (leading hypothesis) |

### JPY States

Reactive-JPY is currently modeled using a Consensus-State Geometry ontology.

The first validated dimension of this ontology is consensus maturity.

Recent investigations suggest that reversal probability depends strongly on whether a consensus state is:

- newly formed,
- developing,
- or fully established.

Additional dimensions may be introduced in future ontology versions, including:

- consensus saturation,
- consensus velocity,
- consensus transition geometry,
- hidden-state representations.

BSVE v1 intentionally restricts the ontology to the maturity dimension in order to validate the simplest explanatory model first.

| State ID               | Entry Condition                       | Behavioral Hypothesis    | Confidence |
| ---------------------- | ------------------------------------- | ------------------------ | ---------- |
| JPY_CONSENSUS_YOUNG    | extreme + maturity < young_boundary   | High reversal risk       | High       |
| JPY_CONSENSUS_MATURING | extreme + maturity in [young, mature) | Transitional             | Medium     |
| JPY_CONSENSUS_MATURE   | extreme + maturity >= mature_boundary | Threshold-dominated exit | High       |
| JPY_NON_EXTREME        | sentiment not extreme                 | Structural baseline      | N/A        |

### CHF States

Reactive-CHF is currently modeled using a Volatility-Conditioned Persistence ontology.

This ontology remains a leading hypothesis rather than a validated behavioral mechanism.

Existing research suggests that crowd persistence behavior varies meaningfully across volatility environments and that volatility context may influence the stability of sentiment states.

BSVE v1 tests this hypothesis directly.

Future ontology revisions may expand the representation if additional explanatory variables emerge.

| State ID                    | Entry Condition              | Behavioral Hypothesis            | Confidence  |
| --------------------------- | ---------------------------- | -------------------------------- | ----------- |
| CHF_LOW_VOL_PERSISTENT      | low vol + extreme sentiment  | High persistence, high info gain | Medium-High |
| CHF_LOW_VOL_NON_PERSISTENT  | low vol + non-extreme        | Control state                    | Low         |
| CHF_ELEVATED_VOL_REACTIVE   | high vol + extreme sentiment | Release/reversion dynamics       | Medium-High |
| CHF_MEDIUM_VOL_TRANSITIONAL | medium vol                   | Boundary/control state           | Low         |
| CHF_NON_EXTREME             | non-extreme sentiment        | Structural baseline              | N/A         |

------

## Calibration

All threshold values in environment specs are initialized as placeholders. They must be replaced with empirically derived values before any validation run is permitted to proceed.

### Scientific Null Outcomes

Calibration is permitted to conclude that no stable behavioral ontology exists for an environment.

Examples:

- no significant maturity structure
- no meaningful volatility-conditioned persistence
- unstable thresholds
- insufficient transfer degradation

These outcomes are scientifically valid and should be documented rather than forced into an ontology definition.

The framework must not assume that every family possesses a useful behavioral partition.

### Null Ontology Artifacts

A calibration process may conclude that no stable behavioral ontology exists for a candidate environment.

Such outcomes are scientifically valid and must be represented explicitly rather than by missing files.

The calibration framework should therefore support versioned null-calibration artifacts.

Example outcomes:

- no significant maturity structure
- unstable thresholds
- insufficient transfer degradation
- no reproducible behavioral differentiation

A null-calibration artifact records that calibration was attempted and failed validation criteria.

This allows downstream tooling to distinguish between:

- ontology never evaluated
- ontology evaluated and rejected

Null-calibration artifacts participate in the same provenance and versioning system as successful calibration artifacts.

### JPY Calibration: Hazard Analysis

**Script:** `bsve/calibration/jpy_maturity_calibration.py`

**Method:** Discrete survival analysis on consensus state lifecycles.

**Steps:**

1. Pool sentiment data across JPY pairs (DL-active window)
2. Derive extreme sentiment threshold from empirical distribution (70th percentile of |sentiment_net|)
3. Extract consensus state episodes (entry → exit or right-censoring)
4. Compute empirical reversal hazard rate as function of maturity
5. Find hazard crossover bar (inflection point after initial high-hazard zone)
6. Derive young and mature boundaries from crossover (rounded to nearest 8-bar session boundary)
7. Compute diagnostic statistics
8. Write versioned calibration artifact with integrity hash

**Primary outputs:**

- `extreme_threshold_net_pct`
- `young_boundary_bars`
- `mature_boundary_bars`

### JPY Calibration Sign-Off Conditions

**Tier 1 — Testable from hazard analysis (current):**
- hazard_crossover_bar is identifiable (not null)
- young_boundary_bars < mature_boundary_bars
- n_episodes_total >= 50
- censoring_rate < 0.30
- Hazard rate in bars 1-6 materially exceeds hazard rate in bars 13+
  (visual confirmation from hazard curve plot)
- Survival curve drops below 0.50 before young_boundary_bars
  (confirms front-loaded reversal risk)

**Tier 2 — Requires exit type labeling (deferred to PR4/state machine):**
- reversal_rate_young > reversal_rate_mature
- reversal_rate_young > 0.15

Note: The Tier 2 conditions are the primary scientific validation of
the maturity hypothesis. They cannot be tested until the state machine
labels threshold exits separately from sentiment resets. The current
artifact correctly records all exits as reversals because threshold
exit labeling is not yet implemented.

**Diagnostic outputs (for visual review):**

- Hazard curve plot with boundary annotations
- Kaplan-Meier survival curve

### CHF Calibration: Volatility Regime Boundaries

**Script:** `bsve/calibration/chf_vol_calibration.py`

**Method:** Two-stage: Jenks natural breaks on ATR% distribution, validated by behavioral significance test on crowd persistence.

**Steps:**

1. Pool ATR% data across CHF pairs (DL-active window)
2. Derive extreme sentiment threshold (shared with JPY method)
3. Find candidate vol boundaries via Jenks natural breaks (cross-checked against percentile method)
4. Measure crowd persistence duration stratified by vol regime
5. Test behavioral significance of boundaries (KS test + Cohen's d
   - persistence ratio — all three must pass)
6. Derive per-pair thresholds for internal coherence check
7. Write versioned calibration artifact with integrity hash

**Primary outputs:**

- `low_vol_threshold_atr_pct`
- `high_vol_threshold_atr_pct`

**Sign-off conditions (all must pass):**

- `low_vol_threshold < high_vol_threshold`
- `median_persistence_low_vol > median_persistence_high_vol`
- `persistence_ratio >= 1.25`
- `ks_pvalue_low_vs_high < 0.05`
- `abs(effect_size_low_vs_high) >= 0.3`
- `pair_threshold_agreement == True`
- No single regime captures > 70% of bars

**Special consideration — SNB era:** The SNB floor removal on 2015-01-15 created a structural break in EURCHF volatility. The calibration script flags this automatically. If the calibration window includes pre-2015 data, vol regime boundaries should be reviewed carefully or the window should be restricted to post-2015.

**Diagnostic outputs (for visual review):**

- ATR% distribution with regime boundaries overlaid
- Crowd persistence distributions per vol regime

### Calibration Artifact Schema

Both calibration tools write a versioned JSON artifact.

Example:

{
  "schema_version": "1.0.0",
  "calibration_id": "reactive_jpy_calibration_v1",
  ...
}

The schema_version field identifies the calibration artifact schema
used to serialize the artifact.

Changes to artifact structure require a schema version increment.
Threshold changes alone do not require a schema version increment.

- All threshold fields are non-null (no placeholders)
- A deterministic SHA-256 hash covers all fields except the hash itself
- The hash is verified by the validation gate before any run proceeds
- The artifact is immutable after writing — re-running calibration produces a new versioned artifact, never overwrites an existing one

Artifacts are stored in `bsve/calibration_artifacts/` and referenced by `calibration_id` in BSVE state surface outputs, providing full provenance tracing from output artifact back to calibration inputs.

------

## Validation Gate

**Script:** `bsve/calibration/validate_calibrations.py`

The validation gate is a hard prerequisite for any BSVE validation run. It:

1. Loads committed calibration artifacts
2. Verifies SHA-256 hash integrity
3. Confirms all required fields are present and non-null
4. Runs all environment-specific sign-off conditions
5. Raises `RuntimeError` if any check fails, preventing the run

The gate is called via `assert_calibrations_valid()` at the start of every validation run. It returns the raw artifact dicts for threshold injection into the state machine at runtime.

This design prevents the most common form of threshold leakage: running "validation" with thresholds that were tuned on the same window being validated.

------

## Three-Part Validation Criterion

A behavioral state partition is considered validated only if all applicable criteria pass. Partial validation is informative but does not authorize downstream use in MPML routing.

### Criterion 1: Behavioral Differentiation

States produce statistically significant differences in downstream behavioral outcomes within the DL-active window (2019-2026).

The objective is to determine whether the ontology partitions behavior into meaningfully different regimes rather than merely creating arbitrary labels.

For Reactive-JPY, the initial validation focuses on maturity-dependent behavioral structure observable from state lifecycles and persistence characteristics.

Examples include:

- episode duration distributions
- survival distributions
- state transition frequencies
- maturity-dependent persistence behavior

Future ontology versions may additionally incorporate exit-type differentiation once threshold-exit labeling is implemented.

**Reactive-JPY tests (v1):**

- KS test on episode duration distributions across maturity states
- Log-rank test (or equivalent survival comparison) across maturity states
- Comparison of state-transition frequencies
- Minimum observations: 50 per state

**Reactive-CHF tests (v1):**

- KS test on persistence duration distributions across volatility-conditioned states
- Comparison of persistence characteristics across volatility regimes
- Minimum observations: 50 per state

**Common requirements:**

- Window: DL-active range (2019-2026)
- Statistical significance threshold: p < 0.05
- Effect sizes must be reported alongside p-values

Criterion 1 is considered satisfied when the proposed ontology produces reproducible behavioral differentiation between states and the observed differences are both statistically significant and practically meaningful.

### Criterion 2: Family Specialization

Behavioral state definitions should exhibit materially stronger differentiation on their intended family than on unrelated families.

The objective is not complete failure under transfer.

Instead, successful behavioral ontologies should show meaningful degradation when applied outside their intended environment.

Examples:

- JPY definitions should perform best on JPY pairs.
- CHF definitions should perform best on CHF pairs.
- Neither definition should outperform established structural ontologies on Persistent-family pairs.

Transfer performance may remain non-zero and still support specialization.

Validation focuses on relative explanatory power rather than binary success/failure.

The objective is to characterize transfer topology across:

- Persistent
- Reactive-CHF
- Reactive-JPY

rather than simply testing for transfer success or failure.

### Criterion 3: Incremental Explanatory Power

Behavioral differentiation produced by sentiment-derived states should provide additional explanatory value beyond structural features alone.

The goal is not strict independence from trend and volatility.

Some overlap is expected because trader positioning and market structure influence each other.

The criterion evaluates whether behavioral states contribute information that is not fully captured by structural features.

Validation compares:

behavioral surface
vs
structural-only baseline

and measures incremental differentiation.

### Criterion 4: Internal Coherence (CHF only)

USDCHF and EURCHF exhibit consistent behavioral organization under the same state definitions.

- Test: Jensen-Shannon divergence on exit type distributions
- Threshold: JSD < 0.15 (low divergence = consistent behavior)
- Prior: strong, based on existing research showing USDCHF/EURCHF agreement

### Criterion 5: Ontology Stability

A behavioral ontology must remain reasonably stable through time.

State definitions calibrated on one period should continue to produce coherent state distributions and meaningful differentiation when applied to later periods.

Suggested protocol:

Calibration:
2019–2022

Evaluation:
2023–2026

Calibration windows and evaluation windows must never overlap.

Measures:

- state frequency stability
- transition frequency stability
- persistence distribution stability
- effect-size stability

The objective is not identical behavior across eras, but resistance to collapse under moderate regime shifts.

------

## 

### Result Classification

| Result              | Conditions                                 |
| ------------------- | ------------------------------------------ |
| Validated           | All applicable criteria pass               |
| Partially validated | Criterion 1 passes; Criterion 2 or 3 fails |
| Not validated       | Criterion 1 fails                          |

Partially validated results are documented, not discarded. They mean behavioral differentiation exists but is not family-specific or not independent of structure. This is scientifically informative.

------

## Validation Protocol (Step Sequence)

```
Step 1: Threshold Calibration
  - Run jpy_maturity_calibration.py
  - Run chf_vol_calibration.py
  - Review diagnostic plots
  - Commit artifacts to bsve/calibration_artifacts/
  - *** SIGN-OFF REQUIRED before Step 2 ***

Step 2: State Machine Dry Run
  - Run state machine over DL-active window
  - Verify minimum observations per state (>= 50)
  - No ambiguous assignments permitted.
  - Verify state distribution is not degenerate (no state > 95% of bars)
  - *** SIGN-OFF REQUIRED before Step 3 ***

Step 3: Criterion Validation
  - Run criterion_tests.py over DL-active window
  - Criterion 1: behavioral differentiation
  - Criterion 2: family specificity (cross-family transfer)
  - Criterion 3: Incremental Explanatory Power
  - Criterion 4: internal coherence (CHF only)
  - Produce validation report

Step 4: Structural Baseline Comparison
  - Run structural-only feature set over full range (2012-2026)
  - Confirms structural layer behavior across full history
  - Provides context for interpreting DL-active findings

Step 5: Transfer Topology Analysis

* Apply JPY ontology to CHF pairs
* Apply CHF ontology to JPY pairs
* Apply JPY ontology to Persistent-family pairs
* Apply CHF ontology to Persistent-family pairs
* Compare explanatory power across all environments

Expected:

Intended family
> Related family
> Unrelated family

The objective is not binary transfer failure.

The objective is to map transfer topology and evaluate candidate ontology ownership relationships.

Transfer results are treated as first-class scientific outputs.
```

### Transfer Validation as a Core Test

Cross-family transfer analysis is a primary scientific validation tool.

For every behavioral surface:

1. Evaluate on intended family.
2. Evaluate on unrelated families.
3. Measure degradation.
4. Document transfer topology.

Transfer analysis directly tests whether the discovered behavioral organization is:

- family-specific,
- universal,
- or partially shared.

Transfer results should be treated as first-class outputs of BSVE validation runs.

------

## Output Artifact Contract (H1)

BSVE produces versioned parquet artifacts at H1 resolution.
These follow the same contract-first design as the MSML DL artifact
schema (schema v2.0.0) and are validated with fail-fast checks before
every write.

| Column | Type | Causal | Description |
|--------|------|--------|-------------|
| entry_time | timestamp (UTC, tz-naive) | Yes | H1 bar open timestamp |
| prediction_available_timestamp | timestamp (UTC, tz-naive) | Yes | entry_time + 1H — MPML causal boundary |
| pair | str | — | Currency pair (e.g. USDJPY) |
| environment_id | str | — | reactive_jpy or reactive_chf |
| state_id | str | — | Named state from environment spec |
| state_version | str | — | Version of the state definition used |
| maturity_bars | int | — | H1 bars since state entry. 0 on entry bar. |
| maturity_class | str | — | young / maturing / mature / n_a |
| state_confidence | float [0,1] | — | 1.0 for rule-based Option A |
| transition_event | str | — | entry / continuation / exit_reversal / exit_threshold / exit_late_reversal / exit_unknown |
| spec_id | str | — | Environment spec filename + version. Provenance. |
| calibration_id | str | — | Calibration artifact identifier. Prevents silent threshold changes. |

**Fail-fast validation checks (applied before every write):**
- `entry_time < prediction_available_timestamp` (no future leakage)
- No null state_ids
- `maturity_bars >= 0`
- `maturity_class` consistent with `maturity_bars` and environment thresholds
- `spec_id` resolvable to a committed spec file
- `calibration_id` resolvable to a committed calibration artifact
- No ambiguous state assignments

**Artifact naming convention:**
bsve/artifacts/<run_id>/bsve_states_<pair><env_id><spec_version>.parquet

> Note:
> Calibration artifacts use `<ontology_id>_<version>_<date>.json` 
> while state surface artifacts use the parquet naming convention.

---

## MPML Aggregation Contract (H1 → D1)

BSVE artifacts are produced at H1. MPML operates at D1. The
aggregation is a deliberate lossy compression appropriate for
strategy routing but not for state discovery or transition geometry
analysis. All BSVE validation runs operate on H1 artifacts directly.

| Field | D1 Aggregation Rule |
|-------|-------------------|
| state_id | Modal H1 state within D1 bar |
| maturity_bars | max(H1 maturity_bars within D1 bar) |
| state_confidence | mean(H1 state_confidence within D1 bar) |
| transition_event | entry if any H1 bar has entry; else exit_* if any exit; else continuation |
| prediction_available_timestamp | max(H1 prediction_available_timestamps within D1 bar) |

The aggregation method is versioned (`modal_state_max_maturity_v1`).
MPML must record which aggregation version was used in its run
manifest, for the same reason MSML records DL artifact schema version.

---

## Relationship to Existing MSML Infrastructure

BSVE is a subsystem of MSML, not a separate repository. It reuses
existing infrastructure wherever possible.

| Existing component | BSVE usage |
|-------------------|------------|
| DL artifact schema (v2.0.0) | BSVE artifact contract follows same design |
| `prediction_available_timestamp` convention | Adopted unchanged |
| `schemas/` module | BSVE adds `bsve_artifact_schema.py` alongside `dl_artifact_schema.py` |
| `research/data/loader.py` | BSVE calibration scripts use existing data loader |
| Existing feature sets (price_trend, trend_vol_only) | BSVE criterion 3 ablation reuses these |
| Cross-family transfer infrastructure | BSVE criterion 2 reuses existing transfer experiment setup |
| Run manifest pattern | BSVE validation runs emit their own manifest |

BSVE does NOT reuse or depend on:
- MPML walk-forward evaluation infrastructure
- MPML regime ontology
- MPML selector or gating logic

These remain strictly downstream consumers of BSVE artifacts.

### Behavioral Surface Concept

BSVE introduces a new artifact category:

Behavioral Surface

A behavioral surface is the behavioral-state analogue of an MSML feature surface.

Examples:

- jpy_consensus_maturity_v1
- chf_volatility_persistence_v1

A behavioral surface contains:

- state assignments
- maturity information
- transition metadata
- calibration provenance

Behavioral surfaces are versioned, immutable artifacts and may coexist even when they describe the same currency family.

The framework should assume multiple competing behavioral surfaces may exist simultaneously.

---

## Relationship to MPML

BSVE artifacts flow into MPML via the same artifact consumption
pattern as DL prediction surfaces. MPML treats BSVE state surface
artifacts as validated external feature surfaces.

BSVE (H1 state surfaces) ↓ D1 aggregation (modal_state_max_maturity_v1) ↓ MPML feature integration layer ↓ Strategy routing / gating selector ↓ Walk-forward evaluation

**Important separation:**
BSVE validates whether behavioral states exist and are family-specific.
MPML studies whether adaptive systems can exploit those states.
MPML findings should not be interpreted as direct evidence for or
against the intrinsic importance of the behavioral states themselves.
This mirrors the existing MSML/MPML separation documented in
RESEARCH_STATE.md.

---

## Future Extension: Dynamic Surface Selection

The current MPML architecture selects surfaces based on static pair
family membership. A natural extension, motivated by the consensus
maturity finding, is time-dependent surface selection:

Current: pair family → select surface → route strategy Future: current behavioral state (time-indexed) → select surface → route strategy


BSVE lays the groundwork for this by producing time-indexed state
annotations as first-class artifacts. The `state_id` and
`maturity_class` columns in the BSVE artifact are the inputs a future
surface selector meta-layer would consume.

This extension is out of scope for BSVE v1 but should be kept in mind
when designing the MPML integration layer. Specifically:

- Surface selection decisions must be logged as first-class artifacts
- The surface selector needs its own causal boundary, separate from
  both DL surface training and MPML walk-forward folds
- MLflow tracking from day one is recommended when this layer is built
  (targeted for MPEX or MPML v2)

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Threshold tuning on validation window | Calibration artifacts committed before validation gate opens. Hash-verified at runtime. |
| Sentiment timestamp leakage (scrape delay) | floor_to_h1_bar_open normalization at ingestion. Versioned pipeline step. |
| SNB structural break in EURCHF (2015-01-15) | Automatic era flag in CHF calibration. Sign-off review required if pre-2015 data included. |
| Sparse DL coverage (sentiment only from 2019) | Explicit DL-active vs full-range window split. Criterion tests use DL-active window only. |
| Ambiguous state assignments | → Emit explicit UNKNOWN state and warning.<br/>→ Never silently coerce observations into another state.<br />**State assignment between episodes:**<br/>Bars where sentiment is not extreme are assigned JPY_NON_EXTREME regardless of proximity to a previous episode. The state machine must not carry forward state identity after a sentiment reset. Each episode is independent. There is no "cooling off" state. |
| Degenerate partition (one state dominates) | Step 2 dry run checks state distribution before criterion tests run. |
| Cross-family leakage in transfer test | Transfer test uses held-out family — no training on target family pairs. |
| Calibration artifact tampering | SHA-256 hash covers all fields. Any manual edit invalidates the hash and blocks the run. |
| Regime ontology mismatch with MPML | BSVE artifact columns are self-contained. MPML reads state_id directly. No shared ontology required. |

---

## Open Questions (per environment)

### Reactive-JPY
- Validation of the calibrated young/maturing/mature partition
  under Criterion 1 behavioral differentiation testing.
- Whether `JPY_CONSENSUS_MATURING` is a genuine intermediate state or
  an artifact of threshold granularity
- Trend persistence influence on consensus maturation probability
  (currently exploratory, medium confidence)
- Cross-broker validation of crowd-short directional asymmetry
- HMM representation of consensus state transitions (Option B, future)

### Reactive-CHF
- Empirical calibration of `low_vol_threshold_atr_pct` and
  `high_vol_threshold_atr_pct` (placeholders until Jenks analysis runs)
- Whether `CHF_MEDIUM_VOL_TRANSITIONAL` is a behaviorally coherent
  state or a boundary artifact
- Mechanism difference between CHF vol-driven reversal and JPY
  consensus failure — are these genuinely distinct processes?
- SNB intervention events as external confounds in CHF vol spikes
- Whether crowd-long asymmetry is stable across 2012-2026 or era-dependent
- Cross-broker validation of crowd-long directional asymmetry

---

## Implementation Strategy

BSVE is being implemented incrementally using a contract-first approach.

The objective of the early pull requests is to establish a reusable behavioral-state research framework before implementing ontology-specific validation workflows.

### PR1 — Foundation Infrastructure ✓

Completed.

Implemented:

- BSVE configuration system
- Environment specifications
- Dataset adapter layer
- Behavioral feature registry
- Artifact contracts and validation infrastructure

Key principle:

Framework infrastructure remains ontology-agnostic.

### PR2 — Calibration Infrastructure ✓

Completed.

Implemented:

- Calibration artifact schema
- Calibration contract
- Calibration registry
- Calibration runner
- Null-calibration support
- Calibration artifact validation
- Calibration artifact persistence

Key principle:

Calibration outcomes are represented as first-class artifacts rather than implicit success/failure states.

### PR3 — Reactive-JPY Calibration ✓

Completed.

Implemented:

- Reactive-JPY maturity calibration plugin
- Hazard-analysis calibration workflow
- Threshold provenance tracking
- Calibration diagnostics
- Plugin bootstrap registration
- Calibration artifact inspection utilities
- Integration tests

Current output:

Reactive-JPY maturity thresholds have been calibrated, reviewed, signed off, and committed as the first production BSVE calibration artifact.

### PR4 — State Assignment Engine ✓

Completed.

Implemented:

- rule_based.py
- calibration loading
- threshold injection
- state assignment
- diagnostics
- manifest generation
- parquet generation
- [ ] Threshold exit labeling
- [ ] Tier 2 maturity validation

### PR5 — Validation Framework

Completed (Phase 1).

Scope:

✓ Criterion 1 validation framework
✓ Validation reporting

Future work:

- Ontology stability testing
- Cross-family transfer evaluation
- Criterion 2–5 validation

### PR6 — Behavioral Outcome Validation ✓

Completed.

Implemented:

- Behavioral outcome analysis framework
- Behavioral outcome report generation
- Fisher exact testing infrastructure
- Cohen's h effect size reporting
- Criterion 1 behavioral-evidence integration
- Behavioral outcome diagnostics

Key finding:

Reactive-JPY Criterion 1 remains INCONCLUSIVE.

The validation framework successfully detects the absence of independently labeled behavioral outcomes in the current state surface artifact.

This outcome supports the existing conclusion that threshold-exit labeling is required before Criterion 1 can be fully evaluated.

### PR7 — MPML Integration

Planned.

Scope:

- H1→D1 aggregation
- Behavioral surface consumption
- Manifest integration
- Walk-forward experimentation

### Development Rule

State assignment must not be implemented until calibration artifacts can be generated and validated.

MPML integration must not be implemented until state surface artifacts exist.

Framework code must remain ontology-independent.

## Implementation Checklist

This checklist reflects the current implementation status rather than the original concept draft.

### Phase 0 — Framework Infrastructure

#### Completed

* [x] `bsve/bsve_config_v1.yaml`
* [x] `bsve/state_specs/reactive_jpy_v1.yaml`
* [x] `bsve/state_specs/reactive_chf_v1.yaml`
* [x] Dataset adapter layer
* [x] Behavioral feature registry
* [x] Artifact validation infrastructure
* [x] Calibration artifact schema
* [x] Calibration contract

### Phase 1 — Calibration Infrastructure

#### Completed

* [x] Calibration registry
* [x] Calibration runner
* [x] Plugin architecture
* [x] Null-calibration support
* [x] Artifact validation
* [x] Artifact persistence
* [x] Calibration artifact inspection utility
* [x] Bootstrap registration mechanism

### Phase 2 — Reactive-JPY Calibration

#### Completed

* [x] `compute_extreme_threshold()`
* [x] `extract_consensus_lifecycles()`
* [x] `compute_hazard_by_maturity()`
* [x] `find_hazard_crossover()`
* [x] `derive_maturity_boundaries()`
* [x] `JPYMaturityCalibrationPlugin`
* [x] Threshold provenance tracking
* [x] Calibration diagnostics
* [x] Integration tests
* [x] Generate first production calibration artifact
* [x] Review calibration diagnostics
* [x] Sign off maturity thresholds
* [x] Commit calibration artifact

### Phase 3 — State Assignment Engine (PR4)

#### Completed

* [x] bsve/state_machine/rule_based.py
* [x] Load environment specification
* [x] Load calibration artifact
* [x] Inject calibrated thresholds
* [x] Fail fast on null calibration artifacts
* [x] Assign behavioral state per H1 bar
* [x] Track maturity counters
* [x] Track transition events
* [x] Generate BSVE state surface artifact
* [x] Generate run manifest
* [x] Verify state coverage and state distributions

- [x] Implement episode sparsity reporting

- [x] Track maturing-state distribution

- [x] Report episode counts
* [x] Report survival counts
* [x] Report duration distributions

### Remaining work after PR4

* [ ] Implement threshold-exit labeling
* [ ] Enable Tier 2 maturity validationFuture enhancements

Future enhancements

* [ ] Determine whether JPY_CONSENSUS_MATURING is independently validatable
* [ ] Emit assignment reason metadata

### Phase 4 — Validation Framework

Status:

* [x] Criterion 1 validation framework implemented
* [x] Validation report generation

Current outcome:

Reactive-JPY Criterion 1 currently returns
INCONCLUSIVE because only duration-derived
(calibration-consistency) diagnostics are available.

Independent behavioral evidence remains
required before Criterion 1 can be considered
satisfied.

Remaining work:

* [ ] Behavioral differentiation testing (independent evidence)
* [ ] Family specialization testing
* [ ] Incremental explanatory power testing
* [ ] Ontology stability testing
* [ ] Internal coherence testing (CHF)
* [ ] Transfer topology analysis

### Phase 5 — Reactive-CHF Calibration

#### Planned

* [ ] CHF volatility calibration plugin
* [ ] Behavioral significance validation
* [ ] Threshold derivation
* [ ] Calibration artifact generation
* [ ] Calibration sign-off

### Phase 6 — MPML Integration

#### Planned

* [ ] H1→D1 aggregation utility
* [ ] Behavioral surface integration
* [ ] Manifest integration
* [ ] Walk-forward experimentation


---

## Copilot Implementation Notes

The following context is relevant for repo-wide Copilot prompting.

**Existing patterns to follow:**
- DL artifact schema and validation:
  `schemas/dl_artifact_schema.py`, `docs/integration/dl_artifact_contract.md`
- Run manifest pattern: `run_manifest.json` in any MPML results archive
- Feature set naming convention: `price_trend`, `trend_vol_only`
  (see MSML README and train CLI)
- Walk-forward causal boundary convention:
  `prediction_available_timestamp` (see DL artifact contract v2)
- Fail-fast validation pattern:
  `write_dl_prediction_artifact()` calls `validate_dl_artifact()` before write

**Key invariants to preserve:**
- Sentiment timestamp normalization must happen at ingestion,
  not at feature engineering time
- Calibration artifacts must be committed and hash-verified before
  any validation run proceeds
- BSVE validation runs operate on H1 artifacts only
- MPML aggregation is downstream and lossy — never aggregate before
  running criterion tests
- State machine ambiguous assignments must never be silent —
  always fall back to non_extreme and emit a warning
- `calibration_id` must be recorded in every BSVE output artifact row
  for full provenance tracing

**Files to read before implementing:**
- `schemas/dl_artifact_schema.py` — artifact schema pattern
- `docs/integration/dl_artifact_contract.md` — timestamp semantics
- `bsve/state_specs/reactive_jpy_v1.yaml` — JPY state definitions
- `bsve/state_specs/reactive_chf_v1.yaml` — CHF state definitions
- `bsve/bsve_config_v1.yaml` — top-level configuration
- `bsve/calibration_artifacts/reactive_jpy_calibration_v1.json` — threshold values
- `bsve/calibration_artifacts/reactive_chf_calibration_v1.json` — threshold values
- `RESEARCH_STATE.md` — research context and confidence rankings

**Implementation order:**
Follow the Phase 0 → 1 → 2 → 3 → 4 sequence in the checklist above.
Do not implement Phase 2 (state machine) before Phase 1 (calibration)
artifacts are committed and validated. The gate in
`assert_calibrations_valid()` enforces this at runtime but the
implementation order should respect it during development as well.

---

### JPY Calibration Observations (2026-06-15: Originally calibrated on v1.5.0, Reproduced unchanged on v1.5.1 after duplicate-row correction)

First production calibration artifact produced from 441 episodes
across USDJPY (118), EURJPY (186), GBPJPY (137).

Key observations:

- Reversal hazard is strongly front-loaded. Approximately 60% of
  episodes terminate by bar 6. This is consistent with the research
  hypothesis that young consensus states carry high reversal risk.
- The hazard crossover is identified at bar 13. Derived thresholds:
  young_boundary = 8 bars, mature_boundary = 24 bars.
- Survival to mature_boundary (bar 24) is approximately 4.8% of
  episodes (21 surviving at bar 24 from 441 total). The resulting
  JPY_CONSENSUS_MATURE state is expected to be relatively rare.
- Median episode duration is 4 bars. The 75th percentile is 8 bars.
  The maturing zone (bars 8-24) and mature zone (bars 24+) together
  represent a minority of all episodes but may carry disproportionate
  behavioral information.
- censoring_rate = 0.0 and reversal_rate = 1.0 for both young and
  mature cohorts. This reflects limitations of the current lifecycle
  labeling procedure rather than a meaningful behavioral result.
  Hazard and survival analyses were used as the primary sign-off
  diagnostics.
- The late-maturity hazard spike at bar 30 (hazard = 0.333) with
  only 12 episodes at risk should be treated as sampling noise.
  It did not influence threshold selection.

Calibration outcome:

The resulting ontology:

Young = 0-8 bars
Maturing = 8-24 bars
Mature = 24+ bars

was judged behaviorally interpretable, empirically supported,
and consistent with prior JPY consensus-maturity research.

Open questions:

- Does the mature state remain sufficiently populated after full
  timeline labeling to support independent criterion validation?

- Is the primary behavioral distinction:
  Young vs Non-Young
  or:
  Young vs Maturing vs Mature

  This question will be revisited after PR4 state assignment and
  surface generation.

---

Reactive JPY calibration artifact v1 (2026-06-15, dataset v1.5.1)
------------------------------------

Dataset: 1.5.1
Episodes: 441
Hazard crossover: 13 bars

Derived thresholds:
- Young: 8 bars
- Mature: 24 bars

Supporting diagnostics:
- Median duration: 4 bars

- Survival ≥24 bars: 21 episodes

- Mature observations after assignment: 209

  The mature state remained sufficiently populated after full state assignment
  (209 observations across 667 episodes), supporting continued evaluation
  as an independent behavioral state.

Calibration artifact:
bsve/calibration_artifacts/reactive_jpy_calibration_v1.json

Supporting figure:
bsve/docs/figures/jpy_hazard_curve_v1.png

---

### Current Implementation Status

Reactive-JPY v1 pipeline complete through Criterion 1 evaluation.

Criterion 1 currently returns INCONCLUSIVE.

Behavioral outcome testing infrastructure exists and executes successfully, but the current state surface artifact does not yet contain independently labeled threshold exits. As a result, no independent behavioral evidence is currently available for maturity-state differentiation.

Threshold-exit labeling is therefore the primary remaining prerequisite for full Criterion 1 evaluation.

---

### Future cleanup

revisit reversal-rate diagnostics,
which currently provide little additional information
beyond the hazard/survival analysis.

