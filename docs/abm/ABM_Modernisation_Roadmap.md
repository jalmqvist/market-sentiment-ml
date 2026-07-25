# ABM Modernisation Roadmap — Market Sentiment ML
# Reactive-JPY Mechanistic Investigation Programme

**Status:** Draft — July 2026
**Audience:** Researchers and contributors to the MSML ABM programme
**Supersedes:** `ABM_EXPERIMENT_DIARY.md` entries through May 2026

---

## 1. Context and Motivation

The MSML research programme has completed its transition from sentiment
prediction toward behavioral representation research. As of July 2026:

- The **Reactive-JPY Behavioral Surface** is frozen, validated under
  the BSVE framework, and fully integrated into the MSML → MPML pipeline.
- **F-006** (preliminary): Reactive-JPY Prediction Artifacts improve
  downstream MPML adaptive strategy selection.
- **F-007** (confirmed): Reactive-JPY and Trend/Volatility Behavioral
  Surfaces produce substantially different prediction artifacts, encoding
  distinct predictive representations rather than alternative
  parameterisations of a common signal.

The ABM programme has not kept pace with this transition. Its last
documented experiments (May 2026) were oriented around generic
Trend/Volatility regime reproduction and Stage-2 decay sensitivity
on EUR-USD. The JPY sign-lock problem was documented but left
unresolved. No mechanistic model exists for the Reactive-JPY ontology.

This document defines the programme required to close that gap.

---

## 2. The Central Scientific Gap

The BSVE calibration of the Reactive-JPY ontology is built around
**consensus lifecycle structure**: discrete episodes of sustained
crowd positioning that form, mature, and dissolve according to an
empirically measured hazard-rate profile.

The calibration artifact encodes this structure precisely:

| Threshold                   | Derivation                                                   |
| --------------------------- | ------------------------------------------------------------ |
| `extreme_threshold_net_pct` | 70th percentile of `abs(net_sentiment)`, pooled across USDJPY, EURJPY, GBPJPY |
| `young_boundary_bars`       | `hazard_crossover_bar × 0.4`, rounded to nearest 8-bar session |
| `mature_boundary_bars`      | `hazard_crossover_bar × 1.6`, rounded to nearest 8-bar session |

The sign-off conditions that define a valid calibration are:

- `reversal_rate_young > reversal_rate_mature` — episodes are fragile
  early and stable once mature
- `reversal_rate_young > 0.15` — the early hazard is non-trivial
- `censoring_rate < 0.30` — enough episodes complete within the window
- `n_episodes_total >= 50` — sufficient sample for reliable estimation

The **current ABM has no concept of an episode**. It produces a
`net_sentiment` timeseries scored against population statistics
(mean, std, autocorrelation, extreme frequency). It cannot ask whether
it generated a consensus episode, how long that episode lasted, or
whether its reversal hazard profile resembles the empirical one.

This is not a calibration failure. It is a fundamental orientation
mismatch: the ABM is calibrated against the wrong scientific object.

The JPY sign-lock problem, documented in the May 2026 experiment diary,
is now interpretable in this light. JPY pairs do not fail to produce
sign flips — they produce **long-lived unidirectional consensus
episodes**. The current ABM generates sign-lock as a degenerate
absorbing state. The real market dwells purposefully in a consensus
state, then exits via a characteristic dissolution process. The
distinction between "absorbed at ±100 forever" and "episode forms,
matures over a characteristic timescale, then dissolves" is exactly
what the mechanistic model must reproduce.

The confirmed independence of Reactive-JPY and T/V prediction
artifacts (F-007) adds a further mechanistic constraint. Whatever
generates Reactive-JPY predictability is not captured by instantaneous
trend or volatility state. It is captured by the **lifecycle position
within a consensus episode** — which determines where the crowd sits
on the hazard-rate curve toward dissolution.

---

## 3. What the ABM Must Eventually Reproduce

Mechanistic success is defined at three levels, in increasing order
of scientific ambition. The levels are cumulative: each requires the
previous to be satisfied first.

### Level 1 — Episode Statistics

The ABM simulation output, when processed through the same
`extract_consensus_lifecycles()` logic used in BSVE calibration,
should produce episode populations whose summary statistics fall
within empirically plausible ranges:

- Episode frequency (episodes per 1000 simulation bars)
- Episode duration distribution: median, shape, and survival counts
  at 8, 16, 24, 32, and 48 bars
- Censoring rate analog: fraction of simulation-window episodes still
  active at simulation end
- Reversal rate by maturity zone: computed from simulated episodes
  against the calibration-derived `young_boundary_bars` and
  `mature_boundary_bars`

Success criterion: simulated `reversal_rate_young > reversal_rate_mature`,
with both rates in empirically plausible ranges. This mirrors the BSVE
sign-off condition applied to real data.

### Level 2 — Hazard Structure

The simulated episode population should reproduce the qualitative shape
of the empirical hazard curve:

- High reversal hazard early in episode life (high-hazard zone)
- Declining hazard as episodes mature
- A recognisable crossover / inflection point
- A concave rather than linear survival curve

Success criterion: visual and quantitative correspondence between the
simulated hazard curve (computed by `compute_hazard_by_maturity()`)
and the `hazard_curve` records stored in the BSVE calibration artifact.
The crossover bar location should be within a plausible range of the
empirical value.

### Level 3 — Predictive Structure

The hardest criterion. The ABM should reproduce the state-dependent
forward-return correlation gradient observed in DL experiments:

> sentiment during CONSENSUS_MATURING predicts forward returns
> better than sentiment during CONSENSUS_ENTRY or CONSENSUS_MATURE

This is the regime-hierarchy equivalent for Reactive-JPY states.
It requires injecting real BSVE state labels as an external
classification axis into `regime_hierarchy_test.py`, replacing
or supplementing the price-only LVTF/HVTF/LVR/HVR classification.

Success criterion: ABM-generated sentiment shows stronger
forward-return correlation during BSVE-labelled MATURING windows
than during ENTRY or MATURE windows — replicating the empirical
DL gradient without any predictive training.

---

## 4. Mechanistic Hypotheses

The following hypotheses are the primary scientific targets of the
programme. They are stated as falsifiable claims about which agent
mechanisms generate the observed episode structure.

### H1 — Persistence and anchoring generate episode duration structure

The existing inertia and anchoring mechanisms (`_INERTIA_THRESHOLD`,
`_SWITCHING_ANCHOR_STRENGTH`) are sufficient to generate episodes
with a characteristic duration distribution, provided the anchor is
calibrated to a non-degenerate regime (below the full sign-lock
threshold identified in the post-PR85 grid).

Prediction: anchor strength is the primary lever governing
`median_episode_duration_bars`. Reducing anchor from 2.0 toward
0.25 should produce a transition from infinite-duration episodes
(absorbing) to a finite, calibratable duration distribution.

### H2 — Volatility-conditioned decay governs the reversal hazard profile

Stage-2 decay (`decay_volatility_scale`) is the primary mechanism
responsible for the shape of the reversal hazard curve. Without decay,
the hazard curve is flat or declining monotonically from the start
(episodes dissolve at a constant or decreasing rate). With decay,
high-volatility periods inject dissolution events, producing a
characteristic hazard peak in the early-to-mid maturity range.

Prediction: `decay_volatility_scale > 0` is necessary to reproduce
`reversal_rate_young > reversal_rate_mature`. The decay mechanism
converts persistent sign-lock into a structured hazard profile.

### H3 — Episode formation requires a crowd-alignment trigger

Pure price-momentum dynamics generate gradual sentiment drift, not
the rapid formation events characteristic of JPY consensus episodes
(which tend to form quickly around macro shock events such as
risk-off yen surges or carry unwinds).

The existing model lacks a shock-injection mechanism. Without it,
simulated episodes form too slowly and too uniformly across market
conditions, producing a flat rather than peaked episode-frequency
distribution.

Prediction: adding a periodic or volatility-triggered crowd-alignment
shock (a fraction of agents simultaneously pushed toward one side)
will increase episode formation rate and produce a more realistic
formation-speed distribution.

### H4 — The Reactive-JPY predictive gradient is an endogenous lifecycle artifact

The DL finding (JPY_CONSENSUS_MATURING is the most predictive state,
confirmed by both MLP and LSTM, F-003/F-007) is interpretable as a
consequence of lifecycle position within an endogenously generated
crowd dynamic — not as a response to exogenous news shocks.

The mechanism: during maturation, episodes that have survived early
dissolution have selected for the most entrenched crowd positions.
The rising dissolution hazard in the maturation zone means an
increasing fraction of the crowd is positioned against imminent
reversal pressure. This creates a structurally predictable contrarian
setup that is invisible to instantaneous classifiers but recoverable
from lifecycle position alone.

Two independent lines of evidence support the endogenous interpretation:

1. **News-shock null (F-008):** Scheduled JPY news events (high +
   medium impact, 6,379 events, 2007–2026) show no meaningful
   association with episode exits. REVERSAL and THRESHOLD exits have
   near-identical news proximity distributions, both at or below the
   Poisson base rate for the observed news calendar density. Observable
   macro shocks are not the primary dissolution trigger.

2. **ABM Stage 3 result:** The MATURING > ENTRY forward-return
   correlation gradient is reproduced by the persistence + decay
   mechanism alone, without any shock injection, at the calibrated
   parameter point (anchor=0.25, beta=0.02). H4 is confirmed on
   GBP-JPY (full gradient) and partially supported on EUR-JPY and
   USD-JPY (MATURING > ENTRY direction holds on 2/3 pairs; MATURE
   cell n=54–89 is statistically unreliable).

This hypothesis predicts that adding a shock mechanism (H3) will
improve episode *formation rate and speed* to better match empirical
BSVE diagnostics, but will not materially strengthen the MATURING
predictive gradient — because that gradient arises from the
dissolution dynamics, not the formation mechanism. The shock stage
should therefore be evaluated against episode formation quality
metrics (H3), not against the H4 correlation gradient.

---

## 5. Implementation Roadmap

The programme is divided into five stages. Each stage produces
concrete, independently runnable artifacts and is designed to be
carried out within the existing repository constraints (single-file
experiment scripts, no modifications to `research/abm/sweep.py`,
backward-compatible defaults).

---

### Stage 0 — Infrastructure Updates
*Prerequisite for all subsequent stages. No new science.*

**0.1 Dataset version bump**

Update all hardcoded dataset version references from `1.2.0` (and the
`trace_one_run.py` reference to `1.3.2`) to `1.6.1`.

Affected files:
- `abm_experiments/sweep_with_volatility.py` — CLI default and docstring
- `abm_experiments/decay_beta_sensitivity.py` — CLI default and docstring
- `abm_experiments/regime_hierarchy_test.py` — CLI default and docstring
- `abm_experiments/trace_one_run.py` — hardcoded `version = "1.3.2"`
- `ABM_RUNBOOK.md` — Section 3 baseline command and dataset version

**0.2 JPY baseline rerun on v1.6.1**

Re-run the existing `decay_beta_sensitivity.py` harness for `usd-jpy`,
`eur-jpy`, and `gbp-jpy` using the post-PR85 calibration configuration
and the v1.6.1 dataset. Record whether the sign-lock profile is
consistent with v1.2.0 observations.

This is a pure verification step. No parameter changes. The purpose
is to confirm that the documented JPY absorbing-state behaviour is
stable across the dataset update before investing in new calibration
infrastructure.

Command template:
```bash
python abm_experiments/decay_beta_sensitivity.py \
    --version 1.6.1 \
    --pair usd-jpy \
    --steps 2000 \
    --beta 0.0 \
    --verbose

python abm_experiments/decay_beta_sensitivity.py \
    --version 1.6.1 \
    --pair usd-jpy \
    --steps 2000 \
    --beta 0.10 \
    --verbose
```

Repeat for `eur-jpy` and `gbp-jpy`. Record results in `ABM_EXPERIMENT_DIARY.md`.

------

### Stage 1 — Episode Extraction and Scoring Infrastructure

*New abm_experiments/ utilities. No changes to research/abm/.*

**1.1 `abm_experiments/episode_utils.py`**

A self-contained utility module (following the single-file constraint) providing:

- `extract_abm_episodes(net_sentiment, extreme_threshold, min_episode_bars)`: Port of `extract_consensus_lifecycles()` from `bsve/calibration/jpy_maturity_calibration.py`, adapted to consume a plain numpy array or pandas Series rather than a BSVE-formatted DataFrame. Returns a list of episode records `(start_step, end_step, duration, exit_type)`.

- `compute_episode_hazard(episodes, max_bars, min_at_risk)`: Port of `compute_hazard_by_maturity()`. Returns a DataFrame with the same schema as the BSVE diagnostic output, enabling direct comparison.

- `score_episode_structure(episodes, hazard_df, calibration_artifact)`: Compute a scalar episode-structure score measuring how closely simulated episode statistics match the BSVE calibration targets. Components:

  - Duration distribution distance (median duration ratio)
  - Reversal rate gradient direction (sign of `reversal_rate_young - reversal_rate_mature`)
  - Hazard crossover location error (simulated vs empirical)
  - Episode frequency plausibility (episodes per 1000 bars)

  The score is designed to be minimised (lower = better match to empirical BSVE structure). It does NOT replace the existing calibration scoring function — it is additive, measuring a different scientific object.

- `load_calibration_artifact(path)`: Thin JSON loader returning the BSVE artifact dict. Used to pull empirical targets into the scoring function without importing from `bsve/`.

**1.2 Validation against known artifact**

Before writing any new experiment harness, validate `episode_utils.py` by running it against a saved BSVE calibration artifact and the real JPY sentiment data. The episode frequency and reversal rate statistics computed by `episode_utils.py` must match the values stored in the artifact's `diagnostics` block.

This is the ground truth check. If `episode_utils.py` does not reproduce the BSVE numbers on real data, it cannot be used as an ABM calibration target.

------

### Stage 2 — Episode-Calibrated ABM Harness

*New experiment script. No changes to research/abm/.*

**2.1 `abm_experiments/reactive_jpy_episode_calibration.py`**

A new experiment harness that replaces the statistical calibration objective with an episode-structure objective.

Structural relationship to existing harnesses:

- Inherits the fixed-configuration philosophy of `decay_beta_sensitivity.py` (one parameter combination per invocation, no internal loops)
- Uses the same ABM pipeline as `decay_beta_sensitivity.py`
- Adds post-processing through `episode_utils.py`
- Accepts a BSVE calibration artifact path as the empirical target

Key parameters:

- `--version` (dataset version, default `1.6.1`)
- `--pair` (single JPY pair)
- `--steps` (simulation length, default `2000`)
- `--beta` (`decay_volatility_scale`)
- `--anchor-strength` (override `_SWITCHING_ANCHOR_STRENGTH`)
- `--calibration-artifact` (path to BSVE JSON artifact — provides `extreme_threshold_net_pct`, `young_boundary_bars`, `mature_boundary_bars`, and empirical diagnostics for scoring)
- `--seed`
- `--verbose`

Output (default, one line per run):

```
beta | anchor | episode_score | ep_freq | median_dur | rev_rate_young | rev_rate_mature | hazard_crossover
```

Output (verbose): full diagnostics plus per-episode records.

This script is the primary tool for Stage 2 and 3 experimentation. It enables systematic exploration of which (anchor, beta) combinations produce episode structures that qualitatively match the empirical BSVE calibration.

**2.2 Anchor sweep — finding the non-degenerate JPY regime**

Using `reactive_jpy_episode_calibration.py`, run a grid over `anchor_strength` from `{0.0, 0.25, 0.5, 1.0, 2.0}` with `beta=0.0`.

Expected finding (per H1): anchor ≥ 1.0 produces degenerate absorbing episodes (very high median duration, zero reversal rate). Anchor ≤ 0.5 should produce a finite duration distribution with non-zero reversal rates in both maturity zones.

**2.3 Beta sweep — finding the hazard structure**

Fix anchor at the best non-degenerate value from 2.2. Run a beta sweep from `{0.0, 0.01, 0.02, 0.05, 0.10, 0.20}`.

Expected finding (per H2): beta = 0.0 produces flat or monotonic hazard (decreasing survival). beta > 0 introduces dissolution events that generate a peaked early hazard and declining hazard as episodes mature. The `reversal_rate_young > reversal_rate_mature` gradient should emerge only with non-zero beta.

**2.4 Combined (anchor, beta) grid — episode structure optimisation**

Run a small factorial grid over the plausible (anchor, beta) space identified in 2.2 and 2.3. Score each combination using the episode structure score from `episode_utils.py`. Identify the region of parameter space that best reproduces the empirical BSVE calibration.

This is the calibration target for the "persistence + decay" mechanism alone, without shock injection.

------

### Stage 3 — Shock-Driven Episode Formation

*Extends Stage 2 with new agent mechanism. No changes to research/abm/sweep.py.*

**3.1 Mechanism design: exogenous crowd-alignment shock**

The existing ABM generates sentiment dynamics entirely from price-signal reaction and crowd herding. This produces gradual drift, not the rapid formation events characteristic of JPY consensus episodes (which tend to form quickly around macro shock events).

A minimal shock mechanism:

- Shock trigger: volatility spike (EMA volatility proxy > threshold) OR periodic (every N bars)
- Shock effect: a fraction of agents (e.g., 30-50%) are simultaneously pushed to the same side (long or short, determined by recent price direction)
- Post-shock: normal agent dynamics resume, with persistence and anchoring sustaining the newly formed consensus

This is implemented as an optional extension in `sweep_with_volatility.py` (which already has volatility-conditioned logic) or as a new `abm_experiments/sweep_with_shocks.py`.

Key parameters:

- `--shock-enable` (flag)
- `--shock-trigger` (`"volatility"` or `"periodic"`)
- `--shock-threshold` (volatility percentile for trigger)
- `--shock-fraction` (fraction of agents affected, 0.0-1.0)
- `--shock-direction` (`"price"` — follow recent price, or `"random"`)

**3.2 Shock + persistence + decay integration**

Combine the shock mechanism with the calibrated (anchor, beta) values from Stage 2. The hypothesis (H3) is that shock injection will:

- Increase episode formation rate (more episodes per 1000 bars)
- Produce a more realistic formation-speed distribution (faster formation than pure price-drift)
- Improve the episode frequency plausibility component of the structure score without degrading the hazard structure

**3.3 Episode structure validation with shocks**

Re-run the episode scoring from Stage 2.3 with shock-enabled configurations. The target is a parameter region where:

- Episode frequency matches empirical (within 2x)
- Median duration matches empirical (within 2x)
- `reversal_rate_young > reversal_rate_mature` holds
- Hazard crossover is in plausible range
- Formation speed distribution is peaked (fast formation) not uniform

This is the "full mechanism" calibration: shock formation + persistence sustaining + decay dissolution = realistic consensus lifecycle.

------

### Stage 4 — Predictive Structure Reproduction

*Extends regime_hierarchy_test.py with BSVE state injection.*

**4.1 BSVE state label injection**

Extend `abm_experiments/regime_hierarchy_test.py` to accept an external state label file. The state labels are the real BSVE Reactive-JPY state assignments (ENTRY / MATURING / MATURE) for the same price series used in the ABM simulation.

New CLI parameters:

- `--bsve-states-path` (path to CSV with columns: `timestamp`, `bsve_state` where state ∈ {`ENTRY`, `MATURING`, `MATURE`, `NEUTRAL`})
- `--use-bsve-states` (flag to use BSVE classification instead of price-only LVTF/HVTF/LVR/HVR)

When `--use-bsve-states` is set, the regime classification function replaces the price-only logic with a lookup into the BSVE state file. The rest of the analysis (forward-return correlation by regime) runs unchanged.

**4.2 Predictive gradient test**

Run the ABM with the calibrated (shock, anchor, beta) configuration from Stage 3. Inject real BSVE state labels. Compute forward-return correlation for:

- `ENTRY` windows
- `MATURING` windows
- `MATURE` windows

The empirical DL finding (confirmed by MSML experiments) is: **MATURING > ENTRY > MATURE** in predictive power.

The mechanistic hypothesis (H4) predicts that ABM-generated sentiment will show the same ordering, without any predictive training, purely as a consequence of the lifecycle dynamics.

**4.3 Sensitivity and robustness**

Test the predictive gradient under:

- Different (anchor, beta) combinations near the calibrated optimum
- Different shock configurations (trigger type, fraction, threshold)
- Different random seeds

The gradient should be robust to modest parameter variations if the mechanism is genuinely explanatory rather than overfitted.

------

### Stage 5 — Documentation and Programme Integration

*Update all ABM documentation to reflect the new programme.*

**5.1 `DL_ABM_RECONCILIATION.md` update**

The existing document reconciles T/V regimes with ABM. Add a second layer reconciling Reactive-JPY lifecycle states with the shock + persistence + decay mechanistic model.

Key additions:

- F-007 as a mechanistic constraint (independence from T/V)
- The lifecycle hypothesis: predictability arises from hazard-rate position, not instantaneous regime
- The three-level success criteria (statistics → hazard → predictive)
- Updated conceptual model:

```
market regime (trend/vol) → governs shock frequency
    shock + price direction → episode formation (rapid, directional)
    persistence + anchoring → episode sustaining (characteristic duration)
    decay (vol-conditioned) → episode dissolution (hazard-rate structure)
        ↓
    consensus lifecycle position → weak predictive signal
        (MATURING most predictive, per empirical DL)
```

**5.2 `ABM_RUNBOOK.md` update**

Add Stage 3 definition (Episode Calibration) with:

- New calibration targets (episode structure, not sentiment statistics)
- New experiment scripts (`reactive_jpy_episode_calibration.py`, `sweep_with_shocks.py`)
- Updated success criteria referencing BSVE sign-off conditions
- Parameter grid recommendations for (anchor, beta, shock) exploration

**5.3 `ABM_EXPERIMENT_DIARY.md` new entry**

Capture the July 2026 pivot:

- Motivation: F-006, F-007, Reactive-JPY integration into MPML
- Realisation: ABM calibrated against wrong target (statistics vs episodes)
- Reinterpretation: JPY sign-lock as episode phenomenology, not bug
- New programme: Stage 3 episode calibration, Stage 4 shock mechanism, Stage 5 predictive reconciliation

------

## 6. Success Criteria Summary

| Stage | Deliverable                           | Success Criterion                                            |
| ----- | ------------------------------------- | ------------------------------------------------------------ |
| 0.1   | Dataset version bump                  | All scripts reference 1.6.1, tests pass                      |
| 0.2   | JPY baseline rerun                    | Sign-lock profile stable vs May 2026 observations            |
| 1.1   | `episode_utils.py`                    | Reproduces BSVE diagnostics on real data                     |
| 1.2   | Validation                            | Episode statistics match artifact within tolerance           |
| 2.1   | `reactive_jpy_episode_calibration.py` | Runs end-to-end, produces episode scores                     |
| 2.2   | Anchor sweep                          | Identifies non-degenerate anchor regime (≤0.5)               |
| 2.3   | Beta sweep                            | `reversal_rate_young > reversal_rate_mature` emerges with β>0 |
| 2.4   | Combined grid                         | Identified (anchor, β) region with good structure score      |
| 3.1   | Shock mechanism                       | Implemented in `sweep_with_shocks.py` or extension           |
| 3.2   | Integration                           | Shock + persistence + decay runs without error               |
| 3.3   | Structure validation                  | All episode statistics within 2x of empirical                |
| 4.1   | BSVE state injection                  | Complete. `--use-bsve-states` flag implemented, state label mapping (JPY_CONSENSUS_* → short labels) applied on load, non-episode rows excluded. |
| 4.2  | Predictive gradient                    | ABM shows MATURING > ENTRY > MATURE on GBP-JPY (full); MATURING > MATURE on EUR-JPY (partial); USD-JPY inconclusive due to n=54 MATURE cell. MATURING is the strongest episode-state predictor on 2/3 pairs at the calibrated parameter point. |
| 4.3   | Robustness                            | Gradient stable across seeds and modest parameter variation  |
| 5.x   | Documentation                         | All three MD files updated, diary entry complete             |

------

## 7. Immediate Next Steps

The following can be executed immediately, in order:

1. **Stage 0.1**: Update dataset version strings (mechanical, 30 min)
2. **Stage 0.2**: Rerun JPY baseline on v1.6.1 (verification, 2-3 hours)
3. **Stage 1.1**: Draft `episode_utils.py` (new code, 4-6 hours)
4. **Stage 1.2**: Validate against BSVE artifact (verification, 1-2 hours)

After Stage 1.2 is complete, we will have:

- Confidence that the episode extraction logic is correct
- A working episode scoring function
- The foundation for all subsequent experimentation

I recommend pausing after Stage 1.2 to review the `episode_utils.py` implementation and the validation results before proceeding to the harness and experimentation stages.

------

## 8. Repository Hygiene and Constraints

Throughout this programme, the following constraints apply (per established repository conventions):

- **No modifications to `research/abm/sweep.py`**
- **No modifications to `research/abm/agents.py` defaults** without explicit documentation in `ABM_RUNBOOK.md`
- **Single-file experiment scripts** for all new harnesses
- **Backward-compatible defaults**: baseline EUR-USD behaviour must remain reachable
- **Deterministic reproducibility**: all scripts accept `--seed`, produce identical outputs for identical inputs
- **Artifact contracts**: all experiment outputs follow the `logs/abm_*_{timestamp}.csv` / `.log` / `.json` pattern

------

## 9. Relationship to Broader MSML Programme

This ABM modernization is not an isolated effort. It directly supports:

| MSML Component | ABM Contribution                                             |
| -------------- | ------------------------------------------------------------ |
| **BSVE**       | Mechanistic explanation for why the Reactive-JPY ontology structure exists |
| **MSML**       | Validation that episode structure (not just state labels) carries predictive information |
| **MPML**       | Confidence that behavioral routing decisions are grounded in reproducible mechanisms |
| **F-007**      | Explanation for why Reactive-JPY and T/V surfaces are independent — they capture different mechanistic layers (lifecycle vs instantaneous regime) |

The long-term scientific goal remains: **connect empirical observation, predictive validation, and mechanistic simulation** into a coherent understanding of how latent behavioral organization emerges from observable market activity.

The Reactive-JPY ontology is the first complete behavioral representation to traverse the full pipeline. The ABM programme outlined here is the mechanistic component required to complete the scientific loop.

------

*Document version: 2026-07-25* *Next review: After Stage 1.2 completion*
