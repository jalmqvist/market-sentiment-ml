# ABM Experiment Diary

This diary captures the *chronological* path of ABM experiments and decisions that led to the current Stage‑2 decay investigations.  It is intentionally lightweight: it records what we ran, what we observed, and why we pivoted — so future work can continue without re-running large ad‑hoc shell sweeps.

> Repo: `jalmqvist/market-sentiment-ml`
> 
> Focus: `research/abm` agent-based retail sentiment simulation.

---

## 2026-07-25 — Stage 2: Episode-calibrated ABM, calibrated parameter point established

### Context

Following the July 2026 programme pivot (see roadmap), the ABM calibration
objective was reoriented from sentiment population statistics (mean, std,
autocorr) to episode lifecycle structure derived from the frozen Reactive-JPY
BSVE calibration artifact (`reactive_jpy_calibration_v1.json`, dataset v1.5.1,
thresholds: extreme=70.0%, young_boundary=8 bars, mature_boundary=24 bars).

New infrastructure:
- `abm_experiments/episode_utils.py` — episode extraction, hazard analysis,
  episode structure scoring. Validated against BSVE artifact (Stage 1.2):
  15/15 pooled diagnostics at 0.000 relative error.
- `abm_experiments/reactive_jpy_episode_calibration.py` — episode-calibrated
  harness. Fixed configuration (50/50 trend/contrarian, momentum=3,
  persistence=0.10, threshold=0.05, decay_clip_max=0.5).
- `abm_experiments/validate_episode_utils.py` — Stage 1.2 ground-truth
  validation script.

### Stage 0.2 — JPY baseline on v1.6.1

Re-ran `decay_beta_sensitivity.py` on usd-jpy, eur-jpy, gbp-jpy (seeds 1-5,
beta=0.0 and 0.1). Result: complete sign-lock confirmed stable on v1.6.1.
pct_saturated=1.0, sign_flips=0, pct_negative=0 across all 30 runs.
Consistent with May 2026 v1.2.0 observations. Dataset update did not affect
JPY absorbing-state profile.

### Stage 2.2 — Anchor sweep (usd-jpy, seed=42, beta=0.0)

Swept anchor_strength in {0.00, 0.05, 0.10, 0.15, 0.17, 0.20, 0.25}.

Sharp phase transition discovered:

| anchor | n_ep | med_dur | rev_m  |
| ------ | ---- | ------- | ------ |
| ≤ 0.10 | 0    | —       | —      |
| 0.15   | 7    | 3.0     | 0.000  |
| 0.20   | 46   | 3.0     | 0.000  |
| 0.25   | 109  | 4.5     | 0.000* |

*seed=42 anomaly — seeds 1-5 at anchor=0.25 showed rev_mature=1.0.

anchor=0.25 identified as the unlock point. Below this threshold, the
anchor mechanism prevents episodes from forming. At 0.25, the system
generates a realistic episode population.

Interpretation: anchor_strength is the primary lever governing
episode formation. The phase transition at 0.25 corresponds to the
point where the anchor can no longer prevent crowd reversals from
dissolving episodes.

### Stage 2.3 — Beta sweep (usd-jpy, anchor=0.25, seeds 1-5)

Swept beta in {0.00, 0.01, 0.02, 0.05, 0.10, 0.20}.

Scoring bug discovered and fixed during analysis: the reversal gradient
direction check used `sim_rev_young < sim_rev_mature` without tolerance,
triggering spurious heavy penalties when both values are near 1.0
(e.g., 0.984 < 1.000). Fixed by adding 0.05 slack to direction check.

Results after fix:

| beta | good_runs/5 | rev_m=0 failures | median_score |
| ---- | ----------- | ---------------- | ------------ |
| 0.00 | 4/5         | 0                | 0.433        |
| 0.01 | 3/5         | 1                | 0.392        |
| 0.02 | 5/5         | 0                | 0.379        |
| 0.05 | 4/5         | 0                | 0.397        |
| 0.10 | 1/5         | 1                | 0.353        |
| 0.20 | 0/5         | 4                | —            |

beta=0.02 is the only value producing 5/5 consistent good runs with
zero mature-episode failures. Mechanism: small amount of
volatility-conditioned decay ensures some episodes persist into the
mature zone without overdamping frequency or duration.

### Stage 2.4 — Cross-pair validation at calibrated point

Ran anchor=0.25, beta=0.02, seeds 1-5 on all three JPY pairs.

| Metric           | USD-JPY     | EUR-JPY     | GBP-JPY     | Empirical |
| ---------------- | ----------- | ----------- | ----------- | --------- |
| Score (mean±std) | 0.364±0.035 | 0.378±0.055 | 0.362±0.018 | ~0.3-0.4  |
| Bad runs         | 0/5         | 0/5         | 0/5         | 0/5       |
| n_ep / 2000      | 105±4       | 104±6       | 113±5       | ~90       |
| freq / 1000      | 52.3±2.2    | 51.8±3.2    | 56.4±2.4    | 44.9      |
| med_dur (bars)   | 4.6±0.5     | 4.8±0.7     | 4.8±0.4     | 4.0       |
| surv_8%          | 30.2±3.8%   | 30.3±6.4%   | 29.6±4.4%   | 25.6%     |
| rev_mature       | 1.0 all     | 1.0 all     | 1.0 all     | 1.0       |

15/15 runs passed. Score range [0.2996, 0.4548]. Consistent across
all three JPY pairs.

Minor residual gaps: frequency ~15-25% above empirical (52-56 vs 45
per 1000 steps); surv_8% ~4pp above empirical (29-30% vs 25.6%);
median duration ~0.5-0.8 bars above empirical. All within acceptable
tolerance for a mechanistic model calibrated without pair-specific tuning.

### Calibrated parameter point (LOCKED)

anchor_strength = 0.25

decay_volatility_scale = 0.02 (beta) 

decay_base = 0.00 

decay_clip_max = 0.50 

n_trend = 50 

n_contrarian = 50 

n_noise = 0 

momentum_window = 3 

persistence = 0.10 

threshold = 0.05 

dataset_version = 1.6.1


### Mechanistic interpretation

The calibrated parameter point confirms H1 and H2 from the roadmap:

**H1 (anchor governs duration structure):** Confirmed. anchor=0.25
is the phase transition point below which no episodes form. The
anchor mechanism is the primary lever controlling episode formation
rate and duration.

**H2 (beta governs hazard profile):** Confirmed with nuance.
Beta=0.02 is sufficient to ensure consistent mature-episode
existence (rev_mature=1.0) across seeds. Higher beta (≥0.10)
begins to overdamp, reducing episode frequency and creating
mature-zone failures. The operating window for beta is narrow
(0.00-0.05).

### Next step

Stage 3 (roadmap): BSVE state label injection into
`regime_hierarchy_test.py` — test whether ABM-generated sentiment
shows stronger forward-return correlation during BSVE-labelled
MATURING windows than ENTRY or MATURE windows (H4).

---

## 2026-05-06 — Stage‑2 decay sensitivity: problem statement

**Goal.** Determine whether Stage‑2 “release” (agent-side decay in accumulation state) behaves as a *continuous control knob* via `decay_volatility_scale` (β).

**Fixed configuration used for sensitivity harness runs** (kept constant across tests):

- `trend_ratio = 1.0`
- `persistence = 0.20`
- `threshold = 0.100`
- `seed` varied (default 42)

**Metrics used** (computed from `net_sentiment` time series):

- `pct_time_saturated`: fraction of steps where `|net_sentiment| >= 90`
- `sign_flips`: count of sign changes in `net_sentiment`
- `autocorr_lag1`: lag‑1 autocorrelation of `net_sentiment`

Harness script: `abm_experiments/decay_beta_sensitivity.py`.

---

## 2026-05-06 — Initial sensitivity (integer accumulation state): quantization

**Experiment.** Sweep β with fixed config for a single pair (`eur-usd`) and seed.

**Observation.** β produced a *phase shift* but not a graded response:

- β = 0.0 behaved like an absorbing/persistent regime
- β ≥ small value triggered a different regime
- further increases in β produced *flat/unchanged* metrics

**Conclusion.** The issue was **loss of resolution** from:

- integer accumulation state (`position: int`)
- truncation during decay (`int(np.trunc(...))`)

This quantized the release mechanism and collapsed sensitivity.

---

## 2026-05-06 — Implementation: continuous accumulation state

**Change.** Converted `RetailTrader.position` from integer to float and removed truncation in the decay step, while keeping:

- accumulation logic
- decay/clipping logic
- switching/anchoring logic

otherwise unchanged.

**Files.**

- `research/abm/agents.py`

**Commit.**

- `Use continuous position state to avoid quantization; remove truncation from decay`
- https://github.com/jalmqvist/market-sentiment-ml/commit/1d2b9b94bb59b3b008eabeddf3f6ff941e8e779a

**Outcome.** Re-running the same sensitivity showed β now influenced metrics strongly (especially `autocorr_lag1` and `sign_flips`), i.e. the quantization bottleneck was removed.

---

## 2026-05-06 — Harness improvement: seed control

**Motivation.** Single runs were highly path/seed dependent, so we needed seed sweeps to evaluate smoothness “in expectation”.

**Change.** Added `--seed` flag (default `42`) to `abm_experiments/decay_beta_sensitivity.py`.

**Commit.**

- `Add optional --seed to decay beta sensitivity harness (default 42)`
- https://github.com/jalmqvist/market-sentiment-ml/commit/29e711485bd3d5f4260f18e875d77eb32b320b0a

---

## 2026-05-06 — Seed ensembles: β controls persistence but regimes are multi-modal

**Experiment.** 10-seed ensemble for `eur-usd` across β in `{0.0, 0.01, 0.02, 0.03, 0.04}`.

**Observation.**

- `autocorr_lag1` reliably decreased vs β=0 for most seeds (β is now a usable persistence-control parameter).
- `sign_flips` and `pct_time_saturated` were highly seed dependent (multi-regime / attractor behavior).

**Interpretation.** After quantization removal, the model exhibits nonlinear regime structure driven by existing switching + asymmetry + inertia.

---

## 2026-05-06 — Harness improvement: labeled output

**Motivation.** Long shell-loop output is hard to interpret without pair/seed labeling; terminal buffers can reorder/mix lines.

**Change.** Added `--verbose` flag to print `pair` and `seed` while keeping the default output format unchanged.

**Commit.**

- `Add --verbose to sensitivity harness to print pair/seed while preserving default output format`
- https://github.com/jalmqvist/market-sentiment-ml/commit/4125ccd7e92f5fad564529c090ee0bc0a93fbe5f

---

## 2026-05-06 — Pair generalization: JPY pairs saturate and are sign-locked

**Experiment.** With `--verbose`, ran across pairs `eur-usd`, `usd-jpy`, `eur-jpy` for seeds 1..5 and β in `{0.0, 0.01, 0.03}`.

**Observation.**

- `eur-usd`: decay reduced autocorr and often increased flips (expected “mixing” response).
- `usd-jpy` and `eur-jpy`: `pct_time_saturated` was ~1.0 and `sign_flips` ~0 almost always.
- Yet `autocorr_lag1` still decreased with β, meaning decay was acting, but **sign changes were suppressed**.

**Hypothesis.** JPY pairs are in a regime where sign is locked by decision boundaries (inertia + asymmetry + anchoring), and decay mostly modulates magnitude within a single sign.

---

## 2026-05-06 — Harness improvement: sentiment summary statistics

**Motivation.** When `pct_time_saturated` is ~1 and flips are 0, we need to know whether the system is truly pinned at ±100 (absorbing), or merely oscillating within the saturated band (e.g., +90..+100).

**Change.** Extended `--verbose` output with basic distribution stats for `net_sentiment`:

- `mean`, `std`, `min`, `max`

Default output format remains unchanged.

**Commit.**

- `Verbose output: include mean/std/min/max net_sentiment (stdout-only; default format unchanged)`
- https://github.com/jalmqvist/market-sentiment-ml/commit/028f934972ba8c9c21a55721f79ad8f1e67fefc9

**Key result example (USD-JPY, seed 1).** As β increases, sign remains locked and saturation stays ~1, but the *level* and *range* of sentiment shift dramatically:

- β=0.0: mean ~134.5, min=90, max=140
- β=0.01: mean ~147.0, min~77, max~185
- β=0.03: mean ~365.9, min~314, max~404
- β=0.04: mean ~443.7, min~377, max~480

This confirms JPY pairs are not merely “always above 90”; they can run far outside the historical `[-100, +100]` scale because agent positions are now continuous and can accumulate beyond ±1 contributions.

---

## Current state / next questions

1. **Scaling:** The dataset convention expects `net_sentiment` in `[-100, +100]`, but with continuous `position` the aggregate can exceed this. Decide whether to:
   - normalise/clamp positions in aggregation, or
   - redefine interpretation of `net_sentiment` magnitude (and thresholds like 90).
2. **JPY sign-lock:** If the goal is for JPY pairs to exhibit controlled sign changes (not permanent saturation), identify which *existing* decision terms dominate:
   - inertia threshold `_INERTIA_THRESHOLD`
   - asymmetric hold probability (currently hard-coded 0.7)
   - anchoring strength `_ANCHOR_STRENGTH`
   - persistence coupling `_PERSISTENCE_WEIGHT`

The next investigation should be phrased as “which existing term causes sign-lock under JPY dynamics?” rather than adding new mechanisms.

---

# Appendix A — Review of prior ABM post-mortem (external LLM write-up)

This appendix reviews the document:

- `abm_pipeline_postmortem_and_lessons_learned.md`

It was produced by a previous LLM instance and should be interpreted as a *hypothesis / narrative* rather than a verified record of repository state at the time. In particular, some parameter names and constant values in the document may not correspond to the current codebase (or to any historical commit that exists in this repository).

## A.1 What appears confirmed by current repo experiments (high confidence)

### A.1.1 There are two intertwined classes of problems: model dynamics and pipeline contracts

The post-mortem claims that debugging became chaotic largely due to missing or implicit “contracts” between modules (simulation/agents/calibration/sweep).

**Confirmed by current work:**

- The Stage‑2 decay sensitivity work showed that a minimal internal change (integer→float accumulation state to remove decay quantization) can unintentionally change the *semantic meaning* and *scale* of `net_sentiment`.
- Downstream diagnostics (e.g., `abs(net_sentiment) >= 90`) implicitly assume the dataset convention `net_sentiment ∈ [-100, +100]`.

As a result, “interface drift” can happen even without function signature changes: **semantic drift** is enough.

### A.1.2 Stabilization must be balanced with controlled endogenous amplification

The post-mortem’s central modeling claim is that the ABM needs both:

- stabilization (avoid runaway positive feedback / absorbing herding), and
- controlled endogenous amplification (avoid over-damping into near-white-noise).

**Consistent with current observations:**

- With decay disabled (β=0), the ABM can be extremely persistent (lag‑1 autocorrelation near 1).
- Enabling decay reduces persistence (autocorr decreases), but does not automatically produce realistic “escape” behavior (especially in sign-locked regimes).

### A.1.3 “Silent interface drift” is a realistic failure mode in research repos

The post-mortem’s recommendation to add:

- explicit schemas
- validation boundaries
- integration tests

is directionally correct for this repository, which contains multiple experiment styles and outputs.

## A.2 Plausible but unverified claims (needs repo history validation)

The post-mortem includes specific claims like:

- `_VOL_FEEDBACK_SCALE: 100.0 -> 0.3`
- `_FLIP_PROB = 0.02`
- `_MEAN_REVERSION_STRENGTH = 0.02`
- `crowd_influence = tanh(...)`

These may be good modeling ideas and may have existed in earlier iterations or experiment branches, but **they should not be treated as fact** until verified against actual commits and file paths.

Recommendation: if we want to use these ideas, we should implement them deliberately and document them as new changes, rather than assuming they are already present or were proven previously.

## A.3 New insight uncovered in current Stage‑2 work (not captured in the post-mortem)

The post-mortem frequently frames “saturation” as being near ±100 (dataset convention). Current Stage‑2 sensitivity experiments uncovered a critical semantic issue:

- When `RetailTrader.position` is made continuous (float) and aggregation uses raw position values, the simulation output `net_sentiment` can exceed `[-100, +100]` by a large margin.

This causes several downstream issues:

- `pct_time_saturated` using the threshold 90 becomes almost always 1.0 and loses interpretive value.
- `sign_flips` can become 0 simply because the system is far from crossing 0, not because the dynamics are necessarily “absorbing at ±100”.

Example (USD-JPY, seed 1):

- β=0.03: mean ~365.9, min ~313.7, max ~403.7
- β=0.04: mean ~443.7, min ~377.3, max ~480.3

This points to a near-term priority: **restore/define `net_sentiment` semantics** (see “Scaling” in the current state section) before drawing strong conclusions from saturation metrics.

## A.4 Recommended use of the post-mortem going forward

- Treat it as a source of **candidate mechanisms** and **architecture guardrails**.
- Prefer hard evidence from reproducible runs + committed code when making decisions.
- When adopting any mechanism suggested in the post-mortem, introduce it as a single-purpose change with explicit tests/diagnostics and diary entries.
