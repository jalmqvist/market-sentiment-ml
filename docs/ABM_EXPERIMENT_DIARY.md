# ABM Experiment Diary

This diary captures the *chronological* path of ABM experiments and decisions that led to the current Stage‑2 decay investigations.  It is intentionally lightweight: it records what we ran, what we observed, and why we pivoted — so future work can continue without re-running large ad‑hoc shell sweeps.

> Repo: `jalmqvist/market-sentiment-ml`
> 
> Focus: `research/abm` agent-based retail sentiment simulation.

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
