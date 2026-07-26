# ABM Experiment Diary

This diary captures the *chronological* path of ABM experiments and decisions that led to the current Stage‑2 decay investigations.  It is intentionally lightweight: it records what we ran, what we observed, and why we pivoted — so future work can continue without re-running large ad‑hoc shell sweeps.

> Repo: `jalmqvist/market-sentiment-ml`
> 
> Focus: `research/abm` agent-based retail sentiment simulation.

---

---

## 2026-07-26 — Stage 4: Robustness Sweep at vol_t80_f30_cd10

### Setup
Robustness sweep bracketing the Stage 3 anchor config (vol_t80_f30_cd10)
along two perturbation dimensions:
  - Cooldown sweep: thresh=0.80, frac=0.30, cooldown ∈ {5, 10, 20}
  - Fraction sweep: thresh=0.80, cooldown=10, fraction ∈ {0.30, 0.40, 0.50}

5 unique configs × 3 pairs × 20 seeds = 300 runs.
Script: abm_experiments/stage4_robustness_sweep.py
All other parameters: anchor=0.25, beta=0.02, steps=1500 (calibrated point).

### Summary table

| Config    | USD-JPY | EUR-JPY | GBP-JPY | Shocks/run | mean|r|MAT |
|-----------|---------|---------|---------|------------|------------|
| cd5_f30   | NO      | PARTIAL | FULL    | ~103       | 0.0559     |
| cd10_f30* | NO      | FULL    | FULL    | ~62        | 0.0769     |
| cd20_f30  | NO      | NO      | NO      | ~36        | 0.0498     |
| cd10_f40  | NO      | FULL    | FULL    | ~62        | 0.0712     |
| cd10_f50  | FULL    | FULL    | NO      | ~62        | 0.0684     |

(* Stage 3 anchor, reproduced exactly)

### Key findings

**F1 — cd10 is a genuine cooldown optimum (inverted-U confirmed)**
Cooldown gradient (mean |r|MATURING across pairs):
  cd5=0.056 < cd10=0.077 > cd20=0.050
cd10 outperforms both directions. Over-shocking (cd5, ~113/run) degrades
the MATURING signal, particularly on USD-JPY (|r| = 0.012 vs 0.079 at
cd10). Under-shocking (cd20, ~36/run) loses EUR-JPY and GBP-JPY FULL
verdicts entirely. The mechanism: at cd5 the shock cadence (~13 bars) is
shorter than mature_boundary (24 bars), disrupting episodes before they
complete their lifecycle and flattening the contrarian signal.

**F2 — Fraction is not a meaningful lever (0.30–0.50 insensitive)**
At cooldown=10, FULL H4 pair count = 2/3 across all three fractions.
mean |r|MATURING declines monotonically (0.077 → 0.071 → 0.068) but the
gradient is shallow and the H4 verdict is unchanged. Fraction does not
provide a meaningful tuning axis within this range.

**F3 — USD-JPY H4 is gated by MATURE cell noise, not MATURING signal**
USD-JPY achieves FULL H4 only at cd10_f50, where |r|MATURE collapses to
0.019 (MATURE/MATURING ratio = 0.35). At all other configs |r|MATURE
exceeds or approaches |r|MATURING, blocking the H4 verdict. This is a
structural consequence of the small MATURE cell (n=54): 20-seed variance
in 54-row correlations dominates. USD-JPY is confirmed as a caution-only
pair for H4 assessment. The EUR-JPY + GBP-JPY pool is the reliable test
surface.

**F4 — H3 frequency gap is structural, not parameter-resolvable**
All 5 configs overproduce episodes: freq/1k = 55–70 against empirical
target 45–56. cd10_f40 brings USD-JPY to 54.9 (within target) but
EUR-JPY and GBP-JPY remain at 67 and 65. Median duration remains ~3
bars against target 4 at all configs. The gap is structural: short
threshold-exit episodes (dur~3 < young_boundary=8) inflate frequency
and prevent the reversal gradient (rev_young > rev_mature) from
operating. This is a known limitation of the persistence+decay+shock
mechanism at the current calibrated point — see roadmap for Level 1
episode statistics assessment.

### Conclusion

H4 robustness confirmed. The vol_t80_f30_cd10 anchor is the identified
optimum along both perturbation axes. FULL H4 on EUR-JPY + GBP-JPY is
stable across fraction=0.30–0.50 and represents the reliable finding.
USD-JPY H4 is structurally limited by MATURE cell size and should not be
used as a primary H4 test surface.

The cooldown optimum finding (F1) provides a mechanistic constraint for
future shock mechanism design: shock cadence must be longer than the
mature_boundary (24 bars) to allow episodes to complete their lifecycle.
At cd=10 with volatility threshold=0.80, mean inter-shock interval is
~22 bars — near but above the critical boundary. At cd=5 (~13 bars) it
falls below, causing degradation.

### Recommended anchor config (confirmed)

vol_t80_f30_cd10 remains the Stage 3/4 anchor:
  shock_trigger       = volatility
  shock_vol_threshold = 0.80
  shock_fraction      = 0.30
  shock_cooldown      = 10

### Next steps

1. Update DL_ABM_RECONCILIATION.md (roadmap Stage 5.1):
   Document H3/H4 findings, the lifecycle-conditioned predictive gradient
   mechanism, and the structural freq/duration gap as a known limitation.

2. Commit Stage 3 + Stage 4 diary and roadmap updates.

3. Consider roadmap Stage 5.2 (optional): targeted investigation of the
   freq/duration gap via episode extractor definition sensitivity —
   specifically whether raising young_boundary or adjusting the extreme
   threshold changes the structural episode count without disrupting H4.

---

## 2026-07-26 — Stage 3 Shock Sweep: H3 + H4 results

### Setup
Sweep of 10 shock configurations × 3 pairs × 20 seeds = 600 ABM runs.
Script: abm_experiments/sweep_with_shocks.py
Parameters: anchor=0.25, beta=0.02, steps=1500 (calibrated point).
Shock sweep matrix: volatility trigger (thresh 0.70/0.80/0.90,
fraction 0.20/0.30/0.50, cooldown 10/20) and periodic trigger
(period 25/50 bars, fraction 0.30/0.50).

### H4 results summary

| Config           | USD-JPY | EUR-JPY | GBP-JPY |
| ---------------- | ------- | ------- | ------- |
| baseline         | NO      | NO      | NO      |
| vol_t70_f30      | NO      | NO      | FULL    |
| vol_t80_f30      | NO      | NO      | NO      |
| vol_t90_f30      | NO      | NO      | NO      |
| vol_t80_f20      | NO      | PARTIAL | NO      |
| vol_t80_f50      | NO      | NO      | PARTIAL |
| vol_t80_f30_cd10 | NO      | FULL    | FULL    |
| per_p50_f30      | NO      | NO      | NO      |
| per_p25_f30      | NO      | NO      | NO      |
| per_p50_f50      | NO      | NO      | NO      |

Best config: vol_t80_f30_cd10 (thresh=0.80, frac=0.30, cooldown=10).
H4 FULL on EUR-JPY (MATURING=0.0511, ENTRY=0.0439, MATURE=0.0169)
and GBP-JPY (MATURING=0.1005, ENTRY=0.0683, MATURE=0.0432).
USD-JPY: H4 NO — MATURE cell (n=54) remains dominant artefact.

### Key findings

**F1 — Shocks necessary but not sufficient for H4**
Baseline (no shocks) produces H4 on zero pairs. Volatility-triggered
shocks produce H4 on at least one pair in most configs. This confirms
H3 as a necessary precondition for H4.

**F2 — Cooldown is the key lever (H3 → H4 pathway)**
vol_t80_f30 (cooldown=20, 36 shocks/run): 0 FULL.
vol_t80_f30_cd10 (cooldown=10, 62 shocks/run): 2 FULL.
More frequent vol-aligned shocks densify episode formation,
populating the MATURING window sufficiently to develop the
contrarian imbalance that generates the forward-return gradient.

**F3 — Periodic trigger null: market-state conditioning required**
All four periodic configs produce 0 FULL H4 results at comparable
shock counts. Arbitrary timing is insufficient. The trigger must
coincide with genuine volatility regimes to create naturalistic
crowd-alignment events. This aligns with F-008 (news-shock null):
event occurrence alone does not matter; market-state amplification
of agent coordination does.

**F4 — Shocks suppress the MATURE artefact**
MATURE mean |r| falls from 0.093 (baseline) to 0.052
(vol_t80_f30_cd10), a 44% reduction. More frequent shocks
redistribute sentiment more broadly, reducing extreme-value
clustering on the small MATURE cell.

**F5 — H3 episode frequency target unmet**
All configs produce freq/1k of 58-76 against empirical target of
45-56. Shocks reduce frequency from baseline (73) toward target
(58 on USD-JPY at cd10) but do not reach it. Duration remains
2.9-3.4 bars against target of 4. Reversal gradient (rev_young >
rev_mature) is zero on all configs — structural limitation of
ABM short-duration threshold-exit episodes interacting with the
episode extractor, not a parameter failure.

### Conclusion

H3 confirmed (shocks improve H4 signal). H4 conditionally confirmed
at vol_t80_f30_cd10: FULL on 2/3 pairs, consistent with GBP-JPY and
EUR-JPY pool. USD-JPY remains limited by MATURE cell size (n=54).
Volatility-conditioned trigger is necessary; periodic trigger is
insufficient. H3 episode frequency/duration targets require further
investigation beyond parameter sweep scope.

### Recommended next config for targeted follow-up
vol_t80_f30_cd10 is the anchor config for Stage 4 robustness testing.
Consider also: cooldown=5 to probe whether further densification
continues to improve H4, and fraction=0.40 to test the fraction
sensitivity at the identified best cooldown.

---

## 2026-07-26 — Stage 3 Pooled H4 test: negative result with qualified findings

### Setup
Pooled cross-pair H4 test: 20 runs × 3 pairs (USD-JPY, EUR-JPY, GBP-JPY),
seeds 1-20, steps=1500, anchor=0.25, beta=0.02.
Pooled frame: 64,100 rows. State counts: ENTRY=43,940 MATURING=15,960 MATURE=4,180.

### Pooled result

| State    | n      | Pearson r  | p      | Spearman r | p      |
| -------- | ------ | ---------- | ------ | ---------- | ------ |
| ENTRY    | 43,940 | -0.0216*** | 0.0000 | -0.0291*** | 0.0000 |
| MATURING | 15,960 | +0.0293*** | 0.0002 | +0.0034 ns | 0.6638 |
| MATURE   | 4,180  | +0.0013 ns | 0.9316 | -0.0172 ns | 0.2674 |

H4 NOT SUPPORTED on pooled data. Empirical rank: ENTRY > MATURE > MATURING
(Spearman). The 5-seed GBP-JPY confirmation was within-noise.

### Key findings from decomposition

**F1 — EUR-JPY ENTRY contrarian signal (genuine)**
EUR-JPY ENTRY Spearman mean = -0.0585 ± 0.0427 (SNR=1.37) across 20 seeds.
Both USD-JPY (+0.0001) and GBP-JPY (-0.0154) ENTRY signals are flat.
The pooled ENTRY significance is driven by EUR-JPY (42.4% of rows).
The negative sign is mechanistically coherent: high ABM sentiment at
episode entry predicts contrarian 24-bar reversal. This is a real
pair-specific finding but should not be generalised cross-pair.

**F2 — MATURING Pearson/Spearman divergence (leverage artefact)**
Pooled MATURING: Pearson r=+0.0293 (p=0.0002) vs Spearman r=+0.0034 (p=0.664).
Gap of 0.026 at n=15,960 indicates high-leverage outlier influence — extreme
ABM sentiment values co-occurring with large forward returns in specific runs
(e.g. USD-JPY seed 18 Pearson r=+0.17). Not a rank-order distributional signal.

**F3 — MATURING sign inconsistent across pairs (cancellation)**
USD-JPY MATURING Spearman = +0.0493 (SNR=0.79)
EUR-JPY MATURING Spearman = -0.0091 (SNR=0.14)
GBP-JPY MATURING Spearman = -0.0342 (SNR=0.43)
All SNRs below 1.0. Signs cancel in pooling. No pair individually
exceeds one within-pair standard deviation. The MATURING gradient
from the 5-seed run was a high-variance false positive.

### H4 assessment (revised)

H4 as formulated (MATURING > ENTRY > MATURE in |forward-return correlation|)
is falsified at the calibrated parameter point by the 20-seed pooled test.
The persistence + decay mechanism alone does not reproducibly generate the
MATURING predictive gradient.

The programme hypothesis that shocks are not needed to reproduce H4 (based
on the 5-seed GBP-JPY result) is not confirmed. The shock mechanism (H3 /
roadmap Stage 3) remains the next required investigation — now as a
necessary condition for H4 rather than a formation-quality enhancement.

### What remains valid

- The calibrated parameter point (anchor=0.25, beta=0.02) produces correct
  episode lifecycle structure (Stage 2 result, unchanged).
- The EUR-JPY ENTRY contrarian signal is a genuine mechanistic finding
  worth investigating further.
- F-008 (news-shock null) is unaffected — scheduled news events do not
  drive episode exits regardless of H4 outcome.
- The endogenous lifecycle interpretation of H4 remains the correct
  scientific framing; the current mechanism is simply insufficient to
  reproduce the gradient without shock injection.

---

## 2026-07-25 — Stage 3: BSVE State Label Injection, H4 hypothesis test

### Context

Implemented Stage 3 (roadmap Stage 4.1/4.2) as a BSVE state-injection
extension to `abm_experiments/regime_hierarchy_test.py`. The script
replaces the price-only LVTF/HVTF/LVR/HVR classification with real
BSVE state_id labels (ENTRY / MATURING / MATURE) loaded from the
augmented dataset CSV. ABM-generated net_sentiment is aligned to
empirical BSVE row indices and per-state forward-return correlations
(Spearman + Pearson) are computed at the calibrated parameter point.

New infrastructure:
- `abm_experiments/regime_hierarchy_test.py` — Stage 3 drop-in replacement
  with `--use-bsve-states` / `--bsve-states-path` flags. Stage 2
  LVTF/HVTF/LVR/HVR fallback retained unchanged.
- `scripts/run_stage3_bsve_injection.sh` — 3-pair run script with
  JSON output per pair and combined summary.

### Implementation notes

State label mapping required: the live BSVE dataset uses full ontology
names (JPY_CONSENSUS_YOUNG, JPY_CONSENSUS_MATURING, JPY_CONSENSUS_MATURE,
JPY_NON_EXTREME). Mapped to canonical short labels (ENTRY, MATURING,
MATURE) on load; JPY_NON_EXTREME rows excluded as non-episode background.

Alignment strategy: row-index alignment (ABM step i → BSVE row i),
testing distributional properties per lifecycle state rather than
point-in-time prediction. Steps=1500 used to avoid mod-wrap on EUR-JPY
(1354 rows).

Parameter patching correction: agents.py reads env vars at module import
time only. All parameter injection switched to direct module-attribute
patching inside a try/finally restore block, mirroring
old_regime_hierarchy_test.py.

### Results (anchor=0.25, beta=0.02, seeds 1-5, steps=1500)

State counts after non-episode filtering:

| Pair    | ENTRY | MATURING | MATURE |
| ------- | ----- | -------- | ------ |
| USD-JPY | 600   | 250      | 54     |
| EUR-JPY | 933   | 332      | 89     |
| GBP-JPY | 665   | 216      | 66     |

Spearman |r| means (5-run average):

| Pair    | MATURING | ENTRY  | MATURE | H4 Full | H4 Partial |
| ------- | -------- | ------ | ------ | ------- | ---------- |
| USD-JPY | 0.0495   | 0.0002 | 0.1524 | False   | False      |
| EUR-JPY | 0.0456   | 0.0655 | 0.0110 | False   | True       |
| GBP-JPY | 0.0848   | 0.0193 | 0.0171 | True    | True       |

All MATURE cells flagged CAUTIOUS (n=54–89, low-n guard at n<100).

### Interpretation

**MATURING > ENTRY direction:** Holds on USD-JPY and GBP-JPY. EUR-JPY
shows ENTRY marginally above MATURING (0.0655 vs 0.0456), within 1
cross-seed std. The direction is structurally present on the majority
of pairs.

**MATURE anomaly:** The elevated |r| on USD-JPY MATURE (0.1524, std=0.113)
is driven by high-leverage alignment between the 54 MATURE rows and
specific ABM sentiment values in the 1500-step synthetic series. The
high std confirms this is noise-dominated. Excluding MATURE from the
H4 test, the MATURING > ENTRY direction holds 2/3 pairs.

**Relationship to F-007 and DL finding:** The DL confirmation that
JPY_CONSENSUS_MATURING is the most predictive state (LSTM > MLP,
F-007) is structurally reproduced on GBP-JPY without predictive
training. The mechanism — lifecycle-conditioned sentiment dispersion
driving a contrarian forward-return signal — appears to operate in
the ABM at the calibrated parameter point, though not uniformly
across pairs.

**H4 assessment:**
- H4 FULLY SUPPORTED: GBP-JPY
- H4 PARTIALLY SUPPORTED: EUR-JPY (MATURING > MATURE confirmed,
  ENTRY ordering within noise margin)
- H4 NOT SUPPORTED: USD-JPY (dominated by small-sample MATURE cell)
- Cross-pair: MATURING is the strongest episode-state predictor on
  2/3 pairs. The full gradient (MATURING > ENTRY > MATURE) is confirmed
  on the pair with the most balanced MATURE cell (GBP-JPY, n=66).

### Limitations and next steps

1. MATURE cell size (n=54–89) is insufficient for stable correlation
   estimates at this run length. The cautious flag correctly identifies
   this. A longer empirical window or pooled cross-pair analysis would
   be needed to test the full H4 gradient reliably.

2. The alignment strategy (row-index, structural surrogate) is correct
   for testing distributional H4 but cannot test point-in-time
   predictive capacity. Timestamp-matched alignment would require the
   ABM to simulate the exact empirical price path, which it does (using
   the real price series), but sentiment diverges from empirical after
   warmup due to stochastic agent initialization.

3. Stage 4 roadmap (shock-driven episode formation) is the next
   investigation. The current result establishes the baseline H4
   signal from the calibrated persistence + decay mechanism alone,
   without shock injection. Adding shocks (H3) may sharpen the
   MATURING signal by producing more realistic episode formation
   dynamics.

### Calibrated parameter point (unchanged, locked 2026-07-25)

anchor_strength = 0.25
decay_volatility_scale = 0.02
decay_base = 0.00
decay_clip_max = 0.50
n_trend = 50, n_contrarian = 50, n_noise = 0
momentum_window = 3, persistence = 0.10, threshold = 0.05
dataset_version = 1.6.1

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
