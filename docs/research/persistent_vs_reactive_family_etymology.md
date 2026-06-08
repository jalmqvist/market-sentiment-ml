# Persistent vs Reactive Family Etymology

## Executive Summary

The Persistent and Reactive pair family labels emerged **incrementally** from a sequence of overlapping research threads rather than from a single algorithmic decision. The families were **not** defined by persistence magnitude metrics (e.g. longest streaks, highest extreme-occupancy). They were defined — implicitly at first, then explicitly — by **differential behaviour in grouped DL multi-pair training experiments**, supported by converging evidence from ABM dynamics, downstream MPML integration asymmetry, and cross-family transfer experiments.

The primary origin event was the observation, during grouped MLP training on the sentiment-driven H1 dataset (v1.3.2), that two natural groupings of pairs — `{EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD}` and `{USDJPY, EURJPY, GBPJPY, EURCHF, USDCHF}` — produced materially different classification metrics under identical architecture, feature set, target horizon, and regime (LVTF). The family labels *persistent* and *reactive* were then assigned post-hoc to describe the qualitative DL behaviour of each group.

The family partition therefore describes **structural DL and behavioural stability** rather than raw persistence magnitude, which explains why direct reconstruction from `side_streak`, `extreme_streak_70/80`, or `sentiment_change` magnitude rankings fails and in fact inverts.

---

## Earliest Evidence

### Phase 1 — JPY-cross hypothesis (archived, invalidated)

The earliest non-universal pair grouping in the repository is the `JPY_cross` vs `non_JPY` binary, used in signal-discovery scripts and surfaced as a canonical feature column:

```
pipeline/features.py:240  add_pair_group() → 'JPY_cross' or 'other'
pipeline/signal.py:79-125  Regime V2 signal restricted to pair_group == 'JPY_cross'
config.py:REGIME_V2_PAIR_GROUP = "JPY_cross"
```

The JPY-cross grouping emerged from early exploratory analysis suggesting JPY pairs exhibited stronger contrarian returns under persistent extreme sentiment. This was formally pre-registered as a validation test (`docs/archive/PRE_REGISTERED_JPY_EFFECT_TEST.md`) and subsequently **invalidated** after corrected walk-forward validation:

> *"The pre-registered JPY effect is not supported… mean differences between JPY and non-JPY ≈ 0"*
> — `docs/archive/PRE_REGISTERED_JPY_EFFECT_TEST.md`

This invalidation is critical context: the project's first pair-grouping attempt based on persistence magnitude and contrarian signal strength was falsified. This forced a methodological pivot toward DL-based behavioral characterisation.

### Phase 2 — Pair-quality and cross-pair persistence audits

Following the JPY invalidation, a series of analysis scripts were written to audit pair-level behavior across persistence metrics:

| Script | Focus |
|---|---|
| `research/analysis/analyze_pair_quality.py` | Return-quality flags per pair (outlier detection) |
| `research/analysis/analyze_persistence.py` | Persistence magnitude by group (major / cross / thin) |
| `research/analysis/analyze_cross_pair_persistence.py` | Per-pair persistence ranking at thr70 and thr80 |
| `research/analysis/analyze_by_pair_group.py` | Threshold summaries by structural pair group |
| `research/analysis/analyze_jpy_cluster_permutation.py` | Permutation test of JPY cluster effect |

These scripts used `pair_group` categories (`major`, `cross`, `thin_exotic`) — **not** the persistent/reactive partition. They ranked pairs by `mean_contrarian_ret` and `hit_rate` at various `extreme_streak_70` thresholds and horizons. No explicit persistent/reactive family assignment appears in any of these scripts.

The problem statement notes that the modern master dataset reconstructed rankings from these metrics were "nearly inverted relative to the historical family labels". This is consistent with the analysis scripts predating the family labels: they were diagnostic audits, not the source of the partition.

### Phase 3 — Grouped DL training experiment (origin event)

The earliest explicit definition of the persistent/reactive partition appears in:

```
docs/behavioral/grouped_pair_family_findings.md
```

This document records the first **grouped multi-pair MLP training experiment** with the two families named and listed explicitly under the headers:

> *"### Persistent / accumulation-oriented"*
> — EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD

> *"### Reactive / release-oriented"*
> — USDJPY, EURJPY, GBPJPY, EURCHF, USDCHF

The experimental setup was:

- Dataset: v1.3.2
- Model: MLP
- Feature set: `price_trend`
- Target: 24-bar directional threshold prediction
- Regime: LVTF
- Training mode: grouped multi-pair

Results that triggered the labels:

| Group | Accuracy | Precision | Recall | F1 | Positive Rate |
|---|---|---|---|---|---|
| Persistent | 0.6020 | 0.288 | 0.431 | 0.345 | 0.364 |
| Reactive | 0.6125 | 0.364 | 0.512 | 0.426 | 0.394 |

The *reactive* group generalized materially better (higher F1, higher precision, higher recall) under identical conditions, suggesting different underlying structural dynamics. This is the **primary empirical event** from which the family labels were assigned.

---

## Relevant Scripts

### Scripts directly involved in the family origin

| Script | Role |
|---|---|
| `research/deep_learning/train.py` | MLP grouped training engine; `--train-pairs` / `--predict-pairs` flags enable cross-family transfer |
| `research/deep_learning/train_lstm.py` | LSTM replication of grouped training |
| `research/deep_learning/feature_sets.py` | Defines `price_trend` and `trend_vol_only` feature sets |

### Scripts that provided preparatory evidence

| Script | Role |
|---|---|
| `research/analysis/analyze_cross_pair_persistence.py` | Per-pair persistence magnitude ranking (cross pairs) |
| `research/analysis/analyze_pair_quality.py` | Pair-level quality filtering (outlier detection) |
| `research/analysis/analyze_persistence.py` | Group-level persistence summaries |
| `research/analysis/analyze_by_pair_group.py` | Group threshold summaries (major/cross/thin) |
| `research/analysis/analyze_jpy_cluster_permutation.py` | Permutation tests for JPY cluster significance |
| `research/analysis/analyze_trend_behavior.py` | `persistent` vs `non_persistent` regime classification (via `extreme_streak_70 >= 3`) |
| `research/analysis/analyze_regime_signal_interaction.py` | JPY × strength × persistence interaction studies |
| `research/signal_discovery/walk_forward_jpy_hypothesis.py` | Walk-forward JPY persistence hypothesis |
| `research/signal_discovery/walk_forward_jpy_regime_signal.py` | Walk-forward JPY regime signal |
| `research/signal_discovery/regime_v2.py` | Regime V2 (JPY-cross restricted signal) |
| `research/signal_discovery/regime_v3.py` | Regime stability across walk-forward years |
| `research/hypothesis_experiments/discover_behavioral_signal.py` | Early per-pair behavioral signal discovery |
| `research/hypothesis_experiments/discovery.py` | Refactored per-pair discovery |
| `research/abm/agents.py` | ABM agent behavior (accumulation / release dynamics) |
| `research/abm/calibration.py` | ABM calibration |

### Scripts confirming and extending the family partition

| Script | Role |
|---|---|
| `research/analysis/analyze_trend_strength_results.py` | JPY × persistent × fight_trend × horizon analysis |
| `research/raw_validation/validate_jpy_effect_preregistered.py` | Pre-registered JPY validation (invalidated) |
| `research/raw_validation/validate_jpy_effect_time_split.py` | Time-split JPY validation |
| `research/raw_validation/validate_jpy_effect_walkforward.py` | Walk-forward JPY validation |

---

## Relevant Metrics

### Metrics used during origin experiments

| Metric | Script | Role in partition |
|---|---|---|
| MLP F1 (grouped family) | `research/deep_learning/train.py` | Primary divergence signal |
| MLP Precision (grouped family) | `research/deep_learning/train.py` | Transition sharpness proxy |
| MLP Recall (grouped family) | `research/deep_learning/train.py` | Continuation/accumulation proxy |
| MLP Accuracy (grouped family) | `research/deep_learning/train.py` | Secondary |
| Positive Rate (grouped family) | `research/deep_learning/train.py` | Signal density proxy |

### Metrics used in preparatory analysis (NOT the partition source)

| Metric | Script |
|---|---|
| `extreme_streak_70` / `extreme_streak_80` | `analyze_cross_pair_persistence.py`, `analyze_persistence.py` |
| `crowd_persistence_bucket_70` | `pipeline/features.py`, `pipeline/signal.py` |
| `contrarian_ret_12b` / `contrarian_ret_48b` | All analysis scripts |
| `hit_rate` (per pair, persistence subset) | `analyze_cross_pair_persistence.py` |
| Sharpe ratio (regime walk-forward) | `regime_v3.py`, `walk_forward_jpy_hypothesis.py` |
| `positive_year_ratio` (sign consistency) | `regime_v3.py` |
| IC (Spearman rank correlation) | `regime_v3.py`, `regime_v8.py` |
| ABM `autocorr_lag1` | `abm_experiments/decay_beta_sensitivity.py` |
| ABM `sign_flips` | `abm_experiments/decay_beta_sensitivity.py` |
| ABM `pct_time_saturated` | `abm_experiments/decay_beta_sensitivity.py` |

---

## Candidate Definitions

### Candidate 1 — DL grouped training performance divergence (Most Likely)

**Confidence: HIGH**

**Evidence:**

The families were first explicitly defined in `docs/behavioral/grouped_pair_family_findings.md` as two groupings in a grouped MLP training experiment. The partition was defined by the observation that the two groups produced materially different F1 / precision / recall metrics under identical conditions:

- *Reactive* group: F1 ≈ 0.43, Precision ≈ 0.36 (higher transition sharpness)
- *Persistent* group: F1 ≈ 0.34, Precision ≈ 0.29 (slower accumulation structure)

The names "persistent" and "reactive" were assigned as interpretive labels for the observed DL behaviour:

> *"The model appears to learn slower accumulation-style structure rather than sharp directional transitions."*
> — persistent family, `grouped_pair_family_findings.md`

> *"The model appears to learn more episodic or release-oriented dynamics."*
> — reactive family, `grouped_pair_family_findings.md`

The grouping itself (which pairs were in each family) appears to have been formed from prior qualitative observation noted in `cross_family_transfer_findings.md`:

> *"JPY/CHF pairs consistently behaved differently from EUR/GBP/NZD pairs"*
> *"ABM accumulation dynamics fit some pairs well but failed on others"*
> *"DL precision and transition behavior differed materially across pair groups"*
> *"downstream MPML integration effects were asymmetric rather than random"*

These prior observations were not the result of a formal ranking or scoring procedure: they were accumulated qualitative findings across multiple research phases. The grouped training experiment then **confirmed and crystallised** those observations into a formal partition.

**What this means for reconstruction:**

The persistent/reactive split describes differential DL learnability under grouped training — specifically the degree to which a shared MLP can learn sharp directional transitions (precision) vs smoother accumulation patterns (recall-dominated). It is **not** derivable from raw persistence magnitude rankings.

---

### Candidate 2 — ABM accumulation vs release dynamics

**Confidence: MEDIUM**

**Evidence:**

The `docs/behavioral/grouped_pair_family_findings.md` document explicitly aligns the families with ABM dynamics:

> *"Persistent-family: accumulation dominates, stable directional clustering, strong anchor dynamics"*
> *"Reactive-family: stronger release dynamics, more boundary-sensitive behavior, volatility-conditioned destabilization"*

The ABM model (`research/abm/agents.py`) simulates agents with:

- `anchor_strength` — governs directional persistence
- `decay_volatility_scale` (β) — governs release / destabilization rate
- `reinforce_strength` — governs positive-feedback accumulation

ABM calibration experiments (`docs/abm/ABM_USDJPY_POST_PR85_CALIBRATION.md`) show that USDJPY required specific near-boundary parameter settings distinct from EUR/GBP/NZD pairs, consistent with the reactive/persistent split. However, the ABM calibration postdates the family labels and appears to validate rather than originate them. There is no evidence of a formal ABM-based clustering that produced the partition.

**What this means for reconstruction:**

The ABM provides mechanistic interpretation of the families, but not a reconstruction algorithm. The ABM alignment is confirmatory, not generative.

---

### Candidate 3 — MPML downstream integration asymmetry

**Confidence: MEDIUM**

**Evidence:**

The `cross_family_transfer_findings.md` document identifies downstream MPML integration effects as a key motivator:

> *"downstream MPML integration effects were asymmetric rather than random"*
> *"effects were structurally coherent rather than random"*

The problem statement notes that from a separate MPML project, the family partition appears connected to:

- information effect
- partition effect
- learning stability
- variance reduction

This is consistent with the finding from `docs/behavioral/sentiment_ablation.md` that:

- Persistent-family: adaptive routing reacted to sentiment removal, but ensemble behavior remained stable
- Reactive-family: performance degraded modestly but remained structurally coherent

The MSML evidence supporting the "structural stability" interpretation is:

- Structure surviving sentiment ablation (`trend_vol_only` experiments)
- Structure surviving architecture changes (MLP ↔ LSTM)
- CHF/JPY split producing weaker differentiation than persistent/reactive split
- Family divergence surviving removal of explicit LVTF regime conditioning

This suggests the families describe **structural stability of learned representations** rather than predictive signal strength, consistent with the MPML findings cited in the problem statement.

**What this means for reconstruction:**

If MPML downstream behavior is the governing criterion, reconstruction would require running grouped DL training experiments and measuring downstream integration stability rather than raw persistence metrics.

---

### Candidate 4 — Cross-period rank stability

**Confidence: LOW**

**Evidence:**

`regime_v3.py` implements sign consistency (`positive_year_ratio`) and Sharpe stability across walk-forward years as regime selection criteria. This is a cross-temporal consistency mechanism for regime selection, not for pair-family classification.

No surviving script applies cross-year Sharpe rank stability or overlap-in-top-decile analysis specifically to produce the persistent/reactive pair partition. The closest evidence is the walk-forward JPY hypothesis scripts, which compare JPY vs non-JPY across years, but this mapping is coarser than the family partition and was invalidated.

**Conclusion:**

Cross-period rank stability was used for regime filtering, not for pair family assignment. This candidate is unlikely to be the origin, but cannot be fully excluded without access to additional research artifacts that may not be committed.

---

### Candidate 5 — Persistence magnitude ranking (Excluded)

**Confidence: VERY LOW (excluded)**

**Evidence against:**

The problem statement itself reports that reconstruction attempts using `side_streak`, `extreme_streak_70`, `extreme_streak_80`, and `sentiment_change` produced rankings "nearly inverted relative to the historical family labels."

The analysis scripts (`analyze_cross_pair_persistence.py`, `analyze_persistence.py`) predate the family labels and use `major/cross/thin_exotic` groupings, not persistent/reactive groupings. The `PAIR_GROUPS` dict in these scripts classifies pairs by liquidity type.

The `analyze_trend_behavior.py` script uses the word "persistent" in the context of `extreme_streak_70 >= 3` as a local regime label (not as a pair family label):

```python
research/analysis/analyze_trend_behavior.py:33-34
out["persistence_bucket"] = np.where(
    out["extreme_streak_70"] >= 3, "persistent",
    np.where(out["extreme_streak_70"] >= 1, "non_persistent", "none")
)
```

This string reuse of "persistent" is a false cognate — it labels rows within a dataset, not currency pairs.

**Conclusion:**

Persistence magnitude was a diagnostic tool used before the family labels existed. The family labels were not derived from persistence magnitude ranking. Any reconstruction via persistence magnitude will produce incorrect results, as the problem statement confirms empirically.

---

## Most Likely Historical Origin

The persistent/reactive family partition most likely originated as follows:

**Step 1 — Prior qualitative observation (multiple scripts)**

Across multiple research phases (JPY hypothesis, ABM calibration, early DL experiments), researchers accumulated the informal observation that:
- JPY/CHF pairs behaved differently from EUR/GBP/NZD pairs in ABM fit quality and downstream MPML effects.
- The ABM worked better on some pairs than others.
- MPML integration effects were not uniform across pairs.

None of these observations produced a formal partition.

**Step 2 — Grouped MLP training experiment (origin event)**

A grouped MLP training experiment was run with the feature set `price_trend` under LVTF regime, using dataset v1.3.2, with target horizon 24 bars. Two natural groupings were formed based on the prior qualitative observations:

- Group A: EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD
- Group B: USDJPY, EURJPY, GBPJPY, EURCHF, USDCHF

The experiment revealed a material F1 / precision divergence (Group B ≈ 25% higher precision). The labels **persistent** and **reactive** were assigned post-hoc to describe the differential DL behavior.

**Step 3 — Confirmation and extension**

Subsequent experiments confirmed the partition:
- Regime-free (no LVTF) grouped experiments: family divergence persisted.
- Sentiment ablation (`trend_vol_only`): family divergence survived.
- LSTM replication: family divergence survived architecture change.
- CHF/JPY subgroup decomposition: weaker differentiation than persistent/reactive split, suggesting the broader partition captures a structural regime boundary.
- MPML downstream experiments: asymmetric integration effects aligned with the partition.

**Step 4 — Formalization in documentation**

The partition was formalized in `docs/behavioral/grouped_pair_family_findings.md` and subsequently propagated to `docs/behavioral/cross_family_transfer_findings.md`, `docs/behavioral/behavioral_model.md`, `RESEARCH_STATE.md`, `README.md`, and `PROJECT_DESCRIPTION.md`.

---

## Recommended Reconstruction Method

Given that the partition originated from grouped DL training performance divergence (not persistence magnitude), the most faithful reconstruction procedure is:

### Method A — Grouped DL training replication (primary recommendation)

1. Use dataset v1.3.2 (or equivalent current dataset).
2. Run grouped MLP training with feature set `price_trend` under LVTF regime, target horizon 24 bars, using `research/deep_learning/train.py` with `--train-pairs`.
3. Compute per-group metrics: accuracy, precision, recall, F1.
4. Rank pairs or groupings by F1 / precision divergence.
5. The grouping that maximises the precision gap between the two families (≈ 25% in historical results) reproduces the partition.

This directly replicates the origin experiment.

### Method B — Cross-family transfer asymmetry (secondary)

1. Train on one candidate family (e.g., reactive-group pairs).
2. Export predictions on the other family (persistent-group pairs) using `--predict-pairs`.
3. Measure transfer degradation (within-family F1 vs cross-family F1).
4. The partition that maximises within-family retention and cross-family degradation is the historical partition.

This method is described in `docs/behavioral/cross_family_transfer_findings.md` and is supported by the `--train-pairs` / `--predict-pairs` infrastructure in `research/deep_learning/train.py`.

### Method C — Sentiment ablation stability contrast (tertiary)

1. Run grouped training with `price_trend` feature set.
2. Re-run with `trend_vol_only` feature set (removing sentiment).
3. Compute delta F1 / precision for each candidate grouping.
4. The partition that shows the strongest contrast between sentiment and no-sentiment conditions (persistent family: higher sentiment sensitivity in routing; reactive family: stable under ablation) replicates the historical findings from `docs/behavioral/sentiment_ablation.md`.

### Method D — Structural stability scoring (experimental, aligned with MPML findings)

1. Construct a stability score per pair based on:
   - F1 variance across walk-forward folds (lower variance = more stable = persistent)
   - Sentiment ablation delta (smaller delta = more structurally driven = reactive)
   - Cross-family transfer degradation (higher degradation = more family-specific = strong boundary)
2. Cluster pairs by stability score.
3. Compare cluster membership to historical partition.

This method is most consistent with the MPML interpretation (information effect, partition effect, learning stability, variance reduction) cited in the problem statement.

---

## Open Questions

1. **Exact grouping algorithm**: The historical grouped training experiment formed the two families from prior qualitative observation. The precise selection criterion for including EURAUD in the persistent family (vs EURCHF in the reactive) is not documented in surviving scripts. It appears to have been informed by ABM fit quality and JPY/CHF behavioral observations rather than a formal algorithm.

2. **LVTF dependency**: `grouped_pair_family_findings.md` planned a follow-up experiment to test whether the family divergence is intrinsic or LVTF-specific. The subsequent regime-free experiments in `cross_family_transfer_findings.md` showed the divergence persisted without LVTF, but the sensitivity to other regimes (HVTF, HVR, LVR) was identified as open.

3. **CHF vs JPY within reactive**: `cross_family_transfer_findings.md` identified possible substructure within the reactive family (CHF-reactive vs JPY-reactive). This raises the question of whether the boundary between persistent and reactive is a hard partition or a continuum.

4. **Dataset version sensitivity**: The historical partition was established on dataset v1.3.2. It is unclear whether the partition is stable across other dataset versions or depends on specific temporal coverage.

5. **Pairs excluded from families**: Several pairs present in the analysis scripts (e.g., AUDUSD, EURCAD, NZDJPY) do not appear in either family. The selection criterion for inclusion/exclusion is not documented.

---

## Suggested Next Experiments

1. **Replicate the origin experiment**: Run `research/deep_learning/train.py` with `--train-pairs EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD` and separately with `--train-pairs USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF` on current dataset under LVTF regime. If the historical precision/F1 gap (~25%) reproduces, the partition is confirmed.

2. **Cross-family transfer matrix**: Train on each family, predict on the other, measure transfer degradation. A strong asymmetric degradation matrix would confirm the partition is a structural boundary.

3. **Exhaustive pair clustering**: Run grouped training experiments for all candidate pair groupings of size 4–6 from the cross-pair universe. Identify the grouping that maximises within-group F1 consistency and between-group F1 divergence. Compare to historical partition.

4. **Stability scoring**: Implement Method D (structural stability scoring) and compare to historical partition. This would operationalise the MPML-aligned interpretation.

5. **Regime expansion**: Test whether the family divergence reproduces under HVTF, HVR, and LVR regimes. If the partition is stable across all regimes, it strengthens the structural stability interpretation.

6. **MPML partition experiment**: Deploy the historical partition into MPML and measure information effect, partition effect, and variance reduction. Compare to a randomised partition as a null baseline. This would directly test the MPML interpretation cited in the problem statement.
