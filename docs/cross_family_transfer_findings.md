# Cross-Family Transfer Findings

## Motivation

Earlier experiments suggested that retail FX sentiment may not behave as a
single universal process across all currency pairs.

Several observations increasingly pointed toward structural divergence:

- JPY/CHF pairs consistently behaved differently from EUR/GBP/NZD pairs
- ABM accumulation dynamics fit some pairs well but failed on others
- DL precision and transition behavior differed materially across pair groups
- downstream MPML integration effects were asymmetric rather than random

This motivated a new research direction:

> test whether learned sentiment structure transfers across FX pair families.

The core question became:

> does retail sentiment encode a universal behavioral process,
> or multiple structurally distinct processes?

---

# Hypothesis

Current working hypothesis:

## Persistent / accumulation-oriented family

Examples:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

Characteristics:

- smoother persistence
- slower positioning decay
- accumulation-dominated behavior
- weaker directional transitions
- lower DL precision

---

## Reactive / release-oriented family

Examples:

- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

Characteristics:

- sharper transition structure
- stronger episodic release behavior
- cleaner directional boundaries
- higher DL precision
- stronger volatility sensitivity

---

# Experimental Design

## Dataset

- dataset version: `1.3.2`
- sentiment-driven H1 dataset
- export window:
  - 2019 → 2024
- target horizon:
  - 24 bars

---

## DL Configuration

Model:

- MLP

Feature set:

- `price_trend`

Core features:

- `trend_12b`
- `trend_48b`
- `vol_12b`
- `vol_48b`
- `net_sentiment`
- `abs_sentiment`
- `sentiment_change`
- `sentiment_z`

---

## Regime

Primary experiments used:

- `LVTF`

Reason:

- strongest and most stable signal regime
- best alignment with persistence-style dynamics

Additional regime-agnostic grouped experiments were later performed.

---

# Grouped DL Training Results

## Persistent-family grouped model

Training set:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

### LVTF results

Observed characteristics:

- weaker directional generalization
- smoother prediction structure
- lower precision
- weaker transition sharpness

Representative metrics:

- F1 ≈ 0.34
- precision ≈ 0.29
- recall ≈ 0.43

Interpretation:

The model appears to learn slower accumulation-style structure rather than
sharp directional transitions.

---

## Reactive-family grouped model

Training set:

- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

### LVTF results

Observed characteristics:

- materially stronger precision
- sharper transition behavior
- stronger directional separability

Representative metrics:

- F1 ≈ 0.43
- precision ≈ 0.36
- recall ≈ 0.51

Interpretation:

The model appears to learn more episodic or release-oriented dynamics.

---

# Regime-Free Grouped Experiments

Subsequent experiments removed explicit regime filtering entirely.

Importantly:

the reactive-family advantage persisted.

This suggests that pair-family divergence is not purely an artifact of
LVTF conditioning.

Current interpretation:

> pair-family structure appears at least partially intrinsic.

---

# Cross-Family Transfer Infrastructure

To explicitly test transferability, the DL export pipeline was extended to
support:

- training on one pair family
- exporting inference predictions on another family

New functionality added:

- `--train-pairs`
- `--predict-pairs`

Example:

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --train-pairs EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD \
  --predict-pairs USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend
```

This enables direct testing of whether learned behavioral structure
 generalizes cross-family.

## LSTM Infrastructure Upgrade

LSTM pipeline was upgraded to support:

- metadata-safe sequence export
- cross-family transfer
- MPML-compatible parquet artifacts
- pair-safe grouped sequence construction

This enabled future sequence-based transfer experiments.

------

# MPML Integration Findings

Cross-family DL surfaces were deployed into `market-phase-ml`
 through the DL artifact integration layer.

Observed effects:

- DL altered downstream strategy behavior
- effects were asymmetric and pair-dependent
- some pairs improved materially
- others degraded substantially

Importantly:

effects were structurally coherent rather than random.

------

# Key Finding

DL appears to encode:

- persistence structure
- transition timing
- release behavior
- state instability

rather than universal directional alpha.

The strongest evidence comes from:

- grouped family experiments
- regime-free grouped experiments
- asymmetric downstream MPML behavior
- ABM persistence/release dynamics

------

# Relationship to ABM

The DL findings increasingly align with the ABM persistence/release framework.

## Persistence dynamics

ABM components:

- anchor strength
- reinforcement
- accumulation

Associated with:

- persistent-family pairs
- smoother transitions
- slower structural evolution

------

## Release dynamics

ABM components:

- volatility-conditioned decay
- destabilization
- boundary behavior

Associated with:

- reactive-family pairs
- sharper directional transitions
- episodic instability

------

# Current Interpretation

The project no longer appears consistent with:

- universal sentiment alpha
- one global behavioral mechanism
- purely additive predictive structure

Instead, current evidence increasingly supports:

> conditional, structurally-dependent behavioral predictability.

More specifically:

retail sentiment may encode multiple partially distinct behavioral processes
 across FX markets.

------

# Next Research Direction

Current transfer findings motivate a controlled sentiment ablation phase.

A new `trend_vol_only` feature set has been added to the DL training pipelines
to remove sentiment-derived inputs while preserving trend/volatility structure.

This enables direct tests of whether observed pair-family behavior survives
without sentiment, or whether sentiment is the dominant organizing signal.

------

# Sentiment Ablation Findings (2026-05)

Controlled sentiment ablation experiments were performed using the new:

- `trend_vol_only`

feature set.

This removed all sentiment-derived inputs while preserving:

- trend features
- volatility features

The experiments were replicated across:

- MLP
- LSTM
- persistent-family grouped models
- reactive-family grouped models

---

# Key Result

Behavioral family structure survived sentiment removal.

This is one of the strongest findings so far because it suggests that:

- pair-family divergence is not purely driven by sentiment inputs
- deeper structural dynamics exist in the underlying market process

---

# Persistent-family effects

Removing sentiment typically produced:

- higher raw classification accuracy
- lower persistence-style recall behavior
- weaker continuation bias

Current interpretation:

sentiment primarily acts as a persistence reinforcement mechanism in these markets.

This aligns closely with:

- ABM anchor dynamics
- accumulation behavior
- slow positioning persistence

---

# Reactive-family effects

Reactive-family structure remained comparatively stable after sentiment removal.

Performance degraded modestly but remained structurally coherent.

Current interpretation:

reactive-family dynamics may be more strongly tied to:

- transition structure
- volatility/release dynamics
- instability geometry

rather than sentiment itself.

---

# Cross-Architecture Consistency

Importantly:

both MLP and LSTM produced qualitatively similar ablation behavior.

This suggests the observed effects are unlikely to be:

- shallow architectural artifacts
- sequence-model-specific effects

Instead, the experiments increasingly support the existence of:

- deeper structural behavioral geometry in FX markets.

---

# Updated Working Interpretation

Current evidence increasingly supports a layered interpretation:

## Structural layer

Driven by:

- price dynamics
- volatility structure
- transition geometry
- persistence/release mechanics

---

## Behavioral layer

Driven by:

- sentiment reinforcement
- crowd anchoring
- persistence amplification
- collective positioning pressure

Under this interpretation:

sentiment modulates structural market behavior rather than fully determining it.

---

# Limitations

These findings remain exploratory.

Important caveats:

- limited dataset duration (sentiment begins ~2019)
- single target horizon (24 bars)
- primarily MLP-based experiments so far
- downstream MPML integration currently affects only stages 3b–3c
- cross-family transfer experiments remain ongoing

No claims of robust tradable alpha are implied.

------

# Immediate Next Steps

## 1. Full cross-family transfer evaluation

Determine whether:

- within-family transfer remains stable
- cross-family transfer degrades materially

This is currently the highest-priority discriminator experiment.

------

## 2. LSTM replication

Test whether sequence models:

- strengthen
   or:
- weaken

the observed family divergence.

------

## 3. Regime expansion

Extend experiments beyond LVTF:

- HVTF
- HVR
- LVR

to determine whether:

- pair-family structure survives regime changes
   or:
- regime dominates pair-family effects.

------

## 4. MPML full-pipeline DL integration

Current MPML integration affects:

- phase prediction
- downstream backtests

but not:

- dynamic selector
- volatility guard
- walk-forward orchestration

Future work will investigate:

- behavior-conditioned strategy weighting
- DL-conditioned selector logic
- adaptive strategy activation

------

# LSTM Replication Findings (2026-05)

The cross-family transfer experiments were replicated using the upgraded
LSTM export pipeline.

Importantly, the overall behavioral picture remained broadly consistent with
the earlier MLP findings:

- within-family transfer generally performed better than cross-family transfer
- persistent-family structure remained more accumulation-oriented
- reactive-family structure remained more transition/release-oriented
- transfer asymmetry survived architecture changes

This is important because it suggests the observed family structure is not
merely an artifact of shallow feedforward modeling.

The experiments also strengthened a possible subdivision inside the reactive
family itself:

## CHF-reactive subgroup

Examples:

- USDCHF
- EURCHF

Observed characteristics:

- cleaner transfer behavior
- more stable downstream MPML improvements
- stronger consistency under same-family transfer

---

## JPY-reactive subgroup

Examples:

- USDJPY
- EURJPY
- GBPJPY

Observed characteristics:

- noisier transfer behavior
- stronger instability
- more heterogeneous downstream effects

This raises the possibility that the broader reactive family may itself contain
multiple distinct mechanisms:

- defensive/release dynamics (CHF)
- macro/event-driven destabilization dynamics (JPY)

This is now an active research direction.

A second important finding was that LSTM did not dramatically outperform MLP.

Current interpretation:

DL models may primarily be learning:

- behavioral state geometry
- persistence structure
- transition boundaries
- release dynamics

rather than pure long-memory directional forecasting.

---

# Conclusion

The project has transitioned from:

> searching for universal sentiment alpha

toward:

> identifying behavioral structure and market-specific dynamics.

Current evidence increasingly suggests that:

- some FX markets behave as persistence-dominated systems
- others behave as reactive/release-dominated systems

and that:

DL + ABM together may be detecting different manifestations of the same
 underlying behavioral structure.
