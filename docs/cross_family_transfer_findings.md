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