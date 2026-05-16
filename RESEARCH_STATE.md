# Research State Summary — Market Sentiment ML

## Objective

Determine whether retail FX sentiment contains causal, exploitable predictive structure.

The project has evolved from:

- “does sentiment predict returns?”
to:
- “under what structural conditions does sentiment become conditionally informative?”

---

# Current Status

| Area                      | Status                           |
| ------------------------- | -------------------------------- |
| Data quality              | ✅ verified                       |
| Pipeline correctness      | ✅ verified                       |
| Validation framework      | ✅ strong                         |
| Price signal              | ✅ stable (~0.14 Sharpe baseline) |
| Sentiment additive signal | ❌ not supported                  |
| DL signal                 | ⚠ weak but structured            |
| MPML integration          | ✅ operational                    |
| ABM baseline              | ✅ stable                         |
| ABM release dynamics      | ✅ emerging evidence              |

---

# Core Finding

Retail sentiment does not appear to provide:

- standalone predictive signal
- simple additive alpha
- universal directional predictability

However:

> weak but persistent conditional structure exists under specific behavioral and market conditions.

The project is now focused on identifying and explaining those conditions.

---

# Current Priority

The immediate goal is not broader architecture expansion, but:

- validating whether behavioral-family structure is real
- determining whether it is:
  - temporal
  - regime-conditioned
  - volatility-conditioned
  - or macro/reactive in nature

Current priority experiments:

1. Cross-family transfer
2. LSTM transfer replication
3. Regime-free grouped experiments
4. ABM persistence vs release reconciliation

---

# Deep Learning — Controlled Signal Cartography

## Setup

Fixed configuration:

- target horizon = 24
- quantile = 0.50

Models:

- MLP
- LSTM

Evaluation:

- weighted F1
- grouped multi-pair experiments
- regime-conditioned and regime-agnostic tests

---

# Key Findings

## Weak signal exists

Typical DL performance:

- F1 ≈ 0.25–0.50
- highly conditional
- unstable as a universal predictor
- but reproducible across experiments

Importantly:

- signal structure survives across architectures
- sequence models often outperform static models
- downstream effects persist after MPML integration

---

# Regime Structure

## Empirical regime hierarchy

| Regime | Interpretation          |
| ------ | ----------------------- |
| LVTF   | strongest / most stable |
| HVR    | moderate                |
| LVR    | sparse / unstable       |
| HVTF   | weak / noisy            |

### Interpretation

Predictability improves when markets exhibit:

- directional persistence
- structural stability
- slower state transitions

Signal structure changes substantially when:

- volatility dominates
- macro/reactive flow dynamics dominate

---

# Pair-Family Structure (NEW)

Recent grouped multi-pair experiments suggest that FX pairs may separate into distinct behavioral “families”.

This divergence survives removal of explicit regime filtering.

That suggests the structure may be intrinsic rather than purely regime-conditioned.

---

## Persistent / accumulation-oriented family

Examples:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

Observed behavior:

- weaker directional generalization
- smoother persistence structure
- accumulation-dominated dynamics
- lower precision
- stable / slow-moving behavior

DL characteristics:

- lower precision
- weaker transition sharpness
- smoother prediction surfaces

Interpretation:

Retail sentiment may behave more like slow-moving positioning pressure.

---

## Reactive / release-oriented family

Examples:

- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

Observed behavior:

- stronger directional generalization
- materially higher precision
- sharper state transitions
- stronger instability / release dynamics

DL characteristics:

- cleaner positive predictions
- sharper boundary behavior
- stronger episodic structure

Interpretation:

Retail sentiment may behave more like reactive / macro-sensitive positioning dynamics.

---

# Regime-Conditioned vs Intrinsic Structure

Earlier experiments suggested that:

- pair differences might primarily emerge inside LVTF regimes

However:

grouped experiments without regime filtering still produced:

- stronger precision
- stronger F1
- cleaner directional structure

for the reactive-family pairs.

Current interpretation:

> pair-family structure appears to be at least partially intrinsic.

---

# Sentiment Ablation Findings (NEW)

Controlled sentiment ablation experiments were performed using the new:

- `trend_vol_only`

feature configuration.

This removes all sentiment-derived inputs while preserving:

- trend structure
- volatility structure

The experiments were replicated across:

- MLP
- LSTM
- persistent-family grouped models
- reactive-family grouped models

---

## Key Finding

Behavioral family structure survives sentiment removal.

This is important because it suggests that:

- pair-family divergence is not purely sentiment-driven
- deeper structural dynamics exist in price/volatility behavior itself

However:

sentiment removal materially changes model behavior.

---

## Persistent-family interpretation

Removing sentiment generally produced:

- higher raw accuracy
- lower recall
- reduced persistence-style continuation behavior

Current interpretation:

sentiment appears to act primarily as:

> a persistence reinforcement / accumulation layer

rather than a pure directional predictor.

This aligns closely with ABM anchor/reinforcement dynamics.

---

## Reactive-family interpretation

Reactive-family structure survived sentiment removal more cleanly.

Performance degraded moderately, but not catastrophically.

Current interpretation:

reactive-family behavior may be more deeply rooted in:

- transition geometry
- release dynamics
- volatility-conditioned instability

rather than sentiment alone.

---

## Current Conceptual Interpretation

The project increasingly suggests a layered market structure:

### Price/volatility layer

Represents:

- structural market geometry
- transition mechanics
- persistence boundaries
- volatility clustering
- instability dynamics

---

### Sentiment layer

Represents:

- collective reinforcement
- positioning persistence
- crowd anchoring
- behavioral amplification

---

Current interpretation:

sentiment modulates underlying structural dynamics rather than fully generating them.

---

---

# MPML Downstream Ablation Findings (NEW)

Controlled DL ablation artifacts were propagated through the
`market-phase-ml` downstream pipeline.

Experiments compared:

- baseline (no DL)
- DL with sentiment
- DL without sentiment (`trend_vol_only`)

across both:

- persistent-family pairs
- reactive-family pairs

---

## Key Finding

DL downstream effects survived sentiment removal surprisingly well.

This is important because it suggests that:

- DL is not primarily extracting raw sentiment alpha
- downstream strategy changes are not purely sentiment-driven
- deeper structural information exists in the price/volatility layer

---

## Persistent-family behavior

Persistent-family pairs continued to show substantial downstream
improvements after sentiment removal.

Examples included:

- EURUSD
- EURGBP
- EURAUD

Interpretation:

persistent-family structure appears strongly tied to:

- accumulation geometry
- persistence timing
- structural continuation behavior

while sentiment primarily reinforces those dynamics.

---

## Reactive-family behavior

Reactive-family pairs also retained substantial DL-driven downstream effects
under ablation.

Interpretation:

reactive-family structure appears strongly connected to:

- transition timing
- instability/release dynamics
- volatility-conditioned structural changes

rather than sentiment alone.

---

## Updated Structural Interpretation

Current evidence increasingly supports a layered interpretation:

### Structural layer

Driven by:

- trend geometry
- volatility structure
- persistence boundaries
- transition dynamics
- release behavior

This layer appears capable of generating substantial downstream DL effects
even without sentiment inputs.

---

### Sentiment layer

Driven by:

- crowd persistence
- reinforcement
- positioning inertia
- behavioral amplification

Current interpretation:

sentiment modulates and amplifies structural dynamics rather than fully
creating them.

---

## Current Working Interpretation

The strongest current interpretation is:

> DL models are primarily learning behavioral state geometry,
> while sentiment acts as a secondary reinforcement layer.

---

# MPML Integration Findings

Cross-repo integration between:

- market-sentiment-ml
and:
- market-phase-ml

is now operational.

DL prediction artifacts:

- export successfully
- attach to MPML phase pipelines
- alter downstream ML behavior
- alter downstream backtest behavior

Importantly:

DL integration effects are:

- asymmetric
- pair-dependent
- structurally coherent

rather than random.

---

# Current Interpretation of DL Effects

DL does not primarily appear to improve:

- raw classification accuracy globally

Instead, current evidence suggests DL may improve:

- transition sensitivity
- persistence estimation
- timing
- trade filtering
- regime-state confidence

This is especially visible in downstream MPML backtests where:

- classification improvements are sometimes modest
- but strategy behavior changes materially

---

# Agent-Based Modeling (ABM)

## Current Model

ABM currently reproduces:

- accumulation
- persistence
- clustering
- path dependence
- asymmetric reinforcement

The baseline ABM is now stable and reproducible.

---

# Stage-2 Release Dynamics

Recent work introduced:

- volatility-conditioned decay / release dynamics

Key finding:

- “release” behavior can emerge without requiring full sign flips

Observed effects:

- reduced internal conviction magnitude
- increased boundary time
- reduced saturation persistence

This increasingly aligns with DL observations in reactive-family pairs.

---

# Current ABM Interpretation

ABM now appears to model two partially distinct processes:

## Persistence dynamics

Driven by:

- anchor strength
- reinforcement
- accumulation

Associated with:

- smoother transitions
- stable positioning
- persistent-family behavior

---

## Release dynamics

Driven by:

- volatility-conditioned decay
- destabilization
- boundary behavior

Associated with:

- reactive-family behavior
- episodic instability
- sharper directional transitions

---

# DL ↔ ABM Relationship

DL is not validating ABM directly.

Instead:

> DL provides empirical constraints on plausible behavioral mechanisms.

ABM increasingly appears capable of explaining:

- persistence structure
- release dynamics
- pair-family divergence
- regime-conditioned predictability

The two approaches are beginning to converge conceptually.

---

# Important Dataset Observation

The sentiment dataset is snapshot-driven rather than strictly bar-driven.

Multiple intra-hour sentiment snapshots can map to the same H1 entry bar.

Operational DL artifacts therefore collapse:

(pair, entry_time)

to a single H1 representation during export.

The underlying dataset remains event-level internally.

---

# What Has Been Ruled Out

The following do not appear sufficient:

- raw sentiment signal
- additive sentiment alpha
- simple threshold conditioning
- static nonlinear interactions
- universal cross-pair signal structure

---

# What Remains Open

## Behavioral structure

- why pair families diverge
- why reactive-family pairs generalize better
- why persistence-family pairs appear smoother but less exploitable

---

## Regime interaction

- how volatility interacts with persistence
- how release dynamics emerge
- how structural transitions occur

---

## DL structure

- whether family-specific DL surfaces transfer cross-family
- whether sequence models exploit transition timing
- whether confidence structure matters more than direction

---

## ABM reconciliation

- whether one unified ABM is sufficient
- whether multiple behavioral families are required
- whether release dynamics explain reactive-family behavior

---

# Immediate Next Steps

## 1. Cross-family transfer experiments

Test whether:

- persistent-family DL surfaces
generalize poorly to:
- reactive-family pairs

and vice versa.

This is currently the highest-value discriminator experiment.

---

## 2. Coverage diagnostics

Quantify:

- DL row coverage
- effective sample reduction
- conditional filtering effects

to separate:

- genuine predictive improvement
from:
- implicit selective sampling.

---

## 3. ABM refinement

Focus on:

- release dynamics
- boundary behavior
- volatility-conditioned destabilization

Avoid major complexity expansion.

---

## 4. Controlled DL continuation

Avoid:

- large hyperparameter sweeps
- architecture proliferation

Prioritize:

- interpretable experiments
- structural clarity
- reproducibility

---

# Research Phase

The project has transitioned from:

“signal discovery”

to:

> structure discovery and behavioral reconciliation

---

# Working Hypothesis

Retail FX sentiment may not represent a single universal behavioral process.

Current evidence suggests at least two broad structures:

## Persistence-dominated markets

Characteristics:

- smoother accumulation
- slower positioning decay
- persistent clustering
- weaker transition signal

Examples:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

---

## Reactive / release-dominated markets

Characteristics:

- sharper boundary behavior
- stronger instability dynamics
- more episodic directional unwinds
- cleaner transition structure

Examples:

- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

---

These findings are now supported across:

- grouped DL experiments
- MPML downstream integration
- ABM persistence/release dynamics

Current interpretation:

ABM may be approximating:

- persistence via anchor dynamics
- release via volatility-conditioned decay

while DL empirically detects the resulting structural differences.

---

# Conclusion

The project no longer supports:

- universal sentiment alpha
- simple additive predictive signal

Instead, current evidence increasingly supports:

> conditional, structurally-dependent behavioral predictability.

The central question is now:

> what mechanisms generate that structure?