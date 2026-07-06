# Behavioral Model (Working Hypothesis)

This document describes the current working behavioral interpretation
emerging from the MSML + ABM + MPML research stack.

This is NOT a finalized theory.

It is an evolving synthesis of:
- DL findings
- transfer experiments
- sentiment ablations
- ABM observations
- downstream MPML behavior

The purpose of this document is to capture the current conceptual model
before the findings become fragmented across experiments.

---

# Core Hypothesis

The current working hypothesis is that FX markets contain:

> partially persistent latent behavioral structure
> conditioned by volatility, trend organization,
> liquidity adaptation, and collective positioning dynamics.

Under this interpretation:

- markets are not fully random
- but neither are they directly predictable
- instead, markets may organize into temporary behavioral geometries

which adaptive systems can partially exploit.

---

# Structural Layer

The strongest current evidence suggests that:

- price structure
- volatility structure
- trend persistence

encode substantial latent organization.

This structural layer appears surprisingly robust across:

- MLP models
- LSTM models
- cross-family transfer
- sentiment ablation
- downstream MPML integration

The current interpretation is that:

> a large portion of market organization may emerge from endogenous
> structural adaptation processes rather than sentiment alone.

Possible contributors include:

- volatility clustering
- adaptive liquidity behavior
- trend reinforcement
- positioning persistence
- collective risk management
- dealer/inventory feedback
- macro release synchronization

---

# Sentiment Layer

Sentiment-derived features still appear to contribute meaningful information.

However, the current evidence increasingly suggests that sentiment behaves more like:

> a conditional modulation layer

than a universal directional predictor.

Sentiment may influence:

- release/reversion behavior
- local instability
- behavioral transitions
- crowd synchronization
- volatility response asymmetry
- reactive regime formation

This interpretation is increasingly consistent with:

- persistent/reactive family differentiation
- CHF/JPY asymmetries
- volatility-conditioned behavior
- MPML routing sensitivity

---

# Persistent vs Reactive Systems

One of the strongest emerging findings is that FX pairs may organize into
partially distinct behavioral families.

---

## Persistent Systems

Examples:
- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

These appear to exhibit:

- more stable transfer behavior
- stronger structural persistence
- smoother downstream integration
- lower sensitivity to sentiment ablation

These systems increasingly resemble:

> persistent adaptive structures.

---

## Reactive Systems

Examples:
- USDJPY
- EURJPY
- GBPJPY
- USDCHF
- EURCHF

These appear to exhibit:

- release/reversion dynamics
- higher local instability
- stronger asymmetry
- more heterogeneous downstream behavior

These systems increasingly resemble:

> reactive release systems.

---

# Latent Geometry

The project increasingly interprets market behavior geometrically.

Rather than:
- independent price predictions

the models increasingly appear to learn:

- state transitions
- behavioral neighborhoods
- persistence/release zones
- conditional transition geometry

Under this interpretation:

- trend
- volatility
- sentiment
- and adaptive positioning

collectively shape a latent behavioral manifold.

---

# ABM Relationship

Several observed DL behaviors increasingly resemble ABM dynamics.

Examples include:

- persistence/release cycles
- volatility-conditioned transitions
- clustered adaptation
- family-specific asymmetries
- reactive instability

This does NOT validate ABM directly.

However, the alignment increasingly suggests that:

> DL and ABM may be observing different manifestations
> of the same underlying behavioral organization.

---

# MPML Interpretation

Recent MPML integration experiments suggest that:

- adaptive downstream systems remain surprisingly stable
  after sentiment ablation
- local routing behavior still changes materially
- selector dynamics remain sensitive to behavioral surfaces

This increasingly suggests that:

> adaptive systems exploit multiple overlapping behavioral channels simultaneously.

Possible channels include:

- structural persistence
- sentiment modulation
- volatility geometry
- phase memory
- adaptive routing
- liquidity response

---

# Current Interpretation

The current interpretation is therefore NOT:

> sentiment predicts markets

but rather:

> behavioral structure emerges conditionally,
> persists temporarily,
> and adaptive systems partially exploit that structure.

---

# Important Caveats

This interpretation remains highly exploratory.

Major unresolved risks include:

- temporal leakage
- sparse DL overlap artifacts
- ontology mismatch
- overfitting
- regime instability
- false manifold discovery

The behavioral model should therefore be interpreted as:

> a working research hypothesis,
> not a finalized theory.

---

# Behavioral Characterization

Behavioral experiments are evaluated by the **Behavioral Characterization Framework** (PR5.1).

Reports produced by this framework answer:

> **What have we learned about this Behavioral Surface?**

Each report distinguishes two independent properties for every finding:

**Scientific Interest**

How important or potentially novel would this finding be if confirmed?

**Scientific Confidence**

How strongly is the finding currently supported by available evidence?

A finding may have high Scientific Interest but low Scientific Confidence early in an experiment
programme. As evidence accumulates across experiments, Confidence increases. Interest may decrease
as findings become well-established.

This distinction helps prioritize future research without overstating current evidence.

Reports end with a single **Research Recommendation** derived from the synthesized findings:

- **Proceed to walk-forward evaluation** — when cross-architecture agreement is high.
- **Repeat with more epochs** — when training has not yet converged.
- **Diagnose and repeat** — when training runs failed.
- **Compare with Reactive CHF or Persistent** — when current surface characterization is complete.
- **Insufficient evidence** — when coverage or data volume is too low for reliable characterization.

