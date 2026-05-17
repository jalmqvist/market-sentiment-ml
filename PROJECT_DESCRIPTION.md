# Project Description

This project investigates whether retail FX sentiment data contains
conditional behavioral structure under different market conditions.

The research combines:

- deep learning
- behavioral modeling (ABM)
- transfer learning
- regime analysis
- downstream adaptive-system integration

to study how behavioral organization emerges in FX markets.

The project no longer treats sentiment as a simple directional predictor.

Instead, the current research focus is:

> understanding when and why behavioral structure becomes conditionally informative.

---

# Core Research Question

The central question is no longer simply:

> "Does sentiment predict returns?"

Current research instead asks:

> Does retail sentiment encode latent behavioral structure
> that interacts with volatility, trend organization,
> and adaptive market participation?

The current interpretation increasingly suggests that:

- markets may contain partially persistent behavioral geometry
- sentiment acts as a conditional modulation layer
- adaptive systems exploit multiple overlapping behavioral channels

rather than sentiment behaving as a standalone predictive signal.

---

# Scope

This is a quantitative behavioral-research and data-engineering project,
not a production trading strategy.

The project focuses on:

- behavioral signal discovery
- latent structure analysis
- transfer-learning experiments
- strict validation methodology
- ABM reconciliation
- downstream behavioral-surface integration

The objective is not merely to maximize predictive performance,
but to understand:

- when structure emerges
- how structure persists
- how adaptive systems react to behavioral surfaces
- and where behavioral information breaks down

---

# Data & Feature Construction

The dataset consists of multi-year time series of:

- retail trader positioning
- FX price data (OHLC)
- derived market features
- behavioral features
- volatility and trend features

The pipeline constructs a unified research dataset by:

- aggregating sentiment snapshots
- aligning sentiment with FX bars
- computing forward evaluation targets
- engineering causal behavioral features

Examples include:

- sentiment extremes
- positioning persistence
- divergence vs price
- volatility-conditioned dynamics
- trend-strength features
- rate-of-change structure

All features are designed to remain:

> causal (past-only)

with strict timestamp alignment and reproducibility guarantees.

---

# Validation Framework

A major contribution of the project is a strict validation methodology.

Signals are evaluated using:

- temporal shift tests
- shuffle/randomization controls
- out-of-sample validation
- walk-forward evaluation
- cross-family transfer tests
- sentiment-ablation experiments

The project increasingly treats:

> negative findings as important scientific results,

rather than failures.

This framework is designed to reduce:

- leakage
- false discovery
- pipeline-induced artifacts
- overfit interpretations

---

# Current Understanding

Current evidence does NOT strongly support:

- universal sentiment alpha
- stable directional prediction
- additive predictive power over price
- globally transferable sentiment behavior

However, current experiments DO increasingly support:

- weak conditional predictive structure
- pair-family asymmetries
- volatility-conditioned organization
- latent structural persistence
- partial transfer learning
- downstream adaptive effects
- partial alignment between DL and ABM dynamics

The strongest current interpretation is therefore:

> sentiment reflects and modulates market behavior
> more than it directly predicts returns.

---

# Structural vs Sentiment Layers

Recent experiments increasingly suggest that market behavior may contain
multiple partially overlapping layers.

---

## Structural Layer

Price, volatility, and trend organization appear to encode substantial
persistent structure.

This layer survives:

- architecture changes
- sentiment ablation
- cross-family transfer
- downstream MPML integration

This suggests that:

> endogenous structural organization may dominate much long-horizon behavior.

---

## Sentiment Layer

Sentiment-derived features still appear to contribute:

- asymmetry
- release/reversion behavior
- reactive transitions
- local instability
- adaptive-routing sensitivity

The current interpretation increasingly suggests that:

> sentiment acts as a conditional modulation layer
> rather than a universal predictive engine.

---

# Research Directions

The project currently explores several complementary directions.

---

## 1. Deep Learning Behavioral Surfaces

The DL pipelines study whether sequence models can detect:

- weak conditional structure
- latent behavioral organization
- transition geometry
- family-specific behavior

Current models include:

- MLP
- LSTM

with support for:

- transfer learning
- grouped-family experiments
- metadata-safe export
- sentiment ablation

---

## 2. Agent-Based Modeling (ABM)

ABM experiments attempt to reproduce observed dynamics through:

- simulated retail participation
- persistence/release behavior
- endogenous regime formation
- agent coordination dynamics

The goal is not simply simulation accuracy,
but theoretical interpretation of observed DL behavior.

---

## 3. MPML Integration

Behavioral surfaces exported from MSML can propagate into:

- walk-forward adaptive systems
- dynamic strategy routing
- selector training
- volatility gating
- downstream ensemble behavior

This downstream layer studies:

> how adaptive systems interact with behavioral information surfaces.

---

# Current Working Hypothesis

The strongest current working hypothesis is now:

> FX markets contain partially persistent latent behavioral structure
> conditioned by volatility, trend organization,
> liquidity adaptation, and collective positioning dynamics.

Within this structure:

- sentiment acts as a conditional modulation layer
- adaptive systems exploit overlapping behavioral channels
- pair families organize differently
- and structural persistence dominates much long-horizon behavior.

---

# Role of This Repository

This repository represents:

- a behavioral-structure research framework
- a validated experimentation environment
- a transfer-learning research platform
- a foundation for ABM/DL reconciliation
- a producer of downstream behavioral surfaces

It should NOT primarily be interpreted as:

> a standalone trading system.

---

# Dataset Versioning

The research dataset is fully versioned and reproducible.

Each dataset version is stored in:

```text
data/output/<version>/
```

alongside manifests containing:

- build parameters
- feature schemas
- preprocessing metadata
- export diagnostics

This supports:

- reproducible ML experiments
- ABM alignment
- downstream MPML integration
- metadata-safe behavioral-surface export

See:

- `docs/data/DATASET.md`

------

# Philosophy

The project increasingly treats markets as:

> adaptive behavioral systems,

rather than:

> stationary prediction problems.

The philosophy of the project is therefore to:

- prioritize correctness over results
- treat negative findings as valuable
- separate discovery from validation
- avoid overclaiming
- and rigorously test behavioral hypotheses

The goal is not to confirm a narrative,
 but to determine:

> whether meaningful behavioral structure exists at all —
>  and how adaptive systems interact with it.
