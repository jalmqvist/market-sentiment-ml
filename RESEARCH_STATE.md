# RESEARCH_STATE.md

Current research interpretation and working hypotheses for
`market-sentiment-ml` (MSML).

---

# Current Research Position

The project has evolved substantially from its original framing.

Early versions of the research focused primarily on:

- directional sentiment prediction
- additive predictive alpha
- supervised forecasting performance

However, the accumulated experimental evidence increasingly suggests that:

> retail sentiment behaves more like a conditional behavioral-state surface
> than a universal directional predictor.

The strongest current evidence supports:

- conditional behavioral structure
- volatility-conditioned organization
- pair-family asymmetries
- latent structural persistence
- sparse downstream behavioral effects

while weakening support for:

- stable universal sentiment alpha
- globally transferable directional prediction
- purely additive sentiment models

The project therefore increasingly studies:

> how behavioral organization emerges under different market conditions,

rather than:

> whether sentiment directly predicts returns.

---

# Current State of Evidence

## Findings with Strongest Support

Current experiments most strongly support:

- persistent/reactive pair-family differentiation
- conditional regime dependence
- volatility-conditioned structure
- latent organization surviving architecture changes
- partial transfer across pair families
- survival of structure after sentiment ablation
- downstream behavioral effects inside adaptive systems
- partial alignment between DL findings and ABM dynamics

---

## Findings with Weak or Mixed Support

Current evidence is weaker or inconsistent for:

- universal predictive alpha
- stable directional forecasting
- globally transferable sentiment behavior
- additive predictive contribution over structural features
- universal regime robustness
- strong standalone trading performance

---

## Current Interpretation

The project increasingly interprets retail sentiment as:

- a contextual behavioral signal
- a conditional modulation layer
- a sparse behavioral-state surface

rather than:

- a standalone predictive engine.

---

# Layered Behavioral Interpretation

A layered interpretation has gradually emerged from the experiments.

---

## Layer 1 — Structural Market Organization

Price, volatility, and trend structure appear to encode substantial
persistent organization.

This structural layer survives:

- sentiment ablation
- architecture changes (MLP ↔ LSTM)
- cross-family transfer
- downstream MPML integration

The current interpretation is that markets may contain:

- endogenous structural persistence
- volatility-conditioned geometry
- adaptive liquidity organization
- self-reinforcing behavioral clustering

that exists independently of explicit sentiment signals.

---

## Layer 2 — Sentiment Modulation

Sentiment-derived features still appear to contribute:

- conditional asymmetries
- release/reversion dynamics
- pair-family differentiation
- localized behavioral transitions
- downstream adaptive-routing effects

The current working hypothesis is therefore:

> sentiment modulates latent behavioral structure
> rather than replacing it.

---

## Layer 3 — Adaptive Exploitation

Downstream experiments in MPML suggest that:

- adaptive systems can preserve substantial behavior
  even after sentiment ablation
- local routing behavior changes more strongly
  than aggregate downstream performance
- structural organization may dominate final ensemble behavior

This suggests that:

> adaptive downstream architectures may exploit multiple partially
> overlapping behavioral channels simultaneously.

---

# Pair-Family Findings

One of the strongest and most persistent findings in the project is that
different FX pair families appear to exhibit meaningfully different
behavioral organization.

---

## Persistent Families

Pairs such as:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

appear to exhibit:

- more stable transfer behavior
- stronger structural persistence
- more robust downstream integration
- less sensitivity to sentiment ablation

These families increasingly resemble:

> persistent structural systems.

---

## Reactive Families

Pairs such as:

- USDJPY
- EURJPY
- GBPJPY
- USDCHF
- EURCHF

appear to exhibit:

- stronger release/reversion dynamics
- higher local instability
- more asymmetric transfer behavior
- more heterogeneous downstream responses

These families increasingly resemble:

> reactive release systems.

---

## CHF vs JPY Separation

Recent findings increasingly suggest that:

- CHF-linked behavior
- and JPY-linked behavior

may themselves contain partially distinct latent structure.

This remains exploratory,
but increasingly appears non-random.

---

# Sentiment Ablation Findings

One of the most important recent findings is that substantial behavioral
structure survives after removing sentiment-derived features.

The `trend_vol_only` feature surface preserves:

- trend features
- volatility features

while removing:

- sentiment-derived inputs

Current results suggest that:

- structure does not collapse completely
- downstream MPML behavior often remains surprisingly stable
- adaptive systems continue functioning coherently
- local routing behavior still changes meaningfully

This increasingly suggests that:

- volatility/trend organization itself encodes latent behavioral geometry
- sentiment acts as a conditional modulation layer
- structural persistence dominates much downstream behavior

---

# Cross-Family Transfer Findings

Cross-family transfer experiments increasingly suggest:

- asymmetric transfer behavior
- partial universality
- family-specific organization
- incomplete generalization

The strongest current interpretation is that:

> markets contain partially shared behavioral structure,
> but organized differently across pair families.

This weakens both extremes:

- purely universal structure
- purely isolated pair behavior

and instead supports:

> partially overlapping latent manifolds.

---

# Architecture Robustness

Observed structure survives across:

- MLP models
- LSTM models
- cross-family transfer
- downstream MPML integration

This weakens the hypothesis that findings are merely:

- architecture artifacts
- optimization artifacts
- isolated modeling effects

while strengthening the interpretation that:

> some form of latent organization genuinely exists in the data.

---

# MPML Integration Findings

Recent full-pipeline experiments integrating MSML DL surfaces into MPML
produced several important observations.

---

## Important Separation

MSML and MPML answer different questions.

### MSML studies:

- behavioral structure
- latent organization
- sentiment-conditioned dynamics
- transfer geometry

### MPML studies:

- adaptive exploitation
- policy routing
- regime-aware downstream behavior
- ensemble robustness

This distinction is extremely important.

MPML findings should therefore NOT be interpreted as direct evidence
for or against the intrinsic importance of sentiment itself.

---

## Current MPML Interpretation

Recent FP experiments suggest that:

- DL signals genuinely propagate through MPML
- walk-forward adaptive systems react to sentiment ablation
- selector behavior changes materially
- fold-level dynamics change materially

However:

- final downstream ensemble behavior often remains surprisingly stable

This increasingly suggests that:

> MPML exploits multiple partially overlapping behavioral channels.

Possible channels include:

- sentiment structure
- volatility structure
- trend persistence
- phase memory
- adaptive routing
- regime geometry

---

# ABM Relationship

Several findings increasingly resemble dynamics observed in ABM work.

Observed parallels include:

- persistence/release cycles
- asymmetric volatility response
- clustered behavioral organization
- reactive regime transitions
- family-dependent behavior

This does NOT validate ABM directly.

However, the alignment increasingly suggests that:

> DL and ABM may be observing different manifestations
> of the same underlying behavioral organization.

---

# Current Scientific Risks

Several important risks remain unresolved.

---

## Leakage Risk

Potential temporal leakage remains one of the most important unresolved
scientific concerns.

This includes:

- rolling normalization leakage
- timestamp alignment leakage
- walk-forward contamination
- export boundary leakage
- future-state contamination

Leakage audits remain a major priority.

---

## Sparse DL Overlap

Current DL surfaces overlap only a subset of the full historical timeline.

This creates several risks:

- sparse-feature dominance
- presence-mask artifacts
- instability in downstream adaptive systems
- misleading aggregate performance stability

DL-era-only experiments remain important.

---

## Ontology Mismatch

MSML and MPML currently use different regime ontologies.

This creates ambiguity when interpreting:

- regime-conditioned behavior
- DL coverage behavior
- downstream routing effects

Regime interpretation therefore requires care.

---

# Immediate Research Priorities

Current priorities include:

---

## Behavioral Research

- CHF vs JPY decomposition
- latent manifold analysis
- transition geometry
- release dynamics
- volatility-conditioned organization

---

## Validation Research

- leakage audits
- randomized DL controls
- shuffled-signal experiments
- sparse-overlap controls
- DL-era-only evaluation

---

## MPML Integration Research

- adaptive routing analysis
- selector sensitivity
- fold-level behavioral dynamics
- multi-surface integration
- downstream robustness analysis

---

## ABM Research

- endogenous regime emergence
- agent coordination
- calibration stability
- persistence/release simulation
- behavioral phase transitions

---

# Current Working Hypothesis

The strongest current working hypothesis is now:

> FX markets contain partially persistent latent behavioral structure
> conditioned by volatility, trend organization, and adaptive participation dynamics.

Within this structure:

- sentiment acts as a conditional modulation layer
- adaptive systems exploit multiple overlapping behavioral channels
- pair families organize differently
- and structural persistence dominates much long-horizon behavior.

The project therefore increasingly studies:

> behavioral geometry and adaptive interaction,

rather than:

> simple directional prediction.

---

# Status

Active exploratory research project.

The current direction increasingly focuses on:

- latent behavioral organization
- structural persistence
- conditional modulation
- adaptive downstream exploitation
- and reconciliation between:
  - DL,
  - ABM,
  - and adaptive policy systems.