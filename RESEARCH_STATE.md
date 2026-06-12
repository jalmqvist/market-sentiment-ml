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

# Current Interpretation

The project increasingly interprets retail sentiment not as a universal directional predictor, but as a behavioral state space whose transition dynamics vary across market environments.

The strongest evidence currently supports:

- family-specific behavioral organization,
- persistence-mediated state evolution,
- conditional information compression,
- localized downstream behavioral effects,
- adaptive exploitation of latent structure.

Under this interpretation, sentiment is valuable primarily because it helps identify and organize behavioral states rather than because it provides stable directional alpha.

Recent investigations further suggest that sentiment-state dynamics themselves may constitute an important object of study.

In particular, JPY environments appear to exhibit consensus-formation and consensus-maturation processes that strongly influence reversal behavior.

This shifts part of the research focus away from:

sentiment
→ returns

and toward:

sentiment-state evolution
→ market behavior.

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

## Consensus-State Dynamics

Recent JPY investigations suggest that sentiment may be most useful when viewed as a dynamic state process rather than as a static explanatory variable.

Several candidate external drivers of sentiment reversals were examined, including:

- high-impact news,
- session structure,
- volatility-related effects.

These variables provided comparatively little explanatory power once sentiment-state maturity was taken into account.

Instead, reversal behavior appears strongly conditioned on the age and maturity of the underlying consensus state.

Current evidence supports the following behavioral chain:

Consensus Formation
→ Consensus Maturation
→ Exit Mechanism
→ Reversal Probability

This finding strengthens the interpretation that sentiment should be studied as a state-transition process rather than as a standalone predictive signal.

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

One of the strongest and most persistent findings in the project is that different FX pair families exhibit meaningfully different behavioral organization.

Importantly, the family distinction does not appear to originate from a single market statistic such as volatility, trend persistence, or sentiment persistence. Historical audits suggest that the family partition emerged empirically through repeated predictive-learning experiments and has subsequently survived multiple independent validation efforts.

Current evidence suggests that pair families should be interpreted as distinct behavioral learning environments rather than simple collections of similar market statistics.

## Persistent Families

Pairs such as:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

consistently exhibit:

- stronger information compression under phase partitioning,
- greater internal cohesion,
- greater benefits from explicit decomposition,
- higher consensus-state maturation rates,
- more stable downstream integration.

These environments increasingly resemble persistent behavioral systems whose predictive structure is comparatively well organized.

## Reactive Families

Pairs such as:

- USDJPY
- EURJPY
- GBPJPY
- USDCHF
- EURCHF

remain meaningfully distinct from Persistent families.

However, recent evidence suggests that Reactive itself is unlikely to be a fully coherent family.

Instead, the Reactive family increasingly appears to contain at least two partially distinct behavioral environments.

## CHF-Reactive Environments

CHF pairs exhibit:

- coherent volatility geometry,
- elevated information gain under persistence-conditioned decomposition,
- volatility-mediated persistence effects,
- strong agreement between EURCHF and USDCHF.

The current working interpretation is:

Volatility Context
→ Crowd-State Persistence
→ Predictive Structure

rather than volatility acting as a direct predictive signal.

## JPY-Reactive Environments

JPY pairs exhibit a different structure.

Recent investigations found that:

- sentiment extremes are overwhelmingly crowd-short,
- reversal probability depends strongly on consensus maturity,
- young consensus states frequently terminate via reversal,
- mature consensus states predominantly decay through threshold exits.

Unlike CHF environments, the strongest explanatory variables appear to arise from the sentiment process itself rather than from external timing variables.

Current evidence therefore supports the provisional interpretation:

Consensus Formation
→ Consensus Maturation
→ Consensus Decay

as the primary organizing geometry of JPY-reactive environments.

The strongest open question is no longer whether JPY environments contain structure, but rather what governs the evolution of their consensus states.

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

The strongest current working hypothesis is that FX markets contain partially persistent behavioral-state structures organized through multiple interacting mechanisms.

Current evidence supports:

- latent structural persistence,
- family-specific information geometry,
- persistence-mediated behavioral organization,
- conditional sentiment-state evolution,
- adaptive downstream exploitation.

Within this framework:

- sentiment acts as a behavioral-state descriptor,
- trend and volatility influence state evolution,
- pair families organize information differently,
- adaptive systems exploit multiple overlapping behavioral channels.

The strongest emerging interpretation is that sentiment contributes less through direct directional prediction and more through its ability to reveal the structure and evolution of behavioral states.

The project therefore increasingly studies:

behavioral geometry,
state-transition dynamics,
consensus formation,
and adaptive interaction,

rather than simple return forecasting.

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

---

# Pair-Family Research Audit (2026-06-12)

This section summarizes the major investigations that contributed to the current pair-family interpretation.

The goal is not to document every experiment, but to preserve the key reasoning steps that motivated the current research state.

------

## Summary Table

| Investigation                       | Main Question                                                | Result                                                       | Confidence  | Implication                                                  |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| Phase decomposition experiments     | Do pairs benefit from explicit regime decomposition?         | Persistent pairs consistently benefited more than Reactive pairs. | High        | First evidence that pair families reflect genuine behavioral differences. |
| Cross-family transfer studies       | Does learned structure transfer between families?            | Transfer was asymmetric and incomplete.                      | High        | Supports partially distinct latent manifolds rather than universal structure. |
| Sentiment ablation experiments      | Does family structure survive sentiment removal?             | Significant structure remained after sentiment removal.      | High        | Family differences are not purely sentiment-driven.          |
| Trend/Volatility-only surfaces      | Can structure survive on price-derived behavioral proxies?   | Significant organization survived.                           | High        | Supports latent behavioral organization beyond sentiment.    |
| MPML integration experiments        | Do family differences survive downstream adaptive exploitation? | Yes. Routing behavior and fold dynamics remained family-dependent. | High        | Family structure propagates into adaptive systems.           |
| CHF decomposition studies           | Why do CHF pairs behave differently?                         | Volatility-conditioned persistence effects repeatedly emerged. | Medium-High | Supports CHF as a volatility-mediated behavioral environment. |
| JPY reversal studies                | Why do JPY pairs exhibit elevated reversal behavior?         | Reversal probability strongly linked to consensus maturity.  | High        | Supports JPY as a consensus-state system.                    |
| JPY news-event analysis             | Are reversals primarily news-driven?                         | Little explanatory power observed.                           | Medium      | Weakens external-event explanations.                         |
| JPY session-dependence analysis     | Are reversals concentrated in specific trading sessions?     | Little explanatory power observed.                           | Medium      | Weakens session-structure explanations.                      |
| JPY state-transition analysis       | How do extreme sentiment states evolve?                      | Young states frequently failed; mature states persisted.     | High        | Consensus maturity emerged as a key state variable.          |
| JPY lifecycle analysis              | What determines reversal vs threshold exits?                 | Immature states died via reversal; mature states via threshold exits. | High        | Reversal risk appears governed by state maturity.            |
| JPY sentiment-reset analysis        | What characterizes reversal events?                          | Reversals frequently associated with large sentiment resets. | High        | Suggests abrupt consensus failure rather than gradual decay. |
| JPY directional asymmetry analysis  | Are all extremes equivalent?                                 | Extremes were overwhelmingly crowd-short.                    | High        | Revealed strong directional asymmetry in JPY sentiment geometry. |
| CHF directional asymmetry analysis  | Are CHF extremes symmetric?                                  | Extremes were overwhelmingly crowd-long.                     | High        | Revealed mirror-image asymmetry relative to JPY.             |
| Trend-strength vs maturity analysis | Does price persistence affect consensus evolution?           | Stronger trends increased maturation probability.            | Medium      | Suggests trend persistence may influence state evolution.    |

------

## Major Conclusions

### Conclusion 1: Pair Families Are Real

The original Persistent vs Reactive distinction was not based on a single metric.

Instead, it emerged repeatedly across:

- decomposition experiments,
- transfer experiments,
- sentiment-ablation experiments,
- downstream MPML integration.

The family structure has survived multiple independent validation attempts and is therefore considered one of the strongest findings in the project.

------

### Conclusion 2: Reactive Is Not a Single Family

Subsequent investigation suggests that the original Reactive family contains at least two distinct behavioral environments:

Reactive-CHF
and
Reactive-JPY.

The distinction emerged independently from:

- volatility studies,
- sentiment-state studies,
- directional crowding studies,
- transition-geometry analysis.

------

### Conclusion 3: CHF and JPY Appear Organized by Different Mechanisms

Current evidence suggests:

CHF:
Volatility Context
→ Persistence Dynamics
→ Predictive Structure

JPY:
Consensus Formation
→ Consensus Maturation
→ Consensus Decay

Both families are reactive, but appear reactive for different reasons.

------

### Conclusion 4: Consensus Maturity Is a Core State Variable

The strongest result from the JPY investigation was the discovery that reversal probability depends heavily on consensus maturity.

Observed pattern:

Young Consensus
→ Reversal-Dominated

Mature Consensus
→ Threshold-Dominated

This result survived multiple independent analyses:

- hazard analysis,
- state-transition analysis,
- lifecycle analysis,
- directional decomposition.

Consensus maturity is therefore currently considered one of the most important state variables identified in the project.

------

### Conclusion 5: External Timing Variables Were Less Important Than Expected

Several candidate explanations for JPY reversals were investigated:

- high-impact news,
- medium-impact news,
- session structure,
- event timing.

None produced explanatory power comparable to consensus maturity.

This shifted the interpretation from:

External Event
→ Reversal

toward:

State Evolution
→ Reversal

------

### Conclusion 6: Trend Persistence May Influence State Evolution

Recent exploratory work found that stronger price-persistence environments were associated with higher consensus-maturation probabilities.

This effect was observed using both:

- 12-bar trend-strength measures,
- 48-bar trend-strength measures.

The finding is currently considered preliminary.

However, it represents one of the few external variables that exhibited a meaningful relationship with sentiment-state evolution.

Current interpretation:

Trend Persistence
→ Consensus Maturation
→ Exit Mechanism
→ Reversal Probability

Further validation remains necessary.

------

## Current Confidence Ranking

Highest Confidence Findings:

1. Pair families are real.
2. Persistent and Reactive environments differ meaningfully.
3. CHF and JPY should not be treated as a single Reactive family.
4. Consensus maturity governs reversal risk.
5. JPY extremes are overwhelmingly crowd-short.
6. CHF extremes are overwhelmingly crowd-long.

Moderate Confidence Findings:

1. CHF organization is volatility-mediated.
2. Trend persistence influences consensus maturation.
3. Consensus-state analysis provides a useful description of JPY behavior.

Exploratory Findings:

1. Trend alignment effects.
2. Hidden-state (HMM) representations of sentiment processes.
3. Cross-broker validation of directional crowding asymmetries.
4. Consensus-state generative modeling.

---

## Supporting Artifacts

The conclusions summarized in this section are supported by a collection of standalone analysis scripts, logs, exploratory studies, and intermediate research artifacts that may not be tracked directly within the main repository.

These artifacts were retained because many of the conclusions emerged through iterative investigation rather than through a single experiment.

The purpose of this audit is therefore not to replace the underlying analyses, but to preserve the reasoning chain that led to the current research interpretation.

Future revisions of the research state should update both:

- the current interpretation,
- and the supporting audit trail,

so that major conclusions remain traceable even as the project evolves.

---

