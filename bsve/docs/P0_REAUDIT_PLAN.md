# P0 Re-Audit Plan

## Persistent Family Re-Evaluation

Status: Planned

Related documents:

- Persistent vs Reactive family etymology
- Reactive-JPY findings
- Reactive-CHF findings
- BSVE Synthesis 01

------

# Motivation

The Persistent family was originally identified through grouped behavioral similarities observed during deep-learning experiments, transfer-learning studies, ABM calibration, and MPML development.

Subsequent work produced the Volatility–Trend framework, which partitions Persistent-family behavior using volatility and trend structure.

The Volatility–Trend framework proved useful in downstream modeling tasks and became an important component of the broader BSVE program.

However, recent work on Reactive-JPY and Reactive-CHF has revealed persistence-related mechanisms that were not part of the original Persistent-family investigation.

In particular:

- Reactive-JPY identified maturity-dependent outcome behavior.
- Reactive-CHF identified volatility-conditioned persistence behavior.
- Both studies independently highlighted persistence as a potentially important explanatory variable.

This raises a fundamental question:

> Was the Volatility–Trend framework the best explanation of Persistent-family behavior, or merely the first useful explanation?

The purpose of the re-audit is to answer that question.

------

# Objective

The objective of P0 is not to replace the Volatility–Trend framework.

The objective is to re-examine the Persistent family using the analytical methods developed during the Reactive-family studies and determine whether alternative behavioral structures emerge.

Particular attention will be given to persistence-related phenomena.

------

# Scope

Persistent-family pairs:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

Reactive-family pairs are excluded except where required for comparison.

------

# Core Questions

## P0A: Persistence Structure

How persistent are sentiment states within Persistent-family pairs?

Questions:

- What do episode-duration distributions look like?
- Are persistence distributions similar across pairs?
- Do distinct persistence regimes emerge naturally?

------

## P0B: Volatility and Persistence

Does volatility influence persistence in Persistent-family pairs?

Questions:

- Do low-volatility environments support longer persistence?
- Do volatility-conditioned persistence structures resemble those observed in CHF pairs?
- Is the volatility–persistence relationship family-specific or more general?

------

## P0C: Persistence and Outcomes

Does persistence influence behavioral outcomes?

Questions:

- Are certain persistence regimes associated with elevated crowd-failure rates?
- Do maturity-like effects emerge naturally from persistence duration?
- Do outcome relationships resemble those observed in Reactive-JPY or Reactive-CHF?

------

# Deliverables

The re-audit will initially focus on descriptive and exploratory analysis.

Expected outputs:

- Episode-duration distributions
- Survival analysis
- Volatility-conditioned persistence analysis
- Persistence-conditioned outcome analysis
- Cross-pair comparison

No ontology changes are planned during P0.

------

# Success Criteria

The re-audit will be considered successful if it establishes one of the following:

1. Volatility–Trend framework remains the most informative explanation of Persistent-family behavior.
2. Persistence provides an equally informative but simpler explanation.
3. Persistence and Volatility–Trend framework describe complementary aspects of the same underlying process.
4. A previously unidentified behavioral structure emerges.

Any of these outcomes would materially improve the current understanding of Persistent-family behavior.

------

# Expected Outcome

The most important result of P0 is not a new ontology.

The most important result is determining whether persistence should be treated as a first-class behavioral concept within BSVE.

Only after answering that question should ontology revision or MPML integration proceed.