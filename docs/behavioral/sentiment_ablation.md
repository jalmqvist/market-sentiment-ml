# Sentiment Ablation

This document summarizes sentiment-ablation experiments performed in MSML
and downstream MPML integration experiments.

The purpose of these experiments is to determine:

> how much observed behavioral structure depends directly on sentiment-derived inputs.

---

# Experiment Design

A new feature surface was introduced:

```text
trend_vol_only
```

This feature surface preserves:

- trend features
- volatility features

while removing:

- sentiment-derived inputs

This allows controlled testing of whether observed structure survives
 without explicit sentiment information.

------

# Hypothesis

If sentiment is the dominant source of behavioral organization, then:

- removing sentiment should strongly degrade:
  - transfer behavior
  - downstream integration
  - adaptive routing
  - MPML stability

If substantial structure survives, then:

- latent organization may exist independently of sentiment.

------

# MSML Findings

Initial MSML experiments showed that:

- structure survives surprisingly well after sentiment removal
- persistent families remain relatively stable
- reactive families degrade somewhat more strongly
- cross-family organization still partially survives

This suggests that:

> volatility/trend organization itself may encode substantial latent structure.

------

# MPML Full-Pipeline Experiments

Recent full-pipeline experiments propagated ablated DL surfaces through:

- walk-forward prediction
- selector training
- dynamic strategy routing
- volatility gating
- adaptive downstream systems

These experiments are especially important because they test whether:

> downstream adaptive systems still behave coherently after sentiment removal.

------

# Persistent Family Results

Persistent-family FP experiments compared:

## A — With sentiment

vs

## B — Without sentiment (`trend_vol_only`)

Pairs:

- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

------

## Final FP Results (Total Return (%))

| Pair   | With Sentiment | Without Sentiment | Delta |
| ------ | -------------- | ----------------- | ----- |
| EURAUD | 55.53          | 55.38             | -0.15 |
| EURGBP | 32.01          | 31.36             | -0.65 |
| EURUSD | 19.96          | 20.00             | +0.04 |
| GBPUSD | -9.45          | -9.87             | -0.42 |
| NZDUSD | 83.01          | 82.19             | -0.82 |

Result:

- final downstream behavior remained remarkably stable

------

## Dynamic Selector Results (Total Return (%))

However, local adaptive behavior changed materially.

Examples:

| Pair   | Dynamic (Sentiment) | Dynamic (Ablated) |
| ------ | ------------------- | ----------------- |
| EURAUD | 98.43               | 107.94            |
| GBPUSD | 27.32               | 40.60             |
| EURUSD | 56.18               | 49.11             |

This suggests that:

- adaptive routing reacts to sentiment removal
- but ensemble-level behavior remains surprisingly robust

------

# Reactive Family Results

Reactive-family FP experiments compared:

## C — With sentiment

vs

## D — Without sentiment (`trend_vol_only`)

Pairs:

- USDJPY
- EURJPY
- GBPJPY
- USDCHF
- EURCHF

------

## Final FP Results (Total Return (%))

| Pair   | With Sentiment | Without Sentiment | Delta |
| ------ | -------------- | ----------------- | ----- |
| EURCHF | 6.41           | 5.78              | -0.63 |
| EURJPY | -0.86          | 2.29              | +3.15 |
| GBPJPY | -9.62          | -9.44             | +0.18 |
| USDCHF | 38.71          | 37.12             | -1.59 |
| USDJPY | -20.32         | -21.90            | -1.58 |

Again:

- final downstream behavior remained relatively stable

------

## Dynamic Selector Results (Total Return (%))

However, adaptive routing behavior again changed materially.

Examples:

| Pair   | Dynamic (Sentiment) | Dynamic (Ablated) |
| ------ | ------------------- | ----------------- |
| EURJPY | -23.39              | -9.95             |
| EURCHF | 25.57               | 29.41             |
| GBPJPY | -12.22              | -23.54            |

------

# Current Interpretation

The strongest current interpretation is now:

> substantial latent behavioral structure survives after sentiment removal.

This increasingly suggests that:

- volatility/trend organization encodes persistent structure
- sentiment acts as a conditional modulation layer
- adaptive downstream systems exploit multiple overlapping channels

------

# Important Nuance

These findings do NOT imply that sentiment is unimportant.

Instead, current evidence suggests that:

- structural organization dominates long-horizon behavior
- while sentiment contributes:
  - asymmetry
  - instability
  - local routing behavior
  - release dynamics
  - conditional transitions

------

# Current Working Hypothesis

The current working interpretation is therefore:

> sentiment modulates latent structural organization
>  rather than creating it entirely.

------

# Important Limitations

Several important limitations remain unresolved.

These include:

- sparse DL overlap
- possible leakage
- ontology mismatch between MSML and MPML
- limited pair counts
- limited regime coverage
- potential adaptive compensation effects inside MPML

These findings should therefore be interpreted as:

> exploratory evidence for structural persistence,
>  not definitive proof.