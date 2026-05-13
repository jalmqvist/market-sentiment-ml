# Grouped Pair-Family Findings

## Motivation

Previous experiments suggested that:
- signal quality varies strongly by pair
- regime conditioning matters
- ABM persistence/release dynamics differ across markets

Hypothesis:
FX pairs may separate into distinct behavioral “families”.

---

## Experimental Setup

Dataset:
- v1.3.2

Model:
- MLP

Features:
- price_trend

Target:
- 24-bar directional threshold prediction

Regime:
- LVTF

Training:
- grouped multi-pair training
- same architecture
- same hyperparameters
- same export logic

---

## Pair Groups

### Persistent / accumulation-oriented
- EURUSD
- GBPUSD
- NZDUSD
- EURGBP
- EURAUD

### Reactive / release-oriented
- USDJPY
- EURJPY
- GBPJPY
- EURCHF
- USDCHF

---

## Results

| Group      | Accuracy | Precision | Recall | F1    | Positive Rate |
| ---------- | -------- | --------- | ------ | ----- | ------------- |
| Persistent | 0.6020   | 0.288     | 0.431  | 0.345 | 0.364         |
| Reactive   | 0.6125   | 0.364     | 0.512  | 0.426 | 0.394         |

---

## Interpretation

The reactive-family generalized materially better despite:
- identical architecture
- identical feature set
- identical target horizon
- identical training procedure

This suggests:
- pair-dependent sentiment dynamics
- potentially different underlying market microstructure
- different stability/persistence properties

---

## ABM Alignment

The findings align with current ABM behavior:

Persistent-family:
- accumulation dominates
- stable directional clustering
- strong anchor dynamics

Reactive-family:
- stronger release dynamics
- more boundary-sensitive behavior
- volatility-conditioned destabilization

---

## Important Dataset Observation

The sentiment dataset is snapshot-driven rather than strictly bar-driven.

Multiple intra-hour sentiment snapshots can map to the same H1 entry bar.

DL export artifacts therefore collapse:
(pair, entry_time)
to a single H1 state representation by averaging probabilities.

The underlying dataset remains event-level internally.

---

## Current Hypothesis

EUR/GBP/NZD:
- persistence-dominated
- slower structural positioning
- accumulation-oriented

JPY/CHF:
- macro/flow reactive
- release-sensitive
- volatility-conditioned

---

## Next Planned Experiment

Repeat grouped-family experiments:
- without regime filtering

Goal:
determine whether the family divergence:
- is intrinsic
or
- emerges specifically under LVTF conditioning.