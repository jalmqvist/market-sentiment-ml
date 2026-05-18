# Market Sentiment ML (MSML)

Behavioral structure research for retail FX sentiment.

---

# Executive Summary

This project studies whether retail FX sentiment contains conditional
behavioral structure under different market conditions.

The research combines:

- deep learning
- behavioral modeling (ABM)
- regime analysis
- cross-pair transfer experiments
- downstream integration into adaptive trading systems

The project no longer treats sentiment as a simple directional predictor.

Instead, the current research focus is:

> understanding when and why behavioral structure becomes conditionally informative.

Current evidence suggests that:

- raw sentiment is mostly noisy
- additive sentiment alpha is weak
- universal predictive behavior does not emerge
- but conditional behavioral structure appears to exist

That structure appears:

- regime-dependent
- pair-family-dependent
- volatility-conditioned
- partially transferable across models
- and partially aligned with ABM-derived behavioral hypotheses

---

# Repository Role

This repository (`market-sentiment-ml`) focuses on:

- behavioral signal discovery
- sentiment structure analysis
- transfer-learning experiments
- deep-learning behavioral surfaces
- sentiment ablation experiments
- ABM reconciliation

Downstream exploitation experiments are performed in:

- `market-phase-ml` (MPML)

where exported DL behavioral surfaces are integrated into a
regime-aware adaptive strategy-selection pipeline.

MSML should therefore primarily be interpreted as:

> a behavioral structure research framework,

rather than:

> a standalone trading system.

---

# Current State of Evidence

The strongest current evidence does **not** support:

- universal sentiment alpha
- stable standalone directional predictability
- globally transferable sentiment behavior
- additive predictive power over price/volatility structure

However, current experiments **do** support:

- weak conditional predictive structure
- pair-family asymmetries
- regime dependence
- volatility-conditioned behavior
- latent structural organization
- persistent downstream effects after DL integration
- partial alignment between DL findings and ABM dynamics

Current evidence increasingly suggests that:

> sentiment may act less like a direct predictor
> and more like a conditional behavioral-state surface.

---

# Behavioral Interpretation

A layered interpretation has gradually emerged from the experiments.

## Structural Layer

Price, volatility, and trend organization appear to encode a large amount
of persistent market structure.

This layer survives:

- sentiment ablation
- architecture changes (MLP ↔ LSTM)
- cross-family transfer
- downstream MPML integration

This suggests that a substantial portion of observed market organization
may emerge from:

- volatility clustering
- trend persistence
- liquidity adaptation
- endogenous structural feedback

rather than sentiment alone.

## Sentiment Layer

Sentiment-derived features still appear to contribute:

- conditional asymmetries
- release/reversion behavior
- pair-family differentiation
- localized behavioral transitions
- sparse downstream routing effects

The current working hypothesis is therefore:

> sentiment modulates behavioral structure
> rather than replacing it.

---

# Research Architecture

The broader research ecosystem is currently organized into three layers.

## 1. ABM Layer

Agent-based models explore:

- emergent market structure
- behavioral persistence
- release dynamics
- agent coordination
- endogenous regime formation

ABM acts as a theoretical substrate for interpreting DL findings.

Relevant docs:

- `docs/abm/`

---

## 2. MSML Layer

MSML performs empirical behavioral discovery through:

- deep learning
- transfer experiments
- feature ablations
- latent structure analysis
- behavioral surface export

Relevant docs:

- `docs/behavioral/`
- `docs/models/`
- `docs/data/`

---

## 3. MPML Layer

MPML acts as a downstream adaptive consumer of behavioral surfaces.

DL surfaces exported from MSML can propagate into:

- walk-forward prediction
- dynamic strategy routing
- selector training
- volatility gating
- adaptive policy selection

This layer studies:

> how adaptive systems interact with behavioral information surfaces.

---

# Key Findings

## Cross-Family Transfer

Persistent pair families and reactive pair families exhibit
meaningfully different behavioral organization.

Experiments increasingly suggest:

- persistent families may contain more stable structural organization
- reactive families may contain more release/reversion dynamics
- CHF and JPY structures appear partially separable
- transfer behavior is asymmetric

See:

- `docs/behavioral/cross_family_transfer_findings.md`
- `docs/behavioral/grouped_pair_family_findings.md`

---

## Sentiment Ablation

Removing sentiment-derived features while preserving:

- trend features
- volatility features

does **not** fully destroy behavioral structure.

This suggests that:

- structural organization survives independently of sentiment
- sentiment acts as a conditional modulation layer
- price/volatility organization itself may encode latent behavioral geometry

See:

- `docs/behavioral/sentiment_ablation.md`

---

## Architecture Robustness

Observed structure survives across:

- MLP models
- LSTM models
- cross-family transfer
- downstream MPML integration

This weakens the hypothesis that findings are merely
architecture-specific artifacts.

---

## DL ↔ ABM Alignment

Several DL findings increasingly resemble behaviors observed in ABM:

- persistence/release cycles
- asymmetric volatility response
- pair-family differentiation
- structural regime transitions

This does not prove ABM correctness,
but increasingly suggests:

> DL and ABM may be observing different manifestations
> of the same underlying behavioral organization.

See:

- `docs/abm/DL_ABM_RECONCILIATION.md`

---

# Deep Learning Models

Current DL pipelines include:

## MLP

Feedforward behavioral-surface model.

See:

- `docs/models/mlp.md`

---

## LSTM

Sequence-aware behavioral-surface model.

Supports:

- sequence-safe export
- grouped pair-family training
- cross-family transfer
- metadata-aligned prediction export

See:

- `docs/models/lstm.md`

---

# DL Prediction Artifacts

MSML exports standardized behavioral surfaces as parquet artifacts.

These artifacts can be consumed downstream by MPML.

The export contract includes:

- metadata-safe row alignment
- pair-safe export semantics
- timestamp alignment guarantees
- standardized feature schemas
- manifest diagnostics

## DL artifact contract (v2, MSML → MPML)

**Status:** live. Schema v2.0.0 is enforced on every export.

The v2 contract resolves a prior ambiguity where MSML artifact-generation
timestamps were misread by MPML as causal prediction-availability timestamps.

### Timestamp semantics (one meaning each)

| Column | Meaning | Causal use? |
|---|---|---|
| `entry_time` | H1 bar open timestamp (UTC tz-naive); the bar being predicted | ✓ (bar key) |
| `prediction_available_timestamp` | Earliest historical timestamp the prediction **could have been known**; `≤ entry_time` | ✓ MPML causal boundary |
| `prediction_generated_timestamp` | Wall-clock inference time | ✗ (diagnostics only) |
| `artifact_created_timestamp` | Wall-clock artifact export time | ✗ (provenance only) |

**Rule**: MPML must use `prediction_available_timestamp` for all causality
checks.  Never use `prediction_generated_timestamp` or
`artifact_created_timestamp` for temporal joins.

### Centralized constants

```python
from schemas.dl_artifact_schema import (
    DL_SCHEMA_VERSION,       # "2.0.0"
    DL_AVAILABLE_TS_COL,     # "prediction_available_timestamp"
    DL_GENERATED_TS_COL,     # "prediction_generated_timestamp"
    DL_ARTIFACT_CREATED_COL, # "artifact_created_timestamp"
    DL_PAIR_COL,             # "pair"
    validate_dl_artifact,
)
```

### Fail-fast validation

`write_dl_prediction_artifact()` calls `validate_dl_artifact()` before
writing any artifact.  Violations raise `ValueError` immediately.

See:

- `docs/integration/dl_artifact_contract.md`
- `docs/integration/DL_SIGNAL_SCHEMA.md`
- `docs/integration/dataset_semantics.md`

---

# DL Export integration status (v1, MSML → MPML)

**Status:** live (proof-of-concept complete). This repo exports per-run H1 DL
prediction artifacts consumed by `market-phase-ml`.

What works:

- Per-run artifact export (`data/output/dl_predictions/*.parquet` + manifest)
- Schema v2.0.0 with explicit timestamp semantics
- Surface identity columns embedded in parquet rows (`model`, `dl_regime`,
  `target_horizon`, `feature_set`)
- Export window controls for MPML-overlapping artifacts (`--export-split`,
  `--export-after-year`, `--export-before-year`)
- Fail-fast contract validation before every write

Known limitations (v1):

- Artifacts may have sparse coverage depending on regime/pair/date overlap;
  this is acceptable for v1
- No multi-surface ensembles or "all-regime" surfaces in v1
- MPML consumer-side strict validation of `prediction_available_timestamp`
  will be implemented in a follow-up PR

### Example: export an overlapping artifact for MPML

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend \
  --export-split all \
  --export-after-year 2019 \
  --export-before-year 2024
```

### Use in market-phase-ml

```bash
export DL_SIGNALS_ENABLED=true
export DL_PREDICTION_ARTIFACT_PATH=../market-sentiment-ml/data/output/dl_predictions/<run_id>.parquet
python -u main.py
```

Notes:

- MPML performs H1→D1 aggregation internally.
- v1 is intentionally minimal: sparse coverage is acceptable; MPML falls
  back per pair when DL is absent.

# Current Limitations

The current research remains exploratory.

Major unresolved questions include:

- temporal leakage risk
- sparse DL overlap effects
- causal interpretation
- stability across market eras
- robustness under randomized controls
- regime ontology mismatch between MSML and MPML

Current findings should therefore be interpreted as:

> evidence for conditional behavioral structure,

not:

> proof of universal predictive alpha.

---

# Research Roadmap

Current priorities include:

## Behavioral Research

- CHF vs JPY decomposition
- latent manifold analysis
- transition geometry
- volatility-conditioned structure
- transfer asymmetry analysis

---

## Integration Research

- DL-era-only experiments
- sparse-overlap controls
- randomized DL controls
- MPML downstream sensitivity analysis

---

## ABM Research

- improved calibration
- endogenous regime emergence
- release dynamics
- persistent/reactive family simulation

---

# Philosophy

This project increasingly treats markets as:

> adaptive behavioral systems,

rather than:

> stationary prediction problems.

The goal is therefore not merely to predict returns,
but to understand:

- when behavioral organization emerges
- how structure persists
- how adaptive systems exploit structure
- and how sentiment interacts with broader market geometry.

---

# Documentation

## Behavioral Research

- `docs/behavioral/`

## ABM

- `docs/abm/`

## Models

- `docs/models/`

## Data

- `docs/data/`

## Integration

- `docs/integration/`

---

# Technical Quickstart

## Train MLP

```bash
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD,GBPUSD,NZDUSD \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend

```

---

## Train LSTM

```
python -m research.deep_learning.train_lstm \
  --dataset-version 1.3.2 \
  --pairs USDJPY,EURJPY,GBPJPY \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend
```

------

## Sentiment Ablation Example

```
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD,GBPUSD,NZDUSD \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set trend_vol_only
```

------

## Cross-Family Transfer Example

```
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --train-pairs EURUSD,GBPUSD,NZDUSD \
  --predict-pairs USDJPY,EURJPY,GBPJPY \
  --regime LVTF \
  --target-horizon 24 \
  --feature-set price_trend
```

------

# Status

Active research project.

The current direction increasingly focuses on:

- conditional behavioral structure
- latent organization
- adaptive downstream exploitation
- and reconciliation between:
  - DL,
  - ABM,
  - and adaptive policy systems.
