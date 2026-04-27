# Project Description

This project investigates whether **retail FX sentiment data** contains predictive information about future market behavior.

The dataset consists of multi-year time series of:

- retail trader positioning (long/short ratios)
- FX price data (OHLC)
- derived behavioral and market features

The central question is:

> Can aggregated retail positioning be transformed into a **causal, predictive signal**?

---

## Scope

This is a **quantitative research and data engineering project**, not a production trading strategy.

The project focuses on:

- building a clean, reproducible research dataset
- constructing and testing signal hypotheses
- eliminating false positives through strict validation
- understanding the limits of behavioral data in financial markets

---

## Data & Feature Construction

The pipeline constructs a unified dataset by:

- aggregating sentiment snapshots across time
- aligning sentiment with FX price bars
- computing forward returns (evaluation targets)
- engineering behavioral features such as:
  - sentiment extremes
  - persistence (streaks)
  - divergence vs price
  - rate-of-change dynamics

All features are designed to be **causal (past-only)**.

---

## Validation Framework

A key contribution of the project is a **strict validation methodology**:

- signals are tested **outside any pipeline**
- mandatory tests include:
  - shift tests (temporal robustness)
  - shuffle tests (randomization control)
  - time-based splits (out-of-sample validation)

This ensures that:

> any detected signal reflects genuine predictive structure, not implementation artifacts.

---

## Current Understanding

Empirical results show that:

- retail sentiment is strongly **correlated with price**
- but does **not provide independent predictive information**
- apparent signals are often explained by:
  - price autocorrelation
  - sampling effects
  - pipeline-induced bias

---

## Research Directions

The project now explores two complementary directions:

### 1. Deep Learning (Predictive Modeling)

- sequence models (LSTM, Transformer)
- nonlinear feature interactions
- goal: detect weak conditional structure, if it exists

### 2. Agent-Based Modeling (Behavioral Modeling)

- simulate retail trader behavior
- reproduce observed sentiment dynamics
- goal: understand *why* predictive signal is absent or unstable

---

## Role of This Project

This repository represents:

- a **validated research environment**
- a **documented negative result**
- a **foundation for behavioral and ML-based exploration**

---

## Dataset Versioning

The research dataset is now **versioned and reproducible**. Each dataset version is stored in a dedicated directory (`data/output/<version>/`) alongside a manifest file that records all build parameters. This supports:

- **ML workflows**: training scripts reference a specific dataset version, ensuring experiments are fully reproducible.
- **ABM workflows**: agent-based models can load the same versioned feature tables used in ML experiments, enabling consistent comparisons across modeling approaches.

Any experiment can be reproduced exactly by specifying its dataset version. See [`docs/DATASET.md`](docs/DATASET.md) for the full reference.

---

## Philosophy

- prioritize **correctness over results**
- treat **negative findings as valuable outcomes**
- enforce **strict separation between discovery and validation**

The objective is not to confirm a hypothesis, but to:

> **rigorously determine whether a signal exists at all**
