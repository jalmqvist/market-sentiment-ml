# FX Retail Sentiment — Behavioral Signal Research

A quantitative research project studying whether retail FX sentiment contains predictive signal.

---

## Executive Summary

**Current status: no validated standalone alpha.**

Extensive experimentation and strict out-of-sample validation have produced a clean negative result:

- Raw retail FX sentiment = noise
- Previous pipeline-based signals were invalidated (caused by pipeline artifacts and data leakage)
- Conditional signal (regime-filtered) appears only weakly and is unstable across time

This is a valid research finding, not a failure. The project continues with hypothesis testing,
deep learning sequence modeling, and agent-based behavioral simulation.

---

## Key Findings

| Component        | Status                       |
| ---------------- | ---------------------------- |
| Raw sentiment    | ❌ No edge (noise)            |
| Signal V2        | ❌ No edge                    |
| Regime filtering | ⚠️ Weak, unstable conditional |
| V20–V21 signal   | ❌ Invalid (pipeline artifact) |

### What went wrong in earlier experiments

The apparent edge in Regime V20–V21 (Sharpe ≈ 0.21–0.22) was caused by:

- Subtle pipeline implementation errors
- Emergent bias from `groupby/apply` + index misalignment
- Implicit selection distortions

**Independent validation outside the pipeline disproved this result.**

Using a clean, minimal validation setup:
- Raw signal Sharpe ≈ 0.00
- Shifted signal Sharpe ≈ 0.00
- Shuffled signal Sharpe ≈ 0.00

---

## Research Direction

Three parallel research tracks are now active:

### 1. Hypothesis Testing (statistical)

Iterative statistical tests on behavioral signal hypotheses. Most results are negative — this is expected and informative. See `research/hypothesis_tests/`.

### 2. Deep Learning (sequence modeling)

Experimental sequence models (LSTM, Transformer) applied to sentiment and price time series to detect nonlinear conditional structure. See `research/deep_learning/`.

### 3. Agent-Based Modeling (behavioral simulation)

Simulation of retail crowd behavior to understand *why* sentiment might be conditionally predictive under certain market regimes. See `research/abm/`.

---

## Repo Structure

```
market-sentiment-ml/
├── research/
│   ├── raw_validation/      # Ground truth testing — validate hypotheses outside any pipeline
│   ├── hypothesis_tests/    # Iterative experiments (regime_v*, signal_v*, etc.) — mostly negative results
│   ├── deep_learning/       # Experimental sequence modeling (WIP)
│   └── abm/                 # Experimental agent-based modeling (WIP)
├── docs/
│   ├── archive/             # Legacy documentation (pipeline_v1, etc.)
│   └── RESEARCH_STRATEGY.md # Research philosophy and approach
├── scripts/                 # Build and data preparation utilities
├── pipeline/                # Core pipeline modules (retained for reference)
├── evaluation/              # Validation and walk-forward evaluation utilities
├── utils/                   # Shared utility functions
└── tests/                   # Unit tests
```

### Directory guide

| Directory | Purpose |
| --------- | ------- |
| `research/raw_validation/` | Ground truth: validate any signal hypothesis using clean, minimal code *outside* the pipeline. This is the primary validation framework. |
| `research/hypothesis_tests/` | All iterative research experiments. Most are negative results. Preserved for reproducibility. |
| `research/deep_learning/` | Experimental deep learning approaches (not yet validated). |
| `research/abm/` | Experimental agent-based modeling (not yet validated). |
| `docs/archive/` | Legacy documentation preserved for historical reference. |

---

## Philosophy

**Negative results are valid.**

> Knowing that retail sentiment is *not* predictive (under naive conditions) is a genuine research contribution.

**Validation outside the pipeline is mandatory.**

> Any new hypothesis must first be validated using a clean, minimal script in `research/raw_validation/` before being integrated into any larger framework. Pipeline complexity is a source of false positives.

**Avoid data leakage.**

> Every feature and signal must be causally clean. Walk-forward evaluation with strict training/test splits is required. No forward-looking information is permitted in any feature construction.

---

## Running validation experiments

```bash
# Validate raw signal (ground truth check)
python research/raw_validation/validate_signal_raw.py

# Pipeline sanity check
python research/raw_validation/pipeline_sanity_check.py
```

---

## Dataset Versioning

Datasets are **versioned and reproducible**. Every build produces:

- A versioned directory under `data/output/<version>/`
- Variant files (`core`, `extended`, `full`) filtered by coverage quality
- A `DATASET_MANIFEST.json` recording the exact build parameters

This ensures that any ML experiment can be reproduced by specifying the dataset version:

```bash
python research/deep_learning/train.py --dataset-version 1.1.0 --feature-set price_sentiment

python research/deep_learning/evaluate.py --dataset-version 1.1.0
```

See [`docs/DATASET.md`](docs/DATASET.md) for the full dataset reference.

---

## Data

See `DATA_AVAILABILITY.md` for data access notes and `INPUT_SCHEMA.md` / `OUTPUT_SCHEMA.md` for data schemas.

---

## Further reading

- [`docs/RESEARCH_STRATEGY.md`](docs/RESEARCH_STRATEGY.md) — Research strategy and modeling philosophy
- [`docs/archive/pipeline_v1.md`](docs/archive/pipeline_v1.md) — Legacy pipeline description (archived)
- [`RESEARCH_STATE.md`](RESEARCH_STATE.md) — Current state of all experiments
