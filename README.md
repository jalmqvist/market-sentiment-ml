# FX Retail Sentiment Research Pipeline

A research pipeline for combining multi-year retail FX sentiment snapshots with hourly FX market data, producing a clean event-level dataset for signal testing and downstream ML workflows.

------

## Executive summary

This project builds and validates a research pipeline for testing whether **retail FX sentiment** contains predictive information.

It demonstrates an end-to-end quantitative workflow:

- large-scale data ingestion and normalization
- timestamp alignment across heterogeneous sources
- feature engineering and target construction
- data-quality diagnostics and filtering
- structured signal validation (permutation, holdout, walk-forward)

------

## Main result (updated)

After correcting methodological biases (notably overlapping signals) and enforcing strict walk-forward validation:

> **A weak but consistent behavioral signal exists — conditional on trend context**

Specifically:

> **Crowd extremes are most exploitable when they occur late in a trend (contrarian alignment)**

Key properties:

- not explained by volatility or macro regime
- not driven by a single pair (multi-pair effect)
- stronger at longer horizons (48 bars)
- structurally **contrarian to price trend**

------

## Evolution of findings

### Phase 1 — Initial discovery (invalidated)

Earlier versions suggested:

- strong JPY-specific effects
- regime dependence (volatility, trend, macro)

These were later shown to be artifacts caused by:

- overlapping signals
- in-sample bias
- improper walk-forward validation

------

### Phase 2 — Corrected validation

After enforcing:

- non-overlapping signals
- true walk-forward evaluation
- regime holdout testing

Results:

- price-based regimes **do not explain the signal**
- most apparent structure disappears
- aggregate performance ≈ noise

This was initially interpreted as a **negative result**

------

### Phase 3 — Behavioral re-framing (current)

Shifting from price regimes → **crowd behavior**

Led to discovery of a new structure:

> **Signal strength depends on interaction between crowd extremes and trend position**

Empirical result:

| Condition                      | Performance |
| ------------------------------ | ----------- |
| Trend-aligned                  | weak        |
| Mixed                          | moderate    |
| **Contrarian (against trend)** | strongest   |

------

## Core insight

> **Retail crowd extremes are most exploitable when they occur late in an existing trend**

Interpretation:

- crowd joins trends late
- positioning becomes saturated
- reversal risk increases
- contrarian trades capture unwind

------

## What the pipeline does

The pipeline transforms raw sentiment snapshots and FX price data into a clean research dataset:

- aggregates multi-year sentiment snapshots
- parses timestamps and aligns timezones
- normalizes pair naming across sources
- merges sentiment to hourly price bars using forward alignment
- computes trading-bar forward returns
- constructs contrarian return targets
- builds persistence and behavioral features

It produces:

- a research dataset (`master_research_dataset_core.csv`)
- a feature contract (`sentiment_features_h1_v1`)

------

## Key findings

### 1. No unconditional edge

- raw sentiment thresholds are weak
- aggregate performance ≈ noise

------

### 2. No price-regime dependency

The following do **not** produce stable effects:

- volatility regimes
- trend strength
- macro periods
- trend alignment (in isolation)

------

### 3. Behavioral conditioning matters

Signal strength emerges when conditioning on:

- crowd persistence
- crowd saturation
- trend context

------

### 4. Cross-pair consistency

- signal exists across multiple JPY crosses
- not driven by a single pair
- remains after removing top contributors

------

### 5. Horizon dependency

- weak at short horizons
- stronger at longer horizons (48 bars)

------

## Validation status

The signal has been tested using:

- pair-level filtering
- subgroup analysis
- permutation testing
- time-based holdout
- strict walk-forward validation

### Result

> **A small but consistent edge survives — conditional on behavioral context**

------

## Current interpretation

The evidence suggests:

- retail traders are not uniformly wrong
- crowd behavior becomes exploitable under specific conditions
- price-based regimes are insufficient

The signal is driven by:

- crowd saturation
- late positioning
- behavioral feedback loops

------

## Research direction: Regime v2 (behavioral regimes)

Focus shifts toward modeling **crowd state**:

Planned features:

- crowd persistence
- sentiment acceleration
- crowd saturation
- trapped positioning (loss regimes)

Goal:

> identify when sentiment becomes predictive based on **behavioral context**

------

## Portfolio construction (current)

A prototype portfolio has been validated:

- multi-pair aggregation
- survivor selection (per-pair validation)
- walk-forward + holdout consistency

Findings:

- diversification improves stability
- contrarian filtering improves Sharpe but reduces capacity
- hybrid approaches likely optimal

------

## Running the project

### 1. Build dataset

```bash
python build_fx_sentiment_dataset.py
```

### 2. Run validation / analysis

```bash
python walk_forward_regime_v2.py
python discover_behavioral_signal.py
python portfolio_behavioral_signal.py
```

------

## Output artifact contract

Key outputs:

- `data/output/master_research_dataset_core.csv`
- `data/output/DATASET_MANIFEST.json`
- `data/output/features/sentiment_features_h1_v1.parquet`

------

## Project structure

(Current structure — subject to refactor)

```
build → analyze → validate → portfolio
```

A future refactor will standardize:

- pipeline modules
- shared utilities
- configuration management

Current file structure:

```
.
├── analyze_by_pair_group.py
├── analyze_cross_pair_persistence.py
├── analyze_jpy_cluster_permutation.py
├── analyze_outliers.py
├── analyze_pair_quality.py
├── analyze_persistence.py
├── analyze_regime_signal_interaction.py
├── analyze_thresholds.py
├── analyze_trend_alignment.py
├── analyze_trend_behavior.py
├── analyze_trend_strength_results.py
├── attach_regimes_to_h1_dataset.py
├── build_fx_sentiment_dataset.py
├── build_sentiment_feature_contract.py
├── data
│   ├── input
│   │   ├── fx
│   │   └── sentiment
│   ├── output
│   │   ├── analysis/
│   │   ├── DATASET_MANIFEST.json
│   │   ├── features
│   │   │   ├── SENTIMENT_FEATURE_MANIFEST_h1_v1.json
│   │   │   └── sentiment_features_h1_v1.parquet
│   │   ├── master_research_dataset_core.csv
│   │   ├── master_research_dataset.csv
│   │   ├── master_research_dataset_extended.csv
│   │   ├── master_research_dataset_with_regime.csv
│   │   └── pair_coverage_summary.csv
│   └── sample
│       ├── fx
│       └── sentiment
├── docs
│   ├── images
│   │   ├── hv_vs_lv_signal_jpy.png
│   │   ├── signal_vs_risk_jpy.png
│   │   ├── trend_strength_jpy.png
│   │   └── yearly_signal_jpy.png
│   └── SENTIMENT_FEATURE_SCHEMA.md
├── DATA_AVAILABILITY.md
├── discover_behavioral_signal.py
├── evaluate_regime_holdout.py
├── evaluate_signal_regime_aware.py
├── experiment_regime_v2_sweep.py
├── INPUT_SCHEMA.md
├── JPY_BEHAVIORAL_HYPOTHESIS.md
├── LICENSE
├── OUTPUT_SCHEMA.md
├── portfolio_behavioral_signal.py
├── PRE_REGISTERED_JPY_EFFECT_TEST.md
├── PROJECT_DESCRIPTION.md
├── README.md
├── validate_jpy_effect_preregistered.py
├── validate_jpy_effect_time_split.py
├── validate_jpy_effect_walkforward.py
├── walk_forward_jpy_hypothesis.py
├── walk_forward_jpy_regime_signal.py
└── walk_forward_regime_v2.py
```

Note: `data/input/` and `data/output/` are **expected local directories** and are not distributed with the repository.

------

## License and data availability

This repository is distributed under a **non-commercial, source-available license** for the original code and repository-authored documentation.

- Personal, educational, academic, and non-commercial research use is allowed
- Commercial use, resale, sublicensing, and inclusion in paid products or services is not allowed without prior written permission

### Data availability

Raw broker-exported FX price data, raw sentiment scrape files, and full derived datasets are **not distributed** in this repository due to licensing and redistribution uncertainty.

The repository contains the code and documentation needed to reproduce the pipeline using data that you have the right to access and use locally.

------

## Final note

This repository documents both:

- **invalidated hypotheses (important)**
- **validated behavioral insight (current)**

It demonstrates:

> how correcting methodology can transform a "false signal" into a **real, conditional edge**

