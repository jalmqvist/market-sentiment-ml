# FX Retail Sentiment Research Pipeline

A research pipeline for combining multi-year retail FX sentiment snapshots with hourly FX market data, producing a clean event-level dataset for signal testing and downstream ML workflows.

## Executive summary

This project builds and validates a research pipeline for testing whether **retail FX sentiment** contains predictive information.

It demonstrates an end-to-end quantitative workflow:

- large-scale data ingestion and normalization
- timestamp alignment across heterogeneous sources
- feature engineering and target construction
- data-quality diagnostics and filtering
- structured signal validation (permutation, holdout, walk-forward)

The main result is **not a broad contrarian effect**, but a **conditional behavioral signal**:

- concentrated in **JPY crosses**
- requiring **persistent extreme sentiment**
- strongly dependent on **trend context**

This suggests that retail underperformance is driven primarily by **timing errors**, not directional bias alone.

---

## Research idea

The core question is whether **retail crowd positioning in FX** contains predictive information.

The working hypothesis is contrarian but conditional:

- extreme positioning may be exploitable
- persistence matters
- market context (trend) determines when the effect appears

The goal of this repository is not to present a trading strategy, but to show how to move from raw data to a **structured, testable hypothesis**.

---

## What the pipeline does

The pipeline transforms raw sentiment snapshots and FX price data into a clean research dataset:

- aggregates multi-year sentiment snapshots
- parses timestamps and aligns timezones
- normalizes pair naming across sources
- merges sentiment to hourly price bars using forward alignment
- computes trading-bar forward returns
- constructs contrarian return targets
- builds persistence and behavioral features

It also produces an **hourly feature contract** (`sentiment_features_h1_v1`) for downstream ML use.

---

## Key findings

### 1. No robust aggregate effect

After cleaning:

- simple threshold-based contrarian signals are weak
- major pairs are mostly flat
- thin/exotic pairs are unstable or misleading

This indicates that **sentiment alone is not broadly predictive**.

---

### 2. Structured signal in JPY crosses

A consistent pattern emerges under specific conditions:

- **pair group:** JPY crosses  
- **sentiment:** `abs_sentiment ≥ 70`  
- **persistence:** `extreme_streak_70 ≥ 3`  

Under these conditions:

- contrarian returns are positive  
- hit rates are consistently above 50%  
- the effect is largely absent in non-JPY pairs  

---

### 3. Behavior depends on trend alignment

Introducing trend context reveals two distinct regimes:

- **fight_trend** (retail against trend → early reversal attempts)
- **follow_trend** (retail with trend → late chasing)

Both can produce positive contrarian returns, but under different conditions.

---

### 4. Trend strength introduces nonlinearity

Conditioning on trend strength reveals a structured pattern:

- For **fight_trend**:
  - mean return increases with trend strength
  - hit rate peaks at **strong** trends
  - extreme trends deliver larger but noisier returns

- For **follow_trend**:
  - signal is strongest in **extreme** trends

  This implies a **risk–return trade-off** rather than a single optimal regime.

---

### Signal vs risk trade-off

![Signal vs risk](docs/images/signal_vs_risk_jpy.png)

- strong trends → best balance of consistency and return  
- extreme trends → higher return, higher variability  

---

### Trend-conditioned behavior (JPY crosses)

![Trend strength vs contrarian returns](docs/images/trend_strength_jpy.png)

Key observations:

- fighting strong trends → most reliable signal  
- following extreme trends → highest payoff but less stable  
- effect is concentrated in JPY crosses  

---

### Practical interpretation

Retail traders fail in two systematic ways:

- **early reversal** → fighting strong trends too soon  
- **late chasing** → joining extreme trends too late  

A simple regime-based interpretation:

- use **strong trends** for more stable signals  
- use **extreme trends** for higher-risk opportunities  

---

## Validation status

The signal has been tested using:

- pair-level outlier filtering  
- subgroup analysis  
- permutation testing  
- time-based holdout  
- walk-forward validation  

### Walk-forward results

- **12-bar horizon:**
  - consistent positive returns across most folds  
  - stable hit rates (~0.55–0.60)  
  - moderate but persistent signal  

- **48-bar horizon:**
  - higher variance  
  - regime-dependent performance  
  - occasional breakdowns  

  Interpretation:

- the signal is **short-horizon and structural**
- longer horizons introduce macro noise and reduce stability

---

## Current interpretation

The evidence supports a conditional behavioral model:

- retail traders are not uniformly wrong  
- errors emerge under specific structural conditions  
- timing relative to trend is the key driver  

This project therefore reframes sentiment from:

> a generic contrarian signal  

to:

> a **regime-dependent behavioral indicator**

---

## Key results

Current exploratory findings suggest that the sentiment effect is **not broad across all FX pairs**, but instead emerges under specific behavioral and market conditions.

### Main observations

- After pair-level quality filtering, the broad aggregate contrarian threshold effect became weak.
- Major pairs appeared mostly flat in the simple threshold framework.
- Thin/exotic pairs were generally weak or unstable after cleaning.
- A subset of **liquid crosses**, especially **JPY crosses**, showed more structured behavior.

### Structured signal candidate

The strongest current signal candidate is:

- **universe:** cross pairs
- **subgroup:** JPY crosses
- **conditions:**
  - `abs_sentiment >= 70`
  - `extreme_streak_70 >= 3`
  - retail crowd **fighting the prevailing trend**
- **interpretation:**
  - retail traders persistently positioned against the trend exhibit the strongest underperformance

### Empirical pattern

Under these conditions:

- contrarian returns are **consistently positive**
- hit rates are meaningfully above 50%
- the effect is materially stronger than in non-JPY crosses under the same conditions

### Behavioral interpretation

The results suggest that retail traders are not simply “wrong,” but fail in **systematic ways**, including:

- **late trend participation** (chasing moves after they are mature)
- **premature reversal attempts** (“the market must turn now”)
- **persistence in losing views** (holding or adding to positions despite adverse moves)

The strongest signal aligns with the combination:

> **persistent extreme sentiment + fighting the trend**

This is consistent with well-known behavioral biases such as:

- mean-reversion bias
- anchoring
- loss aversion and averaging down

### Validation status

This structured effect has so far survived:

- pair-level outlier filtering
- subgroup analysis within crosses
- permutation testing
- time-based holdout validation
- expanding-window walk-forward testing

### Pre-registered final-period check

A stricter locked test was run on the most recent untouched period using the fixed rule:

- `abs_sentiment >= 70`
- `extreme_streak_70 >= 3`
- JPY crosses
- horizons: `12b` and `48b`

That final-period check was **inconclusive** rather than confirmatory. The latest sample was short and uneven, with Q1 2026 largely rangebound and Q2 2026 containing too few qualifying events for reliable inference. As a result, the project now treats the JPY-cross result as a strong exploratory pattern that still requires **regime-conditioned validation**.

---

### Signal vs risk trade-off

![Signal vs risk](docs/images/signal_vs_risk_jpy.png)

This plot shows the trade-off between return magnitude and variability across trend strength regimes.

Key observation:

- Strong trends offer the best balance of consistency and return
- Extreme trends deliver higher average returns but with increased variability

---

### Trend-conditioned behavior (JPY crosses)

The chart below illustrates how contrarian returns vary with **trend strength** under persistent sentiment conditions.

- X-axis: trend strength buckets (weak → extreme)
- Y-axis: mean contrarian return
- Split by:
  - **fight_trend** (retail positioned against trend)
  - **follow_trend** (retail aligned with trend)

    ![Trend strength vs contrarian returns](docs/images/trend_strength_jpy.png)

  Key observation:

- When retail traders fight the trend:

  - The signal becomes stronger as trend strength increases
  - Predictability (hit rate) peaks in strong trends
  - Return magnitude peaks in extreme trends, but with higher variability


This suggests a trade-off between consistency and payoff under extreme market conditions.

- When retail traders **follow the trend**, the strongest signal appears in **extreme trends** (late chasing)

- The effect is concentrated in **JPY crosses** and largely absent in non-JPY pairs

  This supports a behavioral interpretation where retail traders systematically mistime both entries and reversals under different market conditions.

---

### Practical interpretation (decision rule)

The results suggest two distinct behavioral opportunities in JPY crosses under persistent sentiment:

- When retail traders **fight the trend**:
  - strongest and most consistent signal occurs in **strong trends**

- When retail traders **follow the trend**:
  - strongest signal occurs in **extreme trends**

  This implies a regime-dependent approach:

- use **strong trends** for more stable signals
- use **extreme trends** for higher-risk, higher-reward opportunities

---

## Running the project

### 1. Build the research dataset

```bash
python build_fx_sentiment_dataset.py
```

### 2. Run exploratory analysis and validation

Examples:

```
python analyze_thresholds.py
python analyze_outliers.py
python analyze_pair_quality.py
python analyze_by_pair_group.py
python analyze_persistence.py
python analyze_cross_pair_persistence.py
python analyze_jpy_cluster_permutation.py
python validate_jpy_effect_time_split.py
python validate_jpy_effect_walkforward.py
```

These scripts generate summary tables and validation outputs under `data/output/analysis/`.

---

## Output artifact contract

The pipeline now defines a stable output contract for downstream integration.

Key documents:

- `INPUT_SCHEMA.md`
- `OUTPUT_SCHEMA.md`


A machine-readable dataset manifest is written at build time:

- `data/output/DATASET_MANIFEST.json`


The canonical downstream research artifact is:

- `data/output/master_research_dataset_core.csv`


unless otherwise stated.

------

## License and data availability

This repository is distributed under a **non-commercial, source-available license** for the original code and repository-authored documentation.

- Personal, educational, academic, and non-commercial research use is allowed
- Commercial use, resale, sublicensing, and inclusion in paid products or services is not allowed without prior written permission

### Data availability

Raw broker-exported FX price data, raw sentiment scrape files, and full derived datasets are **not distributed** in this repository due to licensing and redistribution uncertainty.

The repository contains the code and documentation needed to reproduce the pipeline using data that you have the right to access and use locally.

------

## Project structure

```
.
├── analyze_by_pair_group.py
├── analyze_cross_pair_persistence.py
├── analyze_jpy_cluster_permutation.py
├── analyze_outliers.py
├── analyze_pair_quality.py
├── analyze_persistence.py
├── analyze_thresholds.py
├── analyze_trend_alignment.py
├── analyze_trend_behavior.py
├── analyze_trend_strength_results.py
├── build_fx_sentiment_dataset.py
├── build_sentiment_feature_contract.py
├── data
│   ├── input
│   │   ├── fx/
│   │   └── sentiment/
│   ├── output
│   │   ├── analysis/
│   │   ├── features/
│   │   │   ├── sentiment_features_h1_v1.parquet
│   │   │   └── SENTIMENT_FEATURE_MANIFEST_h1_v1.json
│   │   ├── DATASET_MANIFEST.json
│   │   ├── master_research_dataset.csv
│   │   ├── master_research_dataset_core.csv
│   │   ├── master_research_dataset_extended.csv
│   │   └── pair_coverage_summary.csv
│   └── sample
│       ├── fx/
│       └── sentiment/
├── docs
│   ├── images
│   │   ├── signal_vs_risk_jpy.png
│   │   └── trend_strength_jpy.png
│   └── SENTIMENT_FEATURE_SCHEMA.md
├── DATA_AVAILABILITY.md
├── INPUT_SCHEMA.md
├── JPY_BEHAVIORAL_HYPOTHESIS.md
├── LICENSE
├── OUTPUT_SCHEMA.md
├── PRE_REGISTERED_JPY_EFFECT_TEST.md
├── PROJECT_DESCRIPTION.md
├── README.md
├── time_alignment_diagram.md
├── validate_jpy_effect_preregistered.py
├── validate_jpy_effect_time_split.py
├── validate_jpy_effect_walkforward.py
└── walk_forward_jpy_hypothesis.py
```

Note: `data/input/` and `data/output/` are **expected local directories** and are not distributed with the repository.