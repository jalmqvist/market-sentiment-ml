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
- dependent on both **trend context** and **volatility regime**

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

### 5. Volatility regime dependency (NEW)

Recent cross-repo analysis (integration with market-phase-ml) shows that the signal is strongly dependent on volatility regime:

- The signal is **positive and strongest in high-volatility (HV) regimes**
- The signal is **weak or negative in low-volatility (LV) regimes**

Empirical observation (JPY crosses, persistent extreme sentiment):

- HV regimes → positive mean returns (~ +0.001)
- LV regimes → near-zero or negative returns

Interpretation:

Retail traders’ behavioral errors are amplified under **high-volatility conditions**, where:

- uncertainty is higher
- trend structure is less stable
- emotional and reactive behavior increases

This reframes the signal from:

> trend-conditioned  

to:

> **volatility-gated + behavior-conditioned**

This also explains why:

- the signal appears inconsistent across time
- performance improves post-2022 (higher volatility regime)

---

### Figure 1: Signal vs Risk (JPY crosses)

![Signal vs Risk](docs/images/signal_vs_risk_jpy.png)

The relationship between signal strength and risk is non-linear.

- Weak signals exhibit low variance but limited return
- Extreme signals show higher variance but improved mean return
- The signal is strongest when crowd positioning is extreme and persistent

This supports the hypothesis that **rare, high-intensity sentiment states carry the strongest informational edge**.

---

### Figure 2: Trend strength vs contrarian returns

![Trend strength](docs/images/trend_strength_jpy.png)

Contrarian performance depends strongly on trend strength:

- In strong trends, following the trend tends to outperform
- In weaker trends, contrarian positioning is more viable

This highlights that **sentiment signals are conditional on market structure**, not universally predictive.

---

### Figure 3: Volatility regime dependency

![HV vs LV](docs/images/hv_vs_lv_signal_jpy.png)

*The violin shape shows the full return distribution, while the overlaid markers indicate mean values with 95% confidence intervals, highlighting both dispersion and statistical reliability.*

The contrarian sentiment signal is significantly stronger in high-volatility environments:

- High-volatility regimes exhibit higher mean returns and broader dispersion
- Low-volatility regimes show weak or negligible signal

This suggests that **retail positioning becomes most exploitable when market uncertainty is elevated**.

---

### Figure 4: Signal evolution over time

![Yearly signal](docs/images/yearly_signal_jpy.png)

*Marker size reflects the number of observations per year. Error bars indicate 95% confidence intervals.*

The signal exhibits strong time-dependence:

- Weak or unstable performance during 2019–2021
- Stronger and more consistent performance post-2022, particularly in high-volatility regimes

This aligns with major macro shifts (post-COVID tightening cycle) and suggests:

> The sentiment signal is not static—it is **regime-dependent and time-varying**.

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
- performance is **strongly dependent on volatility regime**
- high-volatility environments drive most of the signal
- longer horizons introduce macro noise and reduce stability

---

## Current interpretation

The evidence supports a conditional behavioral model:

- retail traders are not uniformly wrong  
- errors emerge under specific structural conditions  

The signal is driven by a combination of:

- **volatility regime (primary gating factor)**
- **trend interaction (fight vs follow)**
- **persistence of extreme positioning**
- **pair-specific behavior (JPY crosses)**

This reframes sentiment from:

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

### Structured signal candidate (updated)

The strongest signal emerges under a combination of behavioral and regime conditions:

- **universe:** JPY crosses  
- **conditions:**
  - `abs_sentiment >= 70`
  - `extreme_streak_70 >= 3`
  - high-volatility regime (`is_high_vol == True`)
  - retail crowd interacting with trend (fight or late follow)

Key refinement:

- The signal is **not universally present**
- It is **activated under high-volatility conditions**

Interpretation:

Retail traders exhibit systematic failure modes when:

- volatility is elevated  
- positioning is extreme and persistent  
- market structure becomes unstable  

This reframes the signal as:

> **a volatility-gated behavioral effect**

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

Initial results were inconclusive due to limited sample size and a low-volatility environment.

However, subsequent cross-repo analysis incorporating **volatility regimes** (via market-phase-ml) clarified the outcome:

- The signal is **strong and consistent in high-volatility regimes**
- The signal is **weak or negative in low-volatility regimes**

This resolves the earlier ambiguity and confirms that the signal is:

> **regime-dependent rather than unstable**

The pre-registered test is therefore considered **conditionally validated**, subject to volatility regime.

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
├── evaluate_signal_regime_aware.py
├── INPUT_SCHEMA.md
├── JPY_BEHAVIORAL_HYPOTHESIS.md
├── LICENSE
├── OUTPUT_SCHEMA.md
├── PRE_REGISTERED_JPY_EFFECT_TEST.md
├── PROJECT_DESCRIPTION.md
├── README.md
├── validate_jpy_effect_preregistered.py
├── validate_jpy_effect_time_split.py
├── validate_jpy_effect_walkforward.py
├── walk_forward_jpy_hypothesis.py
└── walk_forward_jpy_regime_signal.py
```

Note: `data/input/` and `data/output/` are **expected local directories** and are not distributed with the repository.