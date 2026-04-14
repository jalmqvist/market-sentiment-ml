# FX Retail Sentiment Research Pipeline

A research pipeline for combining multi-year retail FX sentiment snapshots with hourly FX market data, producing a clean event-level dataset for signal testing and downstream ML workflows.

## Executive summary

> This project builds and validates a research pipeline for testing whether **retail FX sentiment** contains predictive information.
>
> It demonstrates end-to-end quantitative workflow skills:
>
> - large-scale multi-file data engineering
> - timestamp alignment across heterogeneous market data sources
> - feature engineering and target construction
> - data-quality diagnostics and outlier filtering
> - exploratory signal validation with permutation, holdout, and walk-forward testing
>
>
> A key current finding is that while simple contrarian sentiment effects are weak in the aggregate after cleaning, a more specific and structured effect appears:
>
> - in **JPY crosses**
> - under **persistent extreme retail sentiment**
> - particularly when the retail crowd is **positioned against the prevailing market trend**
>
>
> This suggests that retail underperformance may be driven more by **timing and behavioral errors** than by purely directional mistakes.

## Project goal

This project is primarily about demonstrating:

- robust handling of large financial datasets
- careful timestamp alignment across heterogeneous sources
- reproducible feature engineering
- practical data-quality diagnostics
- research-oriented dataset construction for machine learning and predictive modeling


The goal is **not** to present a production trading system or claim a fully validated profitable strategy.  
Instead, the project shows how to go from messy raw market data to a structured, analysis-ready dataset using sound data engineering and research practices.

## Research idea

The core research question is whether **retail crowd positioning in FX** contains predictive information.

The working hypothesis is mainly **contrarian**:

- when retail positioning becomes strongly one-sided, fading the crowd may have predictive value
- the effect may depend on time horizon
- the effect may depend on persistence of extreme sentiment


This repository focuses on **dataset assembly, validation, and exploratory signal research**, which form the basis for later statistical testing and ML modeling.

---

## Data sources

### 1. Sentiment snapshots

The sentiment dataset consists of many CSV files scraped over several years. Each file represents one sentiment snapshot and contains rows such as:

- `pair`
- `perc`
- `direction`
- `time`

Example interpretation:

- `72 long` means 72% of visible retail crowd is long the pair
- `61 short` means 61% is short

### 2. FX hourly market data

Hourly market data is exported from MT4 `.hst` files into CSV format with columns:

- `time_utc`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`


There is one file per FX pair.

---

## What the pipeline does

The dataset builder performs the following steps:

1. Loads and combines all sentiment snapshot files into one panel
2. Parses snapshot timestamps from filenames
3. Normalizes pair names across sources
4. Corrects timezone differences between sentiment and price data
5. Creates signed sentiment features such as:
   - `net_sentiment`
   - `abs_sentiment`
   - `sentiment_change`
   - `side_streak`
   - `extreme_streak_70`
6. Loads hourly FX data from MT4 CSV exports
7. Aligns each sentiment snapshot to the **first hourly price bar at or after the snapshot time**
8. Uses merge tolerance rules to prevent incorrect long-gap matches
9. Produces pair-level coverage diagnostics
10. Builds filtered research universes based on data completeness
11. Computes **trading-bar forward returns**
12. Computes **contrarian returns** for signal testing

---

## Key implementation choices

### Snapshot-level sentiment alignment

Each sentiment CSV is treated as a single market snapshot, using the timestamp embedded in the filename as the reference time for all rows in that file.

### Timezone correction

During validation, the following was established:

- sentiment timestamps correspond to `UTC+2`
- FX price data corresponds to `UTC+1`


So sentiment snapshot times are shifted by **-1 hour** before merging.

### Pair-by-pair merge

Sentiment and price data are merged **pair by pair** using forward alignment to the next hourly bar. This avoids unstable global asof-merge behavior and makes debugging much easier.

### Merge tolerance

A merge tolerance is used so that a sentiment snapshot is only matched to a nearby future price bar. This prevents silently matching old sentiment rows to price bars that are days, months, or years later in incomplete price files.

### Trading-bar forward returns

Forward returns are computed in **bars ahead**, not wall-clock hours:

- `ret_1b`
- `ret_2b`
- `ret_4b`
- `ret_6b`
- `ret_12b`
- `ret_24b`
- `ret_48b`

This is more robust in FX because weekends and market closures make wall-clock horizons misleading.

### Contrarian return target

A unified target variable is created:

`contrarian_ret_h = -sign(net_sentiment) * ret_h`

Interpretation:

- positive contrarian return → fading the crowd would have made money
- negative contrarian return → the crowd was right

---

## Current research status

The project has progressed beyond raw data assembly into exploratory signal validation.

### Completed work

- built a multi-source FX research pipeline linking:
  - retail sentiment snapshots
  - hourly FX market data
- normalized pair naming across sources
- aligned timestamps across different timezones
- added pair-level coverage diagnostics
- filtered weak/incomplete price histories
- replaced wall-clock forward returns with trading-bar forward returns
- created filtered research universes for cleaner downstream analysis
- added threshold, persistence, pair-group, outlier, permutation, and time-based validation scripts

### Exploratory findings so far

Initial aggregate threshold analysis suggested a contrarian effect, but outlier diagnostics showed that this was largely driven by problematic price behavior in a small number of thin/exotic pairs.

After pair-level quality filtering:

- the broad aggregate threshold effect became weak
- major pairs appeared mostly flat
- thin/exotic pairs were weak or negative in the simple threshold framework
- a subset of liquid crosses looked more promising

  Persistence analysis then suggested that any remaining effect is conditional rather than universal.


The most interesting result so far is:

- within the cleaned cross universe

- persistent extreme sentiment (`abs_sentiment >= 70` and `extreme_streak_70 >= 3`)

- appears stronger in a cluster of **JPY crosses**

Further analysis shows that this effect is not simply a pair-group artifact, but is strongly conditioned on **market context**:

- the effect is significantly stronger when the retail crowd is **fighting the prevailing trend**

- weaker or less consistent when the crowd is aligned with the trend

This JPY-cross clustering has so far survived several increasingly strict checks:

- pair-level outlier filtering

- subgroup analysis within crosses

- a permutation test on pair-cluster membership

- a time-based holdout split

- expanding-window walk-forward validation

### Trend-conditioned behavior (new)

To better understand *how* retail traders fail, the dataset was extended with **trend context features** based on past returns.

This allows separating two behavioral regimes:

- **follow_trend**: crowd aligned with recent trend (late chasing)
- **fight_trend**: crowd positioned against recent trend (early reversal attempts)

#### Key findings

- Trend alone does **not** produce a strong aggregate signal
- However, conditioning on trend reveals **structure in the JPY-cross subset**

Most notably:

- In **JPY crosses**
- Under **persistent extreme sentiment**
- Both:
  - trend-following (late chasing)
  - trend-fighting (early reversal)

  show **positive contrarian returns**

  This suggests that:

- retail traders are not uniformly wrong
- but their behavior becomes exploitable under **specific structural conditions**

#### Interpretation

The evidence supports a refined behavioral model:

- Retail traders:
  - chase trends too late
  - attempt reversals too early
- These behaviors become most visible:
  - under persistent sentiment extremes
  - in structurally trending markets (e.g. JPY crosses)

#### Important note

Trend features are constructed from **past returns only**, ensuring that:

- no future information is used
- analysis avoids mechanical leakage
- results remain interpretable and reproducible

### Current interpretation

This does **not** yet establish a production-ready trading signal.

However, the current evidence supports a more structured interpretation:

- simple threshold-based sentiment fading is not broadly robust in the cleaned universe
- the predictive content of sentiment appears to be **conditional on market context**
- the strongest effects emerge when combining:

  - **persistent extreme sentiment**
  - **trend misalignment (crowd fighting the trend)**
  - **specific pair groups (notably JPY crosses)**


This suggests that retail traders are not uniformly wrong, but instead exhibit **systematic behavioral failure modes**, particularly related to **timing and trend interaction**.

The next step is to further validate and refine this framework using **regime-conditioned analysis** (e.g. trend strength, volatility).

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

This should still be treated as a **research finding**, not as proof of a production-ready trading strategy.

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
├── build_fx_sentiment_dataset.py
├── data
│   ├── input
│   │   ├── fx/
│   │   └── sentiment/
│   ├── output
│   │   ├── analysis/
│   │   ├── DATASET_MANIFEST.json
│   │   ├── master_research_dataset.csv
│   │   ├── master_research_dataset_core.csv
│   │   ├── master_research_dataset_extended.csv
│   │   └── pair_coverage_summary.csv
│   └── sample
│       ├── fx/
│       └── sentiment/
├── DATA_AVAILABILITY.md
├── INPUT_SCHEMA.md
├── LICENSE
├── OUTPUT_SCHEMA.md
├── PROJECT_DESCRIPTION.md
├── README.md
├── sanity.py
├── validate_jpy_effect_time_split.py
└── validate_jpy_effect_walkforward.py
```

Note: `data/input/` and `data/output/` are **expected local directories** and are not distributed with the repository.