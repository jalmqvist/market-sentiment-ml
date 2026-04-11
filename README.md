# FX Retail Sentiment Research Pipeline

A research pipeline for combining multi-year retail FX sentiment snapshots with hourly FX market data, producing a clean event-level dataset for signal testing and downstream ML workflows.

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

This repository currently focuses on **dataset assembly and validation**, which is the foundation for later statistical testing and ML modeling.

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

### Exploratory findings so far

Initial aggregate threshold analysis suggested a contrarian effect, but outlier diagnostics showed that this was largely driven by problematic price behavior in a small number of thin/exotic pairs.

After pair-level quality filtering:

- the broad aggregate threshold effect became weak
- major pairs appeared mostly flat
- thin/exotic pairs were weak or negative in the simple threshold framework
- a subset of liquid crosses looked more promising

Persistence analysis then suggested that any remaining effect is conditional rather than universal.

The most interesting exploratory result so far is:

- within the cleaned cross universe
- persistent extreme sentiment (`abs_sentiment >= 70` and `extreme_streak_70 >= 3`)
- appears stronger in a cluster of **JPY crosses**

A permutation test on pair-cluster membership suggested that this JPY-cross clustering is unlikely under a null of random pair-group assignment within the cross universe.

### Current interpretation

This does **not** yet establish a production-ready trading signal.

At this stage, the more defensible conclusion is:

- simple threshold-based sentiment fading is not broadly robust in the cleaned universe
- but there may be a more specific, regime- or pair-dependent effect
- particularly in persistent extreme sentiment conditions within JPY crosses


The next step is time-based validation and walk-forward testing to determine whether this effect holds up out of sample.

---

## Running the project

### 1. Build the research dataset

```bash
python build_fx_sentiment_dataset.py
```

### 2. Run exploratory analysis

Examples:

```bash
python analyze_thresholds.py
python analyze_pair_quality.py
python analyze_persistence.py
python analyze_by_pair_group.py
python analyze_jpy_cluster_permutation.py
```

These scripts generate summary tables and validation outputs under `data/output/analysis/`.

---

## License and data availability

This repository is distributed under a **non-commercial, source-available license** for the original code and repository-authored documentation.

- Personal, educational, academic, and non-commercial research use is allowed.
- Commercial use, resale, sublicensing, and inclusion in paid products or services is not allowed without prior written permission.

### Data availability

Raw broker-exported FX price data, raw sentiment scrape files, and full derived datasets are **not distributed** in this repository due to licensing and redistribution uncertainty.

The repository contains the code and documentation needed to reproduce the pipeline using data that you have the right to access and use locally.

---

## Project structure

```text
.
├── analyze_by_pair_group.py
├── analyze_cross_pair_persistence.py
├── analyze_jpy_cluster_permutation.py
├── analyze_outliers.py
├── analyze_pair_quality.py
├── analyze_persistence.py
├── analyze_thresholds.py
├── build_fx_sentiment_dataset.py
├── data
│   ├── input
│   │   ├── fx/
│   │   └── sentiment/
│   ├── output
│   │   ├── analysis/
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
├── project_description.md
└── README.md
```

Note: `data/input/` and `data/output/` are **expected local directories**