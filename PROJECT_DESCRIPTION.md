# Project Description

This project investigates whether **retail FX sentiment extremes** contain predictive information about subsequent market behavior.

The initial working hypothesis is **contrarian**:

- when retail traders become strongly one-sided in a currency pair, fading the crowd may have predictive value
- the effect may depend on **time horizon**
- the effect may depend on **persistence** of extreme sentiment
- the effect may depend on **pair type** and possibly **market regime**

## Scope

This is primarily a **data engineering and quantitative research project**, not a claim of a finished trading strategy.

The project demonstrates:

- large-scale multi-file data ingestion
- timestamp alignment across heterogeneous sources
- research-grade feature engineering
- pair-level data-quality filtering
- careful validation of exploratory findings
- structured preparation for downstream ML workflows

## Data used

The research pipeline combines:

- multi-year retail FX sentiment snapshot data
- hourly FX market data exported from MT4 histories

The raw data is not distributed in the repository due to licensing and redistribution uncertainty.

## Pipeline summary

The project currently includes:

- sentiment aggregation across many CSV snapshots
- pair normalization
- timezone correction
- pair-by-pair forward merge to hourly price bars
- merge tolerance to prevent invalid long-gap matches
- construction of trading-bar forward returns
- construction of contrarian return targets
- coverage diagnostics and filtered research universes
- threshold, persistence, subgroup, permutation, holdout, and walk-forward analysis scripts

## Current findings

The broad simple contrarian threshold effect does not appear robust across the cleaned universe.

However, more conditional structure has emerged:

- the aggregate effect weakens substantially after removing problematic outlier pairs
- major pairs look mostly flat in the simple framework
- thin/exotic pairs are generally weak or negative after cleaning
- a more interesting signal candidate appears in **JPY crosses** under **persistent extreme sentiment**

This JPY-cross effect has so far survived:

- pair-level outlier filtering
- subgroup analysis
- permutation testing
- a simple time-based holdout
- expanding-window walk-forward validation

## Current research direction

The current working research direction is:

- validate whether the JPY-cross persistence effect continues to hold under stricter testing
- test whether the effect depends on market regime

  Recent results indicate that the signal is strongly conditioned on:

  - volatility regime (high vs low volatility)
  - trend alignment and strength
  - persistence of sentiment

  This shifts the research direction toward:

  - regime-aware modeling
  - integration with external regime detection (market-phase-ml)
  - evaluation of volatility-gated behavioral signals
- eventually connect the sentiment features to a broader FX ML feature pipeline
