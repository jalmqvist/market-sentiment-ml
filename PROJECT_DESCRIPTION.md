# Project Description

This project investigates whether **retail FX sentiment extremes** contain predictive information about subsequent market behavior.

The initial working hypothesis was **contrarian and conditional**:

- extreme positioning may create exploitable signals  
- persistence of sentiment may matter  
- the effect may depend on pair type and market regime  

## Scope

This is a **data engineering and quantitative research project**, not a finished trading strategy.

It demonstrates:

- large-scale multi-file data ingestion  
- timestamp alignment across heterogeneous sources  
- research-grade feature engineering  
- pair-level data-quality filtering  
- structured validation (permutation, holdout, walk-forward)  
- reproducible research workflow  

---

## Data used

The pipeline combines:

- multi-year retail FX sentiment snapshot data  
- hourly FX market data exported from MT4 histories  

Raw data is not distributed due to licensing constraints.

---

## Pipeline summary

The project includes:

- sentiment aggregation across many CSV snapshots  
- pair normalization  
- timezone correction  
- forward alignment to hourly price bars  
- construction of forward returns and contrarian targets  
- behavioral feature engineering (persistence, streaks, dynamics)  
- structured validation scripts  

---

## Key result (updated)

After correcting for major methodological issues:

- overlapping signals  
- in-sample bias  
- improper walk-forward validation  

👉 **No robust predictive edge remains under price-based conditioning**

Specifically, the following do *not* hold:

- volatility regime (HV vs LV)  
- trend alignment (fight vs follow)  
- trend strength buckets  
- macro regime (pre/post 2022)  

All converge to approximately:

- mean ≈ 0  
- Sharpe ≈ 0  
- hit rate ≈ 50%  

---

## Interpretation

This is a **negative result**.

> Retail sentiment does not produce a stable signal when conditioned on price-based regimes.

Earlier findings (e.g. JPY-cross persistence effects) were driven by:

- overlapping trade exposure  
- clustering in specific time periods  
- pair concentration bias  
- incorrect aggregation methods  

---

## Updated research direction

The failure of price-based regimes suggests:

> If a signal exists, it is governed by **crowd behavior**, not market structure.

The project therefore shifts toward:

### Regime v2 — Behavioral regimes

Instead of conditioning on price (volatility, trend), we model:

- crowd persistence (streaks of extreme positioning)  
- sentiment acceleration (rate of change)  
- crowd saturation (imbalance intensity)  
- behavioral stress (crowd trapped vs correct)  

---

## Current status

The repository now represents:

- a **validated research pipeline**  
- a **documented negative result**  
- a **foundation for behavioral regime modeling (Regime v2)**  

---

## Next phase

The next step is to:

- define minimal behavioral regimes  
- test them under strict walk-forward validation  
- expand analysis beyond JPY crosses to the full FX universe  

---

## Philosophy

This project emphasizes:

- eliminating false positives  
- validating assumptions rigorously  
- documenting negative results clearly  

The goal is not to “find a signal”, but to:

> **build a reliable process for discovering (or rejecting) signals**
