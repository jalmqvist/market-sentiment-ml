# ABM Runbook — Retail FX Sentiment Model

## Purpose

This document defines how to reproduce the baseline Agent-Based Model (ABM) results for retail FX sentiment.

It serves as:

- a reproducibility reference
- a debugging baseline
- a guardrail against pipeline drift

------

# 1. Model Overview

The ABM simulates a population of retail FX traders interacting with:

- real price data (exogenous)
- crowd sentiment (endogenous)

Core components:

- Simulation engine:
- Agent behavior:
- Calibration + scoring: ,
- Sweep pipeline:

------

# 2. Core Mechanism

Each agent updates position based on:

```
score = price_signal
      + crowd_effect
      + noise
      + persistence
```

Key dynamics:

- **Trend following / contrarian behavior**
- **Persistence (position memory)**
- **Inertia (resistance to switching)**
- **Asymmetric switching with anchoring**

Agents operate on **real price series only** (no synthetic generation).

------

# 3. Baseline Experiment (REFERENCE)

## Command

```bash
python research/abm/sweep.py \
    --version 1.2.0 \
    --pair eur-usd \
    --steps 500
```

------

## Dataset

- Version: `1.2.0`
- Source: research dataset
- Required columns:
  - `entry_close`
  - `entry_time`
  - `net_sentiment`
  - `pair`

------

## Output

Files written to:

```
logs/
  abm_sweep_eur-usd_1.2.0_<timestamp>.csv
  abm_sweep-eur-usd_<timestamp>.log
  abm_sweep-eur-usd_<timestamp>.json
```

------

# 4. Parameter Grid

Defined in sweep:

- 

## Dimensions

### Trend ratio

```
[0.0, 0.5, 1.0]
```

Fraction of non-noise agents that follow trend.

------

### Persistence weight

```
[0.0, 0.1, 0.2]
```

Strength of position memory.

------

### Inertia threshold

```
[0.02, 0.05, 0.1]
```

Minimum signal required to change position.

------

## Total runs

```
3 × 3 × 3 = 27 configurations
```

------

# 5. Agent Composition

Fixed population:

- 40 trend-followers
- 40 contrarians
- 20 noise traders

Total: 100 agents

------

# 6. Simulation Details

From :

- Warmup: 48 steps
- Real price series only (no GBM)
- Sentiment computed as:
  - normalized [-1, 1]
  - scaled [-100, 100] for output

Each step:

1. Advance price
2. compute crowd sentiment
3. update agents
4. record state

------

# 7. Calibration Targets

From :

Metrics matched against real data:

- mean
- std
- abs_mean
- autocorrelation
- extreme frequency
- long fraction
- sentiment vs return correlations

------

# 8. Scoring Function

From :

```
score = 0.3 * std_diff
      + 0.3 * abs_mean_diff
      + 0.3 * autocorr_diff
      + 0.1 * extreme_diff
```

Lower = better

------

# 9. Expected Behavior (CRITICAL)

A correct run should show:

### Qualitative

- clustering of sentiment
- persistence over time
- asymmetric positioning
- non-random structure

------

### Quantitative

- finite score (not inf)
- stable ranking across runs
- sensitivity to persistence/inertia

------

### Structural property

> sentiment reacts to price but does not predict returns

------

# 10. Known Model Interpretation

The ABM encodes:

```
signal = f(trend, persistence, inertia)
```

It explains:

- accumulation dynamics
- sentiment clustering
- persistence in retail positioning

------

# 11. Known Limitations

The baseline model does NOT include:

- volatility effects
- regime switching
- macro flow dynamics

This explains failure modes in:

- JPY pairs
- CHF pairs
- high-volatility environments

------

# 12. Reproducibility Checklist

Before modifying anything, verify:

-  Command matches exactly
-  Dataset version = 1.2.0
-  Pair = eur-usd
-  Seed = 42
-  Output files generated
-  Score values finite
-  Results structurally similar

------

# 13. Rules for Future Work

## DO

- create new experiment files
- isolate modifications
- compare against baseline

## DO NOT

- modify sweep.py directly
- change agent defaults globally
- mix pipeline changes with model changes

------

# 14. Relationship to DL Findings

DL experiments show:

```
signal = f(trend, stability)
```

Interpretation:

- ABM captures accumulation
- DL reveals missing dimension: **stability / volatility**

------

## Hypothesis

```
signal = f(trend, persistence, inertia, stability)
```

------

# 15. Next Step (ABM Extension)

Implement minimal extension:

- introduce volatility-dependent behavior
- test:

```
low volatility  → stable accumulation
high volatility → breakdown
```

------

# 16. Status

✔ Baseline reproduced
✔ Pipeline verified
✔ Ready for controlled extension

------

# End of Runbook