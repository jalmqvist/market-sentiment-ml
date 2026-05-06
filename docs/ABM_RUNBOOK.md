# ABM Runbook — Retail FX Sentiment Model

## Purpose

This document defines how to reproduce, validate, and extend the baseline Agent-Based Model (ABM) results for retail FX sentiment.

It serves as:

- a reproducibility reference
- a debugging baseline
- a guardrail against pipeline drift
- a stage-gated log of minimal ABM extensions

------

# 1. Model Overview

The ABM simulates a population of retail FX traders interacting with:

- real price data (exogenous)
- crowd sentiment (endogenous)

Core components:

- **Simulation engine:** `research/abm/simulation.py` (`FXSentimentSimulation`)
- **Agent behavior:** `research/abm/agents.py` (`RetailTrader` + subclasses)
- **Calibration + scoring:** `research/abm/calibration.py`, `research/abm/scoring.py`
- **Sweep pipeline:** `research/abm/sweep.py`

Notes:

- The ABM runs on **real price series only** (no internal GBM price generation).
- The sweep grid is fixed to `3 × 3 × 3 = 27` configurations.

------

# 2. Core Mechanism

Each agent updates position based on:

```text
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
- **Discrete accumulation state** (integer position that can “ratchet”)

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

## Dataset

- Version: `1.2.0`
- Source: research dataset
- Required columns:
  - `entry_close`
  - `entry_time`
  - `net_sentiment`
  - `pair`

## Output

Files written to:

```text
logs/
  abm_sweep_eur-usd_1.2.0_<timestamp>.csv
  abm_sweep-eur-usd_<timestamp>.log
  abm_sweep-eur-usd_<timestamp>.json
```

------

# 4. Parameter Grid

Defined in `research/abm/sweep.py`.

## Dimensions

### Trend ratio

```text
[0.0, 0.5, 1.0]
```

Fraction of non-noise agents that follow trend.

### Persistence weight

```text
[0.0, 0.1, 0.2]
```

Strength of position memory.

### Inertia threshold

```text
[0.02, 0.05, 0.1]
```

Minimum signal required to change position.

## Total runs

```text
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

From `research/abm/simulation.py`:

- Warmup: 48 steps
- Real price series only (no GBM)
- Sentiment computed as:
  - normalized `[-1, 1]` for internal crowd signal
  - scaled `[-100, 100]` for output

Each step:

1. Advance price
2. compute crowd sentiment
3. update agents
4. record state

------

# 7. Calibration Targets

From `research/abm/calibration.py`.

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

From `research/abm/scoring.py`:

```text
score = 0.3 * std_diff
      + 0.3 * abs_mean_diff
      + 0.3 * autocorr_diff
      + 0.1 * extreme_diff
```

Lower = better.

**Important:** Score is used only for ranking sweep configurations; it is NOT a direct measure of stability/release.

------

# 9. Expected Behavior (CRITICAL)

A correct baseline run should show:

### Qualitative

- clustering of sentiment
- persistence over time
- asymmetric positioning
- non-random structure

### Quantitative

- finite score (not inf)
- stable ranking across runs
- sensitivity to persistence/inertia parameters

### Structural property

> sentiment reacts to price but does not predict returns

------

# 10. Known Model Interpretation

The baseline ABM encodes:

```text
signal = f(trend, persistence, inertia)
```

It explains:

- accumulation dynamics
- sentiment clustering
- persistence in retail positioning

------

# 11. Known Limitations (Baseline)

The baseline model does NOT include:

- volatility-conditioned *release* (decay) of accumulated sentiment
- regime switching
- macro flow dynamics

This helps explain failure modes in:

- JPY pairs
- CHF pairs
- high-volatility environments

------

# 12. Reproducibility Checklist

Before modifying anything, verify:

- Command matches exactly
- Dataset version = 1.2.0
- Pair = eur-usd
- Seed = 42
- Output files generated
- Score values finite
- Results structurally similar

------

# 13. Rules for Future Work

## DO

- create new experiment files
- isolate modifications
- compare against baseline
- keep defaults backward compatible (baseline behavior must remain reachable)

## DO NOT

- modify `research/abm/sweep.py` directly
- change agent defaults globally without documenting it here
- mix pipeline changes with model changes

------

# 14. Relationship to DL Findings

DL experiments suggest:

```text
signal = f(trend, stability)
```

Interpretation:

- ABM captures accumulation
- DL reveals missing dimension: **stability / volatility**

Hypothesis:

```text
signal = f(trend, persistence, inertia, stability)
```

------

# 15. Stage 1 Extension (Environment Volatility Perturbation)

## Goal

Test whether *environment-only* volatility perturbations reproduce the DL finding:

- low volatility  → stable accumulation
- high volatility → breakdown of accumulation

## Tooling

Experiment script:

- `abm_experiments/sweep_with_volatility.py`

Mechanism:

- returns = `diff(price)`
- rolling realized volatility = rolling std of returns
- amplify returns by `(1 + volatility_scale * vol_norm_t)`
- run ABM unchanged against adjusted price series

Outputs (experiment):

```text
logs/
  abm_sweep_vol_<pair>_<version>_<timestamp>.csv
  abm_sweep_vol_bestpath_<pair>_<version>_<timestamp>.csv
  abm_sweep-vol-<pair>_<timestamp>.log
  abm_sweep-vol-<pair>_<timestamp>.json
```

## Stage-1 finding (summary)

Environment-only volatility perturbation changes how quickly the model saturates, but does **not** reliably produce a “release” mechanism. In particular, sign flips can remain at/near zero an[...]

This motivates Stage 2.

------

# 16. Stage 2 Extension (Volatility-Conditioned Decay / Release)

## Goal

Introduce a minimal **decay (release)** mechanism that affects accumulated sentiment state (agent accumulation), not the external signal. This is required to prevent absorbing states and to allow[...]

## Design constraints

- Minimal surface area
- Backward compatible defaults
- No refactors / no new modules
- Do not modify `research/abm/sweep.py`

## Implementation (current)

### Agent-side decay (accumulation release)

In `research/abm/agents.py`, accumulation is modified to:

```text
lambda_t = decay_base + decay_volatility_scale * vol_norm_t
lambda_t = clip(lambda_t, 0.0, decay_clip_max)

S_t = (1 - lambda_t) * S_{t-1} + ΔS_t
```

Defaults preserve baseline behavior:

- `decay_base = 0.0`
- `decay_volatility_scale = 0.0`
- `decay_clip_max = 0.2`

#### Important note on sensitivity experiments (β)

A separate sensitivity harness and an experiment diary exist to document the Stage‑2 investigation:

- Sensitivity harness: `abm_experiments/decay_beta_sensitivity.py`
- Experiment diary: [`docs/ABM_EXPERIMENT_DIARY.md`](ABM_EXPERIMENT_DIARY.md)

The Stage‑2 sensitivity work uncovered two interpretation pitfalls:

1. **Quantization:** if the agent accumulation state is stored as an integer and decay uses truncation, β can behave like an on/off switch rather than a continuous control parameter.
2. **Scale drift:** if the agent accumulation state becomes continuous (float) and aggregation uses raw positions, the simulation output `net_sentiment` can exceed the dataset convention `[-100, +100]`. In that case, thresholds such as `abs(net_sentiment) >= 90` no longer represent “near-extreme positioning” and should be re-evaluated, or aggregation should be changed to preserve dataset semantics.

### Simulation-side volatility proxy

In `research/abm/simulation.py`, per-timestep volatility is computed once and passed into agents:

- `ret_t = price_t - price_{t-1}`
- EMA volatility proxy:

```text
alpha = 2 / (vol_window + 1)
vol_ema = alpha * abs(ret_t) + (1 - alpha) * vol_ema
```

- Normalization:

```text
vol_norm = vol_ema / (baseline_vol + 1e-8)
```

- Passed into agent update:

```text
agent.update(..., volatility=vol_norm)
```

Note: `vol_window` is currently fixed in simulation code (default 24).

## Running Stage-2 controlled activation experiments

Use the existing experiment script (do not change sweep pipeline):

```bash
python abm_experiments/sweep_with_volatility.py \
  --version 1.2.0 \
  --pair eur-usd \
  --steps 2000 \
  --volatility-scale 0.0 \
  --decay-base 0.0 \
  --decay-volatility-scale 0.10 \
  --decay-clip-max 0.2
```

Stage-2 CLI flags are supported by:

- `abm_experiments/sweep_with_volatility.py`

Flags:

- `--decay-base`
- `--decay-volatility-scale`
- `--decay-clip-max`

## Analysis focus (do NOT optimize for score)

Extract from best-path CSV:

- `pct_time_saturated` (e.g., abs(net_sentiment) >= 90)
- `sign_flips` (crossings of 0 sign in net_sentiment)
- `first_hit_step` (first time abs(net_sentiment) >= 90)
- `autocorr_lag1` of net_sentiment

Helper script (optional):

- `abm_experiments/summarize_bestpaths.py`

## Stage-2 validation result (eur-usd, version=1.2.0, steps=2000, volatility_scale=0.0)

Baseline (decay disabled):
- `pct_time_saturated ≈ 0.6305`
- `sign_flips = 0`
- `autocorr_lag1 ≈ 0.9997`

Decay enabled (decay_volatility_scale ≥ 0.02 in tested range):
- `pct_time_saturated ≈ 0.4055`
- `sign_flips = 2`
- `autocorr_lag1 ≈ 0.9806`

Interpretation: decay introduces a release mechanism and reduces absorbing saturation vs baseline under identical environment conditions.

------

# 17. Status

✔ Baseline reproduced and stable  
✔ Stage 1 environment-volatility experiment implemented  
✔ Stage 2 decay/release mechanism implemented (defaults off)  
✔ Controlled activation shows reduced saturation and nonzero sign flips at 2000 steps  

------

# End of Runbook
