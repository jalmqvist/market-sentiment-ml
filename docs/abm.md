# Agent-Based Model (ABM) — FX Retail Sentiment

## Purpose

The ABM simulates the collective positioning dynamics of retail FX traders to generate synthetic sentiment data that can be calibrated against real broker positioning data.

It is a **behavioral model**, not a predictive model:

- its goal is to reproduce *how retail positioning behaves* (persistence, extremes, asymmetry)
- any predictive signal is evaluated separately under strict out-of-sample validation

---

## Model Parameters (what they mean, and why we tune them)

This section documents the ABM parameters that materially change the emergent sentiment dynamics.

### Population / composition

| Parameter | Effect on dynamics | Why we include it |
|---|---|---|
| `n_trend` | More trend-followers increases one-sided accumulation during trends; increases saturation frequency. | Captures “trend-chasing” retail behavior; controls directional bias under sustained price moves. |
| `n_contrarian` | More contrarians introduces counter-pressure; reduces lock-in; increases mean-reversion in sentiment. | Captures fading behavior and position trimming near extremes. |
| `n_noise` | Adds random variation; prevents overly deterministic lock-in; increases local jitter. | Real positioning is noisy; helps avoid brittle dynamics. |

### Price signal

| Parameter | Effect on dynamics | Why we tune it |
|---|---|---|
| `momentum_window` | Longer window = slower reaction, smoother positioning; shorter = faster, choppier response. | Represents heterogeneity in how traders perceive “trend” at H1 frequency. |

### Core behavioral controls

| Control | Where it appears | Effect on dynamics | Why we tune it |
|---|---|---|---|
| Trend/contrarian mix (`trend_ratio` in sweeps) | `research/abm/sweep.py` | Shifts balance between trend-chasing vs fading; changes long-run bias and regime dwell time. | Primary lever for pair differences; DL results suggest signal is conditional on trend *and* stability. |
| Persistence weight | `research/abm/sweep.py` | Increases position memory; increases autocorr; lengthens regime dwell time. | Retail positioning is sticky; persistence is required to reproduce long streaks and clustered extremes. |
| Inertia threshold | `research/abm/sweep.py` | Adds switching friction; reduces churn; creates hysteresis (harder to flip sign). | Prevents unrealistic frequent flips; matches observed “slow to change mind” behavior. |
| Herding / crowd effect | `research/abm/agents.py` | Reinforces the majority side; increases lock-in and saturation probability. | Retail crowds exhibit herding; but over-strong herding is a known failure mode (absorbing states). |

### Volatility-conditioned release (Stage‑2)

Stage‑2 introduces a minimal **decay/release** mechanism that can reduce absorbing saturation under volatility.

| Parameter | Effect on dynamics | Why it exists |
|---|---|---|
| `decay_base` | Baseline release even in low vol; increases “leakiness” of accumulated positions. | Lets the model avoid permanent lock-in even in calm periods (defaults to 0). |
| `decay_volatility_scale` | More release during volatile periods; increases boundary time and transitions. | Encodes DL/empirical hypothesis: high volatility disrupts accumulation and forces de-risking. |
| `decay_clip_max` | Prevents unbounded release; stabilizes numeric behavior. | Safety bound / stability guardrail. |

---

## Escape defaults (diagnostics / sensitivity harness)

During Stage‑2 sensitivity testing, we introduced a *diagnostic* notion of “escape” (how often the aggregate sentiment gets near the decision boundary) even when sign flips are rare.

The harness `abm_experiments/decay_beta_sensitivity.py` reports (verbose):

- `pct_time_abs_le_20` — fraction of steps with `|net_sentiment| ≤ 20`
- `pct_time_negative` — fraction of steps with `net_sentiment < 0`

A set of “escape defaults” that produced useful near-boundary behavior without breaking persistence on key JPY crosses was:

```bash
export ABM_ESCAPE_SAT_THRESHOLD=0.3
export ABM_ESCAPE_PROB_SAT=0.08
export ABM_ESCAPE_SHRINK_FACTOR=0.5
export ABM_ESCAPE_ZERO_PROB=0.03
export ABM_ESCAPE_ZERO_COOLDOWN=6
export ABM_ESCAPE_FLIP_PROB=0.0
```

These are **not universal constraints** (some pairs naturally spend time negative), but they serve as a reproducible reference point for later experiments.

---

## How to Run

```bash
python research/abm/run_abm.py \
    --version 1.1.0 \
    --pair eur-usd \
    --steps 500 \
    --seed 42 \
    --n-trend 40 \
    --n-contrarian 40 \
    --n-noise 20 \
    --momentum-window 12 \
    --output logs/eur_usd_sim.csv
```

### Suppress file logging (stdout only)

```bash
python research/abm/run_abm.py \
    --version 1.1.0 \
    --pair eur-usd \
    --no-log-file
```

### Run tests

```bash
python -m pytest tests/test_abm.py -v
```

---

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--version` | *required* | Dataset version directory (e.g. `1.1.0`) |
| `--variant` | `core` | Dataset variant: `full`, `core`, or `extended` |
| `--pair` | *required* | FX pair slug (e.g. `eur-usd`, `usd-jpy`) |
| `--steps` | `500` | Number of simulation steps to record |
| `--seed` | `42` | RNG seed for reproducibility |
| `--n-trend` | `40` | Number of trend-following agents |
| `--n-contrarian` | `40` | Number of contrarian agents |
| `--n-noise` | `20` | Number of noise-trader agents |
| `--momentum-window` | `12` | Look-back window (bars) for price signal |
| `--output` | `None` | Path to save output CSV |
| `--log-level` | `INFO` | Logging verbosity |
| `--no-log-file` | off | Disable file logging; use stdout only |

---

## Agent Types

### TrendFollower

Follows recent price momentum. Goes long after sustained up-moves, short after down-moves. Crowd sentiment adds a herding weight.

### Contrarian

Fades recent momentum. Goes short after up-moves, long after down-moves. Reduces positions when crowd is already on one side.

### NoiseTrader

Random positioning. Provides baseline stochasticity and prevents artificial lock-in.

---

## Outputs

### Log file

`logs/abm_{pair}_{timestamp}Z.log` — full run log with parameters, summary statistics, and calibration comparison table.

### Config snapshot

`logs/abm_{pair}_{timestamp}Z.json` — machine-readable record of every parameter used in the run.

### Output CSV (if `--output` provided)

Columns:

| Column | Description |
|---|---|
| `timestamp` | Bar timestamp from real dataset |
| `price` | Real entry close price |
| `net_sentiment` | Simulated net retail positioning `(fraction_long - fraction_short) * 100`, bounded to `[-100, +100]` |
| `real_net_sentiment` | Actual net retail positioning from dataset |

Aggregation note: internal continuous agent positions are mapped to long/short/neutral
votes with `_AGGREGATION_EPS` before output, so ABM sentiment columns follow dataset
sign/scale semantics (`abs_sentiment = abs(net_sentiment)`, `crowd_side = sign(net_sentiment)`).

---

## Reproducibility

Every ABM run writes two files to `logs/`.

Retrieve the exact command from the JSON snapshot:

```bash
cat logs/abm_eur-usd_20260502T120000Z.json | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```
