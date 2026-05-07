# Feature Definitions

This document defines the semantics of the core behavioral features used across:

- dataset construction (`scripts/build_fx_sentiment_dataset.py`)
- ML experiments (`research/signal_discovery/*`, `docs/mlp.md`)
- ABM outputs (`research/abm/*`)

The goal is to make feature meaning explicit so that model changes do not silently change feature interpretation.

---

## `net_sentiment`

**Definition:** signed retail crowd positioning imbalance.

For a given `(pair, snapshot_time)` sentiment snapshot, the raw source contains:

- `perc` — percentage on one side
- `direction ∈ {long, short}` — which side that percentage refers to

The dataset computes:

```text
net_sentiment = pct_long - pct_short
```

Since the raw snapshot provides only the dominant side percentage, the implementation is:

```text
if direction == "long":  net_sentiment =  +perc
if direction == "short": net_sentiment =  -perc
```

**Range:** `[-100, +100]`

| Value | Interpretation |
|---:|---|
| `+100` | crowd 100% long |
| `0` | balanced |
| `-100` | crowd 100% short |

**Sign meaning:**

- `net_sentiment > 0` → crowd net long
- `net_sentiment < 0` → crowd net short

**Magnitude meaning:**

- `abs(net_sentiment)` measures *extremeness* of crowd positioning

### Implementation reference

- Dataset assembly: `scripts/build_fx_sentiment_dataset.py` (sentiment loading stage)
- Crowd-side label: `pipeline/features.py::compute_crowd_side`

---

## `abs_sentiment`

```text
abs_sentiment = abs(net_sentiment)
```

Used for extreme detection and persistence metrics.

---

## `crowd_side`

A discrete sign label derived from `net_sentiment`:

- `+1` → crowd long
- `-1` → crowd short
- `0` → exactly neutral

---

## Notes on contracts across modules

- The research dataset convention is that `net_sentiment ∈ [-100, +100]`.
- ABM outputs are expected to mirror these semantics when writing `net_sentiment` for downstream tooling.
- If ABM internal state uses continuous accumulation, aggregation must preserve the dataset-scale meaning.
