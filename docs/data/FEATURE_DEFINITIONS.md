# Feature Definitions

This document defines the semantics of the core behavioral features used across:

- dataset construction (`scripts/build_fx_sentiment_dataset.py`)
- ML experiments (`research/signal_discovery/*`, `docs/mlp.md`)
- ABM outputs (`research/abm/*`)

The goal is to make feature meaning explicit so that model changes do not silently change feature interpretation.

---

## `net_sentiment`

**Definition:** signed retail crowd positioning (dominant-side percent with sign).

For a given `(pair, snapshot_time)` sentiment snapshot, the raw source contains:

- `perc` — percentage on one side
- `direction ∈ {long, short}` — which side that percentage refers to

**Canonical assembly (as implemented):**

```text
if direction == "long":  net_sentiment =  +perc
if direction == "short": net_sentiment =  -perc
```

**Range:** approximately `[-100, +100]`

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

## `sentiment_change`

Event-based change in sentiment between consecutive snapshots per pair:

`sentiment_change(k) = net_sentiment(k) − net_sentiment(k−1)`

Computed on the snapshot-event index (not on a forward-filled hourly grid) to
avoid differencing artefacts.

---

## `sentiment_z`

A rolling z-score normalization of sentiment.

### A) Dataset column (`scripts/build_fx_sentiment_dataset.py`)

The master dataset includes a `sentiment_z` column computed in
`add_sentiment_v2_features()`.

Definition (per pair, causal/backward-looking):

- `sentiment = net_sentiment`
- `roll_mean = rolling_mean(sentiment, window=100, min_periods=1)`
- `roll_std  = rolling_std(sentiment,  window=100, min_periods=1)`
- `sentiment_z = (sentiment - roll_mean) / (roll_std + 1e-8)`

Notes:

- this is computed per pair (`groupby("pair")`)
- early rows are less stable because `min_periods=1`
- downstream ML code may still standardize features again on the train split;
  that does not change the definition of the dataset column.

### B) Signal-discovery scripts (local feature engineering)

Several signal discovery scripts compute a *local* `sentiment_z` internally as
part of their own feature engineering (i.e. not necessarily identical to the
dataset column definition).

Examples include:

- `research/signal_discovery/signal_v2.py`
- `research/signal_discovery/regime_v14.py`
- `research/signal_discovery/regime_v17.py`
- `research/signal_discovery/regime_v18.py`
- `research/signal_discovery/regime_v19.py`

These variants typically use a rolling z-score with a **96-bar** window (per
pair) on `net_sentiment`.

**Rule:** unless a given experiment explicitly states otherwise, treat
`sentiment_z` in signal-discovery scripts as an experiment-local feature, and
`sentiment_z` in the dataset as the canonical, reusable column.

---

## Notes on contracts across modules

- The research dataset convention is that `net_sentiment ∈ [-100, +100]`.
- ABM outputs are expected to mirror these semantics when writing `net_sentiment` for downstream tooling.
- If ABM internal state uses continuous accumulation, aggregation must preserve the dataset-scale meaning.
