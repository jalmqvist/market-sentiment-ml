# MLP Experiments — Summary

## Overview

Multi-layer perceptron (MLP) models were trained to predict forward FX returns
from retail sentiment and price-derived features.

Initial experiments appeared to show no predictive signal. However, later
regime-conditioned and pair-conditioned experiments revealed weak but detectable
predictive structure in specific contexts.

The overall conclusion is now:

- static models extract only limited signal
- signal quality is highly regime-dependent
- predictive structure is weak and unstable
- signal varies substantially across currency pairs

MLPs remain useful as a cartography tool for mapping where signal may exist,
even if they are not currently robust enough for deployment.

---

## Input Features

Features were derived from the master research dataset.

### Sentiment Features

| Feature              | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `net_sentiment`      | Signed dominant-side retail positioning: `+perc` if crowd is net long, `-perc` if crowd is net short. Intended range approximately `[-100, +100]`. |
| `abs_sentiment`      | Absolute value of `net_sentiment`                            |
| `sentiment_change`   | First difference of `net_sentiment`                          |
| `sentiment_z`        | Rolling z-score normalization of `net_sentiment`             |
| `extreme_streak_70`  | Consecutive snapshot events with `abs_sentiment >= 70`        |
| `extreme_streak_80`  | Consecutive snapshot events with `abs_sentiment >= 80`        |

### Price Features

| Feature     | Description                        |
| ----------- | ---------------------------------- |
| `trend_12b` | Short-horizon trend proxy          |
| `trend_48b` | Longer-horizon trend proxy         |
| `vol_12b`   | Short-horizon realized volatility  |
| `vol_48b`   | Longer-horizon realized volatility |

### Targets

| Feature   | Description           |
| --------- | --------------------- |
| `ret_12b` | 12-bar forward return |
| `ret_24b` | 24-bar forward return |
| `ret_48b` | 48-bar forward return |

---

## Experimental Design

### Early Experiments

Initial MLP experiments used:

- static feature vectors
- broad cross-pair datasets
- simple directional targets
- standard train/test splits

These experiments generally produced near-random performance.

### DL v2 Experiments

Later experiments introduced:

- regime filtering (`HVTF`, `LVTF`, `HVR`, `LVR`)
- configurable forward horizons
- thresholded classification labels
- improved normalization and logging
- pair-conditioned analysis
- feature-set cartography

This revealed weak predictive structure under certain conditions.

---

## Main Findings

### 1. Signal Is Conditional

Predictive structure is not universal.

Performance depends heavily on:

- volatility regime
- trend regime
- currency pair
- prediction horizon
- feature composition

The strongest results typically appeared in:

- `HVTF` (high-volatility trending)
- `LVTF` (low-volatility trending)

depending on pair.

---

### 2. Static Signal Exists, but Is Weak

MLPs occasionally achieved F1 scores meaningfully above random baseline,
particularly in pair/regime-conditioned experiments.

However:

- results were unstable across seeds
- many effects were small
- cross-pair generalization remained limited

This suggests retail sentiment contains weak structural information, but not
a strong standalone predictive edge.

---

### 3. Pair Dependence Matters

Signal topology differs materially across FX pairs.

Examples observed during cartography experiments:

- `USDJPY` often showed stronger signal
- `EURGBP` frequently showed weak or unstable behavior
- some pairs exhibited regime asymmetry

This implies retail positioning dynamics are not homogeneous across markets.

---

### 4. Regime Conditioning Matters

Contrary to earlier conclusions, regime filtering substantially changed model
behavior.

In particular:

- trending regimes generally produced stronger structure
- ranging regimes often collapsed toward noise
- volatility altered persistence characteristics

This became one of the central findings of later experiments.

---

## Interpretation

Current evidence suggests:

- retail sentiment is not purely random noise
- static snapshots contain limited information
- predictive structure is weak and conditional
- much of the usable signal likely exists in temporal dynamics rather than
  isolated feature values

This aligns with later LSTM findings.

---

## Reproducibility

Every MLP training run writes two files to `logs/`.

### Log file naming

```text
logs/mlp_{tag}_{timestamp}.log
```

Example:

```
logs/mlp_price_trend_20260502T121500Z.log
```

The tag defaults to the `--feature-set` value and can be overridden with
 `--tag`.

### Config JSON

```
logs/mlp_{tag}_{timestamp}.json
```

The JSON snapshot includes:

- `experiment_type`
- `dataset_path`
- `dataset_version`
- `cli_command`
- feature set
- hyperparameters
- regime filter
- pair selection

### Re-running an experiment

Retrieve the exact command from the JSON snapshot:

```
cat logs/mlp_price_trend_20260502T121500Z.json \
  | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```

Then run the printed command.

Example:

```
python research/deep_learning/train.py \
  --dataset-version 1.3.2 \
  --feature-set price_trend \
  --regime HVTF \
  --target-horizon 24
```

------

## Current Status

### What Has Been Established

- weak conditional predictive signal exists
- regime dependence is real
- pair dependence is real
- static models can detect limited structure

### What Remains Unresolved

- robustness across seeds
- economic significance after costs
- stability across time
- cross-pair transferability
- deployment viability

------

## Conclusion

MLPs do not provide a strong or universal predictive model of retail FX
sentiment.

However, later experiments showed that:

- weak predictive structure exists
- signal is highly conditional
- regime and pair context matter substantially

MLPs are therefore best viewed as:

- exploratory mapping tools
- baseline classifiers
- diagnostic models for identifying potentially interesting regions of the
  search space

rather than production-ready predictive systems.
