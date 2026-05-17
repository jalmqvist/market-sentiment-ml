# LSTM Experiments — Summary

## Overview

Long Short-Term Memory (LSTM) models were trained to capture temporal structure
in retail FX sentiment and predict forward returns.

Early experiments appeared to show no temporal signal. However, later
regime-conditioned experiments revealed weak but detectable predictive
structure under specific conditions.

The current interpretation is:

- temporal structure exists

- signal is weak and conditional
- regime filtering materially affects detectability
- sequence models extract information not visible to static models

---

## Sequence Modeling Approach

### Architecture

Typical experiments used:

- single-layer LSTM
- hidden size 32–64
- dropout regularization
- binary directional output head
- Adam optimizer
- BCE / cross-entropy loss

---

### Sequence Construction

| Parameter       | Description                 |
| --------------- | --------------------------- |
| Sequence length | Typically 24 bars           |
| Stride          | 1 bar                       |
| Target          | Direction of forward return |
| Horizons        | 12b / 24b / 48b             |

Sequences were constructed using overlapping rolling windows.

---

## Input Features

### Sentiment Features

| Feature              | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `net_sentiment`      | Signed dominant-side retail positioning: `+perc` if crowd is net long, `-perc` if crowd is net short. Intended range approximately `[-100, +100]`. |
| `abs_sentiment`      | Absolute value of `net_sentiment`                            |
| `sentiment_change`   | First difference of `net_sentiment`                          |
| `sentiment_z`        | Rolling z-score normalization of sentiment                   |
| `extreme_streak_70`  | Consecutive snapshot events with `abs_sentiment >= 70`        |
| `extreme_streak_80`  | Consecutive snapshot events with `abs_sentiment >= 80`        |

### Price Features

| Feature     | Description               |
| ----------- | ------------------------- |
| `trend_12b` | Short-horizon trend       |
| `trend_48b` | Longer-horizon trend      |
| `vol_12b`   | Short-horizon volatility  |
| `vol_48b`   | Longer-horizon volatility |

---

## Normalization Rules

| Feature             | Normalization                                    |
| ------------------- | ------------------------------------------------ |
| `net_sentiment`     | Divide by 100 (maps approximately to `[-1, +1]`) |
| `abs_sentiment`     | Divide by 100                                    |
| Price features      | Standardization using training fold only         |
| Volatility features | Standardization using training fold only         |
| `sentiment_z`       | Already normalized                               |

Normalization statistics were always computed on training data only and applied
to validation/test sets to avoid leakage.

---

## Experimental Evolution

### Early Phase

Initial experiments used:

- broad datasets
- weak labeling schemes
- static thresholds
- minimal regime conditioning

Results were generally indistinguishable from noise.

---

### DL v2 Improvements

Later experiments introduced:

- regime-aware filtering
- configurable horizons
- thresholded labels
- improved feature engineering
- pair-conditioned analysis
- cleaner train/test separation
- extensive logging and aggregation

This substantially improved detectability of weak temporal structure.

---

## Main Findings

### 1. Temporal Structure Exists

Contrary to early conclusions, sequence models occasionally extracted weak
predictive structure.

This signal was generally:

- small
- unstable
- highly conditional

but consistently above random in certain regions of the search space.

---

### 2. Signal Depends on Regime

The strongest results frequently appeared in:

- `HVTF`
- `LVTF`

depending on pair and horizon.

Ranging regimes often produced weaker or noisier behavior.

This became one of the central findings of the DL v2 experiments.

---

### 3. Signal Depends on Pair

Different FX pairs exhibited materially different behavior.

Examples observed during cartography experiments:

- `USDJPY` frequently showed stronger structure
- `EURJPY` sometimes exhibited persistent temporal signal
- `EURGBP` often remained weak or unstable

This suggests retail positioning dynamics differ substantially across markets.

---

### 4. Sequence Models Extract Information Beyond Static Features

LSTM models occasionally outperformed comparable MLP setups.

This implies:

- temporal ordering matters
- persistence and accumulation dynamics matter
- sequence structure contains information not visible in isolated snapshots

This became one of the major conceptual findings of the later experiments.

---

## Interpretation

Current evidence suggests:

- retail sentiment is not purely random
- predictive structure is weak but nonzero
- much of the usable signal exists in temporal organization
- volatility and trend regimes alter accumulation dynamics

These findings later motivated the development of regime-dependent behavioral
ABM hypotheses.

---

## Reproducibility

Every LSTM training run writes two files to `logs/`.

### Log file naming

```text
logs/lstm_{tag}_{timestamp}.log
```

Example:

```
logs/lstm_price_trend_20260502T123000Z.log
```

The tag defaults to the `--feature-set` value and can be overridden with
 `--tag`.

### Config JSON

```
logs/lstm_{tag}_{timestamp}.json
```

The JSON snapshot includes:

- `experiment_type`
- `dataset_path`
- `dataset_version`
- `cli_command`
- feature set
- hyperparameters
- sequence length
- regime filter
- pair selection

### Re-running an experiment

Retrieve the exact command from the JSON snapshot:

```
cat logs/lstm_price_trend_20260502T123000Z.json \
  | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```

Then run the printed command.

Example:

```
python research/deep_learning/train_lstm.py \
  --dataset-version 1.3.2 \
  --feature-set price_trend \
  --regime HVTF \
  --seq-len 24 \
  --target-horizon 24
```

------

## Current Status

### What Has Been Established

- weak temporal signal exists
- regime dependence is real
- pair dependence is real
- sequence structure matters
- temporal accumulation dynamics appear important

### What Remains Unresolved

- robustness across seeds
- economic significance after costs
- stability across time
- cross-pair generalization
- deployment viability

------

## Conclusion

LSTM models revealed weak but meaningful temporal structure in retail FX
sentiment data.

The signal is:

- conditional rather than universal
- regime-dependent
- pair-dependent
- sensitive to labeling and preprocessing choices

Sequence models therefore appear more appropriate than static classifiers for studying retail sentiment dynamics, even though current predictive performance remains far from production quality.

---
