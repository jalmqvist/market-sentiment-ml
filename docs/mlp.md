# MLP Experiments — Summary

## Overview

Multiple layer perceptron (MLP) models were trained to predict forward FX
returns from retail sentiment features. All experiments found no statistically
meaningful signal.

---

## Input Features

Features derived from the master research dataset:

| Feature | Description |
|---|---|
| `net_sentiment` | Raw net retail positioning (% long minus % short) |
| `abs_sentiment` | Absolute value of net sentiment |
| `extreme_streak` | Consecutive bars above extreme threshold (±70) |
| `persistence_bucket` | Categorical persistence label (high / medium / low) |
| `acceleration` | Rate of change of net sentiment |
| `ret_1b` | One-bar return (contemporaneous) |
| `ret_48b` | 48-bar forward return (prediction target) |

Regime-conditioned variants additionally included:

- `phase` (HV_Trend, HV_Ranging, LV_Trend, LV_Ranging)
- `is_trending`, `is_high_vol` flags

---

## Experimental Design

- Train/test split on time (no shuffle across time boundary).
- Walk-forward evaluation with expanding window.
- Targets: directional sign of `ret_12b` and `ret_48b`.
- Evaluation metric: balanced accuracy, AUC-ROC.
- Baseline: predict majority class or random.

---

## Why No Signal Was Found

1. **Retail sentiment is reactive, not predictive.** Correlation analysis
   shows `net_sentiment` correlates weakly with contemporaneous returns
   (`ret_1b`) but not with forward returns (`ret_48b`).

2. **Low signal-to-noise ratio.** After deduplication and quality filtering
   the effective sample is small, making it impossible to distinguish real
   patterns from noise.

3. **No regime-conditioning helped.** Subsetting by volatility regime
   (HV/LV) and trend phase did not reveal a persistent edge.

4. **Leakage-free evaluation confirms no edge.** All shift-tests and
   shuffle-tests produced performance indistinguishable from random chance.

---

## Reproducibility

Every MLP training run writes two files to `logs/`:

### Log file naming

```
logs/mlp_{tag}_{timestamp}.log
```

Example: `logs/mlp_price_sentiment_20260502T121500Z.log`

The tag defaults to the `--feature-set` value and can be overridden with `--tag`.

### Config JSON

```
logs/mlp_{tag}_{timestamp}.json
```

The JSON snapshot includes `experiment_type`, `dataset_path`, `dataset_version`,
`cli_command` (exact command used), and all hyperparameters.

### Re-running an experiment

Retrieve the exact command from the JSON snapshot:

```bash
cat logs/mlp_price_sentiment_20260502T121500Z.json | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```

Then paste and run the printed command, for example:

```bash
python research/deep_learning/train.py --dataset-version 1.1.0 --feature-set price_sentiment --epochs 50
```

---

## Conclusion

MLPs do not extract predictive signal from retail FX sentiment features under
any tested configuration. This is consistent with the broader finding that
retail sentiment is a contrarian indicator at best and not actionable at
standard ML model horizons.

---

## Update (DL v2 Context)

Subsequent experiments using sequence models (LSTM) indicate that:

- weak predictive signal may exist in **temporal structure**
- this signal is not accessible to static models (MLP)

This reinforces the conclusion that:

> MLP failure does not imply absence of signal, only absence of **static signal**
