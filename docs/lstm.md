# LSTM Experiments — Summary

## Overview

Long Short-Term Memory (LSTM) sequence models were trained to capture temporal
structure in retail FX sentiment and predict forward returns. No temporal
signal was found.

---

## Sequence Modeling Approach

### Architecture
- Single-layer LSTM with hidden size 64 and dropout 0.2.
- Linear output head for binary direction prediction.
- Trained with Adam optimizer and cross-entropy loss.

### Sequence construction
- Input window: 24–48 bars of feature history.
- Stride: 1 bar (overlapping sequences).
- Target: sign of `ret_48b` at the last bar of the window.

---

## Normalization Rules

| Feature | Normalization |
|---|---|
| `net_sentiment` | Divide by 100 (maps to [–1, +1]) |
| `abs_sentiment` | Divide by 100 |
| `ret_1b`, `ret_48b` | Z-score within training fold only |
| `extreme_streak` | Clip at 10, divide by 10 |
| `acceleration` | Z-score within training fold only |

**Critical**: normalization statistics are computed on training data only and
applied to validation/test sets. No leakage from future bars.

---

## Walk-Forward Protocol

- Expanding training window, fixed-size validation and test windows.
- Models retrained at each fold boundary.
- No data from test period used in preprocessing or normalization.

---

## Why No Temporal Signal Was Found

1. **Retail sentiment lacks autocorrelation at predictive lags.** The LSTM
   cannot exploit temporal structure that does not exist in the signal.

2. **Short effective sequences.** After quality filtering and deduplication,
   many pairs have fewer than 1 000 usable hourly bars — insufficient to train
   a sequence model reliably.

3. **Sentiment dynamics are stationary.** There is no persistent momentum in
   retail positioning beyond 1–2 bars, which the LSTM cannot exploit across
   24–48 bar windows.

4. **Consistent with MLP results.** The absence of signal in static features
   (MLP) implies the absence of signal in sequences of those features (LSTM).

---

## Reproducibility

Every LSTM training run writes two files to `logs/`:

### Log file naming

```
logs/lstm_{tag}_{timestamp}.log
```

Example: `logs/lstm_sequence-v1_20260502T123000Z.log`

The tag defaults to the `--feature-set` value and can be overridden with `--tag`.

### Config JSON

```
logs/lstm_{tag}_{timestamp}.json
```

The JSON snapshot includes `experiment_type`, `dataset_path`, `dataset_version`,
`cli_command` (exact command used), and all hyperparameters.

### Re-running an experiment

Retrieve the exact command from the JSON snapshot:

```bash
cat logs/lstm_sequence-v1_20260502T123000Z.json | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```

Then paste and run the printed command, for example:

```bash
python research/deep_learning/train_lstm.py --dataset-version 1.1.0 --feature-set price_only --seq-len 24 --epochs 50
```

---

## Conclusion

LSTM models do not find predictive temporal structure in retail FX sentiment
sequences. The negative result is robust across window lengths (24, 48 bars),
hidden sizes (32, 64, 128), and FX pair subsets. No further LSTM experiments
are planned unless new features with demonstrated autocorrelation are
identified.
