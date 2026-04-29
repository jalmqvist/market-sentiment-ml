#!/bin/bash

VERSION=${1:-1.2.0}
EPOCHS=${2:-50}

echo "================================="
echo "LSTM EXPERIMENT - VERSION $VERSION"
echo "================================="

echo "Running LSTM price_only..."
python -m research.deep_learning.train_lstm \
  --dataset-version $VERSION \
  --feature-set price_only \
  --seq-len 24 \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_lstm_price_only.csv

echo "Running LSTM price_sentiment..."
python -m research.deep_learning.train_lstm \
  --dataset-version $VERSION \
  --feature-set price_sentiment \
  --seq-len 24 \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_lstm_price_sentiment.csv

echo "================================="
echo "DONE"
echo "================================="
