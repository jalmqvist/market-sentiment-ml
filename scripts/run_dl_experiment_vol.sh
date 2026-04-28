#!/bin/bash

VERSION=${1:-1.1.0}
EPOCHS=${2:-50}

echo "================================="
echo "DL EXPERIMENT - VERSION $VERSION"
echo "================================="

echo "Running price_vol..."
python -m research.deep_learning.train \
  --dataset-version $VERSION \
  --feature-set price_vol \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_price_vol.csv

echo "Running price_vol_sentiment..."
python -m research.deep_learning.train \
  --dataset-version $VERSION \
  --feature-set price_vol_sentiment \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_price_vol_sentiment.csv

echo "================================="
echo "DONE"
echo "================================="
