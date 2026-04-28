#!/bin/bash

VERSION=${1:-1.1.0}
EPOCHS=${2:-50}

echo "================================="
echo "DL EXPERIMENT - VERSION $VERSION"
echo "================================="

# -------------------------
# Price-only
# -------------------------
echo "Running price_only..."

python -m research.deep_learning.train \
  --dataset-version $VERSION \
  --feature-set price_only \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_price_only.csv

# -------------------------
# Price + sentiment
# -------------------------
echo "Running price_sentiment..."

python -m research.deep_learning.train \
  --dataset-version $VERSION \
  --feature-set price_sentiment \
  --epochs $EPOCHS

python -m research.deep_learning.evaluate \
  --predictions data/output/$VERSION/dl/predictions_price_sentiment.csv

echo "================================="
echo "DONE"
echo "================================="
