#!/usr/bin/env bash

set -e

VERSION=$1
REGIME=$2
PAIRS=$3
EPOCHS=${4:-50}
HORIZON=${5:-24}
LABEL_Q=${6:-0.6}

echo "=============================================="
echo "DL REGIME EXPERIMENT"
echo "  version        : $VERSION"
echo "  regime         : $REGIME"
echo "  pairs          : $PAIRS"
echo "  epochs         : $EPOCHS"
echo "  target_horizon : $HORIZON"
echo "  label_q        : $LABEL_Q"
echo "=============================================="

run_mlp () {
  FEATURE_SET=$1

  echo ""
  echo "--- MLP: $FEATURE_SET ---"

  python -m research.deep_learning.train \
    --dataset-version "$VERSION" \
    --feature-set "$FEATURE_SET" \
    --epochs "$EPOCHS" \
    --pairs "$PAIRS" \
    --regime "$REGIME" \
    --target-horizon "$HORIZON" \
    --label-quantile "$LABEL_Q"
}

run_lstm () {
  FEATURE_SET=$1

  echo ""
  echo "--- LSTM: $FEATURE_SET ---"

  python -m research.deep_learning.train_lstm \
    --dataset-version "$VERSION" \
    --feature-set "$FEATURE_SET" \
    --epochs "$EPOCHS" \
    --pairs "$PAIRS" \
    --regime "$REGIME" \
    --target-horizon "$HORIZON" \
    --label-quantile "$LABEL_Q"
}

# -------------------------
# RUNS
# -------------------------

run_mlp "price_trend"
run_mlp "price_trend_sentiment"

run_lstm "price_trend"
run_lstm "price_trend_sentiment"

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
