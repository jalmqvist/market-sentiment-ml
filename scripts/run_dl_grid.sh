#!/usr/bin/env bash

set -e

VERSION=$1
REGIME=$2
PAIRS=$3
EPOCHS=$4

echo "=============================================="
echo "DL GRID SEARCH (CLEAN)"
echo "  version : $VERSION"
echo "  regime  : $REGIME"
echo "  pairs   : $PAIRS"
echo "  epochs  : $EPOCHS"
echo "=============================================="

FEATURE_SETS=("price_trend" "price_trend_sentiment")
HORIZONS=(12 24 48)

for FS in "${FEATURE_SETS[@]}"; do
  for H in "${HORIZONS[@]}"; do

    echo ""
    echo "=============================================="
    echo "GRID RUN | FS=$FS | H=$H"
    echo "=============================================="

    echo ""
    echo ">>> MLP | $FS | H=$H"
    python -m research.deep_learning.train \
      --dataset-version $VERSION \
      --feature-set $FS \
      --epochs $EPOCHS \
      --pairs $PAIRS \
      --regime $REGIME \
      --target-horizon $H

    echo ""
    echo ">>> LSTM | $FS | H=$H"
    python -m research.deep_learning.train_lstm \
      --dataset-version $VERSION \
      --feature-set $FS \
      --epochs $EPOCHS \
      --pairs $PAIRS \
      --regime $REGIME \
      --target-horizon $H \
      --seq-len 24

  done
done

echo ""
echo "=============================================="
echo "GRID COMPLETE"
echo "=============================================="
