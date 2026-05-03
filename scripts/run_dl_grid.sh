#!/usr/bin/env bash

set -e

VERSION=$1
REGIME=$2
PAIRS=$3
EPOCHS=${4:-50}

if [ -z "$VERSION" ] || [ -z "$REGIME" ] || [ -z "$PAIRS" ]; then
  echo "Usage: run_dl_grid.sh <version> <regime> <pairs_csv> [epochs]"
  exit 1
fi

echo "=============================================="
echo "DL GRID SEARCH (CARTOGRAPHY)"
echo "  version : $VERSION"
echo "  regime  : $REGIME"
echo "  pairs   : $PAIRS"
echo "  epochs  : $EPOCHS"
echo "=============================================="

FEATURE_SETS=("price_trend" "price_trend_sentiment")
HORIZONS=(12 24 48)
QUANTILES=(0.50 0.55 0.60)

for FS in "${FEATURE_SETS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for Q in "${QUANTILES[@]}"; do

      echo ""
      echo "=============================================="
      echo "GRID RUN | FS=$FS | H=$H | Q=$Q"
      echo "=============================================="

      echo ">>> MLP"
      python -m research.deep_learning.train \
        --dataset-version "$VERSION" \
        --feature-set "$FS" \
        --epochs "$EPOCHS" \
        --pairs "$PAIRS" \
        --regime "$REGIME" \
        --target-horizon "$H" \
        --label-quantile "$Q"

      echo ">>> LSTM"
      python -m research.deep_learning.train_lstm \
        --dataset-version "$VERSION" \
        --feature-set "$FS" \
        --epochs "$EPOCHS" \
        --pairs "$PAIRS" \
        --regime "$REGIME" \
        --target-horizon "$H" \
        --label-quantile "$Q"

    done
  done
done

echo ""
echo "=============================================="
echo "GRID COMPLETE"
echo "=============================================="
