#!/usr/bin/env bash

set -e

VERSION=$1
REGIME=$2
PAIRS=$3
EPOCHS=$4

HORIZONS=("12" "24")
QUANTILES=("0.50" "0.55")

IFS=',' read -ra PAIR_LIST <<< "$PAIRS"

echo "=============================================="
echo "DL CARTOGRAPHY RUN"
echo "version : $VERSION"
echo "regime  : $REGIME"
echo "pairs   : ${PAIR_LIST[*]}"
echo "=============================================="

for PAIR in "${PAIR_LIST[@]}"; do

  echo ""
  echo "##############################################"
  echo "PAIR: $PAIR"
  echo "##############################################"

  for H in "${HORIZONS[@]}"; do
    for Q in "${QUANTILES[@]}"; do

      echo ""
      echo "RUN | $PAIR | H=$H | Q=$Q"

      # ---------- MLP ----------
      python -m research.deep_learning.train \
        --dataset-version "$VERSION" \
        --feature-set price_trend_sentiment \
        --epochs "$EPOCHS" \
        --pairs "$PAIR" \
        --regime "$REGIME" \
        --target-horizon "$H" \
        --label-quantile "$Q" \
      || echo "⚠️ MLP failed for $PAIR (skipping)"

      # ---------- LSTM ----------
      python -m research.deep_learning.train_lstm \
        --dataset-version "$VERSION" \
        --feature-set price_trend_sentiment \
        --epochs "$EPOCHS" \
        --pairs "$PAIR" \
        --regime "$REGIME" \
        --target-horizon "$H" \
        --label-quantile "$Q" \
      || echo "⚠️ LSTM failed for $PAIR (skipping)"

    done
  done
done

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
