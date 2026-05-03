#!/bin/sh
for pair in EURUSD GBPUSD NZDUSD USDJPY EURJPY USDCHF USDCAD AUDUSD GBPJPY EURGBP; do
  for regime in HVTF LVTF HVR LVR; do
    python -m research.deep_learning.train \
      --dataset-version 1.3.2 \
      --feature-set price_trend \
      --epochs 50 \
      --pairs $pair \
      --regime $regime \
      --target-horizon 24 \
      --label-quantile 0.50
  done
done
