#!/bin/bash
# scripts/run_dl_regime_experiment.sh
# ====================================
# Run regime-aware DL experiments (MLP + LSTM) for a given dataset version,
# market regime, and optional pair filter.
#
# Usage:
#   ./scripts/run_dl_regime_experiment.sh <version> <regime> <pairs> [epochs] [target_horizon] [label_q]
#
# Arguments:
#   version         Dataset version (e.g. 1.3.0)
#   regime          Market regime filter: HVTF | LVTF | HVR | LVR
#   pairs           Comma-separated pair list (e.g. EURUSD,GBPUSD,NZDUSD)
#   epochs          Number of training epochs (default: 50)
#   target_horizon  Prediction horizon in bars (default: 24)
#   label_q         Quantile threshold for label construction (default: 0.6)
#
# Example:
#   ./scripts/run_dl_regime_experiment.sh 1.3.0 HVTF EURUSD,GBPUSD,NZDUSD 50 24 0.6

set -e

VERSION=${1:?Usage: $0 <version> <regime> <pairs> [epochs] [target_horizon] [label_q]}
REGIME=${2:?Usage: $0 <version> <regime> <pairs> [epochs] [target_horizon] [label_q]}
PAIRS=${3:?Usage: $0 <version> <regime> <pairs> [epochs] [target_horizon] [label_q]}
EPOCHS=${4:-50}
TARGET_HORIZON=${5:-24}
LABEL_Q=${6:-0.6}

echo "=============================================="
echo "DL REGIME EXPERIMENT"
echo "  version        : $VERSION"
echo "  regime         : $REGIME"
echo "  pairs          : $PAIRS"
echo "  epochs         : $EPOCHS"
echo "  target_horizon : $TARGET_HORIZON"
echo "  label_q        : $LABEL_Q"
echo "=============================================="

# -------------------------
# MLP — price_trend
# -------------------------
echo ""
echo "--- MLP: price_trend ---"
python -m research.deep_learning.train \
  --dataset-version "$VERSION" \
  --feature-set price_trend \
  --pairs "$PAIRS" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --target-horizon "$TARGET_HORIZON" \
  --label-mode threshold \
  --label-quantile "$LABEL_Q"

# -------------------------
# MLP — price_trend_sentiment
# -------------------------
echo ""
echo "--- MLP: price_trend_sentiment ---"
python -m research.deep_learning.train \
  --dataset-version "$VERSION" \
  --feature-set price_trend_sentiment \
  --pairs "$PAIRS" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --target-horizon "$TARGET_HORIZON" \
  --label-mode threshold \
  --label-quantile "$LABEL_Q"

# -------------------------
# LSTM — price_trend
# -------------------------
echo ""
echo "--- LSTM: price_trend ---"
python -m research.deep_learning.train_lstm \
  --dataset-version "$VERSION" \
  --feature-set price_trend \
  --pairs "$PAIRS" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --target-horizon "$TARGET_HORIZON" \
  --label-mode threshold \
  --label-quantile "$LABEL_Q"

# -------------------------
# LSTM — price_trend_sentiment
# -------------------------
echo ""
echo "--- LSTM: price_trend_sentiment ---"
python -m research.deep_learning.train_lstm \
  --dataset-version "$VERSION" \
  --feature-set price_trend_sentiment \
  --pairs "$PAIRS" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --target-horizon "$TARGET_HORIZON" \
  --label-mode threshold \
  --label-quantile "$LABEL_Q"

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
