#!/usr/bin/env bash
# =============================================================================
# Stage 0.2 — JPY Baseline Rerun on v1.6.1
# Verify sign-lock profile is stable vs May 2026 observations.
#
# Runs decay_beta_sensitivity.py for all three JPY pairs across:
#   - beta = 0.0  (baseline, decay disabled)
#   - beta = 0.10 (moderate decay, previously tested on v1.2.0)
#
# Fixed configuration (post-PR85 / USDJPY unlock regime):
#   n_trend=50, n_contrarian=50, n_noise=0, momentum_window=3
#   persistence=0.10, threshold=0.05
#
# Seeds 1-5 per (pair, beta) combination — matches May 2026 ensemble size.
#
# Output: printed to stdout. Redirect to a log file for diary entry.
# Usage:
#   bash stage_0_2_jpy_baseline.sh 2>&1 | tee logs/stage_0_2_jpy_baseline.log
# =============================================================================

set -euo pipefail

SCRIPT="../abm_experiments/decay_beta_sensitivity.py"
STEPS=2000
PAIRS=("usd-jpy" "eur-jpy" "gbp-jpy")
BETAS=(0.0 0.10)
SEEDS=(1 2 3 4 5)

# Header for easy parsing / copy-paste into diary
echo "# Stage 0.2 — JPY Baseline v1.6.1"
echo "# $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "# format: pair | seed | beta | pct_saturated | sign_flips | autocorr | mean | std | min | max | pct_abs_le_20 | pct_negative | mean_abs_pos | max_abs_pos | frac_pos_near_zero"
echo ""

for PAIR in "${PAIRS[@]}"; do
    echo "# --- ${PAIR} ---"
    for BETA in "${BETAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            python "${SCRIPT}" \
                --pair "${PAIR}" \
                --steps "${STEPS}" \
                --beta "${BETA}" \
                --seed "${SEED}" \
                --verbose
        done
    done
    echo ""
done

echo "# Stage 0.2 complete."
