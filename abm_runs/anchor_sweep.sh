#!/bin/bash
# Stage 2.2 — Anchor sweep to unlock JPY dynamics
# Run this after placing the script above in abm_experiments/

ARTIFACT="../bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
PAIR="usd-jpy"
STEPS=2000
SEED=42

echo "# Stage 2.2 — Anchor sweep: $PAIR"
echo "# format: pair | seed | anchor | beta | score | n_ep | med_dur | rev_y | rev_m | surv_8"

for ANCHOR in 0.00 0.05 0.10 0.15 0.25; do
    python ../abm_experiments/reactive_jpy_episode_calibration.py \
        --pair $PAIR \
        --steps $STEPS \
        --seed $SEED \
        --anchor-strength $ANCHOR \
        --beta 0.0 \
        --calibration-artifact $ARTIFACT
done
