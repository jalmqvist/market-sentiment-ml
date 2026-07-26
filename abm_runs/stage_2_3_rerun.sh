#!/usr/bin/env bash
# Stage 2.3 rerun — corrected scoring, multi-seed beta sweep at anchor=0.25

ARTIFACT="../bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
STEPS=2000

echo "# Stage 2.3 rerun — corrected scorer, anchor=0.25"
echo "# format: pair | seed | anchor | beta | score | n_ep | med_dur | rev_y | rev_m | surv_8"

for BETA in 0.00 0.01 0.02 0.05 0.10 0.20; do
    for SEED in 1 2 3 4 5; do
        python ../abm_experiments/reactive_jpy_episode_calibration.py \
            --pair usd-jpy \
            --steps $STEPS \
            --seed $SEED \
            --anchor-strength 0.25 \
            --beta $BETA \
            --calibration-artifact $ARTIFACT
    done
done
