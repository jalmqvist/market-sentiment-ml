#!/usr/bin/env bash
# Stage 2.2b — Seed stability at anchor=0.25
# Confirms the unlocked dynamics are reproducible, not seed-lucky.

ARTIFACT="../bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
STEPS=2000

echo "# Stage 2.2b — Seed stability at anchor=0.25, beta=0.0"
echo "# format: pair | seed | anchor | beta | score | n_ep | med_dur | rev_y | rev_m | surv_8"

for PAIR in usd-jpy eur-jpy gbp-jpy; do
    for SEED in 1 2 3 4 5; do
        python ../abm_experiments/reactive_jpy_episode_calibration.py \
            --pair $PAIR \
            --steps $STEPS \
            --seed $SEED \
            --anchor-strength 0.25 \
            --beta 0.0 \
            --calibration-artifact $ARTIFACT
    done
done
