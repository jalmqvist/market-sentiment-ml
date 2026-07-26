#!/usr/bin/env bash
# Stage 2.2c + 2.3 — Fill transition gap and beta sweep at anchor=0.25

ARTIFACT="../bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
PAIR="usd-jpy"
STEPS=2000
SEED=42

echo "# Stage 2.2c — Fill anchor transition gap"
echo "# format: pair | seed | anchor | beta | score | n_ep | med_dur | rev_y | rev_m | surv_8"

for ANCHOR in 0.17 0.20; do
    python ../abm_experiments/reactive_jpy_episode_calibration.py \
        --pair $PAIR \
        --steps $STEPS \
        --seed $SEED \
        --anchor-strength $ANCHOR \
        --beta 0.0 \
        --calibration-artifact $ARTIFACT
done

echo ""
echo "# Stage 2.3 — Beta sweep at anchor=0.25"

for BETA in 0.00 0.01 0.02 0.05 0.10 0.20; do
    python ../abm_experiments/reactive_jpy_episode_calibration.py \
        --pair $PAIR \
        --steps $STEPS \
        --seed $SEED \
        --anchor-strength 0.25 \
        --beta $BETA \
        --calibration-artifact $ARTIFACT
done
