#!/usr/bin/env bash
# =============================================================================
# run_stage3_pooled_h4.sh
#
# Pooled cross-pair H4 test at the calibrated parameter point.
# Runs all three JPY pairs, concatenates aligned DataFrames, and computes
# a single pooled Spearman correlation per BSVE lifecycle state.
#
# Resolves the small-sample MATURE cell problem (n=54-89 per pair)
# by pooling to n=209+ MATURE rows across pairs and runs.
#
# Usage:
#   bash scripts/run_stage3_pooled_h4.sh [--dry-run]
#
# Output:
#   abm_experiments/results/stage3/pooled_h4_result.json
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/abm_experiments/regime_hierarchy_test.py"
BSVE_CSV="$REPO_ROOT/data/output/1.6.1/master_research_dataset_reactive_jpy_v1_core.csv"
ARTIFACT="$REPO_ROOT/bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
OUT_DIR="$REPO_ROOT/abm_experiments/results/stage3"
OUT_JSON="$OUT_DIR/pooled_h4_result.json"

# ---------------------------------------------------------------------------
# Calibrated parameters (locked 2026-07-25)
# ---------------------------------------------------------------------------
ANCHOR=0.25
BETA=0.02
STEPS=1500
RUNS=20          # increased from 5 — tighter confidence intervals
SEED=1
FORWARD=24

# ---------------------------------------------------------------------------
# Dry-run guard
# ---------------------------------------------------------------------------
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Command will be printed but not executed."
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Harness not found: $SCRIPT" >&2; exit 1
fi
if [[ ! -f "$BSVE_CSV" ]]; then
    echo "[ERROR] BSVE dataset not found: $BSVE_CSV" >&2; exit 1
fi

if [[ ! -f "$ARTIFACT" ]]; then
    echo "[WARN]  Calibration artifact not found: $ARTIFACT (continuing without)"
    ARTIFACT_FLAG=""
else
    ARTIFACT_FLAG="--calibration-artifact $ARTIFACT"
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
CMD=(
    python3 "$SCRIPT"
    --pool-pairs
    --pool-pair-list  "usd-jpy,eur-jpy,gbp-jpy"
    --steps           "$STEPS"
    --seed            "$SEED"
    --runs            "$RUNS"
    --anchor-strength "$ANCHOR"
    --beta            "$BETA"
    --forward-horizon "$FORWARD"
    --use-bsve-states
    --bsve-states-path "$BSVE_CSV"
    --output-json      "$OUT_JSON"
    --verbose
)

if [[ -n "${ARTIFACT_FLAG:-}" ]]; then
    CMD+=($ARTIFACT_FLAG)
fi

echo ""
echo "========================================================"
echo "  Stage 3 — Pooled H4 test"
echo "  Pairs: usd-jpy, eur-jpy, gbp-jpy"
echo "  Runs/pair: $RUNS  |  Seeds: $SEED to $((SEED + RUNS - 1))"
echo "  Steps: $STEPS  |  anchor=$ANCHOR  beta=$BETA"
echo "========================================================"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}"
else
    "${CMD[@]}" 2>&1 | tee "$OUT_DIR/pooled_h4_run.log"
fi

echo ""
echo "[done] Pooled H4 test complete."
echo "       JSON : $OUT_JSON"
echo "       Log  : $OUT_DIR/pooled_h4_run.log"
