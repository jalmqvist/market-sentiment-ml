#!/usr/bin/env bash
# =============================================================================
# run_stage3_bsve_injection.sh
#
# Runs the Stage 3 BSVE state-injection harness across all three JPY pairs
# at the calibrated parameter point (anchor=0.25, beta=0.02).
#
# Usage:
#   bash scripts/run_stage3_bsve_injection.sh [--dry-run]
#
# Outputs:
#   abm_experiments/results/stage3/<pair>_h4_result.json  (per-pair)
#   abm_experiments/results/stage3/stage3_summary.json    (combined)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — all relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/abm_experiments/regime_hierarchy_test.py"
BSVE_CSV="$REPO_ROOT/data/output/1.6.1/master_research_dataset_reactive_jpy_v1_core.csv"
ARTIFACT="$REPO_ROOT/bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
OUT_DIR="$REPO_ROOT/abm_experiments/results/stage3"

# ---------------------------------------------------------------------------
# Calibrated parameters (locked 2026-07-25)
# ---------------------------------------------------------------------------
ANCHOR=0.25
BETA=0.02
STEPS=1500
RUNS=5
SEED=1           # seeds 1-5 used (seed + run_idx, 0-indexed => 1,2,3,4,5)
FORWARD=24       # ret_24b

PAIRS=("usd-jpy" "eur-jpy" "gbp-jpy")

# ---------------------------------------------------------------------------
# Dry-run guard
# ---------------------------------------------------------------------------
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Commands will be printed but not executed."
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Harness not found: $SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$BSVE_CSV" ]]; then
    echo "[ERROR] BSVE dataset not found: $BSVE_CSV" >&2
    exit 1
fi

if [[ ! -f "$ARTIFACT" ]]; then
    echo "[WARN]  Calibration artifact not found: $ARTIFACT (continuing without)"
    ARTIFACT_FLAG=""
else
    ARTIFACT_FLAG="--calibration-artifact $ARTIFACT"
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Per-pair runs
# ---------------------------------------------------------------------------
SUMMARY_PARTS=()

for PAIR in "${PAIRS[@]}"; do
    OUT_JSON="$OUT_DIR/${PAIR}_h4_result.json"
    SUMMARY_PARTS+=("$OUT_JSON")

    CMD=(
        python "$SCRIPT"
        --pair            "$PAIR"
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

    # Append artifact flag only if the file exists
    if [[ -n "${ARTIFACT_FLAG:-}" ]]; then
        CMD+=($ARTIFACT_FLAG)
    fi

    echo ""
    echo "========================================================"
    echo "  Pair: $PAIR"
    echo "========================================================"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] ${CMD[*]}"
    else
        "${CMD[@]}"
    fi
done

# ---------------------------------------------------------------------------
# Combine per-pair JSON results into a single summary file
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 0 ]]; then
    export STAGE3_OUT_DIR="$OUT_DIR"   # <-- must be exported for the heredoc subprocess
    python3 - <<'PYEOF'
import json, os
from pathlib import Path

out_dir = os.environ.get("STAGE3_OUT_DIR")
if not out_dir:
    raise RuntimeError("STAGE3_OUT_DIR not set")

pairs   = ["usd-jpy", "eur-jpy", "gbp-jpy"]
summary = {}

for pair in pairs:
    fp = Path(out_dir) / f"{pair}_h4_result.json"
    if fp.exists():
        with open(fp) as f:
            summary[pair] = json.load(f)
    else:
        summary[pair] = {"error": f"result file not found: {fp}"}

print("\n" + "=" * 70)
print("  STAGE 3 SUMMARY — H4 Hypothesis Results")
print("=" * 70)
print(f"  {'Pair':<12} {'H4 Full':>10}  {'H4 Partial':>12}  "
      f"{'|r| MATURING':>13}  {'|r| ENTRY':>10}  {'|r| MATURE':>11}")
print(f"  {'-'*12} {'-'*10}  {'-'*12}  {'-'*13}  {'-'*10}  {'-'*11}")

for pair in pairs:
    d = summary.get(pair, {})
    v = d.get("h4_verdict", {})
    h4  = str(v.get("h4_supported",         "N/A"))
    h4p = str(v.get("h4_partial_supported", "N/A"))
    am  = v.get("abs_MATURING", float("nan"))
    ae  = v.get("abs_ENTRY",    float("nan"))
    at  = v.get("abs_MATURE",   float("nan"))
    print(f"  {pair:<12} {h4:>10}  {h4p:>12}  "
          f"{float(am):>13.4f}  {float(ae):>10.4f}  {float(at):>11.4f}")

print("=" * 70 + "\n")

summary_path = Path(out_dir) / "stage3_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[summary] Written to {summary_path}")
PYEOF
fi
echo "[done] Stage 3 complete. Results in: $OUT_DIR"
