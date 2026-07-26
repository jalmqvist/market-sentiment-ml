#!/usr/bin/env bash
# =============================================================================
# run_stage3_shock_sweep.sh
#
# Sweeps the shock mechanism across trigger types, fractions, and thresholds.
# Runs all three JPY pairs at the calibrated parameter point.
#
# For each configuration, reports both H3 (episode structure) and H4
# (MATURING predictive gradient). The no-shock baseline is always run
# first so results are directly comparable.
#
# Usage:
#   bash scripts/run_stage3_shock_sweep.sh [--dry-run]
#
# Outputs (per configuration):
#   abm_experiments/results/stage3/shocks/<config_id>_<pair>.json
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/abm_experiments/sweep_with_shocks.py"
BSVE_CSV="$REPO_ROOT/data/output/1.6.1/master_research_dataset_reactive_jpy_v1_core.csv"
ARTIFACT="$REPO_ROOT/bsve/calibration_artifacts/reactive_jpy_calibration_v1.json"
OUT_DIR="$REPO_ROOT/abm_experiments/results/stage3/shocks"

# Calibrated parameters (locked 2026-07-25)
ANCHOR=0.25
BETA=0.02
STEPS=1500
RUNS=20
SEED=1
FORWARD=24

PAIRS=("usd-jpy" "eur-jpy" "gbp-jpy")

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Commands will be printed but not executed."
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
if [[ ! -f "$SCRIPT" ]];   then echo "[ERROR] Script not found: $SCRIPT" >&2; exit 1; fi
if [[ ! -f "$BSVE_CSV" ]]; then echo "[ERROR] BSVE CSV not found: $BSVE_CSV" >&2; exit 1; fi
ARTIFACT_FLAG=""
if [[ -f "$ARTIFACT" ]]; then ARTIFACT_FLAG="--calibration-artifact $ARTIFACT"; fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Configuration matrix
# Each entry: "config_id|trigger|vol_thresh|fraction|cooldown|period|enable"
# ---------------------------------------------------------------------------
CONFIGS=(
    # Baseline — no shocks (comparison anchor)
    "baseline|volatility|0.80|0.30|20|50|false"

    # Volatility trigger sweep
    "vol_t70_f30|volatility|0.70|0.30|20|50|true"
    "vol_t80_f30|volatility|0.80|0.30|20|50|true"
    "vol_t90_f30|volatility|0.90|0.30|20|50|true"
    "vol_t80_f20|volatility|0.80|0.20|20|50|true"
    "vol_t80_f50|volatility|0.80|0.50|20|50|true"
    "vol_t80_f30_cd10|volatility|0.80|0.30|10|50|true"

    # Periodic trigger sweep
    "per_p50_f30|periodic|0.80|0.30|20|50|true"
    "per_p25_f30|periodic|0.80|0.30|20|25|true"
    "per_p50_f50|periodic|0.80|0.50|20|50|true"
)

run_config() {
    local cfg_id="$1"
    local trigger="$2"
    local vol_thresh="$3"
    local fraction="$4"
    local cooldown="$5"
    local period="$6"
    local enable="$7"
    local pair="$8"

    local out_json="$OUT_DIR/${cfg_id}_${pair}.json"

    local CMD=(
        python3 "$SCRIPT"
        --pair             "$pair"
        --steps            "$STEPS"
        --seed             "$SEED"
        --runs             "$RUNS"
        --anchor-strength  "$ANCHOR"
        --beta             "$BETA"
        --forward-horizon  "$FORWARD"
        --bsve-states-path "$BSVE_CSV"
        --shock-trigger    "$trigger"
        --shock-vol-threshold "$vol_thresh"
        --shock-fraction   "$fraction"
        --shock-cooldown   "$cooldown"
        --shock-period     "$period"
        --output-json      "$out_json"
    )

    if [[ "$enable" == "true" ]]; then
        CMD+=(--shock-enable)
    fi
    if [[ -n "$ARTIFACT_FLAG" ]]; then
        CMD+=($ARTIFACT_FLAG)
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] ${CMD[*]}"
    else
        echo ""
        echo "  config=$cfg_id  pair=$pair  shock=$enable"
        "${CMD[@]}" 2>&1 | tee -a "$OUT_DIR/${cfg_id}_${pair}.log"
    fi
}

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
total=${#CONFIGS[@]}
n_pairs=${#PAIRS[@]}
echo ""
echo "========================================================"
echo "  Stage 3 Shock Sweep"
echo "  Configs: $total  |  Pairs: $n_pairs  |  Runs/config: $RUNS"
echo "  Steps: $STEPS  |  anchor=$ANCHOR  beta=$BETA"
echo "========================================================"

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg_id trigger vol_thresh fraction cooldown period enable <<< "$config"
    echo ""
    echo "--------------------------------------------------------"
    echo "  Config: $cfg_id  (shock=$enable  trigger=$trigger  "
    echo "          frac=$fraction  vol_thresh=$vol_thresh  cooldown=$cooldown)"
    echo "--------------------------------------------------------"
    for pair in "${PAIRS[@]}"; do
        run_config "$cfg_id" "$trigger" "$vol_thresh" \
                   "$fraction" "$cooldown" "$period" "$enable" "$pair"
    done
done

# ---------------------------------------------------------------------------
# Summary table across all configs and pairs
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 0 ]]; then
    export SHOCK_OUT_DIR="$OUT_DIR"
    python3 - <<'PYEOF'
import json, os, math
from pathlib import Path

out_dir = os.environ.get("SHOCK_OUT_DIR")
configs = [
    "baseline",
    "vol_t70_f30", "vol_t80_f30", "vol_t90_f30",
    "vol_t80_f20", "vol_t80_f50", "vol_t80_f30_cd10",
    "per_p50_f30", "per_p25_f30", "per_p50_f50",
]
pairs = ["usd-jpy", "eur-jpy", "gbp-jpy"]

print("\n" + "=" * 90)
print("  STAGE 3 SHOCK SWEEP SUMMARY")
print("=" * 90)
print(f"  {'Config':<20} {'Pair':<10} {'Shocks/run':>10}  "
      f"{'freq/1k':>8}  {'med_dur':>7}  {'rev_grad':>8}  "
      f"{'|r|MAT':>7}  {'|r|ENT':>7}  {'H4':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*10}  "
      f"{'-'*8}  {'-'*7}  {'-'*8}  "
      f"{'-'*7}  {'-'*7}  {'-'*8}")

for cfg_id in configs:
    for pair in pairs:
        fp = Path(out_dir) / f"{cfg_id}_{pair}.json"
        if not fp.exists():
            print(f"  {cfg_id:<20} {pair:<10} {'MISSING':>10}")
            continue
        with open(fp) as f:
            d = json.load(f)

        ep   = d.get("episode_metrics_mean", {})
        h4   = d.get("h4_verdict", {})
        n_sh = d.get("n_shocks_mean", 0.0)
        freq = ep.get("ep_freq_per_1000", math.nan)
        dur  = ep.get("median_duration_bars", math.nan)
        rg   = ep.get("rev_gradient_correct", None)
        am   = h4.get("abs_MATURING", math.nan)
        ae   = h4.get("abs_ENTRY",    math.nan)

        h4_str = ("FULL"    if h4.get("h4_supported") else
                  "PARTIAL" if h4.get("h4_partial_supported") else "NO")

        rg_str = ("✓" if rg is True else "✗" if rg is False else "?")

        freq_s = f"{freq:.1f}" if not math.isnan(freq) else "nan"
        dur_s  = f"{dur:.1f}"  if not math.isnan(dur)  else "nan"
        am_s   = f"{am:.4f}"   if not math.isnan(am)   else "nan"
        ae_s   = f"{ae:.4f}"   if not math.isnan(ae)   else "nan"

        print(f"  {cfg_id:<20} {pair:<10} {n_sh:>10.1f}  "
              f"{freq_s:>8}  {dur_s:>7}  {rg_str:>8}  "
              f"{am_s:>7}  {ae_s:>7}  {h4_str:>8}")

print("=" * 90 + "\n")

summary_path = Path(out_dir) / "shock_sweep_summary.json"
all_results = {}
for cfg_id in configs:
    all_results[cfg_id] = {}
    for pair in pairs:
        fp = Path(out_dir) / f"{cfg_id}_{pair}.json"
        if fp.exists():
            with open(fp) as f:
                all_results[cfg_id][pair] = json.load(f)
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"[summary] Written to {summary_path}")
PYEOF
fi

echo ""
echo "[done] Stage 3 shock sweep complete."
echo "       Results in: $OUT_DIR"
echo "       Summary:    $OUT_DIR/shock_sweep_summary.json"

