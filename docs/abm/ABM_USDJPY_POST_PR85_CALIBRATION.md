# ABM USD-JPY post-PR85 calibration summary

## 1) Verification on default branch (main)
- `ABM_DISAGREE_HOLD_PROB` env knob: present in `research/abm/agents.py`.
- `ABM_REINFORCE_STRENGTH` env knob: present in `research/abm/agents.py`.
- Contrarian normalized-frame behavior: implemented via `self.signal_sign *= -1` in `Contrarian.__init__`.
- `research/abm/agents.py` has trailing newline.

## 2) Reproducible grid setup
Command baseline (used for all runs):
```bash
python research/abm/run_abm.py --version sample_abm_grid --variant core --pair usd-jpy   --n-trend 50 --n-contrarian 50 --n-noise 0 --momentum-window 3 --steps 2000 --seed 1   --no-log-file --output <csv>
```
Environment variables varied per run: `ABM_DISAGREE_HOLD_PROB`, `ABM_REINFORCE_STRENGTH`, and (follow-up) `ABM_ANCHOR_STRENGTH`.

## 3) Results
| case | hold_prob | reinforce_strength | anchor_strength | mean | std | pct_abs_ge_90 | pct_abs_lt_20 | sign_flips | long_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold0.7_r3.0_a2.0 | 0.70 | 3.00 | 2.00 | 99.973 | 0.231 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.7_r1.0_a2.0 | 0.70 | 1.00 | 2.00 | 99.973 | 0.231 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.5_r3.0_a2.0 | 0.50 | 3.00 | 2.00 | 99.929 | 0.370 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.5_r1.0_a2.0 | 0.50 | 1.00 | 2.00 | 99.929 | 0.370 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.3_r3.0_a2.0 | 0.30 | 3.00 | 2.00 | 99.987 | 0.161 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.3_r1.0_a2.0 | 0.30 | 1.00 | 2.00 | 99.987 | 0.161 | 100.000 | 0.000 | 0 | 1.000 |
| hold0.5_r3.0_a1.00 | 0.50 | 3.00 | 1.00 | 98.266 | 2.125 | 99.350 | 0.000 | 0 | 1.000 |
| hold0.5_r3.0_a0.50 | 0.50 | 3.00 | 0.50 | 82.572 | 10.240 | 29.350 | 0.000 | 0 | 1.000 |
| hold0.5_r3.0_a0.25 | 0.50 | 3.00 | 0.25 | 56.388 | 14.973 | 0.000 | 1.650 | 0 | 1.000 |
| hold0.5_r3.0_a0.00 | 0.50 | 3.00 | 0.00 | 22.579 | 11.122 | 0.000 | 39.400 | 24 | 0.989 |

## 4) Recommendation
- With default anchoring (`ABM_ANCHOR_STRENGTH=2.0`), hold/reinforcement sweeps stay locked near +100 (no sign flips).
- `ABM_REINFORCE_STRENGTH` has negligible effect in this locked regime.
- Lowering anchor unlocks dynamics while retaining persistence. A practical starting point is:
  - `ABM_DISAGREE_HOLD_PROB=0.5`
  - `ABM_REINFORCE_STRENGTH=3.0`
  - `ABM_ANCHOR_STRENGTH=0.25` (near-persistent, non-saturated)
- For stronger regime transitions, test `ABM_ANCHOR_STRENGTH=0.0` (introduces sign flips).
