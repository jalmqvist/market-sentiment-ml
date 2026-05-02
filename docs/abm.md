# Agent-Based Model (ABM) — FX Retail Sentiment

## Purpose

The ABM simulates the collective positioning dynamics of retail FX traders to
generate synthetic sentiment data that can be calibrated against real broker
positioning data. The model tests whether observed sentiment patterns
(mean-reversion tendency, persistence, extreme clustering) can emerge from a
simple heterogeneous-agent interaction.

The ABM does **not** claim to predict returns. The calibration step explicitly
verifies that simulated sentiment does not correlate with future returns —
consistent with the empirical finding that retail sentiment is primarily a
reactive, not predictive, signal.

---

## How to Run

```bash
python research/abm/run_abm.py \
    --version 1.1.0 \
    --pair eur-usd \
    --steps 500 \
    --seed 42 \
    --n-trend 40 \
    --n-contrarian 40 \
    --n-noise 20 \
    --momentum-window 12 \
    --output logs/eur_usd_sim.csv
```

### Suppress file logging (stdout only)

```bash
python research/abm/run_abm.py \
    --version 1.1.0 \
    --pair eur-usd \
    --no-log-file
```

### Run tests

```bash
python -m pytest tests/test_abm.py -v
```

---

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--version` | *required* | Dataset version directory (e.g. `1.1.0`) |
| `--variant` | `core` | Dataset variant: `full`, `core`, or `extended` |
| `--pair` | *required* | FX pair slug (e.g. `eur-usd`, `usd-jpy`) |
| `--steps` | `500` | Number of simulation steps to record |
| `--seed` | `42` | RNG seed for reproducibility |
| `--n-trend` | `40` | Number of trend-following agents |
| `--n-contrarian` | `40` | Number of contrarian agents |
| `--n-noise` | `20` | Number of noise-trader agents |
| `--momentum-window` | `12` | Look-back window (bars) for price signal |
| `--output` | `None` | Path to save output CSV |
| `--log-level` | `INFO` | Logging verbosity |
| `--no-log-file` | off | Disable file logging; use stdout only |

---

## Agent Types

### TrendFollower
Follows recent price momentum. Goes long after sustained up-moves, short after
down-moves. Crowd sentiment adds a herding weight.

### Contrarian
Fades recent momentum. Goes short after up-moves, long after down-moves.
Reduces positions when crowd is already on one side.

### NoiseTrader
Random positioning. Provides baseline stochasticity and prevents artificial
lock-in.

---

## Outputs

### Log file
`logs/abm_{pair}_{timestamp}Z.log` — full run log with parameters,
summary statistics, and calibration comparison table.

### Config snapshot
`logs/abm_{pair}_{timestamp}Z.json` — machine-readable record of
every parameter used in the run.

### Output CSV (if `--output` provided)
Columns:

| Column | Description |
|---|---|
| `timestamp` | Bar timestamp from real dataset |
| `price` | Real entry close price |
| `net_sentiment` | Simulated net retail positioning (–100 to +100) |
| `real_net_sentiment` | Actual net retail positioning from dataset |

---

## Reproducibility

Every ABM run writes two files to `logs/`:

### Log file naming

```
logs/abm_{pair}_{timestamp}.log
```

Example: `logs/abm_eur-usd_20260502T120000Z.log`

### Config JSON

```
logs/abm_{pair}_{timestamp}.json
```

The JSON snapshot includes `experiment_type`, `dataset_path`, `dataset_version`,
`cli_command` (exact command used), and all hyperparameters.

### Re-running an experiment

Retrieve the exact command from the JSON snapshot:

```bash
cat logs/abm_eur-usd_20260502T120000Z.json | python -c "import json,sys; print(json.load(sys.stdin)['cli_command'])"
```

Then paste and run the printed command, for example:

```bash
python research/abm/run_abm.py --version 1.1.0 --pair eur-usd --steps 500 --seed 42
```



- **Mean close to 0**: population is balanced; no systematic directional bias.
- **High autocorr**: sentiment is persistent (sticky positions).
- **High extreme_freq**: agents cluster on one side frequently.
- **corr_contemporaneous ≈ 0**: sentiment does not move with price in the same bar.
- **corr_forward ≈ 0**: sentiment does not predict future returns (expected and validated).

Calibration compares simulated moments to real dataset moments. Small
`rel_diff` values indicate the parameter configuration reproduces empirical
behaviour.
