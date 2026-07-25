from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow repo-root imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.abm.run_abm import _build_agents, _load_real_data
from research.abm.simulation import _AGGREGATION_EPS  # reuse exact contract
from research.abm.simulation import FXSentimentSimulation


_VOL_WINDOW = 24  # must match research/abm/simulation.py


def _aggregate_from_agents(agents) -> float:
    """Return dataset-scale net_sentiment in [-100, 100] using the same contract."""
    positions = np.array([a.position for a in agents], dtype=np.float64)
    votes = np.zeros_like(positions)
    votes[positions > _AGGREGATION_EPS] = 1.0
    votes[positions < -_AGGREGATION_EPS] = -1.0
    return float(votes.mean() * 100.0)


def main() -> None:
    version = "1.6.1"
    pair = "usd-jpy"
    variant = "core"

    seed = 1
    steps_to_record = 50
    momentum_window = 3

    # Small agent counts to make debugging tractable
    n_trend = 1
    n_contrarian = 1
    n_noise = 0

    # Load data
    df, _dataset_path = _load_real_data(version, variant=variant)
    sub = df[df["pair"] == pair].copy().sort_values("entry_time")
    if sub.empty:
        raise ValueError(f"No data found for pair={pair}")

    price_series = sub["entry_close"].to_numpy(dtype=float)
    timestamps = sub["entry_time"].values
    real_sentiment = sub["net_sentiment"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    agents = _build_agents(
        rng,
        pair=pair,
        n_trend=n_trend,
        n_contrarian=n_contrarian,
        n_noise=n_noise,
        momentum_window=momentum_window,
    )

    # Use sim only for warmup_steps value to stay consistent with repo defaults
    sim = FXSentimentSimulation(agents, rng=rng)
    warmup = int(sim.warmup_steps)

    total_required = warmup + steps_to_record + 1
    if len(price_series) < total_required:
        raise ValueError(f"price_series too short: need {total_required}, got {len(price_series)}")

    # Vol proxy (same as simulation.py)
    ema_alpha = 2.0 / (_VOL_WINDOW + 1.0)
    vol_ema = 0.0
    baseline_vol = 0.0
    baseline_alpha = ema_alpha
    prev_price = float(price_series[0])

    price_history = [float(price_series[0])]
    records = []

    print("Agent classes:", [type(a).__name__ for a in agents])
    print("Initial positions:", [a.position for a in agents])
    print("warmup_steps:", warmup)
    print()

    for t in range(1, total_required):
        price = float(price_series[t])
        price_history.append(price)
        ph = np.array(price_history, dtype=np.float64)

        # vol_norm update
        ret_t = price - prev_price
        prev_price = price

        vol_ema = ema_alpha * abs(ret_t) + (1.0 - ema_alpha) * vol_ema
        baseline_vol = baseline_alpha * vol_ema + (1.0 - baseline_alpha) * baseline_vol
        vol_norm = vol_ema / (baseline_vol + 1e-8)

        # crowd sentiment BEFORE update (normalized)
        crowd_norm = _aggregate_from_agents(agents) / 100.0

        # Per-agent debug: raw + normalized signal and position delta
        # (raw_signal requires calling _price_signal; that's the key diagnostic here)
        print(f"t={t:03d} price={price:.5f} crowd_norm={crowd_norm:+.3f} vol_norm={vol_norm:.3f}")
        for i, a in enumerate(agents):
            pos_before = float(a.position)
            raw = float(a._price_signal(ph))  # noqa: SLF001 - debug trace
            norm = float(a.signal_sign) * raw
            a.update(ph, crowd_norm, volatility=vol_norm)
            pos_after = float(a.position)
            print(
                f"  agent{i} {type(a).__name__:12s} "
                f"pos {pos_before:+.3f} -> {pos_after:+.3f} | "
                f"raw_signal={raw:+.1f} norm_signal={norm:+.1f} signal_sign={a.signal_sign:+.0f}"
            )

        # Record only after warmup
        if t > warmup:
            idx = t - warmup - 1
            net_sent = _aggregate_from_agents(agents)

            row = {
                "step": idx,
                "timestamp": timestamps[t],
                "price": price,
                "net_sentiment": net_sent,
                "real_net_sentiment": real_sentiment[warmup + 1 + idx],
                "pos0": float(agents[0].position) if len(agents) > 0 else np.nan,
                "pos1": float(agents[1].position) if len(agents) > 1 else np.nan,
            }
            records.append(row)

        print()

    out = pd.DataFrame(records)
    print("Recorded head:")
    print(out.head(15).to_string(index=False))
    print("\nRecorded tail:")
    print(out.tail(15).to_string(index=False))
    print("\nFinal positions:", [a.position for a in agents])


if __name__ == "__main__":
    main()