from __future__ import annotations

import numpy as np
import pandas as pd

import research.abm.agents as _agents_mod


_EMA_SCALE = 10.0
_VOL_TRIGGER_BASE = 0.2
_VOL_TRIGGER_SCALE = 3.0


class FXSentimentSimulation:
    def __init__(
        self,
        agents,
        rng: np.random.Generator | None = None,
        seed: int = 42,
        warmup_steps: int = 50,
    ) -> None:
        if not agents:
            raise ValueError("agents list must not be empty")

        self.agents = agents

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.warmup_steps = int(warmup_steps)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def run(
        self,
        n_steps: int,
        price_series,
        timestamps=None,
    ) -> pd.DataFrame:

        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")

        price_series = np.asarray(price_series, dtype=float)

        min_len = self.warmup_steps + n_steps
        if len(price_series) < min_len:
            raise ValueError(
                f"price_series too short: need {min_len}, got {len(price_series)}"
            )

        n_agents = self.n_agents

        # ------------------------------------------------------------
        # Warmup
        # ------------------------------------------------------------
        for t in range(self.warmup_steps):
            ph = price_series[: t + 1]
            for agent in self.agents:
                agent.update(ph, crowd_sentiment=0.0, volatility=0.002)

        # ------------------------------------------------------------
        # Regime state
        # ------------------------------------------------------------
        regime_state = "neutral"
        smooth_disagree = 0.0
        smooth_align = 0.0

        # IMPORTANT: previous-step state (causality fix)
        crowd_s_prev = 0.0
        vol_prev = 0.002

        records: list[dict] = []

        # ------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------
        for t in range(n_steps):
            ph = price_series[: self.warmup_steps + t + 1]
            current_price = float(price_series[self.warmup_steps + t])

            # --------------------------------------------------------
            # STEP 1: update agents using PREVIOUS state
            # --------------------------------------------------------
            for agent in self.agents:
                agent.update(ph, crowd_sentiment=crowd_s_prev, volatility=vol_prev)

            # --------------------------------------------------------
            # STEP 2: compute crowd state AFTER update
            # --------------------------------------------------------
            n_long = sum(1 for a in self.agents if a.position == 1)
            n_short = sum(1 for a in self.agents if a.position == -1)
            n_flat = n_agents - n_long - n_short

            net_sentiment = float((n_long - n_short) / n_agents * 100.0)
            crowd_s = net_sentiment / 100.0

            # --------------------------------------------------------
            # STEP 3: disagreement + alignment
            # --------------------------------------------------------
            active = n_long + n_short
            if active > 0:
                agreement = abs(n_long - n_short) / active
                disagreement = 1.0 - agreement
            else:
                disagreement = 0.0

            persistence_weight = float(_agents_mod._PERSISTENCE_WEIGHT)
            inertia_thresh = float(_agents_mod._INERTIA_THRESHOLD)

            smooth_alpha = 1.0 / (1.0 + _EMA_SCALE * persistence_weight)

            smooth_disagree = (
                (1.0 - smooth_alpha) * smooth_disagree + smooth_alpha * disagreement
            )
            smooth_align = (
                (1.0 - smooth_alpha) * smooth_align + smooth_alpha * abs(crowd_s)
            )

            # --------------------------------------------------------
            # STEP 4: regime transition
            # --------------------------------------------------------
            vol_trigger = _VOL_TRIGGER_BASE + _VOL_TRIGGER_SCALE * inertia_thresh

            # improved trend trigger (less hardcoded)
            trend_trigger = 0.3 + 0.5 * inertia_thresh

            if smooth_disagree > vol_trigger:
                regime_state = "volatile"
            elif smooth_align > trend_trigger:
                regime_state = "trend"
            else:
                regime_state = "neutral"

            # --------------------------------------------------------
            # STEP 5: regime volatility
            # --------------------------------------------------------
            if regime_state == "trend":
                vol = 0.001
            elif regime_state == "volatile":
                vol = 0.008
            else:
                vol = 0.003

            # --------------------------------------------------------
            # STEP 6: record
            # --------------------------------------------------------
            records.append(
                {
                    "step": t,
                    "price": current_price,
                    "net_sentiment": net_sentiment,
                    "abs_sentiment": abs(net_sentiment),
                    "crowd_side": int(np.sign(net_sentiment)),
                    "n_long": n_long,
                    "n_short": n_short,
                    "n_flat": n_flat,
                }
            )

            # --------------------------------------------------------
            # STEP 7: store for next step (causality chain)
            # --------------------------------------------------------
            crowd_s_prev = crowd_s
            vol_prev = vol

        df = pd.DataFrame(records)

        if timestamps is not None:
            ts_arr = np.asarray(timestamps)
            df["timestamp"] = ts_arr[
                self.warmup_steps: self.warmup_steps + n_steps
            ]

        return df