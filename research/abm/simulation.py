"""
simulation.py

Research-grade ABM simulation engine for FX sentiment.

Features:
- Backward-compatible (rng or seed)
- Stable numerics (no NaNs / explosions)
- Nonlinear price impact: k * sign(Δs) * |Δs|^1.5
- Volatility clustering (GARCH-lite)
- Robust directional disagreement (filters weak signals)
- Strong regime switching (enforces structure)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FXSentimentSimulation:
    def __init__(self, agents, rng=None, seed=42, warmup_steps=50):
        """
        Parameters
        ----------
        agents : list
        rng : np.random.Generator (optional)
        seed : int
        warmup_steps : int
        """

        self.agents = agents

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.warmup_steps = int(warmup_steps)

    def run(self, price_series, steps=500):
        price_series = np.asarray(price_series, dtype=float)

        if price_series.size == 0:
            raise ValueError("price_series is empty")

        prices = [float(price_series[0])]
        returns = []
        sentiments = []

        for _ in range(int(steps)):
            price_history = np.asarray(prices, dtype=float)
            ret_array = np.asarray(returns, dtype=float)

            last_sentiment = sentiments[-1] if sentiments else 0.0

            # --- agent signals ---
            signals = []
            for agent in self.agents:
                try:
                    s = agent.update(price_history, ret_array, last_sentiment)
                except TypeError:
                    s = agent.update(price_history)

                if not np.isfinite(s):
                    s = 0.0

                signals.append(float(s))

            signals_arr = np.array(signals)
            net_sentiment = float(np.mean(signals_arr))

            # weak mean reversion
            net_sentiment -= 0.05 * net_sentiment

            # =========================================================
            # ROBUST DISAGREEMENT (FIX 1)
            # =========================================================
            threshold = 0.01
            active = np.abs(signals_arr) > threshold

            if np.any(active):
                signs = np.sign(signals_arr[active])
                agreement = abs(np.mean(signs))
                disagreement = 1.0 - agreement
            else:
                disagreement = 0.0

            # dispersion (only meaningful when disagreement exists)
            dispersion = float(np.std(signals_arr))

            # previous volatility
            if len(returns) > 0:
                prev_vol = abs(returns[-1])
            else:
                prev_vol = 0.002

            # base volatility
            vol = (
                0.001
                + 0.6 * prev_vol
                + 0.5 * dispersion * disagreement
                + 1.0 * disagreement
            )

            # =========================================================
            # STRONG REGIME SWITCH (FIX 2)
            # =========================================================
            if disagreement > 0.3:
                # unstable regime (true disagreement)
                vol *= 3.0
                impact_scale = 0.5
            else:
                # stable regime (trend / consensus)
                vol *= 0.5
                impact_scale = 1.5

            # bounds
            vol = max(vol, 0.0005)
            vol = min(vol, 0.01)

            noise = self.rng.normal(0.0, vol)

            # =========================================================
            # RETURNS FROM Δ SENTIMENT (already correct)
            # =========================================================
            if len(sentiments) > 0:
                delta_sentiment = net_sentiment - sentiments[-1]
            else:
                delta_sentiment = net_sentiment

            ret = impact_scale * 0.2 * np.sign(delta_sentiment) * np.abs(delta_sentiment) ** 1.5 + noise

            if not np.isfinite(ret):
                ret = 0.0

            new_price = prices[-1] * (1.0 + ret)

            if not np.isfinite(new_price) or new_price <= 0:
                new_price = prices[-1]

            prices.append(float(new_price))
            returns.append(float(ret))
            sentiments.append(float(net_sentiment))

        df = pd.DataFrame({
            "price": prices[1:],
            "returns": returns,
            "net_sentiment": sentiments,
        })

        if df.isna().any().any():
            raise RuntimeError("Simulation produced NaNs")

        return df