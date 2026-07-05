"""Baseline controls for behavioral experiment evaluation.

Controls allow researchers to estimate how much of any observed metric
difference is explained by partitioning, sample size, or temporal coverage
alone — before attributing effects to Behavioral Surface structure.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from analysis.behavioral.coverage import _coverage_row, _time_column


def _sample_matched(
    df: pd.DataFrame,
    target_size: int,
    target_time_min: pd.Timestamp | None,
    target_time_max: pd.Timestamp | None,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return a random sample from *df* matched for size and time range.

    When the target time window is available, the pool is first restricted to
    rows that fall inside it (to avoid temporal confounding). If fewer rows
    are available than *target_size*, the entire matching pool is returned.
    """
    pool = df.copy()

    if target_time_min is not None and target_time_max is not None:
        time_col = _time_column(pool)
        if time_col:
            ts = pd.to_datetime(pool[time_col], errors="coerce")
            mask = (ts >= target_time_min) & (ts <= target_time_max)
            constrained = pool.loc[mask]
            if len(constrained) >= 1:
                pool = constrained

    n = min(target_size, len(pool))
    return pool.sample(n=n, random_state=seed)


def generate_controls(
    df: pd.DataFrame,
    states: list[dict[str, str]],
    *,
    n_random: int = 3,
    seed: int = 42,
    regime_col: str = "regime",
) -> pd.DataFrame:
    """Generate standardized control partition summaries.

    Parameters
    ----------
    df:
        Full master research dataset (all rows).
    states:
        Discovered behavioral states, as returned by
        ``discover_behavioral_states``.
    n_random:
        Number of randomly sampled controls per state to generate.  Each
        random control is matched for sample size and temporal coverage to the
        corresponding behavioral state partition.
    seed:
        Base random seed.  Random control *i* for state *j* uses
        ``seed + i * 1000 + j``.
    regime_col:
        Name of the column containing regime labels, if present.  Used to
        generate a regime-partition control.

    Returns
    -------
    pd.DataFrame
        One row per control partition with the same coverage columns as
        ``build_coverage_table``, plus ``control_type`` and ``description``.
    """
    rows: list[dict[str, object]] = []

    # --- 1. Canonical full dataset ---
    full_row = _coverage_row("control:full_dataset", df)
    full_row["control_type"] = "full_dataset"
    full_row["description"] = "Complete canonical dataset (no partitioning)"
    full_row["coverage_fraction"] = 1.0
    rows.append(full_row)

    # --- 2. Behavioral partition (all behavioral rows combined) ---
    behavioral = df[df["surface_id"].notna() & df["state_id"].notna()].copy() \
        if {"surface_id", "state_id"}.issubset(df.columns) else pd.DataFrame()
    beh_row = _coverage_row("control:behavioral_partition", behavioral)
    beh_row["control_type"] = "behavioral_partition"
    beh_row["description"] = "All rows with a behavioral surface assignment"
    total = len(df)
    beh_count = int(beh_row["row_count"])
    beh_row["coverage_fraction"] = round(beh_count / total, 6) if total > 0 else None
    rows.append(beh_row)

    # --- 3. Regime partition (if regime column present) ---
    if regime_col in df.columns:
        for regime_val, regime_df in df.groupby(regime_col):
            r_row = _coverage_row(f"control:regime:{regime_val}", regime_df)
            r_row["control_type"] = "regime_partition"
            r_row["description"] = f"Regime partition: {regime_val}"
            r_count = int(r_row["row_count"])
            r_row["coverage_fraction"] = round(r_count / total, 6) if total > 0 else None
            rows.append(r_row)

    # --- 4. Random matched partitions (per-state) ---
    for state_idx, state in enumerate(states):
        state_df = df[
            (df["surface_id"] == state["surface_id"])
            & (df["state_id"] == state["state_id"])
        ].copy() if {"surface_id", "state_id"}.issubset(df.columns) else pd.DataFrame()

        target_size = len(state_df)

        # Time range of the state partition for matching
        time_col = _time_column(state_df) if len(state_df) > 0 else None
        if time_col and len(state_df) > 0:
            ts_series = pd.to_datetime(state_df[time_col], errors="coerce").dropna()
            t_min = ts_series.min() if len(ts_series) > 0 else None
            t_max = ts_series.max() if len(ts_series) > 0 else None
        else:
            t_min = t_max = None

        for i in range(n_random):
            sample_seed = seed + i * 1000 + state_idx
            sample = _sample_matched(df, target_size, t_min, t_max, seed=sample_seed)
            label = (
                f"control:random:{state['surface_id']}:{state['state_id']}:r{i}"
            )
            s_row = _coverage_row(label, sample)
            s_row["control_type"] = "random_matched"
            s_row["surface_id"] = state["surface_id"]
            s_row["state_id"] = state["state_id"]
            s_row["random_seed"] = sample_seed
            s_row["description"] = (
                f"Random sample matched to {state['state_id']} "
                f"(n={target_size}, seed={sample_seed})"
            )
            s_count = int(s_row["row_count"])
            s_row["coverage_fraction"] = round(s_count / total, 6) if total > 0 else None
            rows.append(s_row)

    table = pd.DataFrame(rows)
    for col in ["surface_id", "state_id", "coverage_fraction", "random_seed", "control_type", "description"]:
        if col not in table.columns:
            table[col] = None
    return table
