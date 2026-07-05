from __future__ import annotations

import pandas as pd


def _time_column(df: pd.DataFrame) -> str | None:
    for col in ["timestamp", "snapshot_time", "entry_time", "time"]:
        if col in df.columns:
            return col
    return None


def _coverage_row(scope: str, df: pd.DataFrame) -> dict[str, object]:
    time_col = _time_column(df)
    row: dict[str, object] = {
        "scope": scope,
        "row_count": int(len(df)),
        "pair_count": int(df["pair"].nunique()) if "pair" in df.columns else 0,
    }
    if time_col and len(df) > 0:
        row["timestamp_col"] = time_col
        row["timestamp_min"] = pd.to_datetime(df[time_col]).min()
        row["timestamp_max"] = pd.to_datetime(df[time_col]).max()
        row["timestamp_unique"] = int(pd.to_datetime(df[time_col]).nunique())
    else:
        row["timestamp_col"] = time_col
        row["timestamp_min"] = pd.NaT
        row["timestamp_max"] = pd.NaT
        row["timestamp_unique"] = 0
    return row


def build_coverage_table(df: pd.DataFrame, states: list[dict[str, str]]) -> pd.DataFrame:
    rows = [_coverage_row("full_dataset", df)]

    behavioral = df[df["surface_id"].notna() & df["state_id"].notna()].copy()
    rows.append(_coverage_row("behavioral_coverage", behavioral))

    for state in states:
        state_df = df[
            (df["surface_id"] == state["surface_id"]) & (df["state_id"] == state["state_id"])
        ].copy()
        rows.append(
            {
                **_coverage_row(
                    f"state:{state['surface_id']}:{state['state_id']}",
                    state_df,
                ),
                "surface_id": state["surface_id"],
                "state_id": state["state_id"],
            }
        )

    table = pd.DataFrame(rows)
    for col in ["surface_id", "state_id"]:
        if col not in table.columns:
            table[col] = None

    # Add percentage-based coverage metrics
    total_rows = int(table.loc[table["scope"] == "full_dataset", "row_count"].iloc[0]) if not table.empty else 0
    behavioral_rows = int(table.loc[table["scope"] == "behavioral_coverage", "row_count"].iloc[0]) if not table.empty else 0

    fractions: list[float | None] = []
    state_fractions: list[float | None] = []
    for _, row in table.iterrows():
        scope = row["scope"]
        count = int(row["row_count"])
        if scope == "full_dataset":
            fractions.append(1.0)
            state_fractions.append(None)
        elif scope == "behavioral_coverage":
            fractions.append(count / total_rows if total_rows > 0 else None)
            state_fractions.append(None)
        else:
            fractions.append(count / total_rows if total_rows > 0 else None)
            state_fractions.append(count / behavioral_rows if behavioral_rows > 0 else None)

    table["coverage_fraction"] = fractions
    table["state_fraction_of_behavioral"] = state_fractions
    return table
