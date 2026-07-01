#!/usr/bin/env python3
"""
Extract frozen validation windows from the master research dataset.

This utility partitions the master research dataset into the fixed windows
defined by the BSVE validation protocol.

It performs no ontology-specific processing and may be reused by future
validation studies.

Output structure:

validation/
    development/
    oos/
    excluded/
    holdout/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    if "time" in df.columns:

        time_col = "time"

    elif "timestamp" in df.columns:

        time_col = "timestamp"

    else:

        raise ValueError(
            "Dataset must contain either 'time' or 'timestamp'."
        )

    df[time_col] = pd.to_datetime(
        df[time_col],
        format="mixed",
        errors="raise",
    )

    if time_col != "time":
        df = df.rename(columns={time_col: "time"})

    return df


def _write_window(
    df: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> None:

    window_dir = output_dir / name
    window_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = (
        window_dir
        / "master_research_dataset_core.csv"
    )

    df.to_csv(
        out_file,
        index=False,
    )

    print(
        f"[BSVE] {name:<12}"
        f"{len(df):8,} rows  "
        f"-> {out_file}"
    )


def _describe_window(
    name: str,
    df: pd.DataFrame,
) -> None:

    print()
    print(name.upper())

    print("-" * 72)

    print(f"Rows : {len(df):,}")

    if len(df):

        print(
            f"Start: {df['time'].min()}"
        )

        print(
            f"End  : {df['time'].max()}"
        )

        print()

        print("Pairs:")

        for pair, count in (
            df["pair"]
            .value_counts()
            .sort_index()
            .items()
        ):
            print(
                f"  {pair:<12}{count:8,}"
            )


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_windows(
    dataset: pd.DataFrame,
    *,
    development_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    holdout_start: pd.Timestamp,
):

    development = dataset[
        dataset["time"] <= development_end
    ].copy()

    oos = dataset[
        (dataset["time"] >= oos_start)
        &
        (dataset["time"] <= oos_end)
    ].copy()

    excluded = dataset[
        (dataset["time"] > oos_end)
        &
        (dataset["time"] < holdout_start)
    ].copy()

    holdout = dataset[
        dataset["time"] >= holdout_start
    ].copy()

    return {
        "development": development,
        "oos": oos,
        "excluded": excluded,
        "holdout": holdout,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Extract the frozen BSVE validation windows "
            "from the master research dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--development-end",
        default="2022-12-31",
    )

    parser.add_argument(
        "--oos-start",
        default="2023-01-01",
    )

    parser.add_argument(
        "--oos-end",
        default="2024-08-22",
    )

    parser.add_argument(
        "--holdout-start",
        default="2025-05-10",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()

    print()
    print("=" * 72)
    print("BSVE VALIDATION WINDOW EXTRACTION")
    print("=" * 72)

    print()

    dataset = _load_dataset(
        Path(args.dataset)
    )

    print(
        f"Loaded master dataset: {len(dataset):,} rows"
    )

    windows = extract_windows(
        dataset,
        development_end=pd.Timestamp(
            args.development_end
        ),
        oos_start=pd.Timestamp(
            args.oos_start
        ),
        oos_end=pd.Timestamp(
            args.oos_end
        ),
        holdout_start=pd.Timestamp(
            args.holdout_start
        ),
    )

    output_dir = Path(args.output_dir)

    print()

    for name, df in windows.items():

        _describe_window(
            name,
            df,
        )

        _write_window(
            df,
            output_dir,
            name,
        )

    # ------------------------------------------------------------------
    # Window summary / integrity checks
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("WINDOW SUMMARY")
    print("=" * 72)
    print()

    total_window_rows = 0

    for name in [
        "development",
        "oos",
        "excluded",
        "holdout",
    ]:

        df = windows[name]

        total_window_rows += len(df)

        if len(df):

            start = df["time"].min()
            end = df["time"].max()

            print(
                f"{name.capitalize():<12}"
                f"{start}  →  {end}"
                f"   ({len(df):,} rows)"
            )

        else:

            print(
                f"{name.capitalize():<12}"
                f"<empty>"
            )

    print()

    print(
        f"Original dataset rows : {len(dataset):,}"
    )

    print(
        f"Window rows           : {total_window_rows:,}"
    )

    if total_window_rows == len(dataset):

        print()
        print(
            "✓ All observations assigned to exactly one validation window."
        )
        print()

    else:

        print()
        print(
            "⚠ WARNING: Validation windows do not partition the dataset."
        )

        print(
            f"Difference: "
            f"{len(dataset) - total_window_rows:+d} rows"
        )

    # ------------------------------------------------------------------
    # Verify that windows are mutually exclusive
    # ------------------------------------------------------------------

    all_keys = set()

    overlaps = 0

    for df in windows.values():

        keys = set(
            zip(
                df["pair"],
                df["time"],
            )
        )

        overlaps += len(all_keys & keys)

        all_keys |= keys

    if overlaps == 0:

        print(
            "✓ Validation windows are mutually exclusive."
        )
        print()

    else:

        print(
            f"⚠ WARNING: {overlaps} observations occur in multiple windows."
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()