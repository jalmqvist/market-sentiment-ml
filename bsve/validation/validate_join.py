#!/usr/bin/env python3
"""
Validate one-to-one alignment between a Behavioral Surface and the
master research dataset.

This script verifies that a Behavioral Surface can be joined safely
to the originating research dataset before outcome labeling.

Validation checks:

- identical row counts
- duplicate (timestamp, pair) keys
- missing keys
- inner join cardinality
- pair counts
- crowd_side consistency

The script aborts with a non-zero exit status if any validation fails.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

KEY_COLUMNS = ["timestamp", "pair"]


def fail(message: str) -> None:
    print(f"\n❌ {message}")
    sys.exit(1)


def check_duplicates(df: pd.DataFrame, name: str) -> None:

    dup = df.duplicated(KEY_COLUMNS)

    n = int(dup.sum())

    if n == 0:
        print(f"✓ {name}: no duplicate keys")
        return

    print(df.loc[dup, KEY_COLUMNS].head(20))
    fail(f"{name}: found {n} duplicate key(s)")


def key_set(df: pd.DataFrame):

    return set(zip(df.timestamp, df.pair))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--surface",
        required=True,
        help="Behavioral Surface parquet",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Master research dataset CSV",
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("BSVE JOIN VALIDATION")
    print("=" * 72)

    surface = pd.read_parquet(args.surface)

    dataset = pd.read_csv(args.dataset)

    print("\nSurface columns:")
    print(surface.columns.tolist())

    print("\nDataset columns:")
    print(dataset.columns.tolist())
    print()

    # ------------------------------------------------------------------
    # Resolve timestamp columns
    # ------------------------------------------------------------------

    surface_time_col = "timestamp"

    if "timestamp" in dataset.columns:
        dataset_time_col = "timestamp"
    elif "entry_time" in dataset.columns:
        dataset_time_col = "entry_time"
    else:
        fail(
            "Dataset contains neither 'timestamp' nor 'entry_time'."
        )

    surface[surface_time_col] = pd.to_datetime(surface[surface_time_col])
    dataset[dataset_time_col] = pd.to_datetime(dataset[dataset_time_col])

    # Canonicalize for the remainder of the script
    if dataset_time_col != "timestamp":
        dataset = dataset.rename(columns={dataset_time_col: "timestamp"})

    # ------------------------------------------------------------------
    # Restrict the master dataset to the pairs represented by the
    # Behavioral Surface.
    # ------------------------------------------------------------------

    print()
    print("Behavioral Surface pairs")
    print("-" * 72)

    surface_pairs = sorted(surface["pair"].unique())

    for pair in surface_pairs:
        print(pair)

    dataset_before = len(dataset)

    dataset = dataset[
        dataset["pair"].isin(surface_pairs)
    ].copy()

    print()
    print("Dataset filtering")
    print("-" * 72)

    print(f"Master dataset rows     : {dataset_before:8,d}")
    print(f"Filtered dataset rows   : {len(dataset):8,d}")
    print(f"Behavioral Surface rows : {len(surface):8,d}")

    if len(surface) != len(dataset):
        fail(
            "Filtered dataset and Behavioral Surface "
            "have different row counts."
        )

    print("✓ Pair subset extracted")
    print("✓ Row counts match")

    print()
    print("Duplicate-key validation")
    print("-" * 72)

    check_duplicates(surface, "Behavioral Surface")
    check_duplicates(dataset, "Master dataset")

    print()
    print("Key-set comparison")
    print("-" * 72)

    surface_keys = key_set(surface)
    dataset_keys = key_set(dataset)

    missing_surface = dataset_keys - surface_keys
    missing_dataset = surface_keys - dataset_keys

    print(f"Missing from surface : {len(missing_surface)}")
    print(f"Missing from dataset : {len(missing_dataset)}")

    if missing_surface:
        print("\nExamples missing from surface:")
        print(list(sorted(missing_surface))[:10])
        fail("Surface is missing observations.")

    if missing_dataset:
        print("\nExamples missing from dataset:")
        print(list(sorted(missing_dataset))[:10])
        fail("Dataset is missing observations.")

    print("✓ Key sets identical")

    print()
    print("Join validation")
    print("-" * 72)

    joined = surface.merge(
        dataset,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_surface", "_dataset"),
    )

    print(f"Joined rows : {len(joined):8,d}")

    if len(joined) != len(surface):
        fail("Inner join cardinality mismatch.")

    print("✓ Inner join is one-to-one")

    print()
    print("Pair counts")
    print("-" * 72)

    surface_counts = surface.groupby("pair").size()
    dataset_counts = dataset.groupby("pair").size()

    pairs = sorted(set(surface_counts.index) | set(dataset_counts.index))

    for pair in pairs:

        s = int(surface_counts.get(pair, 0))
        d = int(dataset_counts.get(pair, 0))

        status = "✓" if s == d else "✗"

        print(
            f"{pair:12s}"
            f" surface={s:6d}"
            f" dataset={d:6d}"
            f" {status}"
        )

        if s != d:
            fail(f"Pair counts differ for {pair}")

    print()
    print("Crowd-side consistency")
    print("-" * 72)

    if "crowd_side_dataset" in joined.columns:

        mapping = {
            1: "LONG",
            -1: "SHORT",
            0: "",
        }

        dataset_side = (
            joined["crowd_side_dataset"]
            .map(mapping)
            .fillna(joined["crowd_side_dataset"])
            .astype(str)
            .str.upper()
        )

        surface_side = (
            joined["crowd_side_surface"]
            .astype(str)
            .str.upper()
        )

        mismatch = dataset_side != surface_side

        print(f"Mismatches : {int(mismatch.sum())}")

        if mismatch.any():

            print(
                joined.loc[
                    mismatch,
                    [
                        "timestamp",
                        "pair",
                        "crowd_side_surface",
                        "crowd_side_dataset",
                    ],
                ].head(20)
            )

            fail("Crowd-side mismatch detected.")

        print("✓ Crowd-side values identical")

    else:

        print("Dataset has no crowd_side column (skipped).")

    print()
    print("=" * 72)
    print("PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()