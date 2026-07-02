#!/usr/bin/env python3
"""
Attach outcome labels to a Behavioral Surface.

This utility joins a Behavioral Surface to the master research dataset,
computes future crowd-relative returns and crowd-failure labels, and
exports a canonical labeled Behavioral Surface for subsequent
statistical validation.

The implementation is ontology-agnostic.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

KEY_COLUMNS = ["timestamp", "pair"]

CROWD_SIDE_MAP = {
    "LONG": 1,
    "SHORT": -1,
    "": 0,
}


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_surface(path: str | Path) -> pd.DataFrame:

    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_dataset(path: str | Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    if "entry_time" in df.columns:
        df = df.rename(columns={"entry_time": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_join(surface: pd.DataFrame,
                  dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Validate one-to-one alignment and return the filtered dataset.
    """

    dataset = dataset[
        dataset["pair"].isin(surface["pair"].unique())
    ].copy()

    if len(surface) != len(dataset):

        raise RuntimeError(
            "Behavioral Surface and filtered dataset "
            "have different row counts."
        )

    if surface.duplicated(KEY_COLUMNS).any():

        raise RuntimeError(
            "Behavioral Surface contains duplicate keys."
        )

    if dataset.duplicated(KEY_COLUMNS).any():

        raise RuntimeError(
            "Dataset contains duplicate keys."
        )

    surface_keys = set(zip(surface.timestamp, surface.pair))
    dataset_keys = set(zip(dataset.timestamp, dataset.pair))

    if surface_keys != dataset_keys:

        raise RuntimeError(
            "Surface and dataset key sets differ."
        )

    return dataset


# ---------------------------------------------------------------------
# Outcome labeling
# ---------------------------------------------------------------------

def build_labeled_surface(
    surface: pd.DataFrame,
    dataset: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:

    return_column = f"ret_{horizon}b"

    if return_column not in dataset.columns:

        raise KeyError(
            f"Dataset does not contain '{return_column}'."
        )

    joined = surface.merge(
        dataset[
            KEY_COLUMNS +
            [
                "crowd_side",
                return_column,
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_surface", "_dataset"),
    )

    if len(joined) != len(surface):

        raise RuntimeError(
            "Join cardinality mismatch."
        )

    crowd_sign = (
        joined["crowd_side_surface"]
        .astype(str)
        .str.upper()
        .map(CROWD_SIDE_MAP)
        .fillna(0)
    )

    joined["future_return"] = joined[return_column]

    joined["crowd_relative_return"] = (
        crowd_sign
        * joined["future_return"]
    )

    joined["crowd_failed"] = (
            joined["crowd_relative_return"] < 0
    ).astype(bool)

    joined["horizon_bars"] = horizon

    columns = [
        "timestamp",
        "pair",
        "state_id",
        "episode_id",
        "maturity_bars",
        "crowd_side_surface",
        return_column,  # original dataset column
        "future_return",  # canonical BSVE name
        "crowd_relative_return",
        "crowd_failed",
        "horizon_bars",
    ]

    labeled = (
        joined[columns]
        .rename(
            columns={
                "crowd_side_surface": "crowd_side",
            }
        )
        .sort_values(["pair", "timestamp"])
        .reset_index(drop=True)
    )

    return labeled


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary(df: pd.DataFrame):

    print()
    print("=" * 72)
    print("LABELED BEHAVIORAL SURFACE")
    print("=" * 72)

    print()

    print(f"Rows            : {len(df):,}")

    print(
        f"Horizon         : "
        f"{df['horizon_bars'].iloc[0]} bars"
    )

    print()

    print("State frequencies")
    print("-" * 72)

    print(
        df["state_id"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Crowd-failure frequency by state")
    print("-" * 72)

    failure = (
        df.groupby("state_id")["crowd_failed"]
        .mean()
        .sort_index()
    )

    for state, rate in failure.items():

        print(
            f"{state:30s}"
            f"{100*rate:7.2f}%"
        )

    print()

    overall = df["crowd_failed"].mean()

    print(
        f"Overall crowd-failure rate : "
        f"{100*overall:.2f}%"
    )

    print()

    print(
        "✓ Join preserved "
        f"{len(df):,} observations."
    )

    print()
    print("Pair frequencies")
    print("-" * 72)

    print(
        df["pair"]
        .value_counts()
        .sort_index()
    )

# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def build_manifest(
    labeled: pd.DataFrame,
    surface_path: str | Path,
    dataset_version: str,
    horizon: int,
) -> dict:

    return {
        "artifact_type": "behavioral_surface_labels",
        "schema_version": "1.0.0",
        "generated_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "behavioral_surface": str(surface_path),
        "dataset_version": dataset_version,
        "label_horizon": horizon,
        "rows": int(len(labeled)),
        "pairs": sorted(
            labeled["pair"].unique().tolist()
        ),
    }


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_artifacts(
    labeled,
    manifest,
    output_dir,
    horizon,
    surface_path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    surface_name = Path(surface_path).stem

    if surface_name.startswith("behavioral_surface_"):
        surface_name = surface_name[len("behavioral_surface_"):]

    parquet_path = (
            output_dir
            / f"behavioral_surface_{surface_name}_labeled_{horizon}b.parquet"
    )

    manifest_path = (
        output_dir
        / "behavioral_surface_labels_manifest.json"
    )

    labeled.to_parquet(
        parquet_path,
        index=False,
    )

    with open(manifest_path, "w") as f:

        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print(f"[BSVE] Labeled surface written : {parquet_path}")
    print(f"[BSVE] Manifest written        : {manifest_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Attach outcome labels to a Behavioral Surface."
    )

    parser.add_argument(
        "--surface",
        required=True,
        help="Behavioral Surface parquet.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Master research dataset CSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--dataset-version",
        required=True,
        help="Dataset version string.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forward-return horizon in bars (default: 24).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)

    print()
    print("=" * 72)
    print("BSVE OUTCOME LABELING")
    print("=" * 72)

    print()
    print("Loading Behavioral Surface...")

    surface = load_surface(args.surface)

    print(f"  {len(surface):,} observations")

    print()

    print("Loading master dataset...")

    dataset = load_dataset(args.dataset)

    print(f"  {len(dataset):,} observations")

    print()

    print("Validating join...")

    dataset = validate_join(
        surface,
        dataset,
    )

    print("✓ Join validation passed")

    print()

    print(
        f"Building labeled surface "
        f"({args.horizon}-bar horizon)..."
    )

    labeled = build_labeled_surface(
        surface,
        dataset,
        horizon=args.horizon,
    )

    print_summary(labeled)

    manifest = build_manifest(
        labeled=labeled,
        surface_path=args.surface,
        dataset_version=args.dataset_version,
        horizon=args.horizon,
    )

    export_artifacts(
        labeled=labeled,
        manifest=manifest,
        output_dir=output_dir,
        horizon=args.horizon,
        surface_path=args.surface
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()