"""Independent post-state outcome labeling for Reactive-JPY Criterion 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from bsve.adapters.dataset_adapter import MasterResearchDatasetAdapter

CONSENSUS_STATES = {
    "JPY_CONSENSUS_YOUNG",
    "JPY_CONSENSUS_MATURING",
    "JPY_CONSENSUS_MATURE",
}
DEFAULT_OUTCOME_WINDOW_BARS = 24
DEFAULT_THRESHOLD_COL = "atr_pct"
DEFAULT_PRICE_COL = "entry_close"


def load_state_surface(path: str | Path) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"state-surface artifact not found: {artifact_path}")

    df = pd.read_parquet(artifact_path)
    required = {"pair", "entry_time", "state_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"artifact missing required columns: {missing}")

    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    if out["entry_time"].isna().any():
        raise ValueError("artifact contains invalid entry_time values")
    out["pair"] = out["pair"].astype(str)
    return out


def load_market_dataset(path: str | Path) -> pd.DataFrame:
    adapter = MasterResearchDatasetAdapter.from_artifact(path)
    frame = adapter.get_structural_observations(
        columns=[DEFAULT_PRICE_COL, DEFAULT_THRESHOLD_COL]
    )
    required = {adapter.config.pair_col, adapter.config.timestamp_col, DEFAULT_PRICE_COL, DEFAULT_THRESHOLD_COL}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")

    out = frame.rename(
        columns={
            adapter.config.pair_col: "pair",
            adapter.config.timestamp_col: "entry_time",
        }
    ).copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    if out["entry_time"].isna().any():
        raise ValueError("dataset contains invalid entry_time values")
    out["pair"] = out["pair"].astype(str)
    return out


def reconstruct_consensus_episodes(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["pair", "entry_time"]).copy()
    shifted_pair = ordered["pair"].ne(ordered["pair"].shift())
    shifted_state = ordered["state_id"].ne(ordered["state_id"].shift())
    ordered["episode_id"] = (shifted_pair | shifted_state).cumsum()

    episodes = (
        ordered.groupby(["pair", "state_id", "episode_id"], as_index=False)
        .agg(
            episode_start_time=("entry_time", "min"),
            episode_end_time=("entry_time", "max"),
            episode_bars=("entry_time", "size"),
        )
        .drop(columns=["episode_id"])
    )
    episodes = episodes[episodes["state_id"].isin(CONSENSUS_STATES)].copy()
    episodes["episode_bars"] = episodes["episode_bars"].astype(int)
    return episodes.sort_values(["pair", "episode_start_time"]).reset_index(drop=True)


def _compute_forward_frame(
    market_df: pd.DataFrame,
    *,
    outcome_window_bars: int,
) -> pd.DataFrame:
    ordered = market_df.sort_values(["pair", "entry_time"]).copy()
    grouped = ordered.groupby("pair", sort=False)
    ordered["future_close"] = grouped[DEFAULT_PRICE_COL].shift(-outcome_window_bars)
    ordered["forward_return_24h"] = (
        ordered["future_close"] / ordered[DEFAULT_PRICE_COL] - 1.0
    )
    ordered["success_threshold"] = ordered[DEFAULT_THRESHOLD_COL].abs()
    return ordered[["pair", "entry_time", "forward_return_24h", "success_threshold"]]


def assign_independent_outcome_labels(
    state_surface: pd.DataFrame,
    market_dataset: pd.DataFrame,
    *,
    outcome_window_bars: int = DEFAULT_OUTCOME_WINDOW_BARS,
) -> pd.DataFrame:
    if outcome_window_bars <= 0:
        raise ValueError("outcome_window_bars must be positive")

    episodes = reconstruct_consensus_episodes(state_surface)
    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "state_id",
                "episode_start_time",
                "episode_end_time",
                "episode_bars",
                "outcome_window_bars",
                "forward_return_24h",
                "success_threshold",
                "outcome_label",
                "outcome_available",
            ]
        )

    forward = _compute_forward_frame(
        market_dataset,
        outcome_window_bars=outcome_window_bars,
    )
    merged = episodes.merge(
        forward,
        left_on=["pair", "episode_end_time"],
        right_on=["pair", "entry_time"],
        how="left",
    ).drop(columns=["entry_time"])

    merged["outcome_window_bars"] = int(outcome_window_bars)
    merged["outcome_available"] = (
        merged["forward_return_24h"].notna()
        & merged["success_threshold"].notna()
        & (merged["success_threshold"] > 0)
    )
    merged["outcome_label"] = None

    success = merged["outcome_available"] & (
        merged["forward_return_24h"].abs() >= merged["success_threshold"]
    )
    failure = merged["outcome_available"] & ~success

    merged.loc[success, "outcome_label"] = "SUCCESS"
    merged.loc[failure, "outcome_label"] = "FAILURE"

    return merged[
        [
            "pair",
            "state_id",
            "episode_start_time",
            "episode_end_time",
            "episode_bars",
            "outcome_window_bars",
            "forward_return_24h",
            "success_threshold",
            "outcome_label",
            "outcome_available",
        ]
    ].copy()


def build_outcome_payload(
    state_surface: pd.DataFrame,
    market_dataset: pd.DataFrame,
    *,
    outcome_window_bars: int = DEFAULT_OUTCOME_WINDOW_BARS,
) -> dict[str, Any]:
    outcomes = assign_independent_outcome_labels(
        state_surface,
        market_dataset,
        outcome_window_bars=outcome_window_bars,
    )

    evaluable = outcomes[outcomes["outcome_available"]]
    success_count = int((evaluable["outcome_label"] == "SUCCESS").sum())
    failure_count = int((evaluable["outcome_label"] == "FAILURE").sum())

    payload = {
        "metadata": {
            "module": "bsve.validation.outcome_labeling",
            "outcome_window_bars": int(outcome_window_bars),
            "price_column": DEFAULT_PRICE_COL,
            "threshold_column": DEFAULT_THRESHOLD_COL,
            "success_rule": "abs(forward_return_24h) >= success_threshold",
            "outcome_classes": ["SUCCESS", "FAILURE"],
        },
        "summary": {
            "total_consensus_episodes": int(len(outcomes)),
            "evaluable_episodes": int(len(evaluable)),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": (float(success_count / len(evaluable)) if len(evaluable) else None),
        },
        "independent_outcomes": outcomes.to_dict(orient="records"),
    }
    return payload


def _write_payload(payload: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "independent_outcomes.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return out_path


def _print_summary(payload: dict[str, Any], output_path: Path) -> None:
    summary = payload["summary"]
    print("[BSVE] Independent Outcome Labeling (Reactive-JPY)")
    print("-" * 60)
    print(f"Consensus episodes: {summary['total_consensus_episodes']}")
    print(f"Evaluable episodes: {summary['evaluable_episodes']}")
    print(f"SUCCESS: {summary['success_count']}")
    print(f"FAILURE: {summary['failure_count']}")
    print(f"Output: {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ontology-independent outcome labels from fixed post-episode "
            "forward price behavior."
        ),
        prog="python -m bsve.validation.outcome_labeling",
    )
    parser.add_argument("--artifact", required=True, help="Path to state-surface artifact parquet.")
    parser.add_argument("--dataset-path", required=True, help="Path to master research dataset CSV/parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory where independent_outcomes.json is written.")
    parser.add_argument(
        "--outcome-window-bars",
        type=int,
        default=DEFAULT_OUTCOME_WINDOW_BARS,
        help="Fixed forward horizon (H1 bars) used to label outcomes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        state_surface = load_state_surface(args.artifact)
        market_dataset = load_market_dataset(args.dataset_path)
        payload = build_outcome_payload(
            state_surface,
            market_dataset,
            outcome_window_bars=args.outcome_window_bars,
        )
        output_path = _write_payload(payload, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
