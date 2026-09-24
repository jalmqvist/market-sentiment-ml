"""Calibration plugin for Persistent Commitment Lifecycle surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from bsve.calibration.calibration_contract import CalibrationArtifact, build_calibration_artifact


@dataclass(frozen=True)
class PersistentCalibrationConfig:
    pairs: tuple[str, ...]
    min_level_history: int = 3
    min_trajectory_history: int = 4


class PersistentCalibrationPlugin:
    """Estimate Persistent Level/Trajectory tertile boundaries from training data."""

    ontology_id = "persistent"
    ontology_version = "0.1.0"

    _DEFAULT_PAIRS = (
        "eur-usd",
        "gbp-usd",
        "nzd-usd",
        "eur-gbp",
        "eur-aud",
    )

    def calibrate(
        self,
        dataset_adapter: Any,
        state_spec: dict[str, Any],
        calibration_params: dict[str, Any],
    ) -> CalibrationArtifact:
        cfg = self._config(dataset_adapter, state_spec)

        start = str(calibration_params["calibration_window_start"])
        end = str(calibration_params["calibration_window_end"])
        calibration_id = str(calibration_params["calibration_id"])
        dataset_version = str(calibration_params["dataset_version"])
        calibration_method = str(
            calibration_params.get("calibration_method", "persistent_tertile_training_fold")
        )

        rows = self._eligible_rows(
            dataset_adapter=dataset_adapter,
            pairs=cfg.pairs,
            start=start,
            end=end,
            min_level_history=cfg.min_level_history,
            min_trajectory_history=cfg.min_trajectory_history,
            max_gap=state_spec.get("observed_segments", {}).get("max_gap", "30D"),
        )

        if rows.empty:
            raise ValueError("persistent calibration has no eligible observations")

        level_vals = rows["level"].dropna()
        traj_vals = rows["trajectory"].dropna()

        if level_vals.nunique() < 3:
            raise ValueError("persistent calibration requires at least 3 distinct level values")
        if traj_vals.nunique() < 3:
            raise ValueError("persistent calibration requires at least 3 distinct trajectory values")

        level_q33 = float(level_vals.quantile(1 / 3))
        level_q67 = float(level_vals.quantile(2 / 3))
        traj_q33 = float(traj_vals.quantile(1 / 3))
        traj_q67 = float(traj_vals.quantile(2 / 3))

        if level_q33 == level_q67:
            raise ValueError("persistent calibration degenerate level tertiles (Q33 == Q67)")
        if traj_q33 == traj_q67:
            raise ValueError("persistent calibration degenerate trajectory tertiles (Q33 == Q67)")

        if not self._all_bins_non_empty(level_vals, level_q33, level_q67):
            raise ValueError("persistent calibration produced an empty level bin")
        if not self._all_bins_non_empty(traj_vals, traj_q33, traj_q67):
            raise ValueError("persistent calibration produced an empty trajectory bin")

        thresholds = {
            "level_q33": level_q33,
            "level_q67": level_q67,
            "trajectory_q33": traj_q33,
            "trajectory_q67": traj_q67,
            "min_level_history": cfg.min_level_history,
            "min_trajectory_history": cfg.min_trajectory_history,
            "feature_level": "prior_mean_depth",
            "feature_trajectory": "early_late_commitment_delta",
            "gap_rule": "hard_observational_break",
            "missing_value_policy": "drop_invalid_or_insufficient_history",
            "tie_binning_policy": "half_open_low_mid_high",
            "pairs": list(cfg.pairs),
            "family": "persistent",
            "observed_segment_gap": state_spec.get("observed_segments", {}).get("max_gap", "30D"),
        }

        diagnostics = {
            "eligible_observation_count": int(len(rows)),
            "pair_counts": {str(k): int(v) for k, v in rows["pair"].value_counts().sort_index().items()},
        }

        return build_calibration_artifact(
            calibration_id=calibration_id,
            ontology_id=self.ontology_id,
            ontology_version=self.ontology_version,
            calibration_window_start=start,
            calibration_window_end=end,
            dataset_version=dataset_version,
            calibration_method=calibration_method,
            outcome="success",
            thresholds=thresholds,
            diagnostics=diagnostics,
            calibration_mode=str(calibration_params.get("calibration_mode", "walkforward")),
            threshold_provenance={
                "population": "persistent_family_pooled_training_observations",
                "quantiles": ["Q33", "Q67"],
                "units": "observations",
            },
        )

    def _config(self, dataset_adapter: Any, state_spec: dict[str, Any]) -> PersistentCalibrationConfig:
        from_spec = state_spec.get("environment", {}).get("pairs", [])
        raw_pairs = from_spec if from_spec else self._DEFAULT_PAIRS
        pairs = tuple(dataset_adapter.normalize_pair(p) for p in raw_pairs)
        return PersistentCalibrationConfig(
            pairs=tuple(sorted(set(pairs))),
            min_level_history=int(state_spec.get("minimum_history", {}).get("level", 3)),
            min_trajectory_history=int(state_spec.get("minimum_history", {}).get("trajectory", 4)),
        )

    def _eligible_rows(
        self,
        *,
        dataset_adapter: Any,
        pairs: tuple[str, ...],
        start: str,
        end: str,
        min_level_history: int,
        min_trajectory_history: int,
        max_gap: str,
    ) -> pd.DataFrame:
        df = dataset_adapter.get_sentiment_observations(
            pairs=pairs,
            columns=["net_sentiment", "crowd_side"],
        ).copy()
        if df.empty:
            return df

        ts_col = dataset_adapter.config.timestamp_col
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df[df[ts_col].notna()]
        df = df.sort_values([dataset_adapter.config.pair_col, ts_col], kind="mergesort")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        rows: list[dict[str, Any]] = []
        max_gap_td = pd.Timedelta(max_gap)

        for pair, g in df.groupby(dataset_adapter.config.pair_col, sort=False):
            prior_ts: pd.Timestamp | None = None
            prior_side: str | None = None
            history: list[float] = []

            for r in g.itertuples(index=False):
                ts = getattr(r, ts_col)
                sentiment = getattr(r, "net_sentiment", None)
                if sentiment is None:
                    continue
                try:
                    s = float(sentiment)
                except (TypeError, ValueError):
                    continue
                if pd.isna(s):
                    continue

                side = "LONG" if s > 0 else "SHORT" if s < 0 else ""
                depth = abs(s)

                gap_detected = prior_ts is not None and (ts - prior_ts) > max_gap_td
                side_changed = (
                    prior_side in {"LONG", "SHORT"}
                    and side in {"LONG", "SHORT"}
                    and side != prior_side
                )

                if prior_ts is None or gap_detected or side_changed or side == "":
                    history = []

                level = None
                trajectory = None
                if len(history) >= min_level_history:
                    level = float(pd.Series(history).mean())
                if len(history) >= min_trajectory_history:
                    split_idx = len(history) // 2
                    early = history[:split_idx]
                    late = history[split_idx:]
                    if early and late:
                        trajectory = float(pd.Series(late).mean() - pd.Series(early).mean())

                in_window = start_ts <= ts <= end_ts
                if (
                    in_window
                    and side in {"LONG", "SHORT"}
                    and level is not None
                    and trajectory is not None
                ):
                    rows.append(
                        {
                            "pair": pair,
                            "timestamp": ts,
                            "level": level,
                            "trajectory": trajectory,
                        }
                    )

                if side in {"LONG", "SHORT"}:
                    history = history + [depth]
                else:
                    history = []
                prior_ts = ts
                prior_side = side

        return pd.DataFrame(rows)

    @staticmethod
    def _all_bins_non_empty(values: pd.Series, q33: float, q67: float) -> bool:
        low = int((values < q33).sum())
        mid = int(((values >= q33) & (values < q67)).sum())
        high = int((values >= q67).sum())
        return low > 0 and mid > 0 and high > 0
