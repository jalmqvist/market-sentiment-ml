"""Regression tests for PR5 — Behavioral Evaluation Framework.

Covers:
- trainer-reported artifact discovery
- coverage percentage calculations
- overlap percentage calculations
- report interpretation generation
- experiment comparison (including non-identical state sets)
- control generation
- comparison between experiments with different states
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.behavioral.utils import (
    RunResult,
    parse_reported_artifact_paths,
)
from analysis.behavioral.coverage import build_coverage_table
from analysis.behavioral.compare_predictions import compare_mlp_lstm_predictions
from analysis.behavioral.analyze_manifests import (
    extract_manifest_messages,
    summarize_manifests,
)
from analysis.behavioral.metrics import (
    compute_prediction_metrics,
    compute_prediction_metrics_from_path,
)
from analysis.behavioral.controls import generate_controls
from analysis.behavioral.interpretation import (
    generate_key_observations,
    format_key_observations,
    Observation,
)
from analysis.behavioral.compare_experiments import (
    compare_experiments,
    render_comparison_report,
    compare_coverage,
    compare_prediction_agreement,
)
import analysis.behavioral.run_behavioral_suite as suite


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

DATASET_VERSION = "1.5.1"
DATASET_VARIANT = "reactive_jpy_v1_core"


def _make_dataset(n_per_state: int = 10) -> pd.DataFrame:
    """Create a minimal synthetic dataset with two behavioral states."""
    rows = []
    for state in ["STATE_A", "STATE_B"]:
        for i in range(n_per_state):
            rows.append({
                "pair": "usd-jpy" if i % 2 == 0 else "eur-usd",
                "snapshot_time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                "surface_id": "reactive_jpy",
                "state_id": state,
            })
    # Extra rows with no behavioral assignment (canonical-only)
    for i in range(5):
        rows.append({
            "pair": "usd-jpy",
            "snapshot_time": pd.Timestamp("2024-06-01") + pd.Timedelta(hours=i),
            "surface_id": None,
            "state_id": None,
        })
    return pd.DataFrame(rows)


def _make_pred_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pair": np.where(rng.random(n) > 0.5, "usd-jpy", "eur-usd"),
        "entry_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "pred_prob_up": rng.random(n),
        "signal_strength": rng.standard_normal(n),
    })


def _write_dataset(repo_root: Path) -> None:
    out_dir = repo_root / "data" / "output" / DATASET_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "pair": ["usd-jpy"] * 6,
        "entry_time": pd.date_range("2024-01-01", periods=6, freq="h"),
        "snapshot_time": pd.date_range("2024-01-01", periods=6, freq="h"),
        "surface_id": ["reactive_jpy"] * 6,
        "state_id": [
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURE",
            "JPY_CONSENSUS_MATURE",
            "JPY_CONSENSUS_MATURE",
            "JPY_CONSENSUS_YOUNG",
        ],
        "ret_48b": [0.1, 0.2, -0.1, 0.0, 0.05, -0.03],
    })
    df.to_csv(out_dir / f"master_research_dataset_{DATASET_VARIANT}.csv", index=False)


def _parse_flag(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _build_fake_runner(repo_root: Path, fail_for: set | None = None):
    calls: list[dict] = []
    fail_for = fail_for or set()

    def _runner(*, command: list[str], repo_root: Path, log_path: Path) -> RunResult:
        model = "mlp" if command[1].endswith("train.py") else "lstm"
        surface_id = _parse_flag(command, "--surface")
        state_id = _parse_flag(command, "--state")
        run_id = f"{model}_{surface_id}_{state_id}_{len(calls)}"

        predictions_dir = repo_root / "data" / "output" / "dl_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        pq_path = predictions_dir / f"{run_id}.parquet"
        mf_path = predictions_dir / f"{run_id}.manifest.json"

        pred = pd.DataFrame({
            "pair": ["usd-jpy", "usd-jpy"],
            "entry_time": pd.date_range("2024-01-01", periods=2, freq="h"),
            "pred_prob_up": [0.6, 0.4] if model == "mlp" else [0.55, 0.45],
            "signal_strength": [0.2, -0.2] if model == "mlp" else [0.1, -0.1],
        })
        pred.to_parquet(pq_path, index=False)

        manifest = {
            "identity": {
                "model": model,
                "dl_regime": f"{surface_id}:{state_id}",
                "target_horizon": 24,
                "feature_set": "price_trend",
            },
            "provenance": {
                "dataset_version": DATASET_VERSION,
                "dataset_variant": DATASET_VARIANT,
                "surface_id": surface_id,
                "state_id": state_id,
            },
            "warnings": ["synthetic warning"],
            "missing_provenance_counts": {"dataset_version": 0, "model_version": 1},
        }
        mf_path.write_text(json.dumps(manifest), encoding="utf-8")

        repo_logs = repo_root / "logs"
        repo_logs.mkdir(parents=True, exist_ok=True)
        (repo_logs / f"{run_id}.log").write_text("synthetic trainer log", encoding="utf-8")

        returncode = 1 if (model, surface_id, state_id) in fail_for else 0
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Embed trainer-reported artifact paths in the log text
        log_text = (
            f"$ {' '.join(command)}\n"
            f"INFO:root:artifact_parquet: {pq_path}\n"
            f"INFO:root:artifact_manifest: {mf_path}\n"
        )
        log_path.write_text(log_text, encoding="utf-8")

        calls.append({
            "command": command,
            "model": model,
            "surface_id": surface_id,
            "state_id": state_id,
            "returncode": returncode,
        })

        return RunResult(
            command=command,
            returncode=returncode,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            log_path=log_path,
            reported_parquet_path=pq_path if pq_path.exists() else None,
            reported_manifest_path=mf_path if mf_path.exists() else None,
        )

    return _runner, calls


def _build_args(repo_root: Path, output_root: Path) -> suite.argparse.Namespace:
    return suite._parse_args([
        "--dataset-version", DATASET_VERSION,
        "--dataset-variant", DATASET_VARIANT,
        "--models", "both",
        "--repo-root", str(repo_root),
        "--output-root", str(output_root),
        "--experiment-id", "exp_test",
    ])


# ===========================================================================
# WP1 — Trainer-reported artifact discovery
# ===========================================================================

class TestTrainerReportedArtifacts:
    def test_parse_parquet_and_manifest_from_log(self, tmp_path):
        pq = tmp_path / "pred.parquet"
        mf = tmp_path / "pred.manifest.json"
        pq.write_text("x")
        mf.write_text("{}")

        output = (
            f"$ python train.py\n"
            f"INFO:root:artifact_parquet: {pq}\n"
            f"INFO:root:artifact_manifest: {mf}\n"
        )
        got_pq, got_mf = parse_reported_artifact_paths(output)
        assert got_pq == pq.resolve()
        assert got_mf == mf.resolve()

    def test_returns_none_for_missing_files(self, tmp_path):
        output = (
            "INFO:root:artifact_parquet: /nonexistent/pred.parquet\n"
            "INFO:root:artifact_manifest: /nonexistent/pred.manifest.json\n"
        )
        got_pq, got_mf = parse_reported_artifact_paths(output)
        assert got_pq is None
        assert got_mf is None

    def test_returns_none_when_no_artifact_lines(self):
        output = "some unrelated output\nno artifact lines here"
        got_pq, got_mf = parse_reported_artifact_paths(output)
        assert got_pq is None
        assert got_mf is None

    def test_run_result_carries_reported_paths(self, tmp_path):
        pq = tmp_path / "a.parquet"
        mf = tmp_path / "a.manifest.json"
        pq.write_text("x")
        mf.write_text("{}")
        result = RunResult(
            command=["python", "train.py"],
            returncode=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            log_path=tmp_path / "run.log",
            reported_parquet_path=pq,
            reported_manifest_path=mf,
        )
        assert result.reported_parquet_path == pq
        assert result.reported_manifest_path == mf

    def test_suite_uses_trainer_reported_paths(self, tmp_path, monkeypatch):
        """When runner returns reported paths, suite uses them directly (no filesystem diff)."""
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)

        runner, calls = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)

        args = _build_args(repo_root, tmp_path / "out")
        summary = suite.run_suite(args)

        exp_dir = tmp_path / "out" / "exp_test"
        run_df = pd.read_csv(exp_dir / "summary.csv")
        # All runs should use trainer-reported discovery
        assert (run_df["artifact_discovery"] == "trainer_reported").all()
        assert summary["failures"] == 0


# ===========================================================================
# WP1 — Coverage percentage calculations
# ===========================================================================

class TestCoveragePercentages:
    def test_behavioral_coverage_fraction(self):
        df = _make_dataset(n_per_state=10)
        states = [
            {"surface_id": "reactive_jpy", "state_id": "STATE_A"},
            {"surface_id": "reactive_jpy", "state_id": "STATE_B"},
        ]
        table = build_coverage_table(df, states)

        beh = table[table["scope"] == "behavioral_coverage"].iloc[0]
        total = table[table["scope"] == "full_dataset"].iloc[0]

        # 20 behavioral rows / 25 total
        expected_frac = 20 / 25
        assert abs(float(beh["coverage_fraction"]) - expected_frac) < 1e-9
        assert float(total["coverage_fraction"]) == 1.0

    def test_per_state_fraction_of_behavioral(self):
        df = _make_dataset(n_per_state=10)
        states = [
            {"surface_id": "reactive_jpy", "state_id": "STATE_A"},
            {"surface_id": "reactive_jpy", "state_id": "STATE_B"},
        ]
        table = build_coverage_table(df, states)

        state_a = table[table["scope"] == "state:reactive_jpy:STATE_A"].iloc[0]
        # 10 rows in STATE_A / 20 behavioral rows = 0.5
        assert abs(float(state_a["state_fraction_of_behavioral"]) - 0.5) < 1e-9

    def test_fraction_none_for_full_dataset_scope(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        table = build_coverage_table(df, states)
        full_row = table[table["scope"] == "full_dataset"].iloc[0]
        # full dataset row has no state_fraction_of_behavioral
        assert pd.isna(full_row["state_fraction_of_behavioral"])

    def test_zero_total_rows_handled(self):
        df = pd.DataFrame(columns=["pair", "surface_id", "state_id"])
        states: list[dict] = []
        table = build_coverage_table(df, states)
        assert not table.empty
        # coverage_fraction for empty full_dataset should be safe (1.0 for full)
        full = table[table["scope"] == "full_dataset"].iloc[0]
        assert float(full["coverage_fraction"]) == 1.0


# ===========================================================================
# WP1 — Overlap percentage calculations
# ===========================================================================

class TestOverlapPercentages:
    def _write_artifacts(self, tmp_path: Path):
        mlp_df = pd.DataFrame({
            "pair": ["usd-jpy"] * 4,
            "entry_time": pd.date_range("2024-01-01", periods=4, freq="h"),
            "pred_prob_up": [0.6, 0.4, 0.7, 0.3],
            "signal_strength": [0.2, -0.2, 0.3, -0.1],
        })
        lstm_df = mlp_df.iloc[:2].copy()  # LSTM only has 2 of the 4 rows
        lstm_df["pred_prob_up"] = [0.55, 0.45]
        lstm_df["signal_strength"] = [0.1, -0.1]

        mlp_path = tmp_path / "mlp.parquet"
        lstm_path = tmp_path / "lstm.parquet"
        mlp_df.to_parquet(mlp_path, index=False)
        lstm_df.to_parquet(lstm_path, index=False)
        return mlp_path, lstm_path

    def test_overlap_pct_of_mlp(self, tmp_path):
        mlp_path, lstm_path = self._write_artifacts(tmp_path)
        row = compare_mlp_lstm_predictions(
            mlp_path=mlp_path,
            lstm_path=lstm_path,
            surface_id="reactive_jpy",
            state_id="STATE_A",
        )
        # 2 overlap / 4 mlp = 50%
        assert abs(float(row["overlap_pct_of_mlp"]) - 50.0) < 1e-6

    def test_overlap_pct_of_lstm(self, tmp_path):
        mlp_path, lstm_path = self._write_artifacts(tmp_path)
        row = compare_mlp_lstm_predictions(
            mlp_path=mlp_path,
            lstm_path=lstm_path,
            surface_id="reactive_jpy",
            state_id="STATE_A",
        )
        # 2 overlap / 2 lstm = 100%
        assert abs(float(row["overlap_pct_of_lstm"]) - 100.0) < 1e-6

    def test_overlap_nan_when_no_paths(self):
        row = compare_mlp_lstm_predictions(
            mlp_path=None,
            lstm_path=None,
            surface_id="s",
            state_id="x",
        )
        assert np.isnan(row["overlap_pct_of_mlp"])
        assert np.isnan(row["overlap_pct_of_lstm"])


# ===========================================================================
# WP1 — Manifest message classification
# ===========================================================================

class TestManifestClassification:
    def test_classifies_missing_provenance_as_warning(self):
        manifest = {
            "warnings": [],
            "missing_provenance_counts": {"model_version": 1},
        }
        notes, warnings, errors = extract_manifest_messages(manifest)
        assert any("missing_provenance" in w for w in warnings)
        assert not errors

    def test_classifies_fail_message_as_error(self):
        manifest = {"warnings": ["failed to load checkpoint"]}
        notes, warnings, errors = extract_manifest_messages(manifest)
        assert any("fail" in e.lower() for e in errors)

    def test_classifies_plain_message_as_note(self):
        manifest = {"warnings": ["synthetic note"]}
        notes, warnings, errors = extract_manifest_messages(manifest)
        assert any("synthetic" in n for n in notes)

    def test_summarize_manifests_has_separate_columns(self, tmp_path):
        manifest = {
            "identity": {"model": "mlp"},
            "provenance": {},
            "warnings": ["failed to converge", "missing something"],
        }
        mf_path = tmp_path / "test.manifest.json"
        mf_path.write_text(json.dumps(manifest), encoding="utf-8")
        df = summarize_manifests([mf_path])
        assert "error_count" in df.columns
        assert "warning_count" in df.columns
        assert "note_count" in df.columns


# ===========================================================================
# WP2 — Scientific prediction metrics
# ===========================================================================

class TestPredictionMetrics:
    def test_entropy_near_one_for_uniform_predictions(self):
        df = pd.DataFrame({"pred_prob_up": [0.5] * 100})
        m = compute_prediction_metrics(df)
        # H(0.5) = 1.0 bit
        assert abs(float(m["prediction_entropy_mean"]) - 1.0) < 1e-6

    def test_entropy_zero_for_certain_predictions(self):
        df = pd.DataFrame({"pred_prob_up": [0.0] * 50 + [1.0] * 50})
        m = compute_prediction_metrics(df)
        assert abs(float(m["prediction_entropy_mean"])) < 1e-6

    def test_effective_prediction_coverage(self):
        # 50 confident (> 0.1 from 0.5), 50 uncertain (exactly at 0.5)
        df = pd.DataFrame({"pred_prob_up": [0.9] * 50 + [0.5] * 50})
        m = compute_prediction_metrics(df)
        assert abs(float(m["effective_prediction_coverage"]) - 0.5) < 1e-6

    def test_pair_balance_uniform(self):
        df = pd.DataFrame({"pair": ["a", "b", "c", "d"] * 25})
        m = compute_prediction_metrics(df)
        # Four equal-frequency pairs → maximum entropy → balance = 1.0
        assert abs(float(m["pair_balance"]) - 1.0) < 1e-6

    def test_pair_balance_single_pair(self):
        df = pd.DataFrame({"pair": ["usd-jpy"] * 10})
        m = compute_prediction_metrics(df)
        # Single pair → normalised entropy = 0.0 (or defined as 0 for n=1)
        assert float(m["pair_balance"]) == 0.0

    def test_sharpness_range(self):
        df = _make_pred_df()
        m = compute_prediction_metrics(df)
        assert 0.0 <= float(m["sharpness"]) <= 1.0

    def test_from_path(self, tmp_path):
        df = _make_pred_df()
        path = tmp_path / "pred.parquet"
        df.to_parquet(path, index=False)
        m = compute_prediction_metrics_from_path(path)
        assert m["n_predictions"] == len(df)
        assert m["artifact_file"] == "pred.parquet"

    def test_empty_dataframe_returns_nan_metrics(self):
        df = pd.DataFrame({"pred_prob_up": pd.Series([], dtype=float)})
        m = compute_prediction_metrics(df)
        assert np.isnan(float(m["prediction_entropy_mean"]))

    def test_signal_strength_positive_fraction(self):
        df = pd.DataFrame({
            "pred_prob_up": [0.6] * 10,
            "signal_strength": [1.0] * 7 + [-1.0] * 3,
        })
        m = compute_prediction_metrics(df)
        assert abs(float(m["signal_strength_positive_fraction"]) - 0.7) < 1e-9


# ===========================================================================
# WP3 — Baseline controls
# ===========================================================================

class TestControls:
    def test_full_dataset_control_present(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=1)
        assert "control:full_dataset" in controls["scope"].values

    def test_behavioral_partition_control_present(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=1)
        assert "control:behavioral_partition" in controls["scope"].values

    def test_random_controls_generated(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=3)
        random_rows = controls[controls["control_type"] == "random_matched"]
        assert len(random_rows) == 3

    def test_random_control_size_matched(self):
        df = _make_dataset(n_per_state=8)
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=2)
        random_rows = controls[controls["control_type"] == "random_matched"]
        # Each should have ≤ 8 rows (STATE_A has 8 rows)
        assert (random_rows["row_count"] <= 8).all()

    def test_regime_control_absent_without_regime_col(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=0)
        assert (controls["control_type"] != "regime_partition").all()

    def test_regime_control_present_with_regime_col(self):
        df = _make_dataset()
        df["regime"] = np.where(df.index < 10, "HVTF", "LVTF")
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=0)
        assert (controls["control_type"] == "regime_partition").any()

    def test_coverage_fraction_computed(self):
        df = _make_dataset()
        states = [{"surface_id": "reactive_jpy", "state_id": "STATE_A"}]
        controls = generate_controls(df, states, n_random=0)
        full_row = controls[controls["control_type"] == "full_dataset"].iloc[0]
        assert float(full_row["coverage_fraction"]) == 1.0


# ===========================================================================
# WP4 — Scientific interpretation
# ===========================================================================

class TestInterpretation:
    def _coverage_df(self, behavioral_frac: float = 0.05, state_ratio: float = 1.0) -> pd.DataFrame:
        total = 1000
        beh = int(total * behavioral_frac)
        state_a = beh // (1 + int(state_ratio))
        state_b = beh - state_a
        return pd.DataFrame([
            {"scope": "full_dataset", "row_count": total, "coverage_fraction": 1.0,
             "state_fraction_of_behavioral": None},
            {"scope": "behavioral_coverage", "row_count": beh, "coverage_fraction": behavioral_frac,
             "state_fraction_of_behavioral": None},
            {"scope": "state:s:STATE_A", "row_count": state_a, "coverage_fraction": state_a / total,
             "state_fraction_of_behavioral": state_a / beh if beh > 0 else None,
             "surface_id": "s", "state_id": "STATE_A"},
            {"scope": "state:s:STATE_B", "row_count": state_b, "coverage_fraction": state_b / total,
             "state_fraction_of_behavioral": state_b / beh if beh > 0 else None,
             "surface_id": "s", "state_id": "STATE_B"},
        ])

    def test_low_coverage_generates_observation(self):
        cov_df = self._coverage_df(behavioral_frac=0.05)
        obs = generate_key_observations(cov_df, pd.DataFrame(), pd.DataFrame())
        assert any("5.0%" in o.observed for o in obs)

    def test_high_imbalance_generates_observation(self):
        # 900 vs 100 rows → ratio 9x
        cov_df = self._coverage_df(behavioral_frac=1.0, state_ratio=9.0)
        obs = generate_key_observations(cov_df, pd.DataFrame(), pd.DataFrame())
        assert any("imbalanced" in o.observed.lower() for o in obs)

    def test_low_agreement_generates_observation(self):
        compare_df = pd.DataFrame([{
            "surface_id": "s", "state_id": "STATE_A",
            "agreement_rate": 0.45, "overlap_pct_of_mlp": 100.0, "overlap_pct_of_lstm": 100.0,
        }])
        obs = generate_key_observations(pd.DataFrame(), compare_df, pd.DataFrame())
        assert any("low" in o.observed.lower() and "agreement" in o.observed.lower() for o in obs)

    def test_high_entropy_generates_observation(self):
        metrics_df = pd.DataFrame([{
            "state_id": "STATE_A",
            "artifact_file": "a.parquet",
            "prediction_entropy_mean": 0.95,
            "effective_prediction_coverage": 0.7,
        }])
        obs = generate_key_observations(pd.DataFrame(), pd.DataFrame(), metrics_df)
        assert any("entropy" in o.observed.lower() for o in obs)

    def test_low_effective_coverage_generates_observation(self):
        metrics_df = pd.DataFrame([{
            "state_id": "STATE_A",
            "artifact_file": "a.parquet",
            "prediction_entropy_mean": 0.5,
            "effective_prediction_coverage": 0.2,
        }])
        obs = generate_key_observations(pd.DataFrame(), pd.DataFrame(), metrics_df)
        assert any("effective" in o.observed.lower() for o in obs)

    def test_format_produces_markdown(self):
        obs = [Observation("Observed something.", "It matters.", "Follow up here.")]
        md = format_key_observations(obs)
        assert "## Key Observations" in md
        assert "**Observed:**" in md
        assert "**Why it matters:**" in md
        assert "**Follow-up:**" in md

    def test_no_observations_produces_no_notable(self):
        md = format_key_observations([])
        assert "No notable observations" in md

    def test_report_contains_key_observations_section(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)

        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        assert "## Key Observations" in report


# ===========================================================================
# WP5 — Experiment comparison
# ===========================================================================

def _build_experiment_dir(
    base: Path,
    exp_id: str,
    states: list[dict],
    *,
    agreement_rate: float = 0.7,
) -> Path:
    """Create a minimal but valid completed experiment directory."""
    exp_dir = base / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "prediction_artifacts").mkdir()
    (exp_dir / "manifests").mkdir()
    (exp_dir / "plots").mkdir()
    (exp_dir / "logs").mkdir()

    manifest = {
        "experiment_id": exp_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T01:00:00+00:00",
        "success": True,
        "cli": {"argv": [], "parsed": {"feature_set": "price_trend", "target_horizon": 24}},
        "dataset": {"version": "1.5.1", "variant": "reactive_jpy_v1_core", "path": "/tmp/x.csv"},
        "discovered_states": states,
        "models_executed": ["mlp", "lstm"],
        "git_commit": "abc123",
        "runs": [],
        "manifest_files": [],
        "prediction_artifacts": [],
    }
    (exp_dir / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # metrics.csv
    rows = []
    for state in states:
        scope_key = f"{state['surface_id']}:{state['state_id']}"
        rows.append({
            "metric_group": "coverage",
            "scope": f"state:{state['surface_id']}:{state['state_id']}",
            "row_count": 100,
            "coverage_fraction": 0.1,
            "state_fraction_of_behavioral": 0.5,
            "surface_id": state["surface_id"],
            "state_id": state["state_id"],
        })
        rows.append({
            "metric_group": "mlp_lstm_compare",
            "surface_id": state["surface_id"],
            "state_id": state["state_id"],
            "agreement_common_rows": 50,
            "overlap_pct_of_mlp": 80.0,
            "overlap_pct_of_lstm": 90.0,
            "agreement_rate": agreement_rate,
            "pred_prob_correlation": 0.6,
        })
        rows.append({
            "metric_group": "prediction_metrics",
            "artifact_file": f"{scope_key}.parquet",
            "state_id": state["state_id"],
            "n_predictions": 100,
            "prediction_entropy_mean": 0.8,
            "prediction_confidence_mean": 0.15,
            "effective_prediction_coverage": 0.6,
            "sharpness": 0.3,
            "pair_balance": 0.9,
            "coverage_days": 30.0,
        })
    # Add full_dataset coverage row
    rows.append({
        "metric_group": "coverage",
        "scope": "full_dataset",
        "row_count": 1000,
        "coverage_fraction": 1.0,
        "state_fraction_of_behavioral": None,
    })
    pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)

    run_rows = []
    for state in states:
        for model in ["mlp", "lstm"]:
            run_rows.append({
                "surface_id": state["surface_id"],
                "state_id": state["state_id"],
                "model": model,
                "status": "success",
                "returncode": 0,
            })
    pd.DataFrame(run_rows).to_csv(exp_dir / "summary.csv", index=False)

    return exp_dir


class TestExperimentComparison:
    STATES_A = [
        {"surface_id": "reactive_jpy", "state_id": "STATE_X"},
        {"surface_id": "reactive_jpy", "state_id": "STATE_Y"},
    ]
    STATES_B = [
        {"surface_id": "reactive_jpy", "state_id": "STATE_X"},
        {"surface_id": "reactive_jpy", "state_id": "STATE_Z"},  # different state
    ]

    def test_compare_identical_experiments(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_A)
        result = compare_experiments([exp1, exp2])
        assert len(result["labels"]) == 2
        assert not result["provenance_df"].empty

    def test_compare_different_state_sets(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_B)
        result = compare_experiments([exp1, exp2])
        # STATE_X is shared
        assert "reactive_jpy:STATE_X" in result["shared_states"]
        # STATE_Y and STATE_Z are unique
        assert "reactive_jpy:STATE_Y" in result["unique_states"]
        assert "reactive_jpy:STATE_Z" in result["unique_states"]

    def test_coverage_comparison_includes_both_experiments(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_A)
        result = compare_experiments([exp1, exp2])
        cov_df = result["coverage_df"]
        assert not cov_df.empty
        # Columns for both experiments
        assert any("exp_a" in c for c in cov_df.columns)
        assert any("exp_b" in c for c in cov_df.columns)

    def test_occupancy_comparison_has_state_rows(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        result = compare_experiments([exp1])
        occ_df = result["occupancy_df"]
        assert not occ_df.empty
        assert any("state:" in str(s) for s in occ_df["scope"].values)

    def test_agreement_comparison_tolerates_different_states(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_B)
        result = compare_experiments([exp1, exp2])
        agree_df = result["agreement_df"]
        # STATE_X appears in both → should have agreement for both experiments
        state_x_row = agree_df[agree_df["state_id"] == "STATE_X"]
        assert not state_x_row.empty
        # STATE_Y only in exp_a → LSTM agreement for exp_b should be None/NaN
        state_y_row = agree_df[agree_df["state_id"] == "STATE_Y"]
        assert not state_y_row.empty
        exp_b_col = [c for c in agree_df.columns if "exp_b" in c and "agreement" in c]
        if exp_b_col:
            val = state_y_row.iloc[0][exp_b_col[0]]
            assert val is None or (isinstance(val, float) and np.isnan(val))

    def test_render_report_produces_markdown(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_B)
        result = compare_experiments([exp1, exp2])
        report = render_comparison_report(result)
        assert "# Behavioral Experiment Comparison" in report
        assert "## Provenance" in report
        assert "## Coverage Comparison" in report
        assert "## MLP/LSTM Agreement Comparison" in report

    def test_missing_experiment_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare_experiments([tmp_path / "nonexistent_dir"])

    def test_single_experiment_comparison(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_solo", self.STATES_A)
        result = compare_experiments([exp1])
        assert len(result["labels"]) == 1
        assert not result["provenance_df"].empty

    def test_distribution_comparison(self, tmp_path):
        exp1 = _build_experiment_dir(tmp_path, "exp_a", self.STATES_A)
        exp2 = _build_experiment_dir(tmp_path, "exp_b", self.STATES_B, agreement_rate=0.5)
        result = compare_experiments([exp1, exp2])
        dist_df = result["distribution_df"]
        assert not dist_df.empty
        assert "prediction_entropy_mean" in dist_df.columns


# ===========================================================================
# WP5 — Report includes discovered states
# ===========================================================================

class TestReportContainsDiscoveredStates:
    def test_report_has_states_section(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)

        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        assert "## Discovered Behavioral Surface States" in report
        assert "JPY_CONSENSUS_YOUNG" in report
        assert "JPY_CONSENSUS_MATURE" in report
