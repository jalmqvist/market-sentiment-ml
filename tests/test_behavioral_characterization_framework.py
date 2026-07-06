"""Tests for PR5.1 — Behavioral Characterization Framework.

Covers:
- Finding dataclass (Scientific Interest / Scientific Confidence)
- generate_findings() noise suppression (repeated observations collapsed)
- format_executive_summary() structure
- format_findings() rendering
- derive_research_recommendation() logic
- Named training profiles (smoke / standard / publication)
- Default profile (standard, 10 epochs)
- Report structure: executive summary in primary section, diagnostics in appendix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.behavioral.interpretation import (
    Finding,
    derive_research_recommendation,
    format_executive_summary,
    format_findings,
    generate_findings,
)
from analysis.behavioral.run_behavioral_suite import PROFILES, _parse_args
import analysis.behavioral.run_behavioral_suite as suite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_VERSION = "1.5.1"
DATASET_VARIANT = "reactive_jpy_v1_core"


def _make_coverage_df(beh_frac: float = 0.05) -> pd.DataFrame:
    total = 1000
    beh = int(total * beh_frac)
    return pd.DataFrame([
        {"scope": "full_dataset", "row_count": total, "coverage_fraction": 1.0,
         "state_fraction_of_behavioral": None},
        {"scope": "behavioral_coverage", "row_count": beh, "coverage_fraction": beh_frac,
         "state_fraction_of_behavioral": None},
        {"scope": "state:s:STATE_A", "row_count": beh // 2, "coverage_fraction": beh // 2 / total,
         "state_fraction_of_behavioral": 0.5},
        {"scope": "state:s:STATE_B", "row_count": beh // 2, "coverage_fraction": beh // 2 / total,
         "state_fraction_of_behavioral": 0.5},
    ])


def _make_metrics_df(entropy: float = 0.95, eff_cov: float = 0.2) -> pd.DataFrame:
    """Metrics with 4 rows (2 states × 2 models), all with high entropy."""
    rows = []
    for state in ["STATE_A", "STATE_B"]:
        for model in ["mlp", "lstm"]:
            rows.append({
                "state_id": state,
                "model": model,
                "artifact_file": f"{model}_{state}.parquet",
                "prediction_entropy_mean": entropy,
                "effective_prediction_coverage": eff_cov,
            })
    return pd.DataFrame(rows)


def _make_compare_df(agreement: float = 0.45) -> pd.DataFrame:
    return pd.DataFrame([
        {"surface_id": "s", "state_id": "STATE_A", "agreement_rate": agreement},
        {"surface_id": "s", "state_id": "STATE_B", "agreement_rate": agreement},
    ])


def _write_dataset(repo_root: Path) -> None:
    out_dir = repo_root / "data" / "output" / DATASET_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "pair": ["usd-jpy"] * 6,
        "entry_time": pd.date_range("2024-01-01", periods=6, freq="h"),
        "snapshot_time": pd.date_range("2024-01-01", periods=6, freq="h"),
        "surface_id": ["reactive_jpy"] * 6,
        "state_id": [
            "JPY_CONSENSUS_YOUNG", "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURE", "JPY_CONSENSUS_MATURE",
            "JPY_CONSENSUS_MATURE", "JPY_CONSENSUS_YOUNG",
        ],
        "ret_48b": [0.1, 0.2, -0.1, 0.0, 0.05, -0.03],
    })
    df.to_csv(out_dir / f"master_research_dataset_{DATASET_VARIANT}.csv", index=False)


def _parse_flag(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _build_fake_runner(repo_root: Path):
    from analysis.behavioral.utils import RunResult
    calls: list[dict] = []

    def _runner(*, command: list[str], repo_root: Path, log_path: Path) -> RunResult:
        model = "mlp" if command[1].endswith("train.py") else "lstm"
        surface_id = _parse_flag(command, "--surface")
        state_id = _parse_flag(command, "--state")
        run_id = f"{model}_{surface_id}_{state_id}_{len(calls)}"

        predictions_dir = repo_root / "data" / "output" / "dl_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pq_path = predictions_dir / f"{run_id}.parquet"
        mf_path = predictions_dir / f"{run_id}.manifest.json"

        pd.DataFrame({
            "pair": ["usd-jpy", "usd-jpy"],
            "entry_time": pd.date_range("2024-01-01", periods=2, freq="h"),
            "pred_prob_up": [0.6, 0.4],
            "signal_strength": [0.2, -0.2],
        }).to_parquet(pq_path, index=False)

        mf_path.write_text(json.dumps({
            "identity": {"model": model},
            "provenance": {},
            "warnings": [],
        }), encoding="utf-8")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"INFO:root:artifact_parquet: {pq_path}\nINFO:root:artifact_manifest: {mf_path}\n",
            encoding="utf-8",
        )
        calls.append({"model": model, "surface_id": surface_id, "state_id": state_id})
        return RunResult(
            command=command, returncode=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0, log_path=log_path,
            reported_parquet_path=pq_path,
            reported_manifest_path=mf_path,
        )

    return _runner, calls


def _build_args(repo_root: Path, output_root: Path, extra: list[str] | None = None):
    argv = [
        "--dataset-version", DATASET_VERSION,
        "--dataset-variant", DATASET_VARIANT,
        "--models", "both",
        "--repo-root", str(repo_root),
        "--output-root", str(output_root),
        "--experiment-id", "exp_test",
    ]
    if extra:
        argv.extend(extra)
    return _parse_args(argv)


# ===========================================================================
# PR5.1 — Finding dataclass
# ===========================================================================

class TestFinding:
    def test_finding_has_required_fields(self):
        f = Finding(
            title="Test finding",
            description="Something was observed.",
            evidence=["- State A: 0.95 bits"],
            interest="high",
            confidence="medium",
            follow_up="Run more epochs.",
        )
        assert f.title == "Test finding"
        assert f.interest == "high"
        assert f.confidence == "medium"
        assert len(f.evidence) == 1

    def test_finding_defaults_are_sane(self):
        f = Finding(title="t", description="d")
        assert f.interest in ("low", "medium", "high")
        assert f.confidence in ("low", "medium", "high")
        assert isinstance(f.evidence, list)
        assert isinstance(f.follow_up, str)


# ===========================================================================
# PR5.1 — generate_findings() noise suppression
# ===========================================================================

class TestGenerateFindings:
    def test_high_entropy_all_states_produces_one_finding(self):
        """Four high-entropy rows (2 states × 2 models) → one aggregated finding."""
        metrics_df = _make_metrics_df(entropy=0.96)
        findings = generate_findings(pd.DataFrame(), pd.DataFrame(), metrics_df)
        entropy_findings = [f for f in findings if "entropy" in f.title.lower()]
        assert len(entropy_findings) == 1

    def test_finding_evidence_lists_all_states(self):
        """Supporting evidence should include one line per state/model."""
        metrics_df = _make_metrics_df(entropy=0.96)
        findings = generate_findings(pd.DataFrame(), pd.DataFrame(), metrics_df)
        entropy_f = next(f for f in findings if "entropy" in f.title.lower())
        assert len(entropy_f.evidence) == 4  # 2 states × 2 models

    def test_low_effective_coverage_collapsed(self):
        """Low effective coverage for all states → one finding, not multiple."""
        metrics_df = _make_metrics_df(entropy=0.5, eff_cov=0.1)
        findings = generate_findings(pd.DataFrame(), pd.DataFrame(), metrics_df)
        cov_findings = [f for f in findings if "coverage" in f.title.lower()]
        assert len(cov_findings) == 1

    def test_max_findings_cap_respected(self):
        cov_df = _make_coverage_df(beh_frac=0.05)
        metrics_df = _make_metrics_df(entropy=0.96, eff_cov=0.1)
        compare_df = _make_compare_df(agreement=0.45)
        findings = generate_findings(cov_df, compare_df, metrics_df, max_findings=3)
        assert len(findings) <= 3

    def test_empty_inputs_produce_no_findings(self):
        findings = generate_findings(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert isinstance(findings, list)
        assert len(findings) == 0

    def test_findings_ranked_high_interest_first(self):
        """High-interest findings should appear before low-interest ones."""
        cov_df = _make_coverage_df(beh_frac=0.80)  # high coverage → low interest
        metrics_df = _make_metrics_df(entropy=0.96, eff_cov=0.1)  # → high interest
        findings = generate_findings(cov_df, pd.DataFrame(), metrics_df)
        # Low effective coverage has interest="high", coverage finding has interest="low"
        if len(findings) >= 2:
            interests = [f.interest for f in findings]
            rank = {"high": 0, "medium": 1, "low": 2}
            assert rank[interests[0]] <= rank[interests[-1]]

    def test_finding_interest_and_confidence_are_valid(self):
        metrics_df = _make_metrics_df(entropy=0.96)
        findings = generate_findings(pd.DataFrame(), pd.DataFrame(), metrics_df)
        valid = {"low", "medium", "high"}
        for f in findings:
            assert f.interest in valid, f"Unexpected interest: {f.interest}"
            assert f.confidence in valid, f"Unexpected confidence: {f.confidence}"


# ===========================================================================
# PR5.1 — format_findings()
# ===========================================================================

class TestFormatFindings:
    def test_renders_scientific_findings_heading(self):
        f = Finding(title="Test", description="Desc", interest="high", confidence="medium")
        md = format_findings([f])
        assert "## Scientific Findings" in md

    def test_renders_interest_and_confidence(self):
        f = Finding(title="T", description="D", interest="high", confidence="low")
        md = format_findings([f])
        assert "High" in md
        assert "Low" in md

    def test_renders_evidence_lines(self):
        f = Finding(title="T", description="D", evidence=["- STATE_A: 0.95 bits"])
        md = format_findings([f])
        assert "STATE_A" in md

    def test_renders_follow_up(self):
        f = Finding(title="T", description="D", follow_up="Run more epochs.")
        md = format_findings([f])
        assert "Run more epochs" in md

    def test_empty_list_returns_no_findings_message(self):
        md = format_findings([])
        assert "No significant findings" in md


# ===========================================================================
# PR5.1 — format_executive_summary()
# ===========================================================================

class TestFormatExecutiveSummary:
    def _make_run_df(self, n_success: int = 4, n_failed: int = 0) -> pd.DataFrame:
        rows = [{"status": "success"} for _ in range(n_success)]
        rows += [{"status": "failed"} for _ in range(n_failed)]
        return pd.DataFrame(rows)

    def test_contains_executive_summary_heading(self):
        run_df = self._make_run_df()
        cov_df = _make_coverage_df()
        findings = [Finding(title="T", description="D", interest="high", confidence="medium")]
        md = format_executive_summary(
            experiment_id="test_exp",
            run_df=run_df,
            coverage_df=cov_df,
            discovered_states=[{"surface_id": "s", "state_id": "STATE_A"}],
            findings=findings,
            recommendation="Proceed.",
        )
        assert "## Executive Summary" in md

    def test_shows_experiment_status_success(self):
        run_df = self._make_run_df(n_success=4, n_failed=0)
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=pd.DataFrame(),
            discovered_states=[],
            findings=[],
            recommendation="",
        )
        assert "4/4" in md

    def test_shows_failure_warning(self):
        run_df = self._make_run_df(n_success=3, n_failed=1)
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=pd.DataFrame(),
            discovered_states=[],
            findings=[],
            recommendation="",
        )
        assert "1 failed" in md.lower() or "⚠" in md

    def test_shows_behavioral_surface(self):
        run_df = self._make_run_df()
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=pd.DataFrame(),
            discovered_states=[{"surface_id": "reactive_jpy", "state_id": "STATE_A"}],
            findings=[],
            recommendation="",
        )
        assert "reactive_jpy" in md

    def test_shows_coverage_fraction(self):
        run_df = self._make_run_df()
        cov_df = _make_coverage_df(beh_frac=0.12)
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=cov_df,
            discovered_states=[],
            findings=[],
            recommendation="",
        )
        assert "12.0%" in md

    def test_shows_key_findings_bullets(self):
        run_df = self._make_run_df()
        findings = [Finding(title="High entropy", description="Entropy is high.")]
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=pd.DataFrame(),
            discovered_states=[],
            findings=findings,
            recommendation="Repeat.",
        )
        assert "High entropy" in md

    def test_shows_recommendation(self):
        run_df = self._make_run_df()
        md = format_executive_summary(
            experiment_id="x",
            run_df=run_df,
            coverage_df=pd.DataFrame(),
            discovered_states=[],
            findings=[],
            recommendation="Proceed to walk-forward.",
        )
        assert "Proceed to walk-forward" in md


# ===========================================================================
# PR5.1 — derive_research_recommendation()
# ===========================================================================

class TestDeriveResearchRecommendation:
    def test_failed_runs_recommend_diagnose(self):
        run_df = pd.DataFrame([
            {"status": "success"}, {"status": "failed"},
        ])
        rec = derive_research_recommendation(
            run_df=run_df, findings=[], coverage_df=pd.DataFrame()
        )
        assert "diagnose" in rec.lower() or "failed" in rec.lower()

    def test_empty_runs_recommend_insufficient_evidence(self):
        rec = derive_research_recommendation(
            run_df=pd.DataFrame(), findings=[], coverage_df=pd.DataFrame()
        )
        assert "insufficient evidence" in rec.lower()

    def test_high_entropy_recommends_more_epochs(self):
        findings = [Finding(
            title="High prediction entropy across states",
            description="...",
            interest="medium",
            confidence="high",
        )]
        run_df = pd.DataFrame([{"status": "success"}])
        rec = derive_research_recommendation(
            run_df=run_df, findings=findings, coverage_df=pd.DataFrame()
        )
        assert "epoch" in rec.lower() or "repeat" in rec.lower()

    def test_high_agreement_recommends_walk_forward(self):
        findings = [Finding(
            title="High MLP/LSTM directional agreement",
            description="...",
            interest="medium",
            confidence="high",
        )]
        run_df = pd.DataFrame([{"status": "success"}])
        rec = derive_research_recommendation(
            run_df=run_df, findings=findings, coverage_df=pd.DataFrame()
        )
        assert "walk-forward" in rec.lower() or "pr7" in rec.lower()

    def test_clean_experiment_recommends_continue(self):
        run_df = pd.DataFrame([{"status": "success"}])
        rec = derive_research_recommendation(
            run_df=run_df, findings=[], coverage_df=pd.DataFrame()
        )
        assert len(rec) > 0


# ===========================================================================
# PR5.1 — Named training profiles
# ===========================================================================

class TestTrainingProfiles:
    def test_all_profiles_defined(self):
        assert "smoke" in PROFILES
        assert "standard" in PROFILES
        assert "publication" in PROFILES

    def test_standard_profile_has_10_epochs(self):
        assert PROFILES["standard"]["epochs"] == 10

    def test_smoke_profile_has_fewer_epochs_than_standard(self):
        assert PROFILES["smoke"]["epochs"] < PROFILES["standard"]["epochs"]

    def test_publication_profile_has_more_epochs_than_standard(self):
        assert PROFILES["publication"]["epochs"] > PROFILES["standard"]["epochs"]

    def test_default_profile_is_standard(self):
        args = _parse_args([
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
        ])
        assert args.epochs == 10  # standard profile default

    def test_explicit_epochs_override_profile(self):
        args = _parse_args([
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--profile", "publication",
            "--epochs", "7",
        ])
        assert args.epochs == 7

    def test_smoke_profile_sets_epochs(self):
        args = _parse_args([
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--profile", "smoke",
        ])
        assert args.epochs == PROFILES["smoke"]["epochs"]

    def test_publication_profile_sets_epochs(self):
        args = _parse_args([
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--profile", "publication",
        ])
        assert args.epochs == PROFILES["publication"]["epochs"]

    def test_profile_name_recorded_in_args(self):
        args = _parse_args([
            "--dataset-version", "1.5.1",
            "--dataset-variant", "reactive_jpy_v1_core",
            "--profile", "smoke",
        ])
        assert args.profile_name == "smoke"


# ===========================================================================
# PR5.1 — Report structure: primary section + appendix
# ===========================================================================

class TestReportStructure:
    def test_report_has_executive_summary(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)
        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        assert "## Executive Summary" in report

    def test_report_has_scientific_findings(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)
        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        assert "## Scientific Findings" in report

    def test_report_has_appendix(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)
        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        assert "## Appendix" in report

    def test_executive_summary_appears_before_appendix(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out")
        suite.run_suite(args)
        report = (tmp_path / "out" / "exp_test" / "report.md").read_text(encoding="utf-8")
        exec_pos = report.index("## Executive Summary")
        appendix_pos = report.index("## Appendix")
        assert exec_pos < appendix_pos

    def test_profile_name_in_experiment_manifest(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        _write_dataset(repo_root)
        runner, _ = _build_fake_runner(repo_root)
        monkeypatch.setattr(suite, "run_training_command", runner)
        args = _build_args(repo_root, tmp_path / "out", extra=["--profile", "smoke"])
        suite.run_suite(args)
        manifest = json.loads(
            (tmp_path / "out" / "exp_test" / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        # config_payload records the profile under "profile" key
        cli_parsed = manifest["cli"]["parsed"]
        assert cli_parsed.get("profile") == "smoke"
