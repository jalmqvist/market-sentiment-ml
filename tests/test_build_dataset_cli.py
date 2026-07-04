from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.build_dataset as dataset_cli


def _write_canonical_datasets(version_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "pair": ["eur-usd"],
            "entry_time": ["2024-01-01 00:00:00"],
            "ret_48b": [0.2],
            "vol_48b": [0.1],
        }
    )
    for name in (
        "master_research_dataset.csv",
        "master_research_dataset_core.csv",
        "master_research_dataset_extended.csv",
    ):
        df.to_csv(version_dir / name, index=False)


@pytest.fixture
def _patch_common(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dataset_cli.cfg, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(dataset_cli.cfg, "HORIZONS", [48])
    monkeypatch.setattr(dataset_cli, "setup_experiment_logging", lambda **_: None)


def test_augment_only_requires_behavioral_surface(_patch_common) -> None:
    with pytest.raises(ValueError, match="--augment-only requires --behavioral-surface"):
        dataset_cli.main(["--version", "1.5.1", "--augment-only", "--no-log-file"])


def test_augment_only_fails_when_canonical_dataset_missing(
    _patch_common,
    tmp_path: Path,
) -> None:
    version_dir = tmp_path / "1.5.1"
    version_dir.mkdir(parents=True)
    (version_dir / "master_research_dataset.csv").write_text("pair\n", encoding="utf-8")
    (version_dir / "master_research_dataset_core.csv").write_text("pair\n", encoding="utf-8")
    surface_path = tmp_path / "surface.parquet"
    surface_path.write_bytes(b"not-used")

    with pytest.raises(FileNotFoundError, match="master_research_dataset_extended.csv"):
        dataset_cli.main(
            [
                "--version",
                "1.5.1",
                "--augment-only",
                "--behavioral-surface",
                str(surface_path),
                "--no-log-file",
            ]
        )


def test_augment_only_runs_without_rebuilding_canonical(
    _patch_common,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_dir = tmp_path / "1.5.1"
    version_dir.mkdir(parents=True)
    _write_canonical_datasets(version_dir)
    manifest_path = version_dir / "DATASET_MANIFEST.json"
    manifest_path.write_text("{}", encoding="utf-8")
    surface_path = tmp_path / "surface.parquet"
    surface_path.write_bytes(b"not-used")

    before_full = (version_dir / "master_research_dataset.csv").read_text(encoding="utf-8")

    import scripts.build_fx_sentiment_dataset as builder
    import bsve.dataset_augmentation as augmentation

    def _unexpected_build(**kwargs):  # pragma: no cover - assertion guard
        raise AssertionError("build_master_dataset must not be called in --augment-only mode")

    calls: list[dict] = []

    def _fake_augment(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(builder, "build_master_dataset", _unexpected_build)
    monkeypatch.setattr(augmentation, "run_behavioral_augmentation", _fake_augment)

    dataset_cli.main(
        [
            "--version",
            "1.5.1",
            "--augment-only",
            "--behavioral-surface",
            str(surface_path),
            "--no-log-file",
        ]
    )

    assert len(calls) == 1
    assert set(calls[0]["base_dataset_paths"]) == {"full", "core", "extended"}
    assert calls[0]["base_manifest_path"] == manifest_path
    after_full = (version_dir / "master_research_dataset.csv").read_text(encoding="utf-8")
    assert after_full == before_full


def test_existing_canonical_requires_force(
    _patch_common,
    tmp_path: Path,
) -> None:
    version_dir = tmp_path / "1.5.1"
    version_dir.mkdir(parents=True)
    (version_dir / "master_research_dataset.csv").write_text("pair\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        dataset_cli.main(["--version", "1.5.1", "--no-log-file"])


def test_non_augment_only_build_path_still_runs(
    _patch_common,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "1.5.1"
    version_dir = tmp_path / version
    version_dir.mkdir(parents=True)

    import scripts.build_fx_sentiment_dataset as builder
    import scripts.build_dataset_vol as vol_module

    build_calls: list[dict] = []

    def _fake_build_master_dataset(**kwargs):
        build_calls.append(kwargs)
        _write_canonical_datasets(version_dir)
        return pd.DataFrame()

    def _fake_add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            trend_vol_adj_strength=0.0,
            is_trending=False,
            is_high_vol=False,
        )

    def _fake_add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(vol_12b=0.05, vol_48b=0.10)

    monkeypatch.setattr(builder, "build_master_dataset", _fake_build_master_dataset)
    monkeypatch.setattr(builder, "add_regime_features", _fake_add_regime_features)
    monkeypatch.setattr(vol_module, "add_volatility_features", _fake_add_volatility_features)

    dataset_cli.main(["--version", version, "--force", "--no-log-file"])

    assert len(build_calls) == 1
    out_df = pd.read_csv(version_dir / "master_research_dataset_core.csv")
    assert "target_cls" in out_df.columns
