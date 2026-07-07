from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

import config as cfg


@dataclass
class RunResult:
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str
    duration_seconds: float
    log_path: Path
    # Trainer-reported artifact paths (populated when the trainer emits
    # "artifact_parquet: <path>" / "artifact_manifest: <path>" log lines).
    reported_parquet_path: Path | None = field(default=None)
    reported_manifest_path: Path | None = field(default=None)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def resolve_dataset_csv_path(
    dataset_version: str,
    dataset_variant: str,
    output_dir: Path | None = None,
) -> Path:
    base = output_dir or cfg.OUTPUT_DIR
    suffix = "" if dataset_variant == "full" else f"_{dataset_variant}"
    return Path(base) / dataset_version / f"master_research_dataset{suffix}.csv"


def load_dataset_for_suite(
    dataset_version: str,
    dataset_variant: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    dataset_path = resolve_dataset_csv_path(dataset_version, dataset_variant, output_dir=output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    for col in ["timestamp", "snapshot_time", "entry_time", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


def discover_behavioral_states(
    df: pd.DataFrame,
    selected_surface_id: str | None = None,
) -> list[dict[str, str]]:
    required = {"surface_id", "state_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required behavioral columns: {sorted(missing)}")

    candidates = df[["surface_id", "state_id"]].dropna().drop_duplicates()
    if selected_surface_id is not None:
        candidates = candidates[candidates["surface_id"] == selected_surface_id]

    discovered = [
        {"surface_id": str(row.surface_id), "state_id": str(row.state_id)}
        for row in candidates.sort_values(["surface_id", "state_id"]).itertuples(index=False)
    ]
    if not discovered:
        if selected_surface_id:
            raise ValueError(
                f"No behavioral states found for surface_id={selected_surface_id!r} in selected dataset"
            )
        raise ValueError("No behavioral states found in selected dataset")
    return discovered


def select_models(selection: str) -> list[str]:
    raw = [part.strip().lower() for part in selection.split(",") if part.strip()]
    if not raw:
        raise ValueError("--models must select at least one model")
    allowed = {"mlp", "lstm", "both"}
    invalid = [m for m in raw if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")
    if "both" in raw:
        return ["mlp", "lstm"]
    ordered = []
    for model in raw:
        if model not in ordered:
            ordered.append(model)
    return ordered


def build_training_command(
    *,
    trainer: str,
    dataset_version: str,
    dataset_variant: str,
    surface_id: str,
    state_id: str,
    feature_set: str,
    target_horizon: int,
    epochs: int,
    hidden_dim: int,
    lr: float,
    label_quantile: float,
    seq_len: int,
    train_pairs: str | None,
    predict_pairs: str | None,
    export_split: str,
    walkforward_window: dict[str, str] | None = None,
) -> list[str]:
    if trainer not in {"mlp", "lstm"}:
        raise ValueError(f"Unsupported trainer: {trainer}")

    script = "research/deep_learning/train.py" if trainer == "mlp" else "research/deep_learning/train_lstm.py"
    command = [
        "python",
        script,
        "--dataset-version",
        dataset_version,
        "--dataset-variant",
        dataset_variant,
        "--feature-set",
        feature_set,
        "--surface",
        surface_id,
        "--state",
        state_id,
        "--target-horizon",
        str(target_horizon),
        "--label-quantile",
        str(label_quantile),
        "--epochs",
        str(epochs),
        "--hidden-dim",
        str(hidden_dim),
        "--lr",
        str(lr),
        "--export-split",
        export_split,
    ]

    if train_pairs:
        command.extend(["--train-pairs", train_pairs])
    if predict_pairs:
        command.extend(["--predict-pairs", predict_pairs])
    if trainer == "lstm":
        command.extend(["--seq-len", str(seq_len)])
    if walkforward_window:
        for flag, key in [
            ("--wf-train-start", "train_start"),
            ("--wf-train-end", "train_end"),
            ("--wf-test-start", "test_start"),
            ("--wf-test-end", "test_end"),
        ]:
            value = walkforward_window.get(key)
            if value:
                command.extend([flag, str(value)])

    return command


_ARTIFACT_PARQUET_RE = re.compile(r"artifact_parquet:\s*(\S+)")
_ARTIFACT_MANIFEST_RE = re.compile(r"artifact_manifest:\s*(\S+)")


def parse_reported_artifact_paths(
    output: str,
) -> tuple[Path | None, Path | None]:
    """Extract trainer-reported artifact paths from combined stdout/stderr.

    Trainers emit::

        logging.info("artifact_parquet: %s", pq_path)
        logging.info("artifact_manifest: %s", mf_path)

    This parser finds those lines and returns resolved ``Path`` objects when
    the reported files exist on disk.
    """
    parquet_path: Path | None = None
    manifest_path: Path | None = None
    for line in output.splitlines():
        m = _ARTIFACT_PARQUET_RE.search(line)
        if m and parquet_path is None:
            candidate = Path(m.group(1))
            if candidate.exists():
                parquet_path = candidate.resolve()
        m = _ARTIFACT_MANIFEST_RE.search(line)
        if m and manifest_path is None:
            candidate = Path(m.group(1))
            if candidate.exists():
                manifest_path = candidate.resolve()
    return parquet_path, manifest_path


def run_training_command(
    *,
    command: list[str],
    repo_root: Path,
    log_path: Path,
) -> RunResult:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = datetime.now(timezone.utc)

    combined_output = "\n".join(
        [
            f"$ {' '.join(command)}",
            "\n[stdout]\n",
            completed.stdout,
            "\n[stderr]\n",
            completed.stderr,
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined_output, encoding="utf-8")

    reported_parquet, reported_manifest = parse_reported_artifact_paths(combined_output)

    return RunResult(
        command=command,
        returncode=completed.returncode,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=(finished - started).total_seconds(),
        log_path=log_path,
        reported_parquet_path=reported_parquet,
        reported_manifest_path=reported_manifest,
    )


def snapshot_files(directory: Path, pattern: str) -> set[Path]:
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.glob(pattern)}


def diff_new_files(before: set[Path], after: set[Path]) -> list[Path]:
    return sorted(after - before)


def copy_files(files: Iterable[Path], destination: Path) -> list[Path]:
    copied: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = destination / source.name
        target.write_bytes(source.read_bytes())
        copied.append(target)
    return copied


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_commit(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return proc.stdout.strip() or None
