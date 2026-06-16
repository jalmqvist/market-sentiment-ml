"""Report utilities for BSVE criterion validation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_validation_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Write deterministic Criterion validation JSON report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bsve_validation_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path
