"""Behavioral Surface Registry — Summary Generator.

Generates a concise human-readable summary across all registered Behavioral
Surfaces.

The name "high_score" is intentional.  The summary is a scientific triage
tool, not a ranking.  It presents the current scientific status of every
registered Behavioral Surface so that researchers can quickly identify where
attention is most needed.

Usage
-----
::

    python analysis/registry/high_score.py

    python analysis/registry/high_score.py --repo-root /path/to/repo

    python analysis/registry/high_score.py --format markdown
    python analysis/registry/high_score.py --format csv

The summary focuses on scientific status rather than numeric metrics.  Surfaces
are listed in alphabetical order; they are explicitly *not* ranked by a numeric
score.  The objective is scientific triage rather than competition.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def _registry_root(repo_root: Path) -> Path:
    return repo_root / "registry" / "surfaces"


def load_all_surfaces(registry_root: Path) -> list[dict]:
    """Load all YAML surface entries from *registry_root*.

    Returns a list of parsed dictionaries, one per surface.  Entries that
    cannot be parsed are reported to stderr and skipped.

    Returns
    -------
    list[dict]
        Parsed surface entries, sorted by surface_id.
    """
    yaml_files = sorted(registry_root.glob("*.yaml"))
    surfaces: list[dict] = []
    for path in yaml_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                print(
                    f"WARNING: {path} is not a YAML mapping — skipped.",
                    file=sys.stderr,
                )
                continue
            # Attach the file path for traceability
            data["_registry_path"] = str(path)
            surfaces.append(data)
        except yaml.YAMLError as exc:
            print(f"WARNING: Cannot parse {path}: {exc} — skipped.", file=sys.stderr)
    return surfaces


def _latest_experiment(surface: dict) -> str:
    """Return the most recently promoted experiment directory, or 'none'."""
    exps: list[dict] = surface.get("supporting_experiments") or []
    if not exps:
        return "none"
    # Experiments are appended in chronological order; the last entry is newest.
    last = exps[-1]
    return str(last.get("experiment_dir", "unknown"))


def _shorten_path(path_str: str, max_len: int = 48) -> str:
    """Shorten a long path string for display."""
    if len(path_str) <= max_len:
        return path_str
    # Keep the last portion after analysis/output/ or similar prefix
    for prefix in ("analysis/output/", "output/"):
        idx = path_str.find(prefix)
        if idx >= 0:
            candidate = "…/" + path_str[idx:]
            if len(candidate) <= max_len:
                return candidate
    return "…" + path_str[-(max_len - 1):]


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

_COLUMNS = [
    "Surface",
    "Lifecycle Stage",
    "Scientific Interest",
    "Scientific Confidence",
    "Current Recommendation",
    "Latest Supporting Experiment",
]


def build_summary_rows(surfaces: list[dict]) -> list[dict]:
    """Build summary rows from *surfaces* for display.

    Parameters
    ----------
    surfaces:
        List of parsed registry YAML dictionaries.

    Returns
    -------
    list[dict]
        One dict per surface with keys matching ``_COLUMNS``.
    """
    rows: list[dict] = []
    for s in surfaces:
        recommendation = str(s.get("current_recommendation") or "").strip()
        # Truncate recommendation to first sentence for table display
        if "." in recommendation:
            recommendation = recommendation[: recommendation.index(".") + 1]
        elif len(recommendation) > 80:
            recommendation = recommendation[:77] + "…"

        rows.append({
            "Surface": s.get("surface_id", "unknown"),
            "Lifecycle Stage": s.get("lifecycle_stage", "unknown"),
            "Scientific Interest": s.get("scientific_interest", "unknown"),
            "Scientific Confidence": s.get("scientific_confidence", "unknown"),
            "Current Recommendation": recommendation,
            "Latest Supporting Experiment": _shorten_path(_latest_experiment(s)),
        })
    return rows


def _format_table_markdown(rows: list[dict]) -> str:
    """Render *rows* as a Markdown table."""
    if not rows:
        return "No surfaces registered.\n"

    # Column widths
    widths: dict[str, int] = {col: len(col) for col in _COLUMNS}
    for row in rows:
        for col in _COLUMNS:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    def _pad(text: str, width: int) -> str:
        return str(text).ljust(width)

    header = " | ".join(_pad(col, widths[col]) for col in _COLUMNS)
    separator = " | ".join("-" * widths[col] for col in _COLUMNS)
    lines = [f"| {header} |", f"| {separator} |"]
    for row in rows:
        cells = " | ".join(_pad(str(row.get(col, "")), widths[col]) for col in _COLUMNS)
        lines.append(f"| {cells} |")
    return "\n".join(lines) + "\n"


def _format_table_csv(rows: list[dict]) -> str:
    """Render *rows* as CSV."""
    if not rows:
        return ",".join(_COLUMNS) + "\n"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _format_table_text(rows: list[dict]) -> str:
    """Render *rows* as aligned plain text."""
    if not rows:
        return "No surfaces registered.\n"
    widths: dict[str, int] = {col: len(col) for col in _COLUMNS}
    for row in rows:
        for col in _COLUMNS:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    def _pad(text: str, width: int) -> str:
        return str(text).ljust(width)

    header = "  ".join(_pad(col, widths[col]) for col in _COLUMNS)
    separator = "  ".join("-" * widths[col] for col in _COLUMNS)
    lines = [header, separator]
    for row in rows:
        cells = "  ".join(_pad(str(row.get(col, "")), widths[col]) for col in _COLUMNS)
        lines.append(cells)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(
    *,
    repo_root: Path | None = None,
    fmt: str = "markdown",
) -> str:
    """Generate and return the registry summary as a string.

    Parameters
    ----------
    repo_root:
        Repository root directory.  Defaults to three levels above this script.
    fmt:
        Output format: ``"markdown"`` (default), ``"csv"``, or ``"text"``.

    Returns
    -------
    str
        Formatted summary string.
    """
    resolved_root = repo_root or Path(__file__).parent.parent.parent
    registry_root = _registry_root(resolved_root)

    if not registry_root.is_dir():
        return f"Registry directory not found: {registry_root}\n"

    surfaces = load_all_surfaces(registry_root)
    rows = build_summary_rows(surfaces)

    header_lines = [
        "# Behavioral Surface Registry — Scientific Status Summary",
        "",
        "Scientific triage summary across all registered Behavioral Surfaces.",
        "Surfaces are listed alphabetically.  This summary is not a ranking.",
        "",
    ]
    if not rows:
        return "\n".join(header_lines) + "\nNo surfaces registered.\n"

    header = "\n".join(header_lines)
    if fmt == "csv":
        return _format_table_csv(rows)
    elif fmt == "text":
        return header + _format_table_text(rows)
    else:
        return header + _format_table_markdown(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="high_score.py",
        description=(
            "Generate a concise human-readable summary of the Behavioral Surface Registry.\n\n"
            "The objective is scientific triage rather than numeric ranking.  Surfaces are\n"
            "listed alphabetically and are not scored against each other."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root directory.  Defaults to three levels above this script.",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        default="markdown",
        choices=["markdown", "csv", "text"],
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write output to this file instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else None
    summary = generate_summary(repo_root=repo_root, fmt=args.fmt)

    if args.output:
        Path(args.output).write_text(summary, encoding="utf-8")
        print(f"Registry summary written to {args.output}")
    else:
        print(summary, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
