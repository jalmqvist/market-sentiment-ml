"""
utils/io.py
===========
Standardized I/O helpers for the market-sentiment-ml pipeline.

Provides:
- Logging setup (call ``setup_logging`` once at program entry)
- CSV / Parquet read helpers with debug logging
- Safe output writers that create parent directories automatically
- Path resolution relative to project root
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger.

    Only initializes logging if no handlers are already attached to the root
    logger, preventing double initialization when called multiple times (e.g.
    in tests or notebooks).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).  Defaults to
               ``config.LOG_LEVEL`` which itself falls back to the
               ``LOG_LEVEL`` environment variable or ``"INFO"``.
    """
    level = level or cfg.LOG_LEVEL
    numeric = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        return

    logging.basicConfig(
        stream=sys.stdout,
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.debug("Logging initialised at level %s", level.upper())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(p: str | Path) -> Path:
    """Return an absolute Path.  Relative paths are resolved relative to
    the repository root (``config.REPO_ROOT``).
    """
    p = Path(p)
    if not p.is_absolute():
        p = cfg.REPO_ROOT / p
    return p


def ensure_parent(p: str | Path) -> Path:
    """Create parent directories for *p* and return ``Path(p)``."""
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_csv(
    path: str | Path,
    *,
    parse_dates: Optional[list[str]] = None,
    required_columns: Optional[list[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame.

    Args:
        path: Path to CSV file.
        parse_dates: Column names to parse as dates.
        required_columns: If provided, raise ``ValueError`` if any of these
                          columns are missing after loading.
        **kwargs: Forwarded to ``pandas.read_csv``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are absent.
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.debug("Reading CSV: %s", path)
    df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    logger.debug("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)

    if required_columns:
        _check_required_columns(df, required_columns, context=str(path))

    return df


# ---------------------------------------------------------------------------
# Parquet readers
# ---------------------------------------------------------------------------

def read_parquet(
    path: str | Path,
    *,
    required_columns: Optional[list[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a Parquet file and return a DataFrame.

    Args:
        path: Path to Parquet file.
        required_columns: If provided, raise ``ValueError`` if any of these
                          columns are missing after loading.
        **kwargs: Forwarded to ``pandas.read_parquet``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are absent.
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.debug("Reading Parquet: %s", path)
    df = pd.read_parquet(path, **kwargs)
    logger.debug("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)

    if required_columns:
        _check_required_columns(df, required_columns, context=str(path))

    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write *df* to CSV at *path*, creating parent directories as needed.

    Returns the resolved Path that was written.
    """
    path = ensure_parent(path)
    logger.debug("Writing CSV (%d rows) to %s", len(df), path)
    df.to_csv(path, index=False, **kwargs)
    logger.info("Saved CSV: %s  rows=%d", path, len(df))
    return path


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write *df* to Parquet at *path*, creating parent directories as needed.

    Returns the resolved Path that was written.
    """
    path = ensure_parent(path)
    logger.debug("Writing Parquet (%d rows) to %s", len(df), path)
    df.to_parquet(path, index=False, **kwargs)
    logger.info("Saved Parquet: %s  rows=%d", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required_columns(
    df: pd.DataFrame, columns: list[str], context: str = ""
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        ctx = f" [{context}]" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {missing}")
