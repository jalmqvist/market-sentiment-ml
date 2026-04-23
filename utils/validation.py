"""
utils/validation.py
===================
Schema checks, empty DataFrame guards, and timestamp normalization helpers.

These utilities are used by every stage of the pipeline to fail fast on
bad data rather than propagating NaNs or empty results silently.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column existence checks
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, columns: list[str], context: str = "") -> None:
    """Raise ``ValueError`` if any of *columns* are absent from *df*.

    Args:
        df: DataFrame to inspect.
        columns: Required column names.
        context: Short description of the call site (included in the error
                 message for easier debugging).

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        ctx = f" [{context}]" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {missing}")


def warn_missing_columns(
    df: pd.DataFrame, columns: list[str], context: str = ""
) -> list[str]:
    """Log a warning for each column in *columns* that is absent from *df*.

    Returns the list of missing column names (may be empty).
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        ctx = f" [{context}]" if context else ""
        logger.warning("Missing columns%s: %s", ctx, missing)
    return missing


# ---------------------------------------------------------------------------
# Empty DataFrame checks
# ---------------------------------------------------------------------------

def require_non_empty(df: pd.DataFrame, context: str = "") -> None:
    """Raise ``ValueError`` if *df* is empty.

    Args:
        df: DataFrame to check.
        context: Description of what *df* represents.

    Raises:
        ValueError: If the DataFrame has zero rows.
    """
    if df.empty:
        ctx = f" [{context}]" if context else ""
        raise ValueError(f"Empty DataFrame encountered{ctx}; cannot continue.")


def warn_if_empty(df: pd.DataFrame, context: str = "") -> bool:
    """Log a warning and return ``True`` if *df* is empty; otherwise return
    ``False``.
    """
    if df.empty:
        ctx = f" [{context}]" if context else ""
        logger.warning("Empty DataFrame%s; results may be missing.", ctx)
        return True
    return False


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------

def ensure_utc(series: pd.Series, context: str = "") -> pd.Series:
    """Return *series* coerced to tz-aware UTC datetimes.

    - If the series is tz-naive it is localised to UTC.
    - If it is already tz-aware it is converted to UTC.
    - Non-parseable values are coerced to NaT.

    Args:
        series: A pandas Series of datetime-like values.
        context: Description used in log messages.

    Returns:
        A new Series with UTC-aware datetimes.
    """
    ctx = f" [{context}]" if context else ""
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if s.dt.tz is None:
        logger.debug("Localising tz-naive timestamps to UTC%s", ctx)
        s = s.dt.tz_localize("UTC")
    else:
        logger.debug("Converting timestamps to UTC%s", ctx)
        s = s.dt.tz_convert("UTC")
    n_nat = int(s.isna().sum())
    if n_nat:
        logger.warning("%d NaT values after timestamp parse%s", n_nat, ctx)
    return s


def parse_timestamps(
    df: pd.DataFrame,
    col: str,
    *,
    utc: bool = False,
    format: Optional[str] = None,
    context: str = "",
) -> pd.DataFrame:
    """Parse *col* in *df* as datetimes in-place (returns a copy).

    Args:
        df: Input DataFrame.
        col: Column name to parse.
        utc: If ``True``, ensure the result is tz-aware UTC.
        format: Optional strftime format string.  If ``None``, uses
                ``format="mixed"`` (pandas ≥ 2.0) for robust parsing.
        context: Description used in log messages.

    Returns:
        A copy of *df* with *col* replaced by parsed datetimes.
    """
    require_columns(df, [col], context=context or f"parse_timestamps({col})")
    df = df.copy()
    kw: dict = {"errors": "coerce"}
    if format is not None:
        kw["format"] = format
    else:
        kw["format"] = "mixed"
    df[col] = pd.to_datetime(df[col], **kw)
    if utc:
        df[col] = ensure_utc(df[col], context=context)
    n_nat = int(df[col].isna().sum())
    if n_nat:
        logger.warning(
            "%d NaT values in column '%s'%s",
            n_nat,
            col,
            f" [{context}]" if context else "",
        )
    return df


# ---------------------------------------------------------------------------
# DataFrame shape / stats logging helpers
# ---------------------------------------------------------------------------

def log_shape(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Log the shape and column list of *df* at DEBUG level."""
    logger.debug("%s: %d rows × %d cols", label, len(df), len(df.columns))
