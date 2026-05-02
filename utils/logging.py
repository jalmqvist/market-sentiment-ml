"""
utils/logging.py
================
Standardized experiment logging setup for ABM, MLP, and LSTM experiments.

Provides a single entry-point, ``setup_experiment_logging``, that every
experiment script must call once at startup.  It:

- Clears any existing root-logger handlers (prevents duplication in tests).
- Optionally attaches a ``FileHandler`` writing full logs to
  ``logs/{experiment_type}_{tag}_{timestamp}.log``.
- Always attaches a ``StreamHandler`` at INFO level for minimal stdout output.
- Returns the log-file ``Path`` (or ``None`` when file logging is disabled).

The returned path can be used to derive the matching config JSON path via
``log_file.with_suffix('.json')``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config as cfg

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_experiment_logging(
    experiment_type: str,
    tag: str,
    log_level: str = "INFO",
    no_log_file: bool = False,
    log_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Configure root logger for an experiment run.

    Args:
        experiment_type: One of ``abm``, ``mlp``, or ``lstm``.
        tag:             Short identifier appended to the log filename
                         (e.g. pair name, feature-set slug).
        log_level:       Root log level string (``DEBUG``, ``INFO``, …).
                         Defaults to ``"INFO"``.
        no_log_file:     When ``True``, skip file logging and write to
                         stdout only.
        log_dir:         Directory for log files.  Defaults to
                         ``<repo_root>/logs``.  Override in tests or when
                         running from a non-standard working directory.

    Returns:
        Path to the created log file, or ``None`` if *no_log_file* is set.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    log_file: Optional[Path] = None

    if not no_log_file:
        resolved_log_dir = Path(log_dir) if log_dir is not None else cfg.REPO_ROOT / "logs"
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = resolved_log_dir / f"{experiment_type}_{tag}_{timestamp}.log"

        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    return log_file
