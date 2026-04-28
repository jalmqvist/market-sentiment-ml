# Legacy experiment — not part of current validated approach\n"""
run_regime_v4_sweep.py
======================
Entry-point script for the Regime V4 Robustness Sweep.

All sweep logic lives in ``experiments/regime_v4_sweep.py``; this script is
a thin launcher that configures logging (file-only by default, no stdout) and
delegates to ``experiments.regime_v4_sweep.main``.

What this sweep does
--------------------
Runs a grid search over key parameters of the Signal V2 × Regime Filter
hybrid pipeline (Regime V4 Signal Filter) and evaluates:

* mean Sharpe across walk-forward folds
* Sharpe stability (std across folds)
* mean coverage
* capacity-adjusted Sharpe = mean_sharpe × sqrt(mean_coverage)

The goal is to identify **flat, stable regions** in parameter space, NOT
optimal peaks.

Parameter grid
--------------
``min_n``               : [50, 100, 150, 200]
``min_sharpe``          : [0.0, 0.05, 0.1]
``direction_threshold`` : [0.0002, 0.0005, 0.001]
``threshold``           : [None, 0.5, 1.0]  (Signal V2 position threshold)
``with_direction``      : [True, False]

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Default: log to auto-created timestamped file
    python run_regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv

    # Verbose logging
    python run_regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG

    # Custom log file
    python run_regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/my_sweep.log

    # Quick debugging run (first 10 configs only)
    python run_regime_v4_sweep.py \\
        --data data/output/master_research_dataset.csv \\
        --max-runs 10
"""

from __future__ import annotations

import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from experiments.regime_v4_sweep import main  # noqa: E402

if __name__ == "__main__":
    main()
