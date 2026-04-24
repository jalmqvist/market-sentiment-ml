"""
run_regime_v4_signal_filter.py
================================
Entry-point script for the Signal V2 × Regime Filter hybrid pipeline.

All walk-forward and regime-selection logic lives in
``experiments/regime_v4_signal_filter.py``; this script is a thin launcher
that configures logging (file-only by default, no stdout) and delegates to
``experiments.regime_v4_signal_filter.main``.

Pipeline overview
-----------------
1. Load and prepare the research dataset.
2. Compute causal volatility feature (``vol_24b``) via rolling entry_close
   returns within each pair.
3. Build Signal V2 causal features (divergence, shock, exhaustion) and
   positions (``position = sign(signal_v2_raw)``).
4. For each walk-forward fold:

   a. Compute training-derived quantile cuts for ``vol_regime`` and
      ``trend_strength_bin``.
   b. Build a 4-component regime key per row.
   c. Compute per-regime signal-weighted return statistics (train only):
      ``mean_return = mean(position × ret_48b)``,
      ``sharpe = mean_return / std(position × ret_48b)``.
   d. Select regimes: ``n >= min_n`` **and** ``sharpe >= min_sharpe``.
   e. Apply filter to test set (zero out unselected regimes).
   f. Optionally apply direction modification:
      * ``mean_return > direction_threshold`` → keep
      * ``mean_return < -direction_threshold`` → flip
      * otherwise → zero out

5. Compute per-fold metrics (n, mean, Sharpe, hit rate, coverage,
   n_selected_regimes).
6. Log pooled summary (mean Sharpe, mean coverage).

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Default: log to auto-created timestamped file
    python run_regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv

    # Custom min-n and min-sharpe
    python run_regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv \\
        --min-n 150 --min-sharpe 0.1

    # Enable direction modification
    python run_regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv \\
        --with-direction --direction-threshold 0.0003

    # Signal V2 with threshold + verbose logging
    python run_regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv \\
        --threshold 0.5 --log-level DEBUG \\
        --log-file logs/regime_v4_signal_filter_debug.log

    # Disable direction modification explicitly (the default)
    python run_regime_v4_signal_filter.py \\
        --data data/output/master_research_dataset.csv \\
        --no-direction
"""

from __future__ import annotations

import sys
from pathlib import Path

# Safe repo-root sys.path shim for direct execution
if __package__ is None or __package__ == "":
    _repo_root = Path(__file__).resolve().parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from experiments.regime_v4_signal_filter import main  # noqa: E402

if __name__ == "__main__":
    main()
