"""
run_signal_v2.py
================
Entry-point script for the Signal V2 experiment (sentiment divergence,
shock, and exhaustion).

All feature construction, signal computation, and walk-forward logic lives in
``experiments/signal_v2.py``; this script is a thin launcher that configures
logging (file-only by default, no stdout) and delegates to
``experiments.signal_v2.main``.

Pipeline steps
--------------
1. Load and prepare the research dataset.
2. Compute rolling z-score features (divergence, shock, exhaustion) per pair
   using only past data (strict no-leakage).
3. Evaluate walk-forward signal performance via an expanding window:

   * ``train_df = df[df.year < test_year]``  (used for logging context only)
   * ``test_df  = df[df.year == test_year]``  (signal applied here)

4. Log per-fold diagnostics and a pooled summary.

Logging rules
-------------
* File logging is **ON by default**.  A timestamped log file is created in
  ``logs/`` unless ``--log-file`` is specified explicitly.
* When a log file is used, **no output is written to stdout**.

Usage::

    # Default (auto-timestamped log file, window=96, no threshold)
    python run_signal_v2.py --data data/output/master_research_dataset.csv

    # Custom window and threshold
    python run_signal_v2.py \\
        --data data/output/master_research_dataset.csv \\
        --window 96 --threshold 0.5

    # Explicit log file
    python run_signal_v2.py \\
        --data data/output/master_research_dataset.csv \\
        --log-file logs/signal_v2.log

    # Verbose mode
    python run_signal_v2.py \\
        --data data/output/master_research_dataset.csv \\
        --log-level DEBUG
"""

from experiments.signal_v2 import main

if __name__ == "__main__":
    main()
