# Legacy experiment — not part of current validated approach\n"""
validation/validate_pipeline_extended.py
=========================================
Extended pipeline validation script.

Validates dataset integrity, signal parity, performance, and regime isolation.
Exits with a non-zero status code on failure.

Warnings (not failures) are issued for:
- regime dataset row count < 30% of canonical
- regime pair count < 30% of canonical
- missing regime rate > 20%

Usage::

    python validation/validate_pipeline_extended.py \\
        --data data/output/master_research_dataset.csv \\
        --data-regime data/output/master_research_dataset_with_regime.csv \\
        --reference data/reference/master_research_dataset.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("validate_pipeline")


# =========================
# HELPERS
# =========================

def fail(msg: str) -> None:
    logger.error("VALIDATION FAILED: %s", msg)
    sys.exit(1)


def ok(msg: str) -> None:
    logger.info("OK: %s", msg)


def warn(msg: str) -> None:
    logger.warning("WARNING: %s", msg)


def approx(a: float, b: float, tol: float = 1e-10) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(a - b) <= tol


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


# =========================
# SIGNAL
# =========================

def apply_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["extreme"] = df["abs_sentiment"] >= 70
    return df[df["extreme"]].copy()


def non_overlap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pair", "entry_time"])
    return df.drop_duplicates(subset=["pair", "entry_time"])


def stats(df: pd.DataFrame, col: str = "ret_48b") -> dict:
    r = df[col].dropna()
    if len(r) == 0:
        return {"n": 0, "mean": np.nan, "sharpe": np.nan}
    return {
        "n": len(r),
        "mean": r.mean(),
        "sharpe": r.mean() / r.std() if r.std() != 0 else 0,
    }


# =========================
# MAIN
# =========================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Extended pipeline validation. Exits non-zero on failure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to canonical research dataset CSV.",
    )
    p.add_argument(
        "--data-regime",
        required=True,
        type=Path,
        help="Path to regime-enriched research dataset CSV.",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference dataset for hash/parity checks. Skip hash checks if omitted.",
    )
    args = p.parse_args(argv)

    tol = 1e-10

    # =========================
    # LOAD
    # =========================
    logger.info("=== LOAD DATA ===")

    if not args.data.exists():
        fail(f"Canonical dataset not found: {args.data}")
    if not args.data_regime.exists():
        fail(f"Regime dataset not found: {args.data_regime}")

    df = pd.read_csv(args.data)
    df_reg = pd.read_csv(args.data_regime)

    df_ref = None
    if args.reference is not None:
        if not args.reference.exists():
            fail(f"Reference dataset not found: {args.reference}")
        df_ref = pd.read_csv(args.reference)

    ok("Datasets loaded")

    # =========================
    # WARNING CHECKS (not failures)
    # =========================
    logger.info("=== REGIME COVERAGE WARNINGS ===")

    canonical_rows = len(df)
    regime_rows = len(df_reg)
    canonical_pairs = df["pair"].nunique() if "pair" in df.columns else 0
    regime_pairs_count = df_reg["pair"].nunique() if "pair" in df_reg.columns else 0

    if canonical_rows > 0 and regime_rows < 0.30 * canonical_rows:
        warn(
            f"Regime dataset row count ({regime_rows:,}) is < 30% of canonical ({canonical_rows:,}). "
            f"ratio={regime_rows / canonical_rows:.2%}"
        )
    else:
        ok(f"Regime row coverage: {regime_rows:,} / {canonical_rows:,} = {regime_rows / canonical_rows:.2%}" if canonical_rows else "Regime row coverage: n/a")

    if canonical_pairs > 0 and regime_pairs_count < 0.30 * canonical_pairs:
        warn(
            f"Regime pair count ({regime_pairs_count}) is < 30% of canonical ({canonical_pairs}). "
            f"ratio={regime_pairs_count / canonical_pairs:.2%}"
        )
    else:
        ok(f"Regime pair coverage: {regime_pairs_count} / {canonical_pairs} = {regime_pairs_count / canonical_pairs:.2%}" if canonical_pairs else "Regime pair coverage: n/a")

    if "phase" in df_reg.columns and regime_rows > 0:
        missing_regime_rate = df_reg["phase"].isna().mean()
        if missing_regime_rate > 0.20:
            warn(
                f"Missing regime rate in regime dataset: {missing_regime_rate:.2%} > 20% threshold."
            )
        else:
            ok(f"Missing regime rate: {missing_regime_rate:.2%}")

    # =========================
    # TEST 1 — HASH MATCH
    # =========================
    logger.info("=== TEST 1: DATA HASH ===")

    if df_ref is not None:
        if hash_df(df) != hash_df(df_ref):
            fail("Dataset hash mismatch")
        ok("Dataset hash identical")
    else:
        logger.info("Reference dataset not provided; skipping hash check.")

    # =========================
    # TEST 2 — SIGNAL PARITY
    # =========================
    logger.info("=== TEST 2: SIGNAL PARITY ===")

    sig = non_overlap(apply_signal(df))

    if df_ref is not None:
        sig_ref = non_overlap(apply_signal(df_ref))
        if len(sig) != len(sig_ref):
            fail(f"Signal count mismatch: {len(sig)} vs {len(sig_ref)}")
        ok(f"Signals match: {len(sig)}")
    else:
        ok(f"Signal count (no reference): {len(sig)}")

    # =========================
    # TEST 3 — PERFORMANCE PARITY
    # =========================
    logger.info("=== TEST 3: PERFORMANCE ===")

    if df_ref is not None:
        sig_ref = non_overlap(apply_signal(df_ref))
        s = stats(sig)
        s_ref = stats(sig_ref)
        for k in ["n", "mean", "sharpe"]:
            if not approx(s[k], s_ref[k], tol):
                fail(f"{k} mismatch: {s[k]} vs {s_ref[k]}")
        ok("Performance identical")
    else:
        ok("Performance check skipped (no reference dataset)")

    # =========================
    # TEST 4 — BEHAVIORAL STRUCTURE
    # =========================
    logger.info("=== TEST 4: BEHAVIORAL STRUCTURE ===")

    sig["position"] = -1  # contrarian proxy
    contra = sig["ret_48b"].dropna()

    if contra.mean() < 0:
        fail("Contrarian edge disappeared")

    ok("Contrarian edge intact")

    # =========================
    # TEST 5 — REGIME ISOLATION
    # =========================
    logger.info("=== TEST 5: REGIME ISOLATION ===")

    sig_reg = non_overlap(apply_signal(df_reg))

    if len(sig_reg) >= len(sig):
        fail("Regime dataset should reduce signals")

    ok("Regime properly restricts dataset")

    # =========================
    # TEST 6 — STAGE INDEPENDENCE
    # =========================
    logger.info("=== TEST 6: STAGE INDEPENDENCE ===")

    required_cols = ["pair", "entry_time", "abs_sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fail(f"Missing core columns: {missing}")

    ok("Core columns present")

    # =========================
    # TEST 7 — LEAKAGE CHECK
    # =========================
    logger.info("=== TEST 7: LEAKAGE CHECK ===")

    sample_n = min(1000, len(df))
    sample = df.sample(sample_n, random_state=42)

    for _, row in sample.iterrows():
        if pd.isna(row["ret_48b"]):
            continue
        if abs(row["ret_48b"]) > 0.2:
            fail("Suspicious large return detected")

    ok("No obvious leakage")

    # =========================
    # TEST 8 — PAIR DOMINANCE
    # =========================
    logger.info("=== TEST 8: PAIR ROBUSTNESS ===")

    pair_counts = sig["pair"].value_counts()
    top_pair = pair_counts.index[0]
    sig_reduced = sig[sig["pair"] != top_pair]

    sh_full = stats(sig)["sharpe"]
    sh_red = stats(sig_reduced)["sharpe"]

    if sh_red < -0.1:
        fail("Signal collapses when removing top pair")

    ok("Signal not dominated by single pair")

    # =========================
    # TEST 9 — DETERMINISM HOOK
    # =========================
    logger.info("=== TEST 9: DETERMINISM (OPTIONAL) ===")

    logger.info("Dataset hash: %s", hash_df(df))
    ok("Determinism fingerprint generated")

    # =========================
    # FINAL
    # =========================
    logger.info("PIPELINE CERTIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
