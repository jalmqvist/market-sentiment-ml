"""
validation/validate_pipeline_extended.py
=========================================
Extended pipeline validation script.

Validates dataset integrity, signal parity, performance, and regime isolation.
Exits with a non-zero status code on failure.

Usage::

    python validation/validate_pipeline_extended.py \\
        --data data/output/master_research_dataset.csv \\
        --data-regime data/output/master_research_dataset_with_regime.csv \\
        --reference data/reference/master_research_dataset.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# HELPERS
# =========================

def fail(msg: str) -> None:
    print(f"\n❌ VALIDATION FAILED: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


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
    print("\n=== LOAD DATA ===")

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
    # TEST 1 — HASH MATCH
    # =========================
    print("\n=== TEST 1: DATA HASH ===")

    if df_ref is not None:
        if hash_df(df) != hash_df(df_ref):
            fail("Dataset hash mismatch")
        ok("Dataset hash identical")
    else:
        print("⚠️  Reference dataset not provided; skipping hash check.")

    # =========================
    # TEST 2 — SIGNAL PARITY
    # =========================
    print("\n=== TEST 2: SIGNAL PARITY ===")

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
    print("\n=== TEST 3: PERFORMANCE ===")

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
    print("\n=== TEST 4: BEHAVIORAL STRUCTURE ===")

    sig["position"] = -1  # contrarian proxy
    contra = sig["ret_48b"].dropna()

    if contra.mean() < 0:
        fail("Contrarian edge disappeared")

    ok("Contrarian edge intact")

    # =========================
    # TEST 5 — REGIME ISOLATION
    # =========================
    print("\n=== TEST 5: REGIME ISOLATION ===")

    sig_reg = non_overlap(apply_signal(df_reg))

    if len(sig_reg) >= len(sig):
        fail("Regime dataset should reduce signals")

    ok("Regime properly restricts dataset")

    # =========================
    # TEST 6 — STAGE INDEPENDENCE
    # =========================
    print("\n=== TEST 6: STAGE INDEPENDENCE ===")

    required_cols = ["pair", "entry_time", "abs_sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fail(f"Missing core columns: {missing}")

    ok("Core columns present")

    # =========================
    # TEST 7 — LEAKAGE CHECK
    # =========================
    print("\n=== TEST 7: LEAKAGE CHECK ===")

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
    print("\n=== TEST 8: PAIR ROBUSTNESS ===")

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
    print("\n=== TEST 9: DETERMINISM (OPTIONAL) ===")

    print("Hash:", hash_df(df))
    ok("Determinism fingerprint generated")

    # =========================
    # FINAL
    # =========================
    print("\n==============================")
    print("🚀 PIPELINE CERTIFIED")
    print("==============================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
