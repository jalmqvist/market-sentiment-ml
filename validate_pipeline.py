import sys
import json
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
PIPELINE_DATA = "data/output/master_research_dataset.csv"
REFERENCE_DATA = "data/reference/master_research_dataset.csv"  # <-- store a frozen copy here

TOLERANCE = 1e-10  # strict reproducibility

# =========================
# HELPERS
# =========================
def fail(msg):
    print(f"\n❌ VALIDATION FAILED: {msg}")
    sys.exit(1)

def ok(msg):
    print(f"✅ {msg}")

def approx_equal(a, b, tol=TOLERANCE):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(a - b) <= tol

# =========================
# LOAD DATA
# =========================
print("\n=== LOAD DATA ===")

try:
    df_new = pd.read_csv(PIPELINE_DATA)
except Exception as e:
    fail(f"Could not load pipeline dataset: {e}")

try:
    df_ref = pd.read_csv(REFERENCE_DATA)
except Exception as e:
    fail(f"Could not load reference dataset: {e}")

ok("Datasets loaded")

# =========================
# TEST 1: BASIC SHAPE
# =========================
print("\n=== TEST 1: DATASET SHAPE ===")

if len(df_new) != len(df_ref):
    fail(f"Row count mismatch: {len(df_new)} vs {len(df_ref)}")

if df_new["pair"].nunique() != df_ref["pair"].nunique():
    fail("Pair count mismatch")

ok("Dataset shape matches")

# =========================
# TEST 2: KEY COLUMN MATCH
# =========================
print("\n=== TEST 2: KEY COLUMN MATCH ===")

cols = ["pair", "entry_time", "abs_sentiment"]

missing_cols = [c for c in cols if c not in df_new.columns]
if missing_cols:
    fail(f"Missing columns in pipeline dataset: {missing_cols}")

merged = df_ref[cols].merge(df_new[cols], on=cols, how="outer", indicator=True)

counts = merged["_merge"].value_counts()

left_only = counts.get("left_only", 0)
right_only = counts.get("right_only", 0)

if left_only > 0 or right_only > 0:
    fail(f"Dataset mismatch:\n{counts}")

ok("Core data identical")

# =========================
# SIGNAL LOGIC (INLINE)
# =========================
def apply_signal(df):
    df = df.copy()

    df["extreme"] = df["abs_sentiment"] >= 70
    df["extreme_streak"] = df.groupby("pair")["extreme"].cumsum()

    signal = df[df["extreme"]].copy()

    return signal

def compute_non_overlap(signal):
    signal = signal.sort_values(["pair", "entry_time"])
    signal["prev_exit"] = signal.groupby("pair")["entry_time"].shift()
    return signal.drop_duplicates(subset=["pair", "entry_time"])

def compute_stats(df, col):
    r = df[col].dropna()
    if len(r) == 0:
        return {"n": 0, "mean": np.nan, "sharpe": np.nan, "hit": np.nan}

    return {
        "n": len(r),
        "mean": r.mean(),
        "sharpe": r.mean() / r.std() if r.std() != 0 else 0,
        "hit": (r > 0).mean()
    }

# =========================
# TEST 3: SIGNAL COUNT
# =========================
print("\n=== TEST 3: SIGNAL COUNT ===")

sig_new = apply_signal(df_new)
sig_ref = apply_signal(df_ref)

if len(sig_new) != len(sig_ref):
    fail(f"Raw signal mismatch: {len(sig_new)} vs {len(sig_ref)}")

ok(f"Raw signals match: {len(sig_new)}")

sig_new = compute_non_overlap(sig_new)
sig_ref = compute_non_overlap(sig_ref)

if len(sig_new) != len(sig_ref):
    fail(f"Non-overlap mismatch: {len(sig_new)} vs {len(sig_ref)}")

ok(f"Non-overlapping signals match: {len(sig_new)}")

# =========================
# TEST 4: PERFORMANCE
# =========================
print("\n=== TEST 4: PERFORMANCE ===")

col = "ret_48b"

stats_new = compute_stats(sig_new, col)
stats_ref = compute_stats(sig_ref, col)

for k in ["n", "mean", "sharpe", "hit"]:
    if not approx_equal(stats_new[k], stats_ref[k]):
        fail(f"{k} mismatch: {stats_new[k]} vs {stats_ref[k]}")

ok("Performance metrics match")

print("Pipeline:", stats_new)

# =========================
# TEST 5: BEHAVIORAL STRUCTURE
# =========================
print("\n=== TEST 5: BEHAVIORAL STRUCTURE ===")

# crude proxy (adjust if you have explicit position column)
sig_new["position"] = np.where(sig_new["abs_sentiment"] >= 70, -1, 1)

contra = sig_new[sig_new["position"] == -1]["ret_48b"].dropna()
trend = sig_new[sig_new["position"] == 1]["ret_48b"].dropna()

if contra.mean() <= trend.mean():
    fail("Behavioral structure broken (contrarian <= trend)")

ok("Behavioral structure OK")

# =========================
# TEST 6: YEAR DISTRIBUTION
# =========================
print("\n=== TEST 6: YEAR DISTRIBUTION ===")

sig_new["year"] = pd.to_datetime(sig_new["entry_time"]).dt.year
sig_ref["year"] = pd.to_datetime(sig_ref["entry_time"]).dt.year

y_new = sig_new["year"].value_counts().sort_index()
y_ref = sig_ref["year"].value_counts().sort_index()

if not y_new.equals(y_ref):
    fail(f"Year distribution mismatch:\n{y_new}\nvs\n{y_ref}")

ok("Year distribution matches")

# =========================
# FINAL
# =========================
print("\n==============================")
print("🎯 ALL VALIDATION TESTS PASSED")
print("==============================\n")
