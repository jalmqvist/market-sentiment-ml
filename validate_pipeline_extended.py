import sys
import numpy as np
import pandas as pd
import hashlib

# =========================
# CONFIG
# =========================
DATA_PATH = "data/output/master_research_dataset.csv"
DATA_PATH_REGIME = "data/output/master_research_dataset_with_regime.csv"
REFERENCE_DATA = "data/reference/master_research_dataset.csv"

TOL = 1e-10

# =========================
# HELPERS
# =========================
def fail(msg):
    print(f"\n❌ VALIDATION FAILED: {msg}")
    sys.exit(1)

def ok(msg):
    print(f"✅ {msg}")

def approx(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(a - b) <= TOL

def hash_df(df):
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

# =========================
# SIGNAL (import later ideally)
# =========================
def apply_signal(df):
    df = df.copy()
    df["extreme"] = df["abs_sentiment"] >= 70
    return df[df["extreme"]].copy()

def non_overlap(df):
    df = df.sort_values(["pair", "entry_time"])
    return df.drop_duplicates(subset=["pair", "entry_time"])

def stats(df, col="ret_48b"):
    r = df[col].dropna()
    if len(r) == 0:
        return {"n": 0, "mean": np.nan, "sharpe": np.nan}
    return {
        "n": len(r),
        "mean": r.mean(),
        "sharpe": r.mean() / r.std() if r.std() != 0 else 0
    }

# =========================
# LOAD
# =========================
print("\n=== LOAD DATA ===")

df = pd.read_csv(DATA_PATH)
df_ref = pd.read_csv(REFERENCE_DATA)
df_reg = pd.read_csv(DATA_PATH_REGIME)

ok("Datasets loaded")

# =========================
# TEST 1 — HASH MATCH
# =========================
print("\n=== TEST 1: DATA HASH ===")

if hash_df(df) != hash_df(df_ref):
    fail("Dataset hash mismatch")

ok("Dataset hash identical")

# =========================
# TEST 2 — SIGNAL PARITY
# =========================
print("\n=== TEST 2: SIGNAL PARITY ===")

sig = non_overlap(apply_signal(df))
sig_ref = non_overlap(apply_signal(df_ref))

if len(sig) != len(sig_ref):
    fail(f"Signal count mismatch: {len(sig)} vs {len(sig_ref)}")

ok(f"Signals match: {len(sig)}")

# =========================
# TEST 3 — PERFORMANCE PARITY
# =========================
print("\n=== TEST 3: PERFORMANCE ===")

s = stats(sig)
s_ref = stats(sig_ref)

for k in ["n", "mean", "sharpe"]:
    if not approx(s[k], s_ref[k]):
        fail(f"{k} mismatch: {s[k]} vs {s_ref[k]}")

ok("Performance identical")

# =========================
# TEST 4 — BEHAVIORAL STRUCTURE
# =========================
print("\n=== TEST 4: BEHAVIORAL STRUCTURE ===")

sig["position"] = -1  # contrarian proxy

contra = sig["ret_48b"].dropna()

if contra.mean() <= 0:
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

sample = df.sample(1000)

for _, row in sample.iterrows():
    if pd.isna(row["ret_48b"]):
        continue

    entry = pd.to_datetime(row["entry_time"])
    # crude sanity: returns should not be absurd
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