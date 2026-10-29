"""Audit script to inspect Phase 7J source inputs, hashes, and compare scripts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# 1. Authoritative input files
preds_p = PROJECT_ROOT / "results" / "phase7i" / "phase7i_validation_predictions.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"
y_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy"

print("=== INPUT HASHES AND PROPERTIES ===")
for p in [preds_p, val_meta_p, y_val_p]:
    print(f"File: {p.relative_to(PROJECT_ROOT)}")
    print(f"  Exists: {p.exists()}")
    print(f"  Size: {p.stat().st_size} bytes")
    print(f"  SHA-256: {compute_sha256(p)}")

df_preds = pd.read_csv(preds_p)
val_meta = pd.read_csv(val_meta_p)
y_val = np.load(y_val_p)

print("\n=== SHAPES AND ALIGNMENT ===")
print("df_preds shape:", df_preds.shape)
print("val_meta shape:", val_meta.shape)
print("y_val shape:", y_val.shape)

# Row-by-row alignment check
for col in ["entity_id", "prediction_timestamp", "label", "parent_job_id"]:
    matches = (df_preds[col].fillna(-1) == val_meta[col].fillna(-1)).all()
    print(f"  Alignment {col}: {'PERFECT MATCH' if matches else 'MISMATCH'}")
    assert matches

assert np.array_equal(df_preds["y_true"].values, y_val)
print("  y_true vs y_val.npy: PERFECT MATCH")

# Check class ordering in prediction file
print("\n=== CLASS ORDERING IN PROBABILITY COLUMNS ===")
conditions = ["condition_A", "condition_B", "condition_C", "condition_D"]
for cid in conditions:
    cols = [f"prob_{cid}_c0", f"prob_{cid}_c1", f"prob_{cid}_c2"]
    for c in cols:
        assert c in df_preds.columns, f"Missing {c}"
    # Verify p_fail = c1 + c2
    p_fail_calc = df_preds[f"prob_{cid}_c1"] + df_preds[f"prob_{cid}_c2"]
    p_fail_saved = df_preds[f"p_fail_{cid}"]
    max_diff = np.abs(p_fail_calc - p_fail_saved).max()
    print(f"  {cid}: c0=Normal, c1=Warning, c2=Critical. p_fail == c1+c2 max diff: {max_diff:.2e}")
    assert max_diff < 1e-6

# Compare test_threshold_sweep.py vs run_phase7j_investigation.py
print("\n=== SCRIPT COMPARISON: test_threshold_sweep.py vs run_phase7j_investigation.py ===")
with open(PROJECT_ROOT / "scratch" / "test_threshold_sweep.py", "r") as f:
    code_test = f.read()
with open(PROJECT_ROOT / "scripts" / "run_phase7j_investigation.py", "r") as f:
    code_inv = f.read()

print("Inputs loaded in test_threshold_sweep.py: results/phase7i/phase7i_validation_predictions.csv")
print("Inputs loaded in run_phase7j_investigation.py: results/phase7i/phase7i_validation_predictions.csv, val_metadata.csv, y_val.npy")
print("Threshold grid in both: np.linspace(0.0, 1.0, 201)")
