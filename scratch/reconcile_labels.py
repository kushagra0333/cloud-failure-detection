import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(".").resolve()

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(8192):
            h.update(b)
    return h.hexdigest()

# 1. Authoritative Phase 7C files
y_val_7c_p = root / "data" / "processed" / "phase7c" / "y_val.npy"
meta_val_7c_p = root / "data" / "processed" / "phase7c" / "val_metadata.csv"

y_val_7c = np.load(y_val_7c_p)
meta_val_7c = pd.read_csv(meta_val_7c_p)

# 2. Phase 7K files
val_preds_7k_p = root / "results" / "phase7k" / "validation_predictions.csv"
val_preds_7k = pd.read_csv(val_preds_7k_p)

# 3. Phase 7B / 7E / 7F / 7G / 7H / 7I / 7J
y_val_7b = np.load(root / "data" / "processed" / "phase7b" / "y_val.npy")
meta_val_7b = pd.read_csv(root / "data" / "processed" / "phase7b" / "val_metadata.csv")

# Print shapes and hashes
print("=== FILE HASHES & COUNTS ===")
print(f"Phase 7C y_val.npy: shape={y_val_7c.shape}, sha256={sha256(y_val_7c_p)}")
print(f"Phase 7C val_metadata.csv: shape={meta_val_7c.shape}, sha256={sha256(meta_val_7c_p)}")
print(f"Phase 7K validation_predictions.csv: shape={val_preds_7k.shape}, sha256={sha256(val_preds_7k_p)}")
print(f"Phase 7B y_val.npy: shape={y_val_7b.shape}, sha256={sha256(root / 'data' / 'processed' / 'phase7b' / 'y_val.npy')}")

# Counts
counts_7c_y = pd.Series(y_val_7c).value_counts().to_dict()
counts_7c_meta = meta_val_7c["label"].value_counts().to_dict()
counts_7k_preds = val_preds_7k["y_true"].value_counts().to_dict()
counts_7b_y = pd.Series(y_val_7b).value_counts().to_dict()

print(f"\nPhase 7C y_val.npy counts: {counts_7c_y}")
print(f"Phase 7C val_metadata.csv counts: {counts_7c_meta}")
print(f"Phase 7K validation_predictions.csv y_true counts: {counts_7k_preds}")
print(f"Phase 7B y_val.npy counts: {counts_7b_y}")

# Row-by-row equality check
equal_7c_y_vs_meta = np.array_equal(y_val_7c, meta_val_7c["label"].values)
equal_7c_y_vs_7k = np.array_equal(y_val_7c, val_preds_7k["y_true"].values)
equal_7c_eids_vs_7k = np.array_equal(meta_val_7c["entity_id"].values, val_preds_7k["entity_id"].values)
equal_7c_ts_vs_7k = np.array_equal(meta_val_7c["prediction_timestamp"].values, val_preds_7k["prediction_timestamp"].values)
equal_7c_ttf_vs_7k = np.array_equal(meta_val_7c["time_to_failure"].values, val_preds_7k["time_to_failure"].values)

print(f"\nRow-by-row comparisons:")
print(f"  y_val_7c == meta_val_7c['label']: {equal_7c_y_vs_meta}")
print(f"  y_val_7c == val_preds_7k['y_true']: {equal_7c_y_vs_7k}")
print(f"  meta_val_7c entity_id == val_preds_7k entity_id: {equal_7c_eids_vs_7k}")
print(f"  meta_val_7c timestamp == val_preds_7k timestamp: {equal_7c_ts_vs_7k}")
print(f"  meta_val_7c time_to_failure == val_preds_7k time_to_failure: {equal_7c_ttf_vs_7k}")

# Transition matrix: 7C labels vs 7K labels
trans_mat = pd.crosstab(y_val_7c, val_preds_7k["y_true"], rownames=["Phase_7C_Frozen"], colnames=["Phase_7K_Actual"])
print("\nTransition matrix (Phase 7C Frozen -> Phase 7K Actual):")
print(trans_mat)

# Changed rows
diff_mask = (y_val_7c != val_preds_7k["y_true"].values)
num_diffs = int(diff_mask.sum())
print(f"\nTotal changed rows: {num_diffs}")

# Create label_transition_audit.csv
# If num_diffs == 0, we document every row or all rows with status UNCHANGED
rows_list = []
for idx in range(len(y_val_7c)):
    r_7c = meta_val_7c.iloc[idx]
    r_7k = val_preds_7k.iloc[idx]
    old_label = int(y_val_7c[idx])
    new_label = int(r_7k["y_true"])
    is_diff = (old_label != new_label)
    
    rows_list.append({
        "row_index": idx,
        "entity_id": r_7c["entity_id"],
        "parent_job_id": r_7c["parent_job_id"],
        "prediction_timestamp": float(r_7c["prediction_timestamp"]),
        "time_to_failure": float(r_7c["time_to_failure"]),
        "phase7c_frozen_label": old_label,
        "phase7k_actual_label": new_label,
        "label_changed": is_diff,
        "status": "CHANGED" if is_diff else "EXACT_MATCH"
    })

df_transition = pd.DataFrame(rows_list)
out_dir = root / "results" / "phase7k_audit_reconciliation"
out_dir.mkdir(parents=True, exist_ok=True)
df_transition.to_csv(out_dir / "label_transition_audit.csv", index=False)
print(f"Saved label_transition_audit.csv to {out_dir / 'label_transition_audit.csv'} ({len(df_transition)} rows)")
