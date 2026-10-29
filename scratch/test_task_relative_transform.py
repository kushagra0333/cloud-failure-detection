import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.features.registry import FeatureRegistry

# Load Phase 7C clean arrays
X_train = np.load("data/processed/phase7c/X_train.npy").copy()
y_train = np.load("data/processed/phase7c/y_train.npy").copy()
meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")

X_val = np.load("data/processed/phase7c/X_val.npy").copy()
y_val = np.load("data/processed/phase7c/y_val.npy").copy()
meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")

feature_names = FeatureRegistry().get_feature_names()
raw_cols = ["cpu_utilization", "memory_utilization", "assigned_memory", "disk_io_time"]
raw_indices = [feature_names.index(col) for col in raw_cols]
print(f"Raw feature columns: {raw_cols} at indices {raw_indices}")

# Build causal task-relative transformation:
# For each entity, sort by prediction_timestamp.
# For window k at timestamp t, compute past history strictly from windows < k.
# If history windows >= 2 (representing >= 6 temporal observations including the sequence),
# center raw utilization features by task running mean: x_rel = x - mean_past.
# Fallback: if history < 2, keep x_cluster unchanged.

def apply_causal_task_relative_transform(X_seq: np.ndarray, meta_df: pd.DataFrame, k_min_windows: int = 2) -> np.ndarray:
    X_trans = X_seq.copy()
    df = meta_df.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    
    entities = df["entity_id"].unique()
    transformed_count = 0
    fallback_count = 0
    
    for eid in entities:
        ent_rows = df[df["entity_id"] == eid].sort_values("prediction_timestamp")
        ent_indices = ent_rows["orig_idx"].values
        
        # We track running past measurements of raw features for this entity
        past_measurements = []
        
        for pos, idx in enumerate(ent_indices):
            if len(past_measurements) >= k_min_windows:
                # Calculate running task baseline
                past_arr = np.concatenate(past_measurements, axis=0) # shape [N_past, 31]
                task_mean = np.mean(past_arr[:, raw_indices], axis=0, keepdims=True) # [1, 4]
                task_std = np.std(past_arr[:, raw_indices], axis=0, keepdims=True) + 1e-5 # [1, 4]
                
                # Apply task-relative centering to the sequence:
                curr_feat = X_trans[idx][:, raw_indices] # shape [5, 4]
                X_trans[idx][:, raw_indices] = (curr_feat - task_mean) / task_std
                transformed_count += 1
            else:
                fallback_count += 1
                
            # Add current window step -1 (or steps) to past history for future windows
            past_measurements.append(X_seq[idx, :, :])
            
    print(f"Transform complete: {transformed_count} windows transformed ({transformed_count/len(df)*100:.1f}%), {fallback_count} fell back to cluster scaler ({fallback_count/len(df)*100:.1f}%)")
    return X_trans

print("\nTransforming Train set:")
X_train_rel = apply_causal_task_relative_transform(X_train, meta_train, k_min_windows=2)

print("\nTransforming Validation set:")
X_val_rel = apply_causal_task_relative_transform(X_val, meta_val, k_min_windows=2)

assert not np.isnan(X_train_rel).any()
assert not np.isnan(X_val_rel).any()
print("Assertions passed: Zero NaNs in transformed tensors.")

# Save transformed tensors in results/phase7f/
out_dir = Path("results/phase7f")
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "X_train_task_relative.npy", X_train_rel)
np.save(out_dir / "X_val_task_relative.npy", X_val_rel)
print("Saved transformed tensors in results/phase7f/.")
