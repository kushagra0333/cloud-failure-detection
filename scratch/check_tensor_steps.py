import numpy as np
import pandas as pd

X_val_raw = np.load("data/processed/phase7c/raw_unscaled/X_val_raw.npy")
val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")

print(f"X_val_raw shape: {X_val_raw.shape}")
print(f"Row 0 entity: {val_meta['entity_id'].iloc[0]}, timestamp: {val_meta['prediction_timestamp'].iloc[0]}")
print(f"Row 0 cpu_utilization across 5 steps: {X_val_raw[0, :, 0]}")
print(f"Row 0 task_age across 5 steps (col 29): {X_val_raw[0, :, 29]}")
