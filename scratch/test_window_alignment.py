"""Verify 5-step temporal structure of Candidate A Train and Val windows."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
train_meta = pd.read_csv("data/processed/phase7c/train_metadata.csv")
val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")

X_train = np.load("data/processed/phase7c/X_train.npy")
X_val = np.load("data/processed/phase7c/X_val.npy")

print(f"X_train shape: {X_train.shape}, train_meta len: {len(train_meta)}")
print(f"X_val shape: {X_val.shape}, val_meta len: {len(val_meta)}")

# Check feature names in raw_reconstruction_manifest.json
import json
with open("data/processed/phase7c/raw_unscaled/raw_reconstruction_manifest.json") as f:
    manifest = json.load(f)
cols = manifest["feature_definitions"]["feature_cols"]
print(f"Manifest features count: {len(cols)}")
print(f"First 5 features: {cols[:5]}")
