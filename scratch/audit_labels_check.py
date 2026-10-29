"""Scratch script to audit training and validation label distributions."""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
y_train_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_train.npy"
y_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy"
train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"

y_train = np.load(y_train_p)
y_val = np.load(y_val_p)
train_meta = pd.read_csv(train_meta_p)
val_meta = pd.read_csv(val_meta_p)

print("=== ACTUAL ARRAYS ===")
print("y_train shape:", y_train.shape)
c0_train = int((y_train == 0).sum())
c1_train = int((y_train == 1).sum())
c2_train = int((y_train == 2).sum())
total_train = len(y_train)
print(f"y_train: 0={c0_train} ({c0_train/total_train*100:.2f}%), 1={c1_train} ({c1_train/total_train*100:.2f}%), 2={c2_train} ({c2_train/total_train*100:.2f}%), Total={total_train}")

print("y_val shape:", y_val.shape)
c0_val = int((y_val == 0).sum())
c1_val = int((y_val == 1).sum())
c2_val = int((y_val == 2).sum())
total_val = len(y_val)
print(f"y_val: 0={c0_val} ({c0_val/total_val*100:.2f}%), 1={c1_val} ({c1_val/total_val*100:.2f}%), 2={c2_val} ({c2_val/total_val*100:.2f}%), Total={total_val}")

print("\n=== METADATA CSV ===")
print("train_metadata label counts:")
print(train_meta["label"].value_counts().sort_index())
print("\nval_metadata label counts:")
print(val_meta["label"].value_counts().sort_index())

# Check consistency between arrays and metadata
assert np.array_equal(y_train, train_meta["label"].values), "y_train != train_meta['label']"
assert np.array_equal(y_val, val_meta["label"].values), "y_val != val_meta['label']"
print("\nAssertion passed: arrays exactly match metadata CSV 'label' column!")

# Now let's investigate where 3044, 118, 116 came from!
print("\nInvestigating 3044, 118, 116:")
# Could it be from another dataset or split? Or let's search across the codebase.
