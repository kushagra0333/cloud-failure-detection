import json
import numpy as np
import pandas as pd
from pathlib import Path

X_train = np.load("data/processed/phase7c/X_train.npy")
y_train = np.load("data/processed/phase7c/y_train.npy")
meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")

import sys
sys.path.insert(0, ".")
from src.features.registry import FeatureRegistry

feature_reg = FeatureRegistry()
features = feature_reg.get_feature_names()

# Inspect mean, std, min, max across features at step -1 (t_pred)
tab_train = X_train[:, -1, :]
df_tab = pd.DataFrame(tab_train, columns=features)
df_tab["label"] = y_train

print("\nSummary of features by class in Training Set (Mean values):")
grouped = df_tab.groupby("label").mean().T
grouped.columns = ["Normal (0)", "Warning (1)", "Critical (2)"]
print(grouped.head(15))

# Check missingness indicators
missing_cols = [c for c in features if "missing" in c]
print(f"\nMissingness features: {missing_cols}")
print(df_tab.groupby("label")[missing_cols].mean())
