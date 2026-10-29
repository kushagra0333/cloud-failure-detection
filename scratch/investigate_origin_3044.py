"""Investigate the mathematical origins of 3044, 118, 116."""
import numpy as np
import pandas as pd
from pathlib import Path

train_meta = pd.read_csv("data/processed/phase7c/train_metadata.csv")
val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")

print("Train shape:", train_meta.shape)
print("Val shape:", val_meta.shape)

# Let's check value counts of various columns
print("\nTrain columns:", train_meta.columns.tolist())
print(train_meta["label"].value_counts())

# Check total windows in train: 3278
# If normal was 3044, failing would be 118 + 116 = 234.
# Notice: 234 is almost 237 or 244!
# Wait! In train:
# label 0: 2797
# label 1: 244
# label 2: 237
# Total failing = 244 + 237 = 481
# 3278 - 481 = 2797.
# What is 3278 - 234? 3044!
# Where would 118 and 116 come from?
# Could 118 be warning episodes or entities? Or could 118 be half of 244? 244 / 2 = 122.
# 237 / 2 = 118.5!
# What about parent jobs? In phase 6, total parent jobs = 118!
# Let's check number of unique entities in train:
print("Unique entities in train:", train_meta["entity_id"].nunique())
print("Unique parent jobs in train:", train_meta["parent_job_id"].nunique())

# Check entities by label in train:
normal_eids = set(train_meta[train_meta["label"] == 0]["entity_id"])
warn_eids = set(train_meta[train_meta["label"] == 1]["entity_id"])
crit_eids = set(train_meta[train_meta["label"] == 2]["entity_id"])
print("Entities with normal windows:", len(normal_eids))
print("Entities with warn windows:", len(warn_eids))
print("Entities with crit windows:", len(crit_eids))
fail_eids = warn_eids | crit_eids
pure_normal_eids = normal_eids - fail_eids
print("Pure normal entities in train:", len(pure_normal_eids))
print("Failing entities in train:", len(fail_eids))

# Let's check episodes in train:
prefails = train_meta[train_meta["label"] > 0].copy()
prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
print("Total episodes in train:", len(episodes))

# Wait, what if someone applied val proportions or another dataset?
# Look at val_distribution:
# val: 560 normal (85.24%), 48 warning (7.31%), 49 critical (7.46%)
# 560 / 657 = 0.852359
# 48 / 657 = 0.073059
# 49 / 657 = 0.074581
# If you multiplied 3278 by val proportions:
# 3278 * (560/657) = 2794.03
# 3278 * (48/657) = 239.49
# 3278 * (49/657) = 244.47
# That does NOT give 3044, 118, 116.

# What about 3044 / 3278?
# 3044 / 3278 = 0.928615
# 118 / 3278 = 0.035997
# 116 / 3278 = 0.035387

# Where did 3044, 118, 116 come from?
# Let's check other metadata files in data/ or results/
