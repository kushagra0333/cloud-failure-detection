import pandas as pd
import numpy as np

tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")

all_meta = pd.concat([tr, va, te], ignore_index=True)
fail_windows = all_meta[all_meta["label"] > 0]
print("Total fail windows in dev:", len(fail_windows))
print("Unique failing entities in dev:", fail_windows["entity_id"].nunique())
print("Prediction timestamp summary for fail windows:")
print(fail_windows["prediction_timestamp"].describe())

# How many steps of history did dev entities have?
# In dev entities, what were the min and max timestamps?
counts_per_ent = all_meta.groupby("entity_id").size()
print("\nWindow counts per entity in dev:")
print(counts_per_ent.describe())
print(f"Entities with >= 1 window: {len(counts_per_ent)}")
