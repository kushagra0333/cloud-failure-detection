import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Load dev partitions
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")

dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])
dev_fail_eids = set(tr[tr["label"] > 0]["entity_id"]) | set(va[va["label"] > 0]["entity_id"]) | set(te[te["label"] > 0]["entity_id"])

print(f"dev_eids total: {len(dev_eids)}")
print(f"dev_fail_eids (with label > 0 windows): {len(dev_fail_eids)}")

# Let's inspect all 748 failing entities with telemetry
from scripts.audit_phase6_feasibility import audit_full_trace_failure_census

census_df = audit_full_trace_failure_census()
ge15_df = census_df[census_df["bins_before_fail"] >= 15]

print(f"\nTotal failing tasks in trace with >= 15 bins before fail: {len(ge15_df)}")
print(f"How many in dev_eids: {ge15_df['in_dev_split'].sum()}")
print(f"How many in dev_fail_eids: {ge15_df['in_dev_failing'].sum()}")

# What are the 241 - 192 = 49 entities?
diff_49 = ge15_df[~ge15_df["entity_id"].isin(dev_fail_eids)]
print(f"\nEntities with >= 15 bins before fail that are NOT in dev_fail_eids: {len(diff_49)}")
print("Are they in dev_eids at all?")
print(diff_49["in_dev_split"].value_counts())

# Why are these 49 entities in dev_eids but NOT in dev_fail_eids?
# Let's see their windows in train/val/test!
dev_all = pd.concat([tr, va, te], ignore_index=True)
for eid in diff_49["entity_id"].head(10):
    rows = dev_all[dev_all["entity_id"] == eid]
    f_time = ge15_df[ge15_df["entity_id"] == eid]["first_fail_time"].iloc[0]
    print(f"\nEntity {eid}: rows in dev={len(rows)}, fail_time={f_time}")
    if not rows.empty:
        print(rows[["prediction_timestamp", "label", "time_to_failure"]].head(5))
