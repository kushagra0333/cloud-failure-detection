import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

with open("results/phase4/confirmatory_protocol.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["failure_entities"])
norm_eids = set(protocol["frozen_entity_ids"]["normal_entities"])

from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS

# Load all events for all 300 entities
event_dfs = []
for s_path in protocol["source_event_shards"]:
    p = Path(s_path)
    if p.exists():
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(fail_eids | norm_eids)]
        if not matched.empty:
            event_dfs.append(matched)
events_raw = pd.concat(event_dfs, ignore_index=True)
events_raw["time_sec"] = events_raw["timestamp"] / 1e6

# Load all usage
usage_chunks = []
for s_path in protocol["source_usage_shards"]:
    p = Path(s_path)
    if p.exists():
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(fail_eids | norm_eids)]
            if not matched.empty:
                usage_chunks.append(matched)
usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

grid_counts = usage_df.groupby(["entity_id", "grid_bin"]).size().reset_index()
entity_grid_steps = grid_counts.groupby("entity_id").size()

print("Grid steps for FAIL entities:")
fail_steps = entity_grid_steps.reindex(list(fail_eids)).fillna(0)
print(fail_steps.describe())
print(f"Fail entities with grid steps >= 15: {(fail_steps >= 15).sum()} / {len(fail_eids)}")
print(f"Fail entities with grid steps < 15: {(fail_steps < 15).sum()} / {len(fail_eids)}")

# Look at distribution of grid steps for fail entities
print("\nGrid step count histogram for fail entities:")
print(pd.cut(fail_steps, bins=[0, 5, 10, 14, 15, 20, 30, 100], right=True).value_counts().sort_index())

# Check how many normal entities have >= 15 steps
norm_steps = entity_grid_steps.reindex(list(norm_eids)).fillna(0)
print("\nGrid steps for NORMAL entities:")
print(norm_steps.describe())
print(f"Normal entities with grid steps >= 15: {(norm_steps >= 15).sum()} / {len(norm_eids)}")
