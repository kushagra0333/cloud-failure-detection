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

from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS

# Load events
event_dfs = []
for s_path in protocol["source_event_shards"]:
    p = Path(s_path)
    if p.exists():
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(fail_eids)]
        if not matched.empty:
            event_dfs.append(matched)
events_raw = pd.concat(event_dfs, ignore_index=True)
events_raw["time_sec"] = events_raw["timestamp"] / 1e6

# Load usage
usage_chunks = []
for s_path in protocol["source_usage_shards"]:
    p = Path(s_path)
    if p.exists():
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(fail_eids)]
            if not matched.empty:
                usage_chunks.append(matched)
usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

fail_summary = []
for eid in fail_eids:
    e_events = events_raw[events_raw["entity_id"] == eid]
    f_events = e_events[e_events["event_type"] == 3]
    e_usage = usage_df[usage_df["entity_id"] == eid]
    
    unique_bins = sorted(e_usage["grid_bin"].unique())
    n_bins = len(unique_bins)
    
    # For each fail event, how many grid bins occurred BEFORE the fail event?
    min_f_time = f_events["time_sec"].min() if not f_events.empty else None
    
    bins_before_fail = [b for b in unique_bins if b < min_f_time] if min_f_time else []
    
    fail_summary.append({
        "entity_id": eid,
        "n_grid_bins": n_bins,
        "min_f_time": min_f_time,
        "min_bin": min(unique_bins) if unique_bins else None,
        "max_bin": max(unique_bins) if unique_bins else None,
        "bins_before_fail": len(bins_before_fail),
    })

df_res = pd.DataFrame(fail_summary)
print("Summary of 100 fail entities:")
print(f"Entities with FAIL event: {df_res['min_f_time'].notna().sum()} / 100")
print("Distribution of bins_before_fail:")
print(df_res["bins_before_fail"].value_counts().sort_index())
print(f"Entities with >= 15 bins before fail: {(df_res['bins_before_fail'] >= 15).sum()}")
print(f"Entities with >= 14 bins before fail: {(df_res['bins_before_fail'] >= 14).sum()}")
print(f"Entities with >= 10 bins before fail: {(df_res['bins_before_fail'] >= 10).sum()}")
print(f"Entities with >= 5 bins before fail: {(df_res['bins_before_fail'] >= 5).sum()}")
print(f"Entities with 0 bins before fail: {(df_res['bins_before_fail'] == 0).sum()}")

print("\nSample entities with highest bins_before_fail:")
print(df_res.sort_values("bins_before_fail", ascending=False).head(15))
