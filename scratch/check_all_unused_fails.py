import pandas as pd
import glob
from pathlib import Path
import json

raw_events = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
raw_usage = sorted(glob.glob("data/raw/sample_task_usage*.csv.gz"))

tr = pd.read_csv("data/processed/train_metadata.csv")
val = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")

used_entities = set(tr["entity_id"]) | set(val["entity_id"]) | set(te["entity_id"])

ev_dfs = [pd.read_csv(f, header=None, usecols=[0, 2, 3, 5]) for f in raw_events]
all_ev = pd.concat(ev_dfs, ignore_index=True)
all_ev["entity_id"] = all_ev[2].astype(str) + "_" + all_ev[3].astype(str)
all_ev["time_sec"] = all_ev[0] / 1e6
fail_events = all_ev[all_ev[5] == 3]
all_fail_eids = set(fail_events["entity_id"])
unused_fail_eids = all_fail_eids - used_entities

print(f"Total fail entities in raw events: {len(all_fail_eids)}")
print(f"Used fail entities: {len(all_fail_eids & used_entities)}")
print(f"Unused fail entities: {len(unused_fail_eids)}")

# Check usage for unused fail entities
import numpy as np
usage_chunks = []
for uf in raw_usage:
    for chunk in pd.read_csv(uf, header=None, usecols=[0, 1, 2, 3], chunksize=200000):
        chunk["entity_id"] = chunk[2].astype(str) + "_" + chunk[3].astype(str)
        matched = chunk[chunk["entity_id"].isin(unused_fail_eids)]
        if not matched.empty:
            usage_chunks.append(matched)

if usage_chunks:
    df_u = pd.concat(usage_chunks, ignore_index=True)
    df_u["end_sec"] = df_u[1] / 1e6
    df_u["grid_bin"] = np.ceil(df_u["end_sec"] / 300.0) * 300.0
    
    # Map min fail time per entity
    min_fail = fail_events.groupby("entity_id")["time_sec"].min().to_dict()
    
    stats = []
    for eid, grp in df_u.groupby("entity_id"):
        bins = sorted(grp["grid_bin"].unique())
        f_time = min_fail.get(eid)
        bins_before = [b for b in bins if b < f_time] if f_time else []
        stats.append({
            "entity_id": eid,
            "total_bins": len(bins),
            "bins_before_fail": len(bins_before),
            "f_time": f_time,
        })
    df_stats = pd.DataFrame(stats)
    print(f"\nUnused fail entities found in usage: {len(df_stats)}")
    print("Distribution of bins_before_fail across ALL unused fail entities in the entire trace:")
    print(df_stats["bins_before_fail"].value_counts().sort_index())
    print(f"Entities with >= 15 bins before fail: {(df_stats['bins_before_fail'] >= 15).sum()}")
    print(f"Entities with >= 14 bins before fail: {(df_stats['bins_before_fail'] >= 14).sum()}")
