import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import glob

from src.ingestion.schema import ClusterEventTypeV2, FailureLabel
from src.ingestion.validator import TimeSeriesValidator
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

# Load dev partitions to identify dev entities
tr_meta = pd.read_csv("data/processed/train_metadata.csv")
va_meta = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr_meta["entity_id"]) | set(va_meta["entity_id"]) | set(te["entity_id"])
dev_jobs = set(eid.split("_")[0] for eid in dev_eids)

# Check Phase 5 confirmatory normal entities (from results/phase4/confirmatory_protocol.json)
import json
with open("results/phase4/confirmatory_protocol.json") as f:
    proto5 = json.load(f)
frozen5_norm_eids = set(proto5["frozen_entity_ids"]["normal_entities"])

# Ingest all events from shards
event_files = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
ev_dfs = []
for f in event_files:
    df = pd.read_csv(f, header=None, usecols=[0, 2, 3, 5])
    df["entity_id"] = df[2].astype(str) + "_" + df[3].astype(str)
    df["time_sec"] = df[0] / 1e6
    df["event_type"] = df[5]
    ev_dfs.append(df[["entity_id", "time_sec", "event_type"]])

all_events_df = pd.concat(ev_dfs, ignore_index=True)

# Find all entities with abnormal failure (FAIL==3) anywhere
abnormal_fail_eids = set(all_events_df[all_events_df["event_type"] == ClusterEventTypeV2.FAIL]["entity_id"].unique())
print(f"Total abnormal failing entities: {len(abnormal_fail_eids):,}")

# Pure normal entities: zero FAIL events across entire trace
# And terminating with FINISH (4) or running without termination
all_trace_eids = set(all_events_df["entity_id"].unique())
pure_normal_eids = all_trace_eids - abnormal_fail_eids - dev_eids
print(f"Pure normal unused entities in raw events: {len(pure_normal_eids):,}")

# Test normal label derivation with TargetLabelGenerator on a sample of 200 normal entities
sample_normal = list(frozen5_norm_eids)[:50]
print(f"Testing on 50 sample normal entities from Phase 5 pool...")

# Check telemetry for these 50
usage_files = sorted(glob.glob("data/raw/sample_task_usage*.csv.gz"))
USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]

usage_chunks = []
for uf in usage_files:
    for chunk in pd.read_csv(uf, header=None, names=USAGE_COLS, chunksize=250000):
        chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
        m = chunk[chunk["entity_id"].isin(sample_normal)].copy()
        if not m.empty:
            usage_chunks.append(m)

df_u = pd.concat(usage_chunks, ignore_index=True)
df_u["end_time_sec"] = df_u["end_time"] / 1e6
df_u["grid_bin"] = np.ceil(df_u["end_time_sec"] / 300.0) * 300.0

res = df_u.groupby(["entity_id", "grid_bin"], as_index=False)["end_time_sec"].max()
res["timestamp"] = res["grid_bin"]
res["end_time_sec"] = res["grid_bin"]

events_sample = all_events_df[all_events_df["entity_id"].isin(sample_normal)]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_sample = label_gen.label_telemetry_windows(res, events_sample)

print("Labeled sample distribution:")
print(labeled_sample["label"].value_counts().to_dict())

# Check censorship at trace end:
# How many steps were marked -1 due to telemetry ending without 30m lookahead?
unknowns = labeled_sample[labeled_sample["label"] == -1]
print(f"Steps marked -1 (censored): {len(unknowns)} / {len(labeled_sample)}")
if not unknowns.empty:
    print("Sample censored steps:")
    for eid, grp in unknowns.groupby("entity_id"):
        max_t = grp["end_time_sec"].max()
        print(f"  Entity {eid}: max_telem={max_t}, censored steps={len(grp)}")
        break
