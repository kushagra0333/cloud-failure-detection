"""Inspect trace schema, lifecycle events, and usage columns for Candidate A Train and Val only."""
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()

# 1. Inspect Candidate A Train & Val metadata and shapes
train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"

train_meta = pd.read_csv(train_meta_p)
val_meta = pd.read_csv(val_meta_p)

print(f"Candidate A Train metadata rows: {len(train_meta)}, unique entities: {train_meta['entity_id'].nunique()}, unique jobs: {train_meta['parent_job_id'].nunique()}")
print(f"Candidate A Val metadata rows: {len(val_meta)}, unique entities: {val_meta['entity_id'].nunique()}, unique jobs: {val_meta['parent_job_id'].nunique()}")

train_eids = set(train_meta["entity_id"].unique())
val_eids = set(val_meta["entity_id"].unique())
dev_eids = train_eids.union(val_eids)

print(f"Total dev entities (Train + Val): {len(dev_eids)}")

# Check shapes of X_train, X_val, y_train, y_val
X_train_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "X_train.npy"
y_train_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_train.npy"
X_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "X_val.npy"
y_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy"

X_tr = np.load(X_train_p)
y_tr = np.load(y_train_p)
X_v = np.load(X_val_p)
y_v = np.load(y_val_p)

print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
print(f"X_val shape: {X_v.shape}, y_val shape: {y_v.shape}")

# 2. Inspect Raw Event Schema
EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]

event_shards = [
    PROJECT_ROOT / "data" / "raw" / "sample_task_events.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_events_part1.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_events_part2.csv.gz",
]

dev_events_list = []
for s_path in event_shards:
    if s_path.exists():
        df = pd.read_csv(s_path, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(dev_eids)].copy()
        dev_events_list.append(matched)
        print(f"From {s_path.name}: matched {len(matched)} event rows for dev entities.")

dev_events = pd.concat(dev_events_list, ignore_index=True)
dev_events["time_sec"] = dev_events["timestamp"] / 1e6

print("\n--- DEV EVENTS SUMMARY (Train + Val Entities) ---")
print(f"Total dev events: {len(dev_events)}")
print(f"Unique entities with events: {dev_events['entity_id'].nunique()} / {len(dev_eids)}")
print("Event type distribution (0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL, 6=LOST, 7=UPDATE_PEND, 8=UPDATE_RUN):")
print(dev_events["event_type"].value_counts().sort_index())

# Check missingness of event columns in dev_events
print("\nEvent column missingness:")
for col in EVENT_COLS:
    null_cnt = dev_events[col].isna().sum()
    print(f"  {col:25s}: {null_cnt:5d} / {len(dev_events)} ({null_cnt/len(dev_events)*100:.2f}%) nulls")

# Check event types per entity: are there multiple SCHEDULE events (rescheduling)? EVICT? FAIL?
ev_counts_per_ent = dev_events.groupby(["entity_id", "event_type"]).size().unstack(fill_value=0)
print("\nEvent counts summary across entities:")
print(ev_counts_per_ent.describe().T[["mean", "min", "50%", "max"]])

# Check causal availability of events relative to prediction_timestamp
# Merge dev_events with prediction timestamps
dev_meta = pd.concat([train_meta, val_meta], ignore_index=True)

# For each window (entity_id, prediction_timestamp), what events occurred at or before t?
print("\nChecking causal event occurrence relative to prediction_timestamp t:")
prior_event_counts = {et: 0 for et in range(9)}
concurrent_fail = 0
events_by_window = []

for _, row in dev_meta.iterrows():
    eid = row["entity_id"]
    t = row["prediction_timestamp"]
    ent_evs = dev_events[dev_events["entity_id"] == eid]
    prior_evs = ent_evs[ent_evs["time_sec"] <= t]
    for et in range(9):
        if (prior_evs["event_type"] == et).any():
            prior_event_counts[et] += 1

print(f"Across all {len(dev_meta)} dev windows (Train + Val):")
for et, count in prior_event_counts.items():
    print(f"  Event type {et} observed at or before t: in {count} / {len(dev_meta)} windows ({count/len(dev_meta)*100:.2f}%)")

# 3. Inspect Raw Usage Schema
USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]

usage_shards = [
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage_part1.csv.gz",
]

# Read sample of usage to check columns and missingness
print("\n--- RAW USAGE SUMMARY (Sample from shards) ---")
usage_sample_list = []
for s_path in usage_shards:
    if s_path.exists():
        print(f"Sampling {s_path.name}...")
        for chunk in pd.read_csv(s_path, header=None, names=USAGE_COLS, chunksize=100000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(dev_eids)].copy()
            if not matched.empty:
                usage_sample_list.append(matched)
            if len(usage_sample_list) >= 10:  # Sample enough chunks
                break

if usage_sample_list:
    sample_usage_df = pd.concat(usage_sample_list, ignore_index=True)
    print(f"Sampled {len(sample_usage_df)} usage rows for dev entities.")
    print("Usage column missingness in dev sample:")
    for col in USAGE_COLS:
        null_cnt = sample_usage_df[col].isna().sum()
        print(f"  {col:32s}: {null_cnt:5d} / {len(sample_usage_df)} ({null_cnt/len(sample_usage_df)*100:.2f}%) nulls")
