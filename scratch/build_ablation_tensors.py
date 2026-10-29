"""Extract and build feature matrices for M0, M1, M2, M3, M4 across Train and Val."""
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(".").resolve()

train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"
train_meta = pd.read_csv(train_meta_p)
val_meta = pd.read_csv(val_meta_p)

dev_meta = pd.concat([train_meta, val_meta], ignore_index=True)
dev_eids = set(dev_meta["entity_id"].unique())

print(f"Candidate A Train: {len(train_meta)} windows; Val: {len(val_meta)} windows")

# 1. Load baseline M0 tensors
X_tr_m0_raw = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled" / "X_train_raw.npy")
X_v_m0_raw = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled" / "X_val_raw.npy")
y_tr = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_train.npy")
y_v = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy")

assert X_tr_m0_raw.shape == (3278, 5, 31)
assert X_v_m0_raw.shape == (657, 5, 31)

# 2. Ingest Events for J2 and J3
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

dev_events = pd.concat(dev_events_list, ignore_index=True)
dev_events["time_sec"] = dev_events["timestamp"] / 1e6
events_by_eid = dev_events.groupby("entity_id")

# Static attributes dict
static_dict = {}
for eid, grp in events_by_eid:
    prio = grp["priority"].iloc[0] / 10.0
    sched = grp["scheduling_class"].iloc[0] / 3.0
    rcpu = grp["req_cpu"].iloc[0]
    rram = grp["req_ram"].iloc[0]
    static_dict[eid] = np.array([prio, sched, rcpu, rram], dtype=np.float32)

# 3. Ingest Usage for J1
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
usage_chunks = []
for s_path in usage_shards:
    if s_path.exists():
        for chunk in pd.read_csv(s_path, header=None, names=USAGE_COLS, chunksize=500000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(dev_eids)].copy()
            if not matched.empty:
                usage_chunks.append(matched)

usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

# Resample to 300s bins
agg_cols = {
    "cpu_rate": "mean",
    "canonical_memory": "mean",
    "assigned_memory": "mean",
    "max_cpu_rate": "mean",
    "max_memory": "mean",
    "total_page_cache": "mean",
}
resampled = usage_df.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_cols)
resampled["cpu_burstiness"] = resampled["max_cpu_rate"] - resampled["cpu_rate"]
resampled["memory_headroom"] = resampled["assigned_memory"] - resampled["max_memory"]

# Create fast lookup map: (entity_id, grid_bin) -> [max_cpu_rate, cpu_burstiness, max_memory, memory_headroom, total_page_cache]
j1_cols = ["max_cpu_rate", "cpu_burstiness", "max_memory", "memory_headroom", "total_page_cache"]
resampled["key"] = resampled["entity_id"] + "@" + resampled["grid_bin"].astype(str)
j1_lookup = dict(zip(resampled["key"], resampled[j1_cols].values))

# Function to build additional features for a given metadata DataFrame
def build_additional_tensors(meta_df: pd.DataFrame):
    n = len(meta_df)
    j1_tensor = np.zeros((n, 5, 5), dtype=np.float32)
    j2_tensor = np.zeros((n, 5, 4), dtype=np.float32)
    j3_tensor = np.zeros((n, 5, 2), dtype=np.float32)

    for i, row in meta_df.iterrows():
        eid = row["entity_id"]
        t = row["prediction_timestamp"]
        
        # J2: Static scheduler metadata (replicated across 5 steps)
        static_feats = static_dict[eid]
        for step in range(5):
            j2_tensor[i, step, :] = static_feats
            
        # Timestamps for 5 steps: [t - 1200, t - 900, t - 600, t - 300, t]
        step_times = [t - 1200.0 + step * 300.0 for step in range(5)]
        
        # J3: Dynamic preemption events at each step
        evs = events_by_eid.get_group(eid)
        for step, step_t in enumerate(step_times):
            # J1 usage lookup
            k_key = f"{eid}@{step_t}"
            if k_key in j1_lookup:
                j1_tensor[i, step, :] = j1_lookup[k_key]
            else:
                # Fallback to preceding available step if present
                j1_tensor[i, step, :] = np.nan
                
            # J3 events lookup strictly <= step_t
            causal_evs = evs[evs["time_sec"] <= step_t]
            num_evicts = float((causal_evs["event_type"] == 2).sum())
            sched_evs = causal_evs[causal_evs["event_type"] == 1]
            last_sched = sched_evs["time_sec"].max() if not sched_evs.empty else step_t
            time_since_sched_min = float((step_t - last_sched) / 60.0)
            
            j3_tensor[i, step, 0] = num_evicts
            j3_tensor[i, step, 1] = time_since_sched_min
            
    return j1_tensor, j2_tensor, j3_tensor

print("Building additional features for Train...")
tr_j1, tr_j2, tr_j3 = build_additional_tensors(train_meta)
print(f"  Train J1 shape: {tr_j1.shape}, J2 shape: {tr_j2.shape}, J3 shape: {tr_j3.shape}")

print("Building additional features for Val...")
val_j1, val_j2, val_j3 = build_additional_tensors(val_meta)
print(f"  Val J1 shape: {val_j1.shape}, J2 shape: {val_j2.shape}, J3 shape: {val_j3.shape}")

# Check missingness in J1
tr_j1_nans = np.isnan(tr_j1).sum()
val_j1_nans = np.isnan(val_j1).sum()
print(f"J1 NaNs - Train: {tr_j1_nans} ({tr_j1_nans/tr_j1.size*100:.2f}%), Val: {val_j1_nans} ({val_j1_nans/val_j1.size*100:.2f}%)")
print(f"J2 NaNs - Train: {np.isnan(tr_j2).sum()}, Val: {np.isnan(val_j2).sum()}")
print(f"J3 NaNs - Train: {np.isnan(tr_j3).sum()}, Val: {np.isnan(val_j3).sum()}")

# Save unscaled raw tensors
out_raw_dir = PROJECT_ROOT / "results" / "phase7k" / "raw_features"
out_raw_dir.mkdir(parents=True, exist_ok=True)
np.save(out_raw_dir / "tr_j1.npy", tr_j1)
np.save(out_raw_dir / "tr_j2.npy", tr_j2)
np.save(out_raw_dir / "tr_j3.npy", tr_j3)
np.save(out_raw_dir / "val_j1.npy", val_j1)
np.save(out_raw_dir / "val_j2.npy", val_j2)
np.save(out_raw_dir / "val_j3.npy", val_j3)

print("Saved raw additional feature tensors.")
