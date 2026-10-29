"""Detailed audit of lifecycle events and candidate usage features for dev entities."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()

train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"

train_meta = pd.read_csv(train_meta_p)
val_meta = pd.read_csv(val_meta_p)
dev_meta = pd.concat([train_meta, val_meta], ignore_index=True)
dev_eids = set(dev_meta["entity_id"].unique())

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

# Check distribution of static attributes per entity
print("=== STATIC ATTRIBUTES PER DEV ENTITY ===")
ent_attrs = dev_events.groupby("entity_id").agg({
    "scheduling_class": "first",
    "priority": "first",
    "req_cpu": "first",
    "req_ram": "first",
    "req_disk": "first",
    "diff_machine_restriction": "first"
})

print("Scheduling class distribution:")
print(ent_attrs["scheduling_class"].value_counts())
print("\nPriority distribution:")
print(ent_attrs["priority"].value_counts())
print("\nResource requests describe:")
print(ent_attrs[["req_cpu", "req_ram", "req_disk"]].describe())

# Check lifecycle events occurring strictly before/at prediction_timestamp
print("\n=== DYNAMIC LIFECYCLE EVENTS OBSERVABLE AT t ===")
# Group dev_events by entity
events_by_eid = dev_events.groupby("entity_id")

window_event_stats = []
for _, row in dev_meta.iterrows():
    eid = row["entity_id"]
    t = row["prediction_timestamp"]
    y = row["label"]
    split = row["assigned_split"]
    
    if eid in events_by_eid.groups:
        evs = events_by_eid.get_group(eid)
        obs_evs = evs[evs["time_sec"] <= t]
        
        num_submits = int((obs_evs["event_type"] == 0).sum())
        num_schedules = int((obs_evs["event_type"] == 1).sum())
        num_evicts = int((obs_evs["event_type"] == 2).sum())
        num_fails = int((obs_evs["event_type"] == 3).sum())
        
        # Reschedule count: additional schedules beyond the first
        reschedule_count = max(0, num_schedules - 1)
        
        # Time since latest schedule
        sched_evs = obs_evs[obs_evs["event_type"] == 1]
        time_since_schedule = (t - sched_evs["time_sec"].max()) if not sched_evs.empty else np.nan
        
        # Priority and scheduling class
        prio = obs_evs["priority"].iloc[-1] if not obs_evs.empty else np.nan
        sched_cls = obs_evs["scheduling_class"].iloc[-1] if not obs_evs.empty else np.nan
        req_cpu = obs_evs["req_cpu"].iloc[-1] if not obs_evs.empty else np.nan
        req_ram = obs_evs["req_ram"].iloc[-1] if not obs_evs.empty else np.nan
        
        window_event_stats.append({
            "entity_id": eid,
            "prediction_timestamp": t,
            "label": y,
            "split": split,
            "num_submits": num_submits,
            "num_schedules": num_schedules,
            "reschedule_count": reschedule_count,
            "num_evicts": num_evicts,
            "num_fails": num_fails,
            "time_since_schedule": time_since_schedule,
            "priority": prio,
            "scheduling_class": sched_cls,
            "req_cpu": req_cpu,
            "req_ram": req_ram
        })

df_win_evs = pd.DataFrame(window_event_stats)
print(f"Total evaluated windows: {len(df_win_evs)}")
print("\nDynamic event counts summary at t:")
print(f"Windows with num_evicts > 0: {(df_win_evs['num_evicts'] > 0).sum()} / {len(df_win_evs)} ({(df_win_evs['num_evicts'] > 0).mean()*100:.2f}%)")
print(f"Windows with reschedule_count > 0: {(df_win_evs['reschedule_count'] > 0).sum()} / {len(df_win_evs)} ({(df_win_evs['reschedule_count'] > 0).mean()*100:.2f}%)")
print(f"Windows with num_fails > 0: {(df_win_evs['num_fails'] > 0).sum()} / {len(df_win_evs)} ({(df_win_evs['num_fails'] > 0).mean()*100:.2f}%)")

print("\nMean dynamic events by label (0=Normal, 1=Warning, 2=Critical):")
print(df_win_evs.groupby("label")[["num_submits", "num_schedules", "reschedule_count", "num_evicts", "priority", "scheduling_class", "req_cpu", "req_ram"]].mean())

print("\nDistribution of reschedule_count by label:")
print(pd.crosstab(df_win_evs["label"], df_win_evs["reschedule_count"]))

print("\nDistribution of num_evicts by label:")
print(pd.crosstab(df_win_evs["label"], df_win_evs["num_evicts"]))

print("\nDistribution of priority by label:")
print(pd.crosstab(df_win_evs["label"], df_win_evs["priority"]))

print("\nDistribution of scheduling_class by label:")
print(pd.crosstab(df_win_evs["label"], df_win_evs["scheduling_class"]))
