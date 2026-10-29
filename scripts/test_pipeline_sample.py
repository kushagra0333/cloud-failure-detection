"""Test scratch script to verify rapid filtering, feature engineering, labeling, and windowing."""

from pathlib import Path
import pandas as pd
import numpy as np

USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]

EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]

def test():
    events_path = Path("data/raw/sample_task_events.csv.gz")
    usage_path = Path("data/raw/sample_task_usage.csv.gz")

    print("1. Loading events...")
    events = pd.read_csv(events_path, header=None, names=EVENT_COLS)
    events["entity_id"] = events["job_id"].astype(str) + "_" + events["task_index"].astype(str)
    events["time_sec"] = events["timestamp"] / 1e6

    # Identify all failed tasks
    fail_tasks = set(events[events["event_type"] == 3]["entity_id"].unique())
    finish_tasks = set(events[events["event_type"] == 4]["entity_id"].unique())
    print(f"Failed task IDs: {len(fail_tasks)}, Finish task IDs: {len(finish_tasks)}")

    # We want ALL failed tasks + a balanced sample of 1,500 finished/normal tasks
    rng = np.random.RandomState(42)
    selected_finish = set(rng.choice(list(finish_tasks), size=min(1500, len(finish_tasks)), replace=False))
    selected_eids = fail_tasks | selected_finish
    print(f"Target tasks to extract: {len(selected_eids)} tasks")

    # Fast chunk scan of usage_path
    print("2. Scanning task_usage for selected tasks...")
    chunks = []
    for chunk in pd.read_csv(usage_path, header=None, names=USAGE_COLS, chunksize=500000):
        chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
        matched = chunk[chunk["entity_id"].isin(selected_eids)].copy()
        if not matched.empty:
            chunks.append(matched)

    usage_df = pd.concat(chunks, ignore_index=True)
    print(f"Extracted {len(usage_df)} telemetry rows across {usage_df['entity_id'].nunique()} tasks!")
    print(f"Tasks per window count:")
    counts = usage_df.groupby("entity_id").size()
    print(f"  >= 15 windows: {(counts >= 15).sum()} tasks")
    print(f"  < 15 windows : {(counts < 15).sum()} tasks")

if __name__ == "__main__":
    test()
