"""Profile the complete 2.52M row task_usage and 450K row task_events shards.

Computes task lifespans, telemetry coverage, failure linkages, and window eligibility.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

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

def profile():
    events_path = Path("data/raw/sample_task_events.csv.gz")
    usage_path = Path("data/raw/sample_task_usage.csv.gz")

    print("[1] Loading events shard...")
    events_df = pd.read_csv(events_path, header=None, names=EVENT_COLS)
    events_df["time_sec"] = events_df["timestamp"] / 1e6
    print(f"  Total events: {len(events_df):,}")
    print(f"  Time span: [{events_df['time_sec'].min():.1f}s, {events_df['time_sec'].max():.1f}s] ({ (events_df['time_sec'].max() - events_df['time_sec'].min())/3600:.2f} hours)")

    # Lifecycle events breakdown
    print("  Event counts by type:")
    for etype, count in events_df["event_type"].value_counts().sort_index().items():
        print(f"    Type {etype}: {count:,}")

    # Failures
    fails = events_df[events_df["event_type"] == 3] # v2.1 FAIL
    fail_tasks = set(zip(fails["job_id"], fails["task_index"]))
    print(f"  Total FAIL events: {len(fails):,} across {len(fail_tasks):,} unique tasks")

    # Finishes
    finishes = events_df[events_df["event_type"] == 4] # v2.1 FINISH
    finish_tasks = set(zip(finishes["job_id"], finishes["task_index"]))
    print(f"  Total FINISH events: {len(finishes):,} across {len(finish_tasks):,} unique tasks")

    # Kills
    kills = events_df[events_df["event_type"] == 5] # v2.1 KILL
    kill_tasks = set(zip(kills["job_id"], kills["task_index"]))
    print(f"  Total KILL events: {len(kills):,} across {len(kill_tasks):,} unique tasks")

    # Evicts
    evicts = events_df[events_df["event_type"] == 2] # v2.1 EVICT
    evict_tasks = set(zip(evicts["job_id"], evicts["task_index"]))
    print(f"  Total EVICT events: {len(evicts):,} across {len(evict_tasks):,} unique tasks")

    print("\n[2] Scanning complete task_usage shard in chunks of 500k...")
    task_window_counts = Counter()
    matched_fail_telemetry_windows = Counter()
    total_usage_rows = 0
    t_min = float("inf")
    t_max = float("-inf")

    chunk_size = 500000
    for chunk in pd.read_csv(usage_path, header=None, names=USAGE_COLS, chunksize=chunk_size):
        total_usage_rows += len(chunk)
        t_start_s = chunk["start_time"].min() / 1e6
        t_end_s = chunk["end_time"].max() / 1e6
        t_min = min(t_min, t_start_s)
        t_max = max(t_max, t_end_s)

        # Count windows per task
        keys = list(zip(chunk["job_id"], chunk["task_index"]))
        task_window_counts.update(keys)

        # Track failure matches
        for k in keys:
            if k in fail_tasks:
                matched_fail_telemetry_windows[k] += 1

    print(f"  Total task_usage records: {total_usage_rows:,}")
    print(f"  Time span: [{t_min:.1f}s, {t_max:.1f}s] ({ (t_max - t_min)/3600:.2f} hours)")
    print(f"  Unique tasks in task_usage: {len(task_window_counts):,}")

    # Analyze window length distribution
    lengths = list(task_window_counts.values())
    lengths_arr = np.array(lengths)
    print("\n[3] Task Duration Distribution (in 5-min intervals):")
    print(f"  - Tasks with exactly 1 window (<= 5 min)  : {(lengths_arr == 1).sum():,} ({(lengths_arr == 1).mean()*100:.1f}%)")
    print(f"  - Tasks with >= 3 windows (>= 15 min)     : {(lengths_arr >= 3).sum():,} ({(lengths_arr >= 3).mean()*100:.1f}%)")
    print(f"  - Tasks with >= 6 windows (>= 30 min)     : {(lengths_arr >= 6).sum():,} ({(lengths_arr >= 6).mean()*100:.1f}%)")
    print(f"  - Tasks with >= 15 windows (>= 75 min)    : {(lengths_arr >= 15).sum():,} ({(lengths_arr >= 15).mean()*100:.1f}%)")
    print(f"  - Tasks with >= 30 windows (>= 2.5 hours) : {(lengths_arr >= 30).sum():,} ({(lengths_arr >= 30).mean()*100:.1f}%)")
    print(f"  - Maximum windows for a single task       : {lengths_arr.max():,}")

    print("\n[4] Linkage to Failure Episodes:")
    print(f"  - Total unique tasks in usage that had FAIL: {len(matched_fail_telemetry_windows):,}")
    fail_lengths = [matched_fail_telemetry_windows[k] for k in matched_fail_telemetry_windows]
    fail_lengths_arr = np.array(fail_lengths)
    print(f"  - Failed tasks with >= 1 telemetry window : {len(fail_lengths_arr):,}")
    print(f"  - Failed tasks with >= 3 windows (>= 15m) : {(fail_lengths_arr >= 3).sum():,}")
    print(f"  - Failed tasks with >= 6 windows (>= 30m) : {(fail_lengths_arr >= 6).sum():,}")
    print(f"  - Failed tasks with >= 15 windows (>= 75m): {(fail_lengths_arr >= 15).sum():,}")

if __name__ == "__main__":
    profile()
