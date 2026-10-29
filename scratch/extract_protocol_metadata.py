import sys
from pathlib import Path
import glob
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.schema import ClusterEventTypeV2
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

# 1. Dev entities
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])
dev_jobs = set(eid.split("_")[0] for eid in dev_eids)

# 2. Ingest raw events
event_files = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
ev_dfs = []
for f in event_files:
    df = pd.read_csv(f, header=None, usecols=[0, 2, 3, 5])
    df["entity_id"] = df[2].astype(str) + "_" + df[3].astype(str)
    df["time_sec"] = df[0] / 1e6
    df["event_type"] = df[5]
    ev_dfs.append(df[["entity_id", "time_sec", "event_type"]])
all_events_df = pd.concat(ev_dfs, ignore_index=True)
all_fails_df = all_events_df[all_events_df["event_type"] == ClusterEventTypeV2.FAIL]
raw_failing_eids = set(all_fails_df["entity_id"].unique())
unused_failing_eids = raw_failing_eids - dev_eids

# 3. Ingest usage shards for unused failing tasks
usage_files = sorted(glob.glob("data/raw/sample_task_usage*.csv.gz"))
USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]

chunks = []
for uf in usage_files:
    for chunk in pd.read_csv(uf, header=None, names=USAGE_COLS, chunksize=250000):
        chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
        m = chunk[chunk["entity_id"].isin(unused_failing_eids)].copy()
        if not m.empty:
            chunks.append(m)

df_u = pd.concat(chunks, ignore_index=True)
df_u["start_time_sec"] = df_u["start_time"] / 1e6
df_u["end_time_sec"] = df_u["end_time"] / 1e6

df_u["grid_bin"] = np.ceil(df_u["end_time_sec"] / 300.0) * 300.0
agg_funcs = {
    "start_time_sec": "min",
    "end_time_sec": "max",
    "cpu_rate": "mean",
    "canonical_memory": "mean",
    "assigned_memory": "mean",
    "disk_io_time": "mean",
    "cycles_per_instruction": "mean",
    "memory_accesses_per_instruction": "mean",
}
for c in ["unmapped_page_cache", "total_page_cache", "max_memory", "local_disk_space", "max_cpu_rate", "max_disk_io_time"]:
    if c in df_u.columns:
        agg_funcs[c] = "mean"

usage_resampled = df_u.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_funcs)
usage_resampled["end_time_sec"] = usage_resampled["grid_bin"]
usage_resampled["timestamp"] = usage_resampled["end_time_sec"]
usage_resampled["cpu_utilization"] = usage_resampled["cpu_rate"]
usage_resampled["memory_utilization"] = usage_resampled["canonical_memory"]

validator = TimeSeriesValidator(expected_interval_sec=300.0)
usage_clean, _ = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")

filtered_events = all_events_df[all_events_df["entity_id"].isin(df_u["entity_id"].unique())]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(usage_clean, filtered_events)

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(labeled_df)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

wg5 = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
X5, y5, meta5 = wg5.build_dataset_tensors(featured_df, feature_cols)

prefail_windows = meta5[meta5["label"] > 0].copy()
prefail_windows["fail_timestamp"] = (prefail_windows["prediction_timestamp"] + prefail_windows["time_to_failure"]).round(1)

# Group by entity and episode
episodes_df = prefail_windows.groupby(["entity_id", "fail_timestamp"]).agg(
    num_windows=("label", "count"),
    num_warning=("label", lambda s: (s == 1).sum()),
    num_critical=("label", lambda s: (s == 2).sum()),
    earliest_pred=("prediction_timestamp", "min"),
    latest_pred=("prediction_timestamp", "max"),
).reset_index()

episodes_df["parent_job_id"] = episodes_df["entity_id"].apply(lambda x: x.split("_")[0])
episodes_df["is_parent_job_in_dev"] = episodes_df["parent_job_id"].isin(dev_jobs)
episodes_df["episode_id"] = episodes_df["entity_id"] + "@" + episodes_df["fail_timestamp"].astype(str)

print("EPISODES SUMMARY:")
print(f"Total episodes: {len(episodes_df)}")
print(f"Total unique entities: {episodes_df['entity_id'].nunique()}")
print(f"Total unique parent jobs: {episodes_df['parent_job_id'].nunique()}")
print(f"Episodes from dev-present jobs: {(episodes_df['is_parent_job_in_dev']).sum()}")
print(f"Episodes from dev-absent (disjoint) jobs: {(~episodes_df['is_parent_job_in_dev']).sum()}")

# Entities summary
entities_df = episodes_df.groupby("entity_id").agg(
    parent_job_id=("parent_job_id", "first"),
    is_parent_job_in_dev=("is_parent_job_in_dev", "first"),
    num_episodes=("episode_id", "count"),
    total_prefail_windows=("num_windows", "sum"),
    warning_windows=("num_warning", "sum"),
    critical_windows=("num_critical", "sum")
).reset_index()

print("\nENTITIES SUMMARY:")
print(f"Total failing entities: {len(entities_df)}")
print(f"Entities from dev-present jobs: {entities_df['is_parent_job_in_dev'].sum()}")
print(f"Entities from dev-absent jobs: {(~entities_df['is_parent_job_in_dev']).sum()}")

# Parent jobs summary
jobs_df = entities_df.groupby("parent_job_id").agg(
    is_in_dev=("is_parent_job_in_dev", "first"),
    num_entities=("entity_id", "count"),
    num_episodes=("num_episodes", "sum"),
    total_prefail_windows=("total_prefail_windows", "sum")
).reset_index()

print("\nPARENT JOBS SUMMARY:")
print(jobs_df.to_string())

# Save metadata to scratch for protocol freezing
episodes_df.to_json("scratch/failing_episodes_metadata.json", orient="records", indent=2)
entities_df.to_json("scratch/failing_entities_metadata.json", orient="records", indent=2)
jobs_df.to_json("scratch/failing_jobs_metadata.json", orient="records", indent=2)
print("\nSaved metadata files to scratch/")
