import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import glob

from src.ingestion.schema import ClusterEventTypeV2, FailureLabel
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

# Load dev partitions to identify dev entities
tr_meta = pd.read_csv("data/processed/train_metadata.csv")
va_meta = pd.read_csv("data/processed/val_metadata.csv")
te_meta = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr_meta["entity_id"]) | set(va_meta["entity_id"]) | set(te_meta["entity_id"])
print(f"Total dev entities: {len(dev_eids)}")

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
all_fails_df = all_events_df[all_events_df["event_type"] == ClusterEventTypeV2.FAIL]
raw_failing_eids = set(all_fails_df["entity_id"].unique())
print(f"1. Raw failing tasks (unique entity_ids with FAIL event): {len(raw_failing_eids):,}")

# Filter to unused failing tasks (not in dev)
unused_failing_eids = raw_failing_eids - dev_eids
print(f"   - Used in dev: {len(raw_failing_eids & dev_eids)}")
print(f"   - Unused in dev: {len(unused_failing_eids)}")

# Ingest telemetry for unused failing tasks
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
        m = chunk[chunk["entity_id"].isin(unused_failing_eids)].copy()
        if not m.empty:
            usage_chunks.append(m)

df_usage = pd.concat(usage_chunks, ignore_index=True)
df_usage["start_time_sec"] = df_usage["start_time"] / 1e6
df_usage["end_time_sec"] = df_usage["end_time"] / 1e6
failing_with_telemetry = set(df_usage["entity_id"].unique())
print(f"2. Unused failing tasks with telemetry: {len(failing_with_telemetry)}")

# Grid aggregation
df_usage["grid_bin"] = np.ceil(df_usage["end_time_sec"] / 300.0) * 300.0
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
    if c in df_usage.columns:
        agg_funcs[c] = "mean"

usage_resampled = df_usage.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_funcs)
usage_resampled["end_time_sec"] = usage_resampled["grid_bin"]
usage_resampled["timestamp"] = usage_resampled["end_time_sec"]
usage_resampled["cpu_utilization"] = usage_resampled["cpu_rate"]
usage_resampled["memory_utilization"] = usage_resampled["canonical_memory"]

validator = TimeSeriesValidator(expected_interval_sec=300.0)
usage_clean, _ = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")

# Label with TargetLabelGenerator
filtered_events = all_events_df[all_events_df["entity_id"].isin(failing_with_telemetry)]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(usage_clean, filtered_events)

print("\nObservation label distribution across all unused failing tasks:")
print(labeled_df["label"].value_counts().to_dict())

# Check how many unused entities have pre-failure labels (1 or 2)
entities_with_prefails = labeled_df[labeled_df["label"] > 0]["entity_id"].unique()
print(f"Unused failing entities with at least one Warning/Critical step: {len(entities_with_prefails)}")

# Now test WindowGenerator with L=5
registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(labeled_df)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

wg5 = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
X5, y5, meta5 = wg5.build_dataset_tensors(featured_df, feature_cols)

print(f"\n3. Windows generated with L=5:")
print(f"   Total valid windows: {len(X5)}")
print("   Window label distribution:", pd.Series(y5).value_counts().to_dict())
print(f"   Entities represented in windows: {meta5['entity_id'].nunique()}")

prefail_windows_5 = meta5[meta5["label"] > 0]
print(f"   Pre-failure windows (Warning + Critical): {len(prefail_windows_5)}")
print(f"   Warning windows (label == 1): {(meta5['label'] == 1).sum()}")
print(f"   Critical windows (label == 2): {(meta5['label'] == 2).sum()}")
print(f"   Failing entities contributing pre-failure windows: {prefail_windows_5['entity_id'].nunique()}")

# Distinct failure episodes among pre-failure windows
if not prefail_windows_5.empty:
    prefail_windows_5["fail_timestamp"] = (prefail_windows_5["prediction_timestamp"] + prefail_windows_5["time_to_failure"]).round(1)
    episodes = prefail_windows_5.groupby(["entity_id", "fail_timestamp"]).size()
    print(f"   Distinct failure episodes: {len(episodes)}")
