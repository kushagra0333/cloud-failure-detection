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

# Dev entities
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])

# Ingest all events
event_files = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
ev_dfs = []
for f in event_files:
    df = pd.read_csv(f, header=None, usecols=[0, 2, 3, 5])
    df["entity_id"] = df[2].astype(str) + "_" + df[3].astype(str)
    df["time_sec"] = df[0] / 1e6
    df["event_type"] = df[5]
    ev_dfs.append(df[["entity_id", "time_sec", "event_type"]])
all_events_df = pd.concat(ev_dfs, ignore_index=True)

# Phase 4/5 normal entities
with open("results/phase4/confirmatory_protocol.json") as f:
    p5 = json.load(f)
p5_norm = p5["frozen_entity_ids"]["normal_entities"]
print(f"Phase 5 normal entities in protocol: {len(p5_norm)}")

# Check usage shards
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
        m = chunk[chunk["entity_id"].isin(p5_norm)].copy()
        if not m.empty:
            chunks.append(m)

df_u = pd.concat(chunks, ignore_index=True)
df_u["start_time_sec"] = df_u["start_time"] / 1e6
df_u["end_time_sec"] = df_u["end_time"] / 1e6

telem_eids = set(df_u["entity_id"].unique())
print(f"p5_norm entities with telemetry in usage shards: {len(telem_eids)} / {len(p5_norm)}")

# Let's see which ones were missing telemetry:
missing_telem = set(p5_norm) - telem_eids
print(f"Entities with zero telemetry: {len(missing_telem)}")

# Now test grid aggregation and L=5 window generation on p5_norm
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

filtered_events = all_events_df[all_events_df["entity_id"].isin(telem_eids)]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(usage_clean, filtered_events)

print("Observation label counts across p5_norm:")
print(labeled_df["label"].value_counts().to_dict())

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(labeled_df)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

wg5 = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
X5, y5, meta5 = wg5.build_dataset_tensors(featured_df, feature_cols)

print(f"\np5_norm under L=5:")
print(f"Total valid windows: {len(X5)}")
print(f"Window label distribution: {pd.Series(y5).value_counts().to_dict()}")
print(f"Unique entities with valid windows: {meta5['entity_id'].nunique()}")

# Are there any non-zero labels?
assert (y5 == 0).all(), "All valid windows from normal cohort must have label == 0!"
print("Assertion PASSED: All windows from p5_norm are strictly label == 0 (Normal).")
missing_norm = sorted(list(set(p5_norm) - set(meta5["entity_id"].unique())))
print("Missing normal entities:", missing_norm)
