import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import glob
import pandas as pd
import numpy as np
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator
from src.ingestion.validator import TimeSeriesValidator

with open("results/phase6/protocol_l5.json") as f:
    proto = json.load(f)

f_137 = set(proto["frozen_entity_ids"]["all_failure_entities"])

# Dev entities
tr = pd.read_csv("data/processed/train_metadata.csv")
val = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr["entity_id"]) | set(val["entity_id"]) | set(te["entity_id"])

# Raw events
raw_events = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
ev_dfs = [pd.read_csv(f, header=None, usecols=[0, 2, 3, 5]) for f in raw_events]
all_ev = pd.concat(ev_dfs, ignore_index=True)
all_ev["entity_id"] = all_ev[2].astype(str) + "_" + all_ev[3].astype(str)
all_ev["time_sec"] = all_ev[0] / 1e6
all_ev["event_type"] = all_ev[5]
fail_events = all_ev[all_ev["event_type"] == 3]
all_fail_eids = set(fail_events["entity_id"])
unused_fail_eids = all_fail_eids - dev_eids

# Raw usage
raw_usage = sorted(glob.glob("data/raw/sample_task_usage*.csv.gz"))
USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]
chunks = []
for uf in raw_usage:
    for chunk in pd.read_csv(uf, header=None, names=USAGE_COLS, chunksize=250000):
        chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
        m = chunk[chunk["entity_id"].isin(unused_fail_eids)].copy()
        if not m.empty:
            chunks.append(m)
df_u = pd.concat(chunks, ignore_index=True)
df_u["start_time_sec"] = df_u["start_time"] / 1e6
df_u["end_time_sec"] = df_u["end_time"] / 1e6
df_u["grid_bin"] = np.ceil(df_u["end_time_sec"] / 300.0) * 300.0
agg_funcs = {
    "start_time_sec": "min", "end_time_sec": "max", "cpu_rate": "mean",
    "canonical_memory": "mean", "assigned_memory": "mean", "disk_io_time": "mean",
    "cycles_per_instruction": "mean", "memory_accesses_per_instruction": "mean"
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

filtered_events = all_ev[all_ev["entity_id"].isin(set(usage_clean["entity_id"]))]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(usage_clean, filtered_events)

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(labeled_df)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

wg5 = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
X5, y5, meta5 = wg5.build_dataset_tensors(featured_df, feature_cols)

total_entities = meta5["entity_id"].nunique()
print(f"Total entities in valid L=5 windows: {total_entities}")
all_198 = set(meta5["entity_id"].unique())
diff_61 = sorted(list(all_198 - f_137))
print(f"Difference count: {len(diff_61)}")

meta_61 = meta5[meta5["entity_id"].isin(diff_61)]
print(f"61 entities window count: {len(meta_61)}")
print("61 entities label distribution:", meta_61["label"].value_counts().to_dict())

meta_137 = meta5[meta5["entity_id"].isin(f_137)]
print(f"137 entities window count: {len(meta_137)}")
print("137 entities label distribution:", meta_137["label"].value_counts().to_dict())

# Save diff_61 to scratch
with open("scratch/excluded_61_failing_tasks.json", "w") as f:
    json.dump(diff_61, f, indent=2)
print("Saved 61 excluded task IDs to scratch/excluded_61_failing_tasks.json")
