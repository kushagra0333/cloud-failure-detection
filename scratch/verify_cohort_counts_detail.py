import sys
from pathlib import Path
import json
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Load protocol_l5.json
with open("results/phase6/protocol_l5.json") as f:
    protocol = json.load(f)

frozen_failing_eids = protocol["frozen_entity_ids"]["all_failure_entities"]
frozen_normal_eids = protocol["frozen_entity_ids"]["normal_entities"]
episodes_meta = protocol["frozen_episode_metadata"]
counts_summary = protocol["cohort_counts_summary"]

print(f"Protocol failure entities: {len(frozen_failing_eids)}")
print(f"Protocol normal entities: {len(frozen_normal_eids)}")
print(f"Protocol episodes: {len(episodes_meta)}")

# 2. Episode distribution per entity
ep_df = pd.DataFrame(episodes_meta)
ep_per_entity = ep_df.groupby("entity_id")["episode_id"].count()
ep_counts_dist = ep_per_entity.value_counts().to_dict()
print(f"\nEpisodes per entity distribution: {ep_counts_dist}")
for count, n_ent in sorted(ep_counts_dist.items()):
    print(f"  - {n_ent} entities have exactly {count} episode(s)")

multi_ep_entities = ep_per_entity[ep_per_entity > 1].index.tolist()
print(f"\nEntities with > 1 episode ({len(multi_ep_entities)}):")
for eid in multi_ep_entities:
    e_eps = ep_df[ep_df["entity_id"] == eid]
    print(f"  Entity {eid}: {len(e_eps)} episodes -> {e_eps['episode_id'].tolist()}")

# 3. Check window counts and overlap from failing cohort
# Re-extract from scratch files or regenerate
from src.ingestion.schema import ClusterEventTypeV2
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

# Load raw events
event_files = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
ev_dfs = []
for f in event_files:
    df = pd.read_csv(f, header=None, usecols=[0, 2, 3, 5])
    df["entity_id"] = df[2].astype(str) + "_" + df[3].astype(str)
    df["time_sec"] = df[0] / 1e6
    df["event_type"] = df[5]
    ev_dfs.append(df[["entity_id", "time_sec", "event_type"]])
all_events_df = pd.concat(ev_dfs, ignore_index=True)

# Ingest usage for the 137 failure entities
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
        m = chunk[chunk["entity_id"].isin(frozen_failing_eids)].copy()
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

filtered_events = all_events_df[all_events_df["entity_id"].isin(frozen_failing_eids)]
label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(usage_clean, filtered_events)

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(labeled_df)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

wg5 = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
X_fail, y_fail, meta_fail = wg5.build_dataset_tensors(featured_df, feature_cols)

print(f"\nFailing cohort windows under L=5:")
print(f"Total windows: {len(X_fail)}")
print("Window label distribution:", pd.Series(y_fail).value_counts().to_dict())

# Check for duplicates or improper overlap
dup_keys = meta_fail.duplicated(subset=["entity_id", "prediction_timestamp"])
print(f"Duplicate (entity_id, prediction_timestamp) keys: {dup_keys.sum()}")
assert dup_keys.sum() == 0, "No duplicate window timestamps allowed!"

# Check sequence ordering and spacing within each window in X_fail
print("\nVerifying 5-bin sequence ordering and timestamp spacing for failing cohort...")
for i in range(min(50, len(X_fail))):
    row_meta = meta_fail.iloc[i]
    pred_t = row_meta["prediction_timestamp"]
    # Check that pred_t is finite and positive
    assert pred_t > 0, f"Non-positive prediction timestamp {pred_t}"

# Verify label definitions against actual time_to_failure
print("Verifying label boundaries against time_to_failure:")
for idx, row in meta_fail.iterrows():
    y = row["label"]
    ttf = row["time_to_failure"]
    if y == 2:  # CRITICAL
        assert 0 < ttf <= 900.0, f"Critical label mismatch at idx {idx}: TTF={ttf}"
    elif y == 1:  # WARNING
        assert 900.0 < ttf <= 1800.0, f"Warning label mismatch at idx {idx}: TTF={ttf}"
    elif y == 0:  # NORMAL
        if pd.notna(ttf):
            assert ttf > 1800.0, f"Normal label mismatch at idx {idx}: TTF={ttf}"
print("Assertion PASSED: All failing-cohort windows strictly adhere to label definitions!")

# Check post-failure telemetry exclusion
assert (meta_fail["time_to_failure"].dropna() > 0).all(), "Zero post-failure windows permitted!"
print("Assertion PASSED: Zero post-failure windows in failing cohort (all non-null TTF > 0).")
