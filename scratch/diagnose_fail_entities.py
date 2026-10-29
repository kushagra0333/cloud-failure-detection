import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

with open("results/phase4/confirmatory_protocol.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["failure_entities"])

from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

# Load events
event_dfs = []
for s_path in protocol["source_event_shards"]:
    p = Path(s_path)
    if p.exists():
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(fail_eids)]
        if not matched.empty:
            event_dfs.append(matched)
events_raw = pd.concat(event_dfs, ignore_index=True)
events_raw["time_sec"] = events_raw["timestamp"] / 1e6

# Load usage
usage_chunks = []
for s_path in protocol["source_usage_shards"]:
    p = Path(s_path)
    if p.exists():
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(fail_eids)]
            if not matched.empty:
                usage_chunks.append(matched)
usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

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
    if c in usage_df.columns:
        agg_funcs[c] = "mean"

usage_resampled = usage_df.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_funcs)
usage_resampled["end_time_sec"] = usage_resampled["grid_bin"]
usage_resampled["timestamp"] = usage_resampled["end_time_sec"]
usage_resampled["cpu_utilization"] = usage_resampled["cpu_rate"]
usage_resampled["memory_utilization"] = usage_resampled["canonical_memory"]

validator = TimeSeriesValidator(expected_interval_sec=300.0)
usage_clean, _ = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(usage_clean)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")
labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)

step_counts = labeled_df.groupby("entity_id").size()
eligible_41 = step_counts[step_counts >= 15].index.tolist()
print(f"Number of fail entities with >= 15 steps: {len(eligible_41)}")

wg = WindowGenerator(sequence_length=15, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)

success_eids = []
reason_counts = {"has_gap": 0, "all_unknown_at_eval_steps": 0, "success": 0}

for eid in eligible_41:
    sub = labeled_df[labeled_df["entity_id"] == eid].sort_values("end_time_sec").reset_index(drop=True)
    wins = wg.create_windows_for_entity(sub, feature_cols)
    if len(wins) > 0:
        success_eids.append(eid)
        reason_counts["success"] += 1
    else:
        # diagnose
        num_rows = len(sub)
        timestamps = sub["end_time_sec"].values
        # check gaps
        has_gap = False
        valid_history_count = 0
        unknown_at_valid = 0
        for pred_idx in range(14, num_rows):
            h_times = timestamps[pred_idx - 14 : pred_idx + 1]
            if wg.is_valid_contiguous_history(h_times):
                valid_history_count += 1
                if sub["label"].iloc[pred_idx] == -1:
                    unknown_at_valid += 1
            else:
                has_gap = True
        if valid_history_count == 0:
            reason_counts["has_gap"] += 1
        elif unknown_at_valid == valid_history_count:
            reason_counts["all_unknown_at_eval_steps"] += 1

print("Diagnostic on 41 entities with >= 15 steps:")
print(reason_counts)
print(f"Entities with windows: {len(success_eids)}")

# Let's inspect the fail events for these 41 entities
print("\nFail events for the 41 entities:")
f_events = events_raw[(events_raw["entity_id"].isin(eligible_41)) & (events_raw["event_type"] == 3)]
print(f"Total FAIL event records: {len(f_events)} across {f_events['entity_id'].nunique()} entities")

# Let's look at one example of all_unknown_at_eval_steps
for eid in eligible_41:
    sub = labeled_df[labeled_df["entity_id"] == eid].sort_values("end_time_sec").reset_index(drop=True)
    wins = wg.create_windows_for_entity(sub, feature_cols)
    if len(wins) == 0:
        evs = events_raw[events_raw["entity_id"] == eid].sort_values("time_sec")
        print(f"\n--- Diagnostic for {eid} ---")
        print(f"Telemetry end_time_sec: min={sub['end_time_sec'].min()}, max={sub['end_time_sec'].max()}, len={len(sub)}")
        print("Events:")
        print(evs[["time_sec", "event_type"]])
        print("Telemetry steps:")
        print(sub[["end_time_sec", "label", "time_to_failure_sec"]].tail(10))
        break
