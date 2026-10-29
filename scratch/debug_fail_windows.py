import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator
from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS

with open("results/phase4/confirmatory_protocol.json") as f:
    protocol = json.load(f)

fail_eids = list(protocol["frozen_entity_ids"]["failure_entities"])[:5]
print("Testing fail eids:", fail_eids)

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
usage_clean, num_dups = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")

registry = FeatureRegistry()
engineer = FeatureEngineer(registry=registry)
featured_df = engineer.transform(usage_clean)
feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]

label_gen = TargetLabelGenerator(
    critical_horizon_sec=900.0,
    warning_horizon_sec=1800.0,
    trace_version="2011",
)
labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)

print("Labeled DF:")
for eid in fail_eids:
    sub = labeled_df[labeled_df["entity_id"] == eid]
    print(f"\nEntity {eid}: rows={len(sub)}, labels={sub['label'].value_counts().to_dict()}")
    print("Timestamps:", sub["end_time_sec"].values)
    deltas = np.diff(sub["end_time_sec"].values)
    print("Deltas:", deltas)
    
    # Try WindowGenerator
    wg = WindowGenerator(sequence_length=15, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
    wins = wg.create_windows_for_entity(sub, feature_cols)
    print(f"Windows created: {len(wins)}")
    if len(wins) == 0:
        # Check why is_valid_contiguous_history failed
        start_idx = 14
        for pred_idx in range(start_idx, len(sub)):
            h_times = sub["end_time_sec"].values[pred_idx - 14 : pred_idx + 1]
            deltas_h = np.diff(h_times)
            is_valid = wg.is_valid_contiguous_history(h_times)
            print(f"  pred_idx={pred_idx}, label={sub['label'].iloc[pred_idx]}, is_valid={is_valid}")
            print(f"    deltas min={deltas_h.min()}, max={deltas_h.max()}, total_dur={h_times[-1] - h_times[0]}")
