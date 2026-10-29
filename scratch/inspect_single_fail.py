import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

with open("results/phase4/confirmatory_protocol.json") as f:
    protocol = json.load(f)

eid = "6251995937_1195"

from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

event_dfs = []
for s_path in protocol["source_event_shards"]:
    p = Path(s_path)
    if p.exists():
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"] == eid]
        if not matched.empty:
            event_dfs.append(matched)
events_raw = pd.concat(event_dfs, ignore_index=True)
events_raw["time_sec"] = events_raw["timestamp"] / 1e6

usage_chunks = []
for s_path in protocol["source_usage_shards"]:
    p = Path(s_path)
    if p.exists():
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"] == eid]
            if not matched.empty:
                usage_chunks.append(matched)
usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

print(f"Usage for {eid}: rows={len(usage_df)}")
print(f"Start times: min={usage_df['start_time_sec'].min()}, max={usage_df['start_time_sec'].max()}")
print(f"End times: min={usage_df['end_time_sec'].min()}, max={usage_df['end_time_sec'].max()}")
print("Unique grid bins:", sorted(usage_df["grid_bin"].unique()))
print(f"Grid bin count: {len(usage_df['grid_bin'].unique())}")

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

print("\nLabeled steps for entity:")
print(labeled_df[["end_time_sec", "label", "time_to_failure_sec"]])

print("\nEvents for entity:")
print(events_raw[["time_sec", "event_type"]])

wg = WindowGenerator(sequence_length=15, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
wins = wg.create_windows_for_entity(labeled_df, feature_cols)
print(f"\nWindows generated: {len(wins)}")
if len(wins) == 0:
    for pred_idx in range(14, len(labeled_df)):
        h_times = labeled_df["end_time_sec"].values[pred_idx - 14 : pred_idx + 1]
        valid = wg.is_valid_contiguous_history(h_times)
        lbl = labeled_df["label"].iloc[pred_idx]
        print(f"pred_idx={pred_idx}, t={labeled_df['end_time_sec'].iloc[pred_idx]}, valid={valid}, label={lbl}")
