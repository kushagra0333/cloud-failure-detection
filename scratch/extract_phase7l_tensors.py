"""Extract L=5 and L=12 unscaled raw tensors for Candidate A Train and Validation."""
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.schema import ClusterEventTypeV2, FailureLabel
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

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

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_l5_and_l12(output_dir: Path):
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_split_csv = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
    cand_df = pd.read_csv(candidate_split_csv)
    
    # We only process train and val splits to strictly comply with quarantine!
    dev_cand_df = cand_df[cand_df["assigned_split"].isin(["train", "val"])].copy()
    target_entities = set(dev_cand_df["entity_id"].unique())
    print(f"Target entities in Train & Val: {len(target_entities)} (Train=284, Val=69, Total=353)")
    assert len(target_entities) == 353

    raw_event_shards = [
        "data/raw/sample_task_events.csv.gz",
        "data/raw/sample_task_events_part1.csv.gz",
        "data/raw/sample_task_events_part2.csv.gz",
    ]
    raw_usage_shards = [
        "data/raw/sample_task_usage.csv.gz",
        "data/raw/sample_task_usage_part1.csv.gz",
    ]

    # Ingest Events
    event_dfs = []
    for s_path in raw_event_shards:
        p = PROJECT_ROOT / s_path
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(target_entities)].copy()
        if not matched.empty:
            event_dfs.append(matched)

    events_raw = pd.concat(event_dfs, ignore_index=True)
    events_raw["time_sec"] = events_raw["timestamp"] / 1e6

    # Ingest Telemetry
    usage_chunks = []
    for s_path in raw_usage_shards:
        p = PROJECT_ROOT / s_path
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=500000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(target_entities)].copy()
            if not matched.empty:
                usage_chunks.append(matched)

    usage_df = pd.concat(usage_chunks, ignore_index=True)
    usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
    usage_df["end_time_sec"] = usage_df["end_time"] / 1e6

    # 300s Grid Alignment
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
    assert len(feature_cols) == 31

    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)

    # Build sliding windows with L=15 history
    window_gen = WindowGenerator(
        sequence_length=15,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X_full, y_full, meta_df = window_gen.build_dataset_tensors(labeled_df, feature_cols)
    print(f"Total dev sliding windows generated: N = {len(meta_df):,}")

    meta_df["key"] = meta_df["entity_id"] + "@" + meta_df["prediction_timestamp"].astype(str)
    dev_cand_df["key"] = dev_cand_df["entity_id"] + "@" + dev_cand_df["prediction_timestamp"].astype(str)

    cand_key_to_split = dict(zip(dev_cand_df["key"], dev_cand_df["assigned_split"]))
    cand_key_to_job = dict(zip(dev_cand_df["key"], dev_cand_df["parent_job_id"]))
    cand_key_to_label = dict(zip(dev_cand_df["key"], dev_cand_df["label"]))

    # Filter to only Candidate A Train & Val
    mask_in_dev = meta_df["key"].isin(cand_key_to_split)
    meta_df = meta_df[mask_in_dev].copy().reset_index(drop=True)
    X_full = X_full[mask_in_dev]
    y_full = y_full[mask_in_dev]

    meta_df["parent_job_id"] = meta_df["key"].map(cand_key_to_job)
    meta_df["assigned_split"] = meta_df["key"].map(cand_key_to_split)
    expected_labels = meta_df["key"].map(cand_key_to_label).values

    assert (meta_df["label"].values != expected_labels).sum() == 0

    # Partition Train and Val
    # Slices
    X_l5 = X_full[:, -5:, :]
    X_l12 = X_full[:, -12:, :]

    for s_name in ["train", "val"]:
        s_mask = (meta_df["assigned_split"] == s_name).values
        s_indices = np.where(s_mask)[0]

        X_5_split = X_l5[s_indices].copy()
        X_12_split = X_l12[s_indices].copy()
        y_split = y_full[s_indices].copy()
        meta_split = meta_df.iloc[s_indices].copy().reset_index(drop=True).drop(columns=["key"])

        np.save(output_dir / f"X_{s_name}_raw_l5.npy", X_5_split)
        np.save(output_dir / f"X_{s_name}_raw_l12.npy", X_12_split)
        np.save(output_dir / f"y_{s_name}.npy", y_split)
        meta_split.to_csv(output_dir / f"{s_name}_metadata.csv", index=False)
        print(f"[{s_name:5s}] Saved: X_l5={X_5_split.shape}, X_l12={X_12_split.shape}, y={y_split.shape}, meta={len(meta_split)}")

    elapsed = time.time() - start_time
    print(f"Extraction completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    out_d = PROJECT_ROOT / "results" / "phase7l_longer_context" / "raw_features"
    extract_l5_and_l12(out_d)
