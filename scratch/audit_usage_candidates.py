"""Audit raw usage candidate features across Candidate A Train and Val."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()

train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"

train_meta = pd.read_csv(train_meta_p)
val_meta = pd.read_csv(val_meta_p)
dev_meta = pd.concat([train_meta, val_meta], ignore_index=True)
dev_eids = set(dev_meta["entity_id"].unique())

USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]

usage_shards = [
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage_part1.csv.gz",
]

usage_chunks = []
for s_path in usage_shards:
    if s_path.exists():
        for chunk in pd.read_csv(s_path, header=None, names=USAGE_COLS, chunksize=500000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(dev_eids)].copy()
            if not matched.empty:
                usage_chunks.append(matched)

usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

# Aggregate to 300s bins
agg_cols = {
    "cpu_rate": "mean",
    "canonical_memory": "mean",
    "assigned_memory": "mean",
    "disk_io_time": "mean",
    "max_cpu_rate": "mean",
    "max_memory": "mean",
    "total_page_cache": "mean",
    "unmapped_page_cache": "mean",
    "local_disk_space": "mean",
    "max_disk_io_time": "mean",
    "cycles_per_instruction": "mean",
    "memory_accesses_per_instruction": "mean",
}

resampled = usage_df.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_cols)
resampled.rename(columns={"grid_bin": "prediction_timestamp"}, inplace=True)

# Merge with dev_meta (3935 windows)
merged = pd.merge(dev_meta, resampled, on=["entity_id", "prediction_timestamp"], how="inner")
print(f"Merged dev windows with resampled usage: {len(merged)} / {len(dev_meta)} windows")

print("\n--- CANDIDATE USAGE COLUMNS SUMMARY (Merged Dev Windows) ---")
candidate_cols = ["max_cpu_rate", "max_memory", "total_page_cache", "unmapped_page_cache", "local_disk_space", "max_disk_io_time"]
for col in candidate_cols:
    nulls = merged[col].isna().sum()
    print(f"{col:25s}: {nulls:4d} nulls ({nulls/len(merged)*100:.2f}%) | mean={merged[col].mean():.5f}, std={merged[col].std():.5f}, min={merged[col].min():.5f}, max={merged[col].max():.5f}")

# Compute candidate derived features:
merged["cpu_burstiness"] = merged["max_cpu_rate"] - merged["cpu_rate"]
merged["memory_headroom"] = merged["assigned_memory"] - merged["max_memory"]
merged["memory_max_ratio"] = merged["max_memory"] / (merged["assigned_memory"] + 1e-5)
merged["page_cache_ratio"] = merged["total_page_cache"] / (merged["canonical_memory"] + 1e-5)

print("\n--- CANDIDATE DERIVED USAGE FEATURES BY LABEL ---")
derived_cols = ["cpu_burstiness", "memory_headroom", "memory_max_ratio", "page_cache_ratio", "local_disk_space"]
print(merged.groupby("label")[derived_cols].mean())

print("\nCorrelations with binary failure (label > 0):")
merged["is_failure"] = (merged["label"] > 0).astype(int)
for col in candidate_cols + derived_cols:
    corr = merged[col].corr(merged["is_failure"])
    print(f"  {col:25s}: correlation = {corr:+.4f}")
