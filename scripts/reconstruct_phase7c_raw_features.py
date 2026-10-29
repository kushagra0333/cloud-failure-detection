"""Phase 7C.2 — Raw-Feature Reconstruction from Google Borg Trace Shards.

Reconstructs the Candidate A development dataset directly from raw Borg cluster trace files:
- data/raw/sample_task_events*.csv.gz
- data/raw/sample_task_usage*.csv.gz

Strict Non-Negotiable Invariants:
1. Reconstructs feature rows and entity/window identifiers using frozen Candidate A assignments.
2. Preserves genuine missing values as NaN until train-only imputation. Zero historical constants (e.g. 2.5).
3. Evaluates all features strictly causal (available at prediction time t <= t_pred).
4. Strictly excludes post-failure observations via TargetLabelGenerator (Policy A).
5. Joins reconstructed windows to Candidate A assignments using stable keys (entity_id, prediction_timestamp).
6. Outputs raw, unscaled feature tensors and metadata to data/processed/phase7c/raw_unscaled/.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

def reconstruct_raw_features(
    raw_event_shards: List[str],
    raw_usage_shards: List[str],
    candidate_split_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 7C.2: RAW-FEATURE RECONSTRUCTION FROM GOOGLE BORG TRACE")
    print("=" * 80)
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Candidate A frozen split assignments
    print(f"\n[Step 1] Loading frozen Candidate A split: {candidate_split_csv}")
    cand_df = pd.read_csv(candidate_split_csv)
    cand_split_hash = compute_sha256(candidate_split_csv)
    print(f"  - Frozen Split SHA-256: {cand_split_hash}")
    print(f"  - Total Candidate A Windows: {len(cand_df):,}")
    print(f"  - Unique Entities: {cand_df['entity_id'].nunique():,}")
    print(f"  - Unique Parent Jobs: {cand_df['parent_job_id'].nunique():,}")

    target_entities = set(cand_df["entity_id"].unique())

    # 2. Ingest Lifecycle Events for Candidate A entities
    print(f"\n[Step 2] Ingesting lifecycle events for {len(target_entities)} entities across {len(raw_event_shards)} shards...")
    event_dfs = []
    event_shard_hashes = {}
    for s_path in raw_event_shards:
        p = PROJECT_ROOT / s_path
        if not p.exists():
            raise FileNotFoundError(f"Missing required event shard: {p}")
        h = compute_sha256(p)
        event_shard_hashes[s_path] = h
        print(f"  - Scanning {p.name} ({p.stat().st_size:,} bytes | SHA-256: {h[:16]}...)...")
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(target_entities)].copy()
        if not matched.empty:
            event_dfs.append(matched)

    events_raw = pd.concat(event_dfs, ignore_index=True)
    events_raw["time_sec"] = events_raw["timestamp"] / 1e6
    print(f"  - Extracted {len(events_raw):,} lifecycle events across target entities.")
    matched_ev_eids = set(events_raw["entity_id"].unique())
    assert target_entities.issubset(matched_ev_eids), f"Missing events for {len(target_entities - matched_ev_eids)} entities!"
    print(f"  - Verified 100% event coverage for all {len(target_entities)} Candidate A entities.")

    # 3. Ingest Raw Telemetry for Candidate A entities
    print(f"\n[Step 3] Scanning raw telemetry shards for {len(target_entities)} entities...")
    usage_chunks = []
    usage_shard_hashes = {}
    for s_path in raw_usage_shards:
        p = PROJECT_ROOT / s_path
        if not p.exists():
            raise FileNotFoundError(f"Missing required usage shard: {p}")
        h = compute_sha256(p)
        usage_shard_hashes[s_path] = h
        print(f"  - Scanning {p.name} ({p.stat().st_size:,} bytes | SHA-256: {h[:16]}...)...")
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=500000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(target_entities)].copy()
            if not matched.empty:
                usage_chunks.append(matched)

    usage_df = pd.concat(usage_chunks, ignore_index=True)
    usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
    usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
    print(f"  - Extracted {len(usage_df):,} raw telemetry records.")
    matched_usage_eids = set(usage_df["entity_id"].unique())
    assert target_entities.issubset(matched_usage_eids), f"Missing usage for {len(target_entities - matched_usage_eids)} entities!"
    print(f"  - Verified 100% usage coverage for all {len(target_entities)} Candidate A entities.")

    # 4. Grid-Aligned 300s Aggregation
    print("\n[Step 4] Aligning telemetry to standard 300-second physical grid...")
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

    # 5. Data Validation & Duplicate Resolution
    print("\n[Step 5] Validating time series and resolving duplicates...")
    validator = TimeSeriesValidator(expected_interval_sec=300.0)
    usage_clean, num_dups = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")
    print(f"  - Duplicate entity-timestamp pairs resolved: {num_dups}")
    print(f"  - Cleaned telemetry rows: {len(usage_clean):,}")

    # 6. Feature Engineering (Preserving Genuine NaNs)
    print("\n[Step 6] Computing trailing mathematical features (Groups A-I)...")
    registry = FeatureRegistry()
    engineer = FeatureEngineer(registry=registry)
    featured_df = engineer.transform(usage_clean)
    feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]
    print(f"  - Features engineered: {len(feature_cols)} columns")

    # 7. Target Label Generation (Adhering to Policy A)
    print("\n[Step 7] Deriving ground-truth future failure labels (Policy A)...")
    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)

    # 8. Sliding Window Generation (History L=15, Consumed L=5)
    print("\n[Step 8] Building 15-step sliding history windows...")
    window_gen = WindowGenerator(
        sequence_length=15,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X_full, y_full, meta_df = window_gen.build_dataset_tensors(labeled_df, feature_cols)
    print(f"  - Valid sliding windows generated: N = {len(meta_df):,}")

    # Slice L=5 trailing sequence inputs
    X_l5 = X_full[:, -5:, :]
    assert X_l5.shape == (len(meta_df), 5, len(feature_cols)), f"Unexpected X_l5 shape: {X_l5.shape}"

    # 9. Exact Match & Join with Frozen Candidate A Split
    print("\n[Step 9] Joining reconstructed windows with Candidate A split assignments...")
    meta_df["key"] = meta_df["entity_id"] + "@" + meta_df["prediction_timestamp"].astype(str)
    cand_df["key"] = cand_df["entity_id"] + "@" + cand_df["prediction_timestamp"].astype(str)

    recon_keys = set(meta_df["key"])
    cand_keys = set(cand_df["key"])

    missing_in_recon = cand_keys - recon_keys
    excess_in_recon = recon_keys - cand_keys

    if missing_in_recon or excess_in_recon:
        raise ValueError(
            f"FATAL: Reconstructed window keys do not match Candidate A split!\n"
            f"Missing in reconstructed: {len(missing_in_recon)}\n"
            f"Excess in reconstructed: {len(excess_in_recon)}"
        )

    print(f"  - 100% Exact 1-to-1 Key Match: All {len(cand_keys):,} Candidate A windows reproduced perfectly!")

    # Map candidate A assignments and verify labels
    cand_key_to_split = dict(zip(cand_df["key"], cand_df["assigned_split"]))
    cand_key_to_job = dict(zip(cand_df["key"], cand_df["parent_job_id"]))
    cand_key_to_label = dict(zip(cand_df["key"], cand_df["label"]))

    meta_df["parent_job_id"] = meta_df["key"].map(cand_key_to_job)
    meta_df["assigned_split"] = meta_df["key"].map(cand_key_to_split)
    expected_labels = meta_df["key"].map(cand_key_to_label).values

    label_diffs = (meta_df["label"].values != expected_labels).sum()
    assert label_diffs == 0, f"Label mismatch detected in {label_diffs} reconstructed windows!"
    print(f"  - Label Invariant Verified: Reconstructed ground-truth labels match Candidate A assignments 100%.")

    # 10. Partition into Candidate A Splits
    print("\n[Step 10] Partitioning unscaled raw tensors by Candidate A assignments...")
    splits_data = {}
    for s_name in ["train", "val", "test"]:
        s_mask = (meta_df["assigned_split"] == s_name).values
        s_indices = np.where(s_mask)[0]

        X_split = X_l5[s_indices].copy()
        y_split = y_full[s_indices].copy()
        meta_split = meta_df.iloc[s_indices].copy().reset_index(drop=True)
        # Drop temporary join key
        meta_split = meta_split.drop(columns=["key"])

        splits_data[s_name] = {
            "X": X_split,
            "y": y_split,
            "meta": meta_split,
        }
        print(f"  - Partition '{s_name:5s}': {len(meta_split):,d} windows | X: {X_split.shape} | Entities: {meta_split['entity_id'].nunique()} | Parent Jobs: {meta_split['parent_job_id'].nunique()}")

    # 11. Analyze and Record Raw Missingness
    print("\n[Step 11] Analyzing genuine raw missingness before imputation...")
    missingness_stats = {}
    for s_name in ["train", "val", "test"]:
        X_s = splits_data[s_name]["X"]
        n_obs = X_s.shape[0] * X_s.shape[1]
        nan_counts = np.isnan(X_s).sum(axis=(0, 1))

        missingness_stats[s_name] = {
            "total_windows": X_s.shape[0],
            "total_sequence_steps": n_obs,
            "features_missingness": {}
        }
        for idx, col in enumerate(feature_cols):
            cnt = int(nan_counts[idx])
            pct = float(cnt / n_obs * 100.0)
            missingness_stats[s_name]["features_missingness"][col] = {
                "nan_count": cnt,
                "nan_pct": pct,
            }
            if cnt > 0:
                print(f"    * [{s_name}] {col:32s}: {cnt:5,d} / {n_obs:,d} ({pct:5.2f}%) genuine NaNs")

    # 12. Save Reconstructed Raw Arrays and Manifest
    print(f"\n[Step 12] Serializing unscaled raw arrays to {output_dir}...")
    manifest = {
        "reconstruction_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - start_time, 2),
        "source_shards": {
            "event_shards": event_shard_hashes,
            "usage_shards": usage_shard_hashes,
        },
        "candidate_a_split": {
            "path": str(candidate_split_csv),
            "sha256": cand_split_hash,
            "total_windows": len(cand_df),
        },
        "feature_definitions": {
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
        },
        "partitions": {},
        "missingness_summary": missingness_stats,
    }

    for s_name in ["train", "val", "test"]:
        X_p = output_dir / f"X_{s_name}_raw.npy"
        y_p = output_dir / f"y_{s_name}.npy"
        meta_p = output_dir / f"{s_name}_metadata.csv"

        np.save(X_p, splits_data[s_name]["X"])
        np.save(y_p, splits_data[s_name]["y"])
        splits_data[s_name]["meta"].to_csv(meta_p, index=False)

        manifest["partitions"][s_name] = {
            "windows": len(splits_data[s_name]["meta"]),
            "entities": int(splits_data[s_name]["meta"]["entity_id"].nunique()),
            "parent_jobs": int(splits_data[s_name]["meta"]["parent_job_id"].nunique()),
            "X_shape": list(splits_data[s_name]["X"].shape),
            "X_sha256": compute_sha256(X_p),
            "y_sha256": compute_sha256(y_p),
            "meta_sha256": compute_sha256(meta_p),
        }

    manifest_p = output_dir / "raw_reconstruction_manifest.json"
    with open(manifest_p, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  - Serialized manifest: {manifest_p}")
    print("=" * 80)
    print("RAW-FEATURE RECONSTRUCTION COMPLETE")
    print("=" * 80)
    return manifest

def main():
    raw_event_shards = [
        "data/raw/sample_task_events.csv.gz",
        "data/raw/sample_task_events_part1.csv.gz",
        "data/raw/sample_task_events_part2.csv.gz",
    ]
    raw_usage_shards = [
        "data/raw/sample_task_usage.csv.gz",
        "data/raw/sample_task_usage_part1.csv.gz",
    ]
    candidate_split_csv = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
    output_dir = PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled"

    reconstruct_raw_features(
        raw_event_shards=raw_event_shards,
        raw_usage_shards=raw_usage_shards,
        candidate_split_csv=candidate_split_csv,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()
