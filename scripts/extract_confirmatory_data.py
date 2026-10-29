"""Extracts and prepares frozen confirmatory evaluation dataset from raw shards.

Strict Scientific Invariants:
- Ingests strictly the 300 frozen entities pre-registered in results/phase4/confirmatory_protocol.json.
- Zero replacement, zero resampling.
- Uses frozen Phase 2 pipeline: 300s grid aggregation, 31 causal features, 15-step sequence windows.
- Uses frozen scaler fitted strictly on training data (configs/fitted_scaler.json).
- Audits and documents every single entity's window generation status.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.schema import ClusterEventTypeV2
from src.ingestion.validator import TimeSeriesValidator
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.features.scaler import LeakageSafeScaler
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]

USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]


def extract_confirmatory_dataset(
    protocol_path: str = "results/phase4/confirmatory_protocol.json",
    output_dir: str = "data/processed/confirmatory",
    scaler_path: str = "configs/fitted_scaler.json",
):
    print("=" * 78)
    print("PHASE 5: CONFIRMATORY DATASET EXTRACTION & PROCESSING")
    print("=" * 78)

    with open(protocol_path) as f:
        protocol = json.load(f)

    fail_eids = set(protocol["frozen_entity_ids"]["failure_entities"])
    norm_eids = set(protocol["frozen_entity_ids"]["normal_entities"])
    all_target_eids = fail_eids | norm_eids
    print(f"Target Frozen Confirmatory Population: {len(all_target_eids)} entities")
    print(f"  - Failure Entities: {len(fail_eids)}")
    print(f"  - Normal Entities : {len(norm_eids)}")

    # 1. Ingest Events
    print("\n[Step 1] Ingesting lifecycle events for confirmatory entities...")
    event_dfs = []
    for s_path in protocol["source_event_shards"]:
        p = Path(s_path)
        if p.exists():
            df = pd.read_csv(p, header=None, names=EVENT_COLS)
            df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
            matched = df[df["entity_id"].isin(all_target_eids)]
            if not matched.empty:
                event_dfs.append(matched)

    events_raw = pd.concat(event_dfs, ignore_index=True)
    events_raw["time_sec"] = events_raw["timestamp"] / 1e6
    print(f"  - Extracted {len(events_raw)} event records across {events_raw['entity_id'].nunique()} entities")

    # 2. Ingest Telemetry
    print("\n[Step 2] Ingesting telemetry for confirmatory entities...")
    usage_chunks = []
    t0 = time.time()
    for s_path in protocol["source_usage_shards"]:
        p = Path(s_path)
        if p.exists():
            for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
                chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
                matched = chunk[chunk["entity_id"].isin(all_target_eids)].copy()
                if not matched.empty:
                    usage_chunks.append(matched)

    t1 = time.time()
    usage_df = pd.concat(usage_chunks, ignore_index=True)
    print(f"  - Extracted {len(usage_df):,} raw telemetry records in {t1 - t0:.2f}s")
    print(f"  - Telemetry entities present: {usage_df['entity_id'].nunique()} of {len(all_target_eids)}")

    usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
    usage_df["end_time_sec"] = usage_df["end_time"] / 1e6

    # 2.1 300s Grid-aligned aggregation
    print("\n[Step 3] Applying 300s grid aggregation...")
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

    # 3. Validation & Duplicate resolution
    validator = TimeSeriesValidator(expected_interval_sec=300.0)
    usage_clean, num_dups = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")
    print(f"  - Cleaned grid telemetry rows: {len(usage_clean):,}")

    # 4. Feature Engineering
    print("\n[Step 4] Computing 31 causal features...")
    registry = FeatureRegistry()
    engineer = FeatureEngineer(registry=registry)
    featured_df = engineer.transform(usage_clean)
    feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]
    print(f"  - Extracted {len(feature_cols)} feature columns")

    # 5. Target Labeling
    print("\n[Step 5] Deriving ground-truth failure horizons (Warning=30m, Critical=15m)...")
    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)
    print("  - Label distribution across steps:", labeled_df["label"].value_counts().to_dict())

    # 6. Window Generation (L=15 steps = 75 minutes)
    print("\n[Step 6] Constructing 15-step sliding windows...")
    window_gen = WindowGenerator(
        sequence_length=15,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X_raw, y_raw, meta_df = window_gen.build_dataset_tensors(labeled_df, feature_cols)
    print(f"  - Generated {len(X_raw):,} valid sliding windows")
    print("  - Window-level label distribution:", pd.Series(y_raw).value_counts().to_dict())
    print(f"  - Unique entities represented in windows: {meta_df['entity_id'].nunique()}")

    # 7. Apply Frozen Scaler
    print(f"\n[Step 7] Scaling features using frozen scaler from {scaler_path}...")
    scaler = LeakageSafeScaler.load(scaler_path)
    X_scaled = np.zeros_like(X_raw)
    for step in range(X_raw.shape[1]):
        step_df = pd.DataFrame(X_raw[:, step, :], columns=feature_cols)
        X_scaled[:, step, :] = scaler.transform(step_df).values

    # Check finite
    assert np.isfinite(X_scaled).all(), "Non-finite values detected in scaled confirmatory tensors!"

    # 8. Save Processed Confirmatory Artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_confirmatory.npy", X_scaled)
    np.save(out_dir / "y_confirmatory.npy", y_raw)
    meta_df.to_csv(out_dir / "confirmatory_metadata.csv", index=False)

    print(f"\nSaved confirmatory dataset to {out_dir}:")
    print(f"  - X_confirmatory.npy: {X_scaled.shape}")
    print(f"  - y_confirmatory.npy: {y_raw.shape}")
    print(f"  - confirmatory_metadata.csv: {len(meta_df)} rows")

    # Document entity coverage
    evaluated_eids = set(meta_df["entity_id"].unique())
    missing_eids = all_target_eids - evaluated_eids
    print(f"\n[Entity Coverage Summary]")
    print(f"  - Total Frozen Entities: {len(all_target_eids)}")
    print(f"  - Successfully Evaluated Entities: {len(evaluated_eids)} ({len(evaluated_eids & fail_eids)} fail, {len(evaluated_eids & norm_eids)} normal)")
    print(f"  - Entities with <15 steps (excluded): {len(missing_eids)}")
    if missing_eids:
        print(f"    Excluded entities: {sorted(list(missing_eids))}")

    return {
        "total_frozen_entities": len(all_target_eids),
        "evaluated_entities": len(evaluated_eids),
        "evaluated_fail_entities": len(evaluated_eids & fail_eids),
        "evaluated_norm_entities": len(evaluated_eids & norm_eids),
        "excluded_entities": sorted(list(missing_eids)),
        "total_windows": len(X_scaled),
        "window_distribution": pd.Series(y_raw).value_counts().to_dict(),
    }


if __name__ == "__main__":
    extract_confirmatory_dataset()
