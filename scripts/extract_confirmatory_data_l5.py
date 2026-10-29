"""Extracts and prepares the frozen Phase 6.2 L=5 confirmatory evaluation dataset from raw shards.

Strict Scientific Invariants:
1. Ingests strictly the ratified primary population pre-registered in results/phase6/protocol_l5_amendment_2_1.json.
2. Quarantines and rejects the 61 non-pre-failing census tasks.
3. Uses frozen Phase 2 pipeline: 300s grid aggregation, 31 causal features, 5-step sequence windows (L=5).
4. Uses frozen scaler fitted strictly on training data (configs/fitted_scaler.json).
5. Verifies that extracted windows strictly match the locked primary evaluation denominators:
   - Exactly 5,511 total sliding windows (400 Critical, 300 Warning, 4,811 Normal).
   - Exactly 4,515 pure-normal windows + 296 failing-cohort normal windows.
   - Exactly 137 failure entities + 198 evaluable pure-normal entities = 335 evaluable entities.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

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
from src.evaluation.confirmatory_evaluator_l5 import (
    load_ratified_l5_protocol,
    validate_evaluation_manifest,
)

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


def extract_l5_confirmatory_dataset(
    protocol_path: str = "results/phase6/protocol_l5_amendment_2_1.json",
    output_dir: Optional[str] = None,
    scaler_path: str = "configs/fitted_scaler.json",
    dry_run: bool = False,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 6.2.3: L=5 CONFIRMATORY DATASET EXTRACTION & VALIDATION")
    print("=" * 80)

    # 1. Load ratified protocol amendment
    print(f"\n[Step 1] Loading ratified protocol amendment from {protocol_path}...")
    protocol = load_ratified_l5_protocol(protocol_path)
    print(f"  - Protocol Version: Amendment {protocol['amendment_version']}")
    print(f"  - Status: {protocol['status']}")

    frozen_eids = protocol["frozen_entity_ids"]
    fail_eids = set(frozen_eids["primary_failure_entities"])
    norm_eids = set(frozen_eids["primary_normal_entities_sampled"])
    eval_norm_eids = set(frozen_eids["primary_normal_entities_evaluable"])
    excluded_census_eids = set(frozen_eids["excluded_census_failure_tasks"])

    all_target_eids = fail_eids | norm_eids
    print(f"Target Primary Population: {len(all_target_eids)} entities")
    print(f"  - Failure Entities (pre-fail yielding): {len(fail_eids)}")
    print(f"  - Sampled Pure-Normal Entities: {len(norm_eids)} ({len(eval_norm_eids)} evaluable under L=5)")
    print(f"  - Quarantined Census Failure Tasks: {len(excluded_census_eids)} (strictly prohibited from ingestion)")

    # Assert zero overlap
    assert len(all_target_eids & excluded_census_eids) == 0, "Quarantined census tasks found in target cohort!"

    # 2. Ingest Lifecycle Events
    print("\n[Step 2] Ingesting lifecycle events for confirmatory entities...")
    event_files = sorted(Path("data/raw").glob("sample_task_events*.csv.gz"))
    event_dfs = []
    for ep in event_files:
        df = pd.read_csv(ep, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(all_target_eids)]
        if not matched.empty:
            event_dfs.append(matched)

    events_raw = pd.concat(event_dfs, ignore_index=True)
    events_raw["time_sec"] = events_raw["timestamp"] / 1e6
    print(f"  - Ingested {len(events_raw)} event records across {events_raw['entity_id'].nunique()} entities")

    # 3. Ingest Telemetry
    print("\n[Step 3] Ingesting telemetry for confirmatory entities...")
    usage_files = sorted(Path("data/raw").glob("sample_task_usage*.csv.gz"))
    usage_chunks = []
    t0 = time.time()
    for uf in usage_files:
        for chunk in pd.read_csv(uf, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(all_target_eids)].copy()
            if not matched.empty:
                usage_chunks.append(matched)
    t1 = time.time()

    usage_df = pd.concat(usage_chunks, ignore_index=True)
    print(f"  - Extracted {len(usage_df):,} raw telemetry records in {t1 - t0:.2f}s")
    print(f"  - Entities with telemetry: {usage_df['entity_id'].nunique()} of {len(all_target_eids)}")

    usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
    usage_df["end_time_sec"] = usage_df["end_time"] / 1e6

    # 4. 300s Grid-aligned aggregation
    print("\n[Step 4] Applying 300s grid aggregation...")
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

    # 5. Validation & Duplicate resolution
    validator = TimeSeriesValidator(expected_interval_sec=300.0)
    usage_clean, num_dups = validator.detect_and_resolve_duplicates(usage_resampled, resolution="mean")
    print(f"  - Cleaned grid telemetry rows: {len(usage_clean):,}")

    # 6. Feature Engineering
    print("\n[Step 5] Computing 31 causal features...")
    registry = FeatureRegistry()
    engineer = FeatureEngineer(registry=registry)
    featured_df = engineer.transform(usage_clean)
    feature_cols = [f.name for f in registry.list_features() if f.name in featured_df.columns]
    print(f"  - Extracted {len(feature_cols)} feature columns")

    # 7. Target Labeling (0=Normal, 1=Warning [15-30m], 2=Critical [0-15m])
    print("\n[Step 6] Deriving ground-truth target horizons...")
    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(featured_df, events_raw)
    print("  - Label distribution across steps:", labeled_df["label"].value_counts().to_dict())

    # 8. Window Generation (L=5 steps = 25 minutes physical span)
    print("\n[Step 7] Constructing L=5 sliding windows (require_full_history=True)...")
    window_gen = WindowGenerator(
        sequence_length=5,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X_raw, y_raw, meta_df = window_gen.build_dataset_tensors(labeled_df, feature_cols)
    print(f"  - Total windows generated: {len(X_raw):,}")
    print("  - Window-level label distribution:", pd.Series(y_raw).value_counts().to_dict())
    print(f"  - Unique entities represented in windows: {meta_df['entity_id'].nunique()}")

    # 9. Rigorous Manifest Validation against Protocol Amendment 2.1
    print("\n[Step 8] Validating extracted manifest against Protocol Amendment 2.1 locked denominators...")
    val_report = validate_evaluation_manifest(meta_df, protocol)
    print("  - Validation status: PASSED")
    print(f"  - Total Windows: {val_report['total_windows']} (expected 5,511)")
    print(f"  - Critical Windows: {val_report['critical_windows']} (expected 400)")
    print(f"  - Warning Windows: {val_report['warning_windows']} (expected 300)")
    print(f"  - Normal Windows: {val_report['normal_windows']} (expected 4,811)")
    print(f"    * Pure-Normal Windows: {val_report['pure_normal_windows']} (expected 4,515)")
    print(f"    * Failing-Cohort Normal Windows: {val_report['failing_cohort_normal_windows']} (expected 296)")
    print(f"  - Evaluable Entities: {val_report['evaluable_failure_entities']} fail + {val_report['evaluable_pure_normal_entities']} pure-normal = 335 total")
    print("  - Excluded Census Tasks Ingestion: ZERO (Quarantine strictly maintained)")

    # 10. Scale features if output_dir requested
    if output_dir and not dry_run:
        print(f"\n[Step 9] Scaling features with frozen scaler from {scaler_path}...")
        scaler = LeakageSafeScaler.load(scaler_path)
        X_scaled = np.zeros_like(X_raw)
        for step in range(X_raw.shape[1]):
            step_df = pd.DataFrame(X_raw[:, step, :], columns=feature_cols)
            X_scaled[:, step, :] = scaler.transform(step_df).values

        assert np.isfinite(X_scaled).all(), "Non-finite values detected in scaled tensors!"

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        np.save(out_p / "X_confirmatory_l5.npy", X_scaled)
        np.save(out_p / "y_confirmatory_l5.npy", y_raw)
        meta_df.to_csv(out_p / "confirmatory_l5_metadata.csv", index=False)
        print(f"  - Saved scaled tensors and metadata to {out_p}")

    return {
        "status": "VALIDATED",
        "validation_report": val_report,
        "feature_count": len(feature_cols),
        "total_windows": len(X_raw),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract L=5 confirmatory evaluation dataset")
    parser.add_argument("--protocol", default="results/phase6/protocol_l5_amendment_2_1.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    extract_l5_confirmatory_dataset(
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
