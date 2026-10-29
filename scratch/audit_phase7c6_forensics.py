"""Phase 7C.6 Independent Scientific Audit Forensic Probe.

Performs independent verification across Tasks A, B, C, D, E, F:
- Provenance and hash verification
- Parent-job and entity isolation
- Machine overlap and temporal span across splits
- Scaler recomputation on window-flattened vs unique physical steps
- Temporal causality of historical steps in X_raw
- Target label correctness and failure timestamp boundaries
- Detailed evaluation of Phase 7C tests
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("c:/Users/hp/Desktop/Research project")
sys.path.insert(0, str(PROJECT_ROOT))

def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_forensic_audit():
    print("================================================================================")
    print("PHASE 7C.6 INDEPENDENT FORENSIC SCIENTIFIC PROBE")
    print("================================================================================")
    results = {}

    # -------------------------------------------------------------------------
    # Task A: Artifact Provenance & Integrity
    # -------------------------------------------------------------------------
    print("\n--- TASK A: ARTIFACT PROVENANCE & INTEGRITY ---")
    cand_split_path = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
    cand_split_hash = compute_sha256(cand_split_path)
    cand_df = pd.read_csv(cand_split_path)
    print(f"Candidate A Split Hash: {cand_split_hash}")
    print(f"Total rows in Candidate A CSV: {len(cand_df):,}")

    expected_hash = "f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946"
    assert cand_split_hash == expected_hash, f"Hash mismatch: {cand_split_hash} != {expected_hash}"

    # Check phase7c data files
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    raw_unscaled_dir = p7c_dir / "raw_unscaled"

    data_files = {
        "X_train.npy": p7c_dir / "X_train.npy",
        "y_train.npy": p7c_dir / "y_train.npy",
        "train_metadata.csv": p7c_dir / "train_metadata.csv",
        "X_val.npy": p7c_dir / "X_val.npy",
        "y_val.npy": p7c_dir / "y_val.npy",
        "val_metadata.csv": p7c_dir / "val_metadata.csv",
        "X_test.npy": p7c_dir / "X_test.npy",
        "y_test.npy": p7c_dir / "y_test.npy",
        "test_metadata.csv": p7c_dir / "test_metadata.csv",
        "fitted_scaler_phase7c.json": PROJECT_ROOT / "configs" / "phase7c" / "fitted_scaler_phase7c.json",
        "X_train_raw.npy": raw_unscaled_dir / "X_train_raw.npy",
        "X_val_raw.npy": raw_unscaled_dir / "X_val_raw.npy",
        "X_test_raw.npy": raw_unscaled_dir / "X_test_raw.npy",
    }

    file_hashes = {}
    for name, path in data_files.items():
        assert path.exists(), f"Missing file: {path}"
        h = compute_sha256(path)
        file_hashes[name] = {"path": str(path), "sha256": h, "size_bytes": path.stat().st_size}
        print(f"  {name:26s} | {path.stat().st_size:10,d} B | SHA-256: {h[:16]}...")

    # Load arrays
    X_tr = np.load(data_files["X_train.npy"])
    y_tr = np.load(data_files["y_train.npy"])
    m_tr = pd.read_csv(data_files["train_metadata.csv"])

    X_va = np.load(data_files["X_val.npy"])
    y_va = np.load(data_files["y_val.npy"])
    m_va = pd.read_csv(data_files["val_metadata.csv"])

    X_te = np.load(data_files["X_test.npy"])
    y_te = np.load(data_files["y_test.npy"])
    m_te = pd.read_csv(data_files["test_metadata.csv"])

    X_tr_raw = np.load(data_files["X_train_raw.npy"])
    X_va_raw = np.load(data_files["X_val_raw.npy"])
    X_te_raw = np.load(data_files["X_test_raw.npy"])

    results["task_a"] = {
        "candidate_a_hash": cand_split_hash,
        "candidate_a_hash_verified": cand_split_hash == expected_hash,
        "file_hashes": file_hashes,
        "shapes": {
            "train": {"X": list(X_tr.shape), "y": list(y_tr.shape), "meta_rows": len(m_tr)},
            "val": {"X": list(X_va.shape), "y": list(y_va.shape), "meta_rows": len(m_va)},
            "test": {"X": list(X_te.shape), "y": list(y_te.shape), "meta_rows": len(m_te)},
        }
    }

    # -------------------------------------------------------------------------
    # Task B: Audit Raw-Data Reconstruction & Window Matching
    # -------------------------------------------------------------------------
    print("\n--- TASK B: AUDIT RAW-DATA RECONSTRUCTION & WINDOW MATCHING ---")
    # Verify entity_id@prediction_timestamp exact match
    cand_df["key"] = cand_df["entity_id"].astype(str) + "@" + cand_df["prediction_timestamp"].astype(str)
    all_recon_meta = pd.concat([m_tr, m_va, m_te], ignore_index=True)
    all_recon_meta["key"] = all_recon_meta["entity_id"].astype(str) + "@" + all_recon_meta["prediction_timestamp"].astype(str)

    recon_keys = set(all_recon_meta["key"])
    cand_keys = set(cand_df["key"])

    print(f"Total Candidate A keys : {len(cand_keys):,}")
    print(f"Total Reconstructed keys: {len(recon_keys):,}")
    print(f"Key Intersection       : {len(recon_keys & cand_keys):,}")
    print(f"Keys Missing in Recon  : {len(cand_keys - recon_keys)}")
    print(f"Excess Keys in Recon   : {len(recon_keys - cand_keys)}")

    # Check labels alignment
    key_to_cand_label = dict(zip(cand_df["key"], cand_df["label"]))
    label_mismatches = 0
    for _, row in all_recon_meta.iterrows():
        k = row["key"]
        if row["label"] != key_to_cand_label[k]:
            label_mismatches += 1
    print(f"Label mismatches against Candidate A: {label_mismatches}")

    # Check prediction timestamp range
    print(f"Prediction timestamp range: {all_recon_meta['prediction_timestamp'].min():.1f}s to {all_recon_meta['prediction_timestamp'].max():.1f}s")

    results["task_b"] = {
        "cand_keys_count": len(cand_keys),
        "recon_keys_count": len(recon_keys),
        "exact_key_match": cand_keys == recon_keys,
        "missing_keys": len(cand_keys - recon_keys),
        "excess_keys": len(recon_keys - cand_keys),
        "label_mismatches": label_mismatches,
    }

    # -------------------------------------------------------------------------
    # Task C: Audit Feature and Label Correctness
    # -------------------------------------------------------------------------
    print("\n--- TASK C: AUDIT FEATURE & LABEL CORRECTNESS ---")
    # Check feature definitions
    with open(raw_unscaled_dir / "raw_reconstruction_manifest.json") as f:
        recon_man = json.load(f)
    feature_cols = recon_man["feature_definitions"]["feature_cols"]
    print(f"Total features: {len(feature_cols)}")

    # Check finite integrity
    non_finite_X_tr = int((~np.isfinite(X_tr)).sum())
    non_finite_X_va = int((~np.isfinite(X_va)).sum())
    non_finite_X_te = int((~np.isfinite(X_te)).sum())
    print(f"Non-finite in X_train: {non_finite_X_tr}")
    print(f"Non-finite in X_val  : {non_finite_X_va}")
    print(f"Non-finite in X_test : {non_finite_X_te}")

    # Policy A check against raw events
    # Load events for all candidate entities
    raw_event_shards = [
        "data/raw/sample_task_events.csv.gz",
        "data/raw/sample_task_events_part1.csv.gz",
        "data/raw/sample_task_events_part2.csv.gz",
    ]
    event_cols = [
        "timestamp", "missing_info", "job_id", "task_index", "machine_id",
        "event_type", "user_name", "scheduling_class", "priority",
        "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
    ]
    target_eids = set(cand_df["entity_id"].unique())
    ev_dfs = []
    for s_path in raw_event_shards:
        p = PROJECT_ROOT / s_path
        df = pd.read_csv(p, header=None, names=event_cols)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(target_eids)].copy()
        if not matched.empty:
            ev_dfs.append(matched)
    all_events = pd.concat(ev_dfs, ignore_index=True)
    all_events["time_sec"] = all_events["timestamp"] / 1e6

    # In 2011 trace, event_type == 5 is FAIL
    fail_events = all_events[all_events["event_type"] == 5].copy()
    first_fails = fail_events.groupby("entity_id")["time_sec"].min().to_dict()
    print(f"Entities with failure events in trace: {len(first_fails)} of {len(target_eids)}")

    post_fail_violations = 0
    for _, row in all_recon_meta.iterrows():
        eid = row["entity_id"]
        t_pred = row["prediction_timestamp"]
        if eid in first_fails:
            t_fail = first_fails[eid]
            if t_pred >= t_fail:
                post_fail_violations += 1

    print(f"Post-failure prediction windows (t_pred >= t_fail): {post_fail_violations}")
    assert post_fail_violations == 0, f"Violation of Policy A: {post_fail_violations} windows occur post-failure!"

    # Label distribution
    class_dist = {
        "train": m_tr["label"].value_counts().to_dict(),
        "val": m_va["label"].value_counts().to_dict(),
        "test": m_te["label"].value_counts().to_dict(),
    }
    print(f"Class distributions:")
    print(f"  Train: {class_dist['train']}")
    print(f"  Val  : {class_dist['val']}")
    print(f"  Test : {class_dist['test']}")

    results["task_c"] = {
        "feature_count": len(feature_cols),
        "non_finite_counts": {"train": non_finite_X_tr, "val": non_finite_X_va, "test": non_finite_X_te},
        "entities_with_failures": len(first_fails),
        "post_failure_violations": post_fail_violations,
        "class_distributions": class_dist,
    }

    # -------------------------------------------------------------------------
    # Task D: Audit Missingness & Preprocessing (Window Overlap Analysis)
    # -------------------------------------------------------------------------
    print("\n--- TASK D: AUDIT MISSINGNESS & PREPROCESSING ---")
    # 1. Genuine missingness in raw unscaled arrays
    raw_nans = {
        "train": {col: int(np.isnan(X_tr_raw[:, :, idx]).sum()) for idx, col in enumerate(feature_cols)},
        "val": {col: int(np.isnan(X_va_raw[:, :, idx]).sum()) for idx, col in enumerate(feature_cols)},
        "test": {col: int(np.isnan(X_te_raw[:, :, idx]).sum()) for idx, col in enumerate(feature_cols)},
    }
    features_with_nans = [col for col, cnt in raw_nans["train"].items() if cnt > 0]
    print(f"Features with NaNs in Train Raw: {features_with_nans}")
    for col in features_with_nans:
        print(f"  {col}: Train={raw_nans['train'][col]:,}, Val={raw_nans['val'][col]:,}, Test={raw_nans['test'][col]:,}")

    # 2. Scaler Recomputation
    with open(data_files["fitted_scaler_phase7c.json"]) as f:
        saved_scaler = json.load(f)

    tr_flat = X_tr_raw.reshape(-1, len(feature_cols))
    scaler_diffs = []
    for idx, col in enumerate(feature_cols):
        obs = tr_flat[:, idx]
        valid_obs = obs[~np.isnan(obs)]
        med = float(np.median(valid_obs))
        mean = float(np.mean(valid_obs))
        std = float(np.std(valid_obs))
        if std < 1e-6:
            std = 1.0

        diff_med = abs(med - saved_scaler["medians"][col])
        diff_mean = abs(mean - saved_scaler["means"][col])
        diff_std = abs(std - saved_scaler["stds"][col])
        scaler_diffs.append((col, diff_med, diff_mean, diff_std))

    max_diff_med = max(d[1] for d in scaler_diffs)
    max_diff_mean = max(d[2] for d in scaler_diffs)
    max_diff_std = max(d[3] for d in scaler_diffs)
    print(f"Max difference between independent recomputation and saved scaler:")
    print(f"  Median: {max_diff_med:.2e} | Mean: {max_diff_mean:.2e} | Std: {max_diff_std:.2e}")

    # 3. CRITICAL AUDIT PROBE: Window Overlap Impact on Preprocessing
    # In X_train_raw, shape is (3278, 5, 31). Flattened is 16,390 steps.
    # What is the number of UNIQUE physical telemetry steps in train?
    # To find out, let's load the cleaned telemetry dataframe from raw usage for training entities
    print("\nEvaluating Window Overlap on Training Preprocessing Population:")
    # We can reconstruct the unique (entity_id, end_time_sec) steps present in training windows
    # For each window in m_tr: pred_time = row['prediction_timestamp']. The 5 steps are pred_time - 4*300 to pred_time.
    train_unique_steps = set()
    for _, row in m_tr.iterrows():
        eid = row["entity_id"]
        t_pred = row["prediction_timestamp"]
        for step_idx in range(5):
            t_step = t_pred - (4 - step_idx) * 300.0
            train_unique_steps.add((eid, t_step))
    print(f"  Total window-step observations in Train : {len(m_tr) * 5:,} (3,278 x 5)")
    print(f"  Unique (entity_id, timestamp) steps in Train: {len(train_unique_steps):,}")
    step_multiplicity = (len(m_tr) * 5) / len(train_unique_steps)
    print(f"  Average observation multiplicity: {step_multiplicity:.2f}x per unique time-step")

    # Let's measure how much means/stds differ between window-weighted vs unique-step-weighted
    # We can extract unique steps from X_tr_raw by indexing:
    # Notice that step 4 of window i (ending at t) is step 3 of window i+1 (ending at t+300)
    # Let's map each (entity, timestamp) to its unique feature vector
    unique_step_dict = {}
    for w_idx, row in m_tr.iterrows():
        eid = row["entity_id"]
        t_pred = row["prediction_timestamp"]
        for s_idx in range(5):
            t_step = t_pred - (4 - s_idx) * 300.0
            k = (eid, t_step)
            if k not in unique_step_dict:
                unique_step_dict[k] = X_tr_raw[w_idx, s_idx, :]

    unique_step_mat = np.array(list(unique_step_dict.values()))
    print(f"  Extracted unique step matrix shape: {unique_step_mat.shape}")

    unique_vs_window_comp = {}
    for idx, col in enumerate(feature_cols):
        # Window-flattened stats
        w_obs = tr_flat[:, idx]
        w_valid = w_obs[~np.isnan(w_obs)]
        w_med = float(np.median(w_valid))
        w_mean = float(np.mean(w_valid))
        w_std = float(np.std(w_valid))

        # Unique-step stats
        u_obs = unique_step_mat[:, idx]
        u_valid = u_obs[~np.isnan(u_obs)]
        u_med = float(np.median(u_valid))
        u_mean = float(np.mean(u_valid))
        u_std = float(np.std(u_valid))

        mean_rel_diff = abs(w_mean - u_mean) / (abs(u_mean) + 1e-6)
        std_rel_diff = abs(w_std - u_std) / (abs(u_std) + 1e-6)
        unique_vs_window_comp[col] = {
            "window_mean": w_mean, "unique_mean": u_mean, "mean_rel_diff": mean_rel_diff,
            "window_std": w_std, "unique_std": u_std, "std_rel_diff": std_rel_diff,
            "window_med": w_med, "unique_med": u_med
        }

    # Print top 5 features with largest difference
    sorted_by_mean_diff = sorted(unique_vs_window_comp.items(), key=lambda x: x[1]["mean_rel_diff"], reverse=True)
    print("\n  Top 5 features with largest relative difference (Window-Weighted vs Unique-Step-Weighted):")
    for col, d in sorted_by_mean_diff[:5]:
        print(f"    * {col:32s}: w_mean={d['window_mean']:.5f}, u_mean={d['unique_mean']:.5f} (rel_diff={d['mean_rel_diff']*100:.2f}%)")

    # Is this leakage across splits?
    # NO: Both populations come 100% strictly from Candidate A Train!
    # But it is an important methodological nuance:
    # Standardizing tensors flattened over windows effectively weights observations by their window participation.

    results["task_d"] = {
        "raw_nan_counts": raw_nans,
        "max_scaler_recomputation_diff": {"median": max_diff_med, "mean": max_diff_mean, "std": max_diff_std},
        "window_overlap": {
            "total_window_steps": len(m_tr) * 5,
            "unique_physical_steps": len(train_unique_steps),
            "multiplicity_ratio": step_multiplicity,
            "top_differences": dict(sorted_by_mean_diff[:5])
        }
    }

    # -------------------------------------------------------------------------
    # Task E: Audit Split Independence & Evaluation Validity
    # -------------------------------------------------------------------------
    print("\n--- TASK E: AUDIT SPLIT INDEPENDENCE & EVALUATION VALIDITY ---")
    tr_jobs = set(m_tr["parent_job_id"].astype(str).unique())
    va_jobs = set(m_va["parent_job_id"].astype(str).unique())
    te_jobs = set(m_te["parent_job_id"].astype(str).unique())

    print(f"Parent jobs: Train={len(tr_jobs)}, Val={len(va_jobs)}, Test={len(te_jobs)}")
    print(f"Overlap Train-Val : {len(tr_jobs & va_jobs)}")
    print(f"Overlap Train-Test: {len(tr_jobs & te_jobs)}")
    print(f"Overlap Val-Test  : {len(va_jobs & te_jobs)}")

    tr_eids = set(m_tr["entity_id"].astype(str).unique())
    va_eids = set(m_va["entity_id"].astype(str).unique())
    te_eids = set(m_te["entity_id"].astype(str).unique())

    print(f"Entities: Train={len(tr_eids)}, Val={len(va_eids)}, Test={len(te_eids)}")
    print(f"Entity Overlap Train-Val : {len(tr_eids & va_eids)}")
    print(f"Entity Overlap Train-Test: {len(tr_eids & te_eids)}")
    print(f"Entity Overlap Val-Test  : {len(va_eids & te_eids)}")

    # Check temporal spans
    tr_t_min, tr_t_max = m_tr["prediction_timestamp"].min(), m_tr["prediction_timestamp"].max()
    va_t_min, va_t_max = m_va["prediction_timestamp"].min(), m_va["prediction_timestamp"].max()
    te_t_min, te_t_max = m_te["prediction_timestamp"].min(), m_te["prediction_timestamp"].max()

    print(f"Temporal prediction timestamp spans:")
    print(f"  Train: [{tr_t_min:10.1f}s, {tr_t_max:10.1f}s] (span = {(tr_t_max - tr_t_min)/3600:.1f} hours)")
    print(f"  Val  : [{va_t_min:10.1f}s, {va_t_max:10.1f}s] (span = {(va_t_max - va_t_min)/3600:.1f} hours)")
    print(f"  Test : [{te_t_min:10.1f}s, {te_t_max:10.1f}s] (span = {(te_t_max - te_t_min)/3600:.1f} hours)")

    # Temporal overlap: All three partitions span the entire ~600 hour (25-day) trace!
    # Because Candidate A is a JOB-GROUPED split, not a temporal split!
    print("  Note: Candidate A parent jobs run concurrently across the trace timeline.")

    # Check machine_id overlap across splits
    # Inspect raw events or usage to find which machine_id each entity was scheduled on
    print("\nAuditing Machine ID Overlap across splits:")
    # In events_raw: machine_id is column 4
    ent_machines = all_events.dropna(subset=["machine_id"]).groupby("entity_id")["machine_id"].unique().to_dict()
    tr_machines = set(m for eid in tr_eids if eid in ent_machines for m in ent_machines[eid])
    va_machines = set(m for eid in va_eids if eid in ent_machines for m in ent_machines[eid])
    te_machines = set(m for eid in te_eids if eid in ent_machines for m in ent_machines[eid])

    print(f"  Unique Machines in Train : {len(tr_machines):,}")
    print(f"  Unique Machines in Val   : {len(va_machines):,}")
    print(f"  Unique Machines in Test  : {len(te_machines):,}")
    print(f"  Machine Overlap Train-Val: {len(tr_machines & va_machines):,} / {len(va_machines)} ({len(tr_machines & va_machines)/len(va_machines)*100:.1f}% of Val)")
    print(f"  Machine Overlap Train-Test: {len(tr_machines & te_machines):,} / {len(te_machines)} ({len(tr_machines & te_machines)/len(te_machines)*100:.1f}% of Test)")

    # This is a critical finding for generalization:
    # Parent jobs are strictly disjoint, but in a shared cluster, different jobs run on overlapping physical machines.
    # This is standard for cluster workloads, but must be documented!

    results["task_e"] = {
        "job_overlap": {
            "train_val": len(tr_jobs & va_jobs),
            "train_test": len(tr_jobs & te_jobs),
            "val_test": len(va_jobs & te_jobs)
        },
        "entity_overlap": {
            "train_val": len(tr_eids & va_eids),
            "train_test": len(tr_eids & te_eids),
            "val_test": len(va_eids & te_eids)
        },
        "temporal_spans": {
            "train": [float(tr_t_min), float(tr_t_max)],
            "val": [float(va_t_min), float(va_t_max)],
            "test": [float(te_t_min), float(te_t_max)],
        },
        "machine_overlap": {
            "unique_machines": {"train": len(tr_machines), "val": len(va_machines), "test": len(te_machines)},
            "train_val_overlap": len(tr_machines & va_machines),
            "train_test_overlap": len(tr_machines & te_machines),
        }
    }

    # -------------------------------------------------------------------------
    # Task F: Audit Tests
    # -------------------------------------------------------------------------
    print("\n--- TASK F: AUDIT TESTS IN TEST SUITE ---")
    test_evaluations = {
        "test_raw_missing_values_remain_nan_in_unscaled_arrays": {
            "verdict": "RIGOROUS",
            "reason": "Directly loads saved raw unscaled .npy arrays and asserts genuine NaNs exist in memory_accesses_per_instruction across train, val, and test."
        },
        "test_imputation_and_scaling_statistics_derived_exclusively_from_train": {
            "verdict": "RIGOROUS",
            "reason": "Independently computes medians, means, and stds from X_train_raw and verifies np.isclose against configs/phase7c/fitted_scaler_phase7c.json."
        },
        "test_val_test_data_cannot_influence_preprocessing_state": {
            "verdict": "RIGOROUS",
            "reason": "Tests LeakageSafeScaler behavior by passing extreme outlier val data to transform and verifying internal medians, means, and stds remain unchanged."
        },
        "test_entity_grouping_prevents_lag_or_rolling_cross_contamination": {
            "verdict": "RIGOROUS",
            "reason": "Constructs synthetic multi-entity DataFrame with boundary step and asserts lag features do not bleed across entity boundaries."
        },
        "test_feature_timestamps_do_not_exceed_prediction_timestamp": {
            "verdict": "WEAK",
            "reason": "Claims to verify all historical steps within sliding window are <= prediction timestamp, but implementation only asserts (meta_df['prediction_timestamp'] > 0).all()."
        },
        "test_policy_a_strictly_excludes_post_failure_observations": {
            "verdict": "RIGOROUS",
            "reason": "Tests TargetLabelGenerator directly on synthetic failure event at t=5000s; verifies t=5000s and t=5300s yield UNKNOWN (-1) and t=4500s yields CRITICAL."
        },
        "test_no_parent_job_overlap_among_candidate_a_partitions": {
            "verdict": "RIGOROUS",
            "reason": "Extracts parent_job_id from train, val, and test metadata and verifies pairwise intersections are strictly empty."
        },
        "test_processed_arrays_are_completely_finite": {
            "verdict": "RIGOROUS",
            "reason": "Directly tests X_train.npy, X_val.npy, X_test.npy for np.isfinite and absence of NaNs."
        },
        "test_feature_label_metadata_alignment": {
            "verdict": "RIGOROUS",
            "reason": "Verifies shapes (N, 5, 31) and confirms y array equals metadata['label'].values row-for-row across all partitions."
        },
        "test_missing_or_duplicate_join_keys_cause_explicit_failure": {
            "verdict": "WEAK",
            "reason": "Uses a mock toy DataFrame with 2 keys instead of testing the actual reconstruct_phase7c_raw_features.py joining logic on actual candidate split."
        }
    }

    for name, ev in test_evaluations.items():
        print(f"  {name:65s}: [{ev['verdict']:8s}] - {ev['reason'][:60]}...")

    results["task_f"] = {
        "test_evaluations": test_evaluations,
    }

    # Save probe results
    scratch_out = PROJECT_ROOT / "scratch" / "phase7c6_probe_results.json"
    scratch_out.parent.mkdir(parents=True, exist_ok=True)
    with open(scratch_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nForensic probe completed. Serialized probe results to {scratch_out}")

if __name__ == "__main__":
    run_forensic_audit()
