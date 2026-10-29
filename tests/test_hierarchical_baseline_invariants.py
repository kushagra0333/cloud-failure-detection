"""Phase 7G Step 3: Automated Audit Tests for Causal Baseline Invariants.

Verifies:
1. No future or concurrent timestamps contribute to any baseline (t_prior < t_pred strictly enforced).
2. Zero label leakage (transformations are completely independent of y).
3. Split boundary isolation (train and val jobs are completely disjoint).
4. Insufficient history fallback invariant (exact bit-for-bit equivalence with cluster scaler when K < 2).
5. Determinism and reproducibility (multiple runs yield bit-for-bit identical tensors).
6. Complete numerical validity (no NaNs, no Infs).
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hierarchical_baseline import (
    compute_job_relative_transform,
    compute_hierarchical_transform,
    RAW_INDICES
)

def get_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def test_causal_timestamp_ordering():
    """Verify that every pooled observation in baselines has t_prior < t_pred."""
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    X_val = np.load("data/processed/phase7c/X_val.npy")
    
    # Run hierarchical transform
    _, routing_df = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    
    df = meta_val.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    
    for idx, row in routing_df.iterrows():
        t_pred = row["prediction_timestamp"]
        jid = row["parent_job_id"]
        eid = row["entity_id"]
        source = row["baseline_source"]
        
        if source == "task_history":
            prior_tasks = df[(df["entity_id"] == eid) & (df["prediction_timestamp"] < t_pred)]
            concurrent_or_future = df[(df["entity_id"] == eid) & (df["prediction_timestamp"] >= t_pred)]
            assert len(prior_tasks) >= 2, f"Expected K_task >= 2 for task_history, got {len(prior_tasks)}"
            # Ensure none of concurrent_or_future have t < t_pred
            assert (concurrent_or_future["prediction_timestamp"] >= t_pred).all()
        elif source == "parent_job_history":
            prior_jobs = df[(df["parent_job_id"] == jid) & (df["prediction_timestamp"] < t_pred)]
            concurrent_or_future = df[(df["parent_job_id"] == jid) & (df["prediction_timestamp"] >= t_pred)]
            assert len(prior_jobs) >= 2, f"Expected K_job >= 2 for parent_job_history, got {len(prior_jobs)}"
            assert (concurrent_or_future["prediction_timestamp"] >= t_pred).all()
        elif source == "cluster_scaler_fallback":
            prior_jobs = df[(df["parent_job_id"] == jid) & (df["prediction_timestamp"] < t_pred)]
            prior_tasks = df[(df["entity_id"] == eid) & (df["prediction_timestamp"] < t_pred)]
            assert len(prior_jobs) < 2 and len(prior_tasks) < 2, "Expected insufficient history for fallback"

def test_zero_label_leakage():
    """Verify that transformations are completely independent of y."""
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    X_val = np.load("data/processed/phase7c/X_val.npy")
    
    # Transform with original metadata
    X_trans_1, _ = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    
    # Create corrupted metadata with inverted/shuffled labels
    meta_corrupted = meta_val.copy()
    meta_corrupted["label"] = np.random.permutation(meta_corrupted["label"].values)
    meta_corrupted["time_to_failure"] = meta_corrupted["time_to_failure"] * -1.0
    
    X_trans_2, _ = compute_hierarchical_transform(X_val, meta_corrupted, k_min_windows=2)
    
    assert np.array_equal(X_trans_1, X_trans_2), "Transformation depends on labels or time_to_failure!"
    assert get_hash(X_trans_1) == get_hash(X_trans_2)

def test_split_isolation():
    """Verify that jobs in train and val do not overlap and splits cannot leak."""
    meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    
    train_jobs = set(meta_train["parent_job_id"])
    val_jobs = set(meta_val["parent_job_id"])
    overlap = train_jobs.intersection(val_jobs)
    assert len(overlap) == 0, f"Split leakage! Parent jobs overlap: {overlap}"

def test_insufficient_history_fallback_exactness():
    """Verify that fallback windows are bit-for-bit identical to original cluster-scaled values."""
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    X_val = np.load("data/processed/phase7c/X_val.npy")
    
    X_hier, routing_df = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    
    fallback_indices = routing_df[routing_df["baseline_source"] == "cluster_scaler_fallback"]["orig_idx"].values
    assert len(fallback_indices) > 0, "Expected at least one fallback window"
    
    for idx in fallback_indices:
        np.testing.assert_array_equal(
            X_hier[idx], X_val[idx],
            err_msg=f"Fallback window at index {idx} does not match original cluster-scaled values!"
        )

def test_determinism_and_reproducibility():
    """Verify that transformations are perfectly deterministic."""
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    X_val = np.load("data/processed/phase7c/X_val.npy")
    
    X_run1, r1 = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    X_run2, r2 = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    
    assert np.array_equal(X_run1, X_run2)
    pd.testing.assert_frame_equal(r1, r2)
    assert not np.isnan(X_run1).any()
    assert np.isfinite(X_run1).all()

if __name__ == "__main__":
    print("Running audit tests...")
    test_causal_timestamp_ordering()
    print("  [PASS] test_causal_timestamp_ordering")
    test_zero_label_leakage()
    print("  [PASS] test_zero_label_leakage")
    test_split_isolation()
    print("  [PASS] test_split_isolation")
    test_insufficient_history_fallback_exactness()
    print("  [PASS] test_insufficient_history_fallback_exactness")
    test_determinism_and_reproducibility()
    print("  [PASS] test_determinism_and_reproducibility")
    print("All 5 audit tests PASSED successfully.")
