"""Hierarchical and Parent-Job Causal Baseline Feature Transformations for Phase 7G.

Enforces strict causal constraints:
1. Only completed historical windows with timestamp strictly prior to prediction timestamp (t_prior < t_pred) are used.
2. Concurrent sibling windows (t == t_pred) and future windows (t > t_pred) are strictly excluded.
3. Labels are never accessed or referenced.
4. If eligible history count K < k_min (default 2), falls back gracefully.
5. All routing decisions, history depths, and fallback reasons are recorded per window.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

RAW_COLS = ["cpu_utilization", "memory_utilization", "assigned_memory", "disk_io_time"]
RAW_INDICES = [0, 1, 2, 3]

def compute_job_relative_transform(
    X_seq: np.ndarray,
    meta_df: pd.DataFrame,
    k_min_windows: int = 2,
    raw_indices: List[int] = RAW_INDICES
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Experiment A: Parent-Job Historical Baseline with cluster-scaler fallback.
    
    For each window at t_pred in parent job J:
      - Collect all windows from parent job J with t_prior < t_pred.
      - If len(prior_job_windows) >= k_min_windows:
          normalize raw features by job running mean and std.
      - Else:
          fall back to frozen cluster scaler (leave values unchanged).
    """
    X_trans = X_seq.copy()
    df = meta_df.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    
    # Pre-index windows by parent_job_id
    jobs = df["parent_job_id"].unique()
    
    routing_records = []
    
    for jid in jobs:
        job_rows = df[df["parent_job_id"] == jid].sort_values("prediction_timestamp")
        job_indices = job_rows["orig_idx"].values
        job_timestamps = job_rows["prediction_timestamp"].values
        
        for idx, t_pred in zip(job_indices, job_timestamps):
            # Find all strictly prior windows in this job (any entity, t_prior < t_pred)
            prior_mask = (job_timestamps < t_pred)
            prior_indices = job_indices[prior_mask]
            k_job = len(prior_indices)
            
            if k_job >= k_min_windows:
                # Pool all observations from prior windows in this job
                # Shape: [k_job * seq_len, num_features]
                past_obs = np.concatenate([X_seq[p_idx] for p_idx in prior_indices], axis=0)
                job_mean = np.mean(past_obs[:, raw_indices], axis=0, keepdims=True)
                job_std = np.std(past_obs[:, raw_indices], axis=0, keepdims=True) + 1e-5
                
                curr_feat = X_trans[idx][:, raw_indices]
                X_trans[idx][:, raw_indices] = (curr_feat - job_mean) / job_std
                
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": df.loc[idx, "entity_id"],
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "parent_job_history",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": "none"
                })
            else:
                # Fall back to cluster scaler
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": df.loc[idx, "entity_id"],
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "cluster_scaler_fallback",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": f"insufficient_job_history (K_job={k_job} < {k_min_windows})"
                })
                
    routing_df = pd.DataFrame(routing_records).sort_values("orig_idx").reset_index(drop=True)
    return X_trans, routing_df

def compute_hierarchical_transform(
    X_seq: np.ndarray,
    meta_df: pd.DataFrame,
    k_min_windows: int = 2,
    raw_indices: List[int] = RAW_INDICES,
    sigma_floor: float = 0.0
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Hierarchical Causal Baseline with optional variance floor.
    
    For each window at t_pred of entity E in parent job J:
      1. Preference: Eligible prior windows of entity E with t_prior < t_pred.
         If K_task >= k_min_windows:
           normalize by task baseline (floored at sigma_floor).
      2. Second preference: Eligible prior windows across job J with t_prior < t_pred.
         If K_job >= k_min_windows:
           normalize by job baseline (floored at sigma_floor).
      3. Fallback: Revert to cluster scaler.
    """
    X_trans = X_seq.copy()
    df = meta_df.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    
    jobs = df["parent_job_id"].unique()
    routing_records = []
    
    for jid in jobs:
        job_rows = df[df["parent_job_id"] == jid].sort_values("prediction_timestamp")
        job_indices = job_rows["orig_idx"].values
        job_timestamps = job_rows["prediction_timestamp"].values
        job_entities = job_rows["entity_id"].values
        
        for idx, t_pred, eid in zip(job_indices, job_timestamps, job_entities):
            # 1. Task prior windows (same entity, t_prior < t_pred)
            task_mask = (job_entities == eid) & (job_timestamps < t_pred)
            task_prior_indices = job_indices[task_mask]
            k_task = len(task_prior_indices)
            
            # 2. Job prior windows (any entity in job, t_prior < t_pred)
            job_mask = (job_timestamps < t_pred)
            job_prior_indices = job_indices[job_mask]
            k_job = len(job_prior_indices)
            
            if k_task >= k_min_windows:
                # Use Task History
                past_obs = np.concatenate([X_seq[p_idx] for p_idx in task_prior_indices], axis=0)
                task_mean = np.mean(past_obs[:, raw_indices], axis=0, keepdims=True)
                raw_std = np.std(past_obs[:, raw_indices], axis=0, keepdims=True) + 1e-5
                task_std = np.maximum(raw_std, sigma_floor) if sigma_floor > 0 else raw_std
                
                curr_feat = X_trans[idx][:, raw_indices]
                X_trans[idx][:, raw_indices] = (curr_feat - task_mean) / task_std
                
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "task_history",
                    "history_depth_windows": int(k_task),
                    "fallback_reason": "none"
                })
            elif k_job >= k_min_windows:
                # Fall back to Parent-Job History
                past_obs = np.concatenate([X_seq[p_idx] for p_idx in job_prior_indices], axis=0)
                job_mean = np.mean(past_obs[:, raw_indices], axis=0, keepdims=True)
                raw_std = np.std(past_obs[:, raw_indices], axis=0, keepdims=True) + 1e-5
                job_std = np.maximum(raw_std, sigma_floor) if sigma_floor > 0 else raw_std
                
                curr_feat = X_trans[idx][:, raw_indices]
                X_trans[idx][:, raw_indices] = (curr_feat - job_mean) / job_std
                
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "parent_job_history",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": f"insufficient_task_history_using_job (K_task={k_task} < {k_min_windows}, K_job={k_job})"
                })
            else:
                # Fall back to Cluster Scaler
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "cluster_scaler_fallback",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": f"insufficient_task_and_job_history (K_task={k_task}, K_job={k_job} < {k_min_windows})"
                })
                
    routing_df = pd.DataFrame(routing_records).sort_values("orig_idx").reset_index(drop=True)
    return X_trans, routing_df

def compute_hierarchical_robust_transform(
    X_seq: np.ndarray,
    meta_df: pd.DataFrame,
    k_min_windows: int = 2,
    raw_indices: List[int] = RAW_INDICES,
    mad_floor: float = 0.05,
    clip_bound: float = 5.0
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Ablation C: Robust Causal Scaling with Median, MAD, and Soft Tanh Bounding.
    
    z = (x - median) / max(MAD * 1.4826, mad_floor)
    z_clipped = clip_bound * tanh(z / clip_bound)
    """
    X_trans = X_seq.copy()
    df = meta_df.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    
    jobs = df["parent_job_id"].unique()
    routing_records = []
    
    for jid in jobs:
        job_rows = df[df["parent_job_id"] == jid].sort_values("prediction_timestamp")
        job_indices = job_rows["orig_idx"].values
        job_timestamps = job_rows["prediction_timestamp"].values
        job_entities = job_rows["entity_id"].values
        
        for idx, t_pred, eid in zip(job_indices, job_timestamps, job_entities):
            task_mask = (job_entities == eid) & (job_timestamps < t_pred)
            task_prior_indices = job_indices[task_mask]
            k_task = len(task_prior_indices)
            
            job_mask = (job_timestamps < t_pred)
            job_prior_indices = job_indices[job_mask]
            k_job = len(job_prior_indices)
            
            if k_task >= k_min_windows:
                past_obs = np.concatenate([X_seq[p_idx] for p_idx in task_prior_indices], axis=0)
                med = np.median(past_obs[:, raw_indices], axis=0, keepdims=True)
                mad = np.median(np.abs(past_obs[:, raw_indices] - med), axis=0, keepdims=True) * 1.4826 + 1e-5
                scale = np.maximum(mad, mad_floor)
                
                curr_feat = X_trans[idx][:, raw_indices]
                z = (curr_feat - med) / scale
                z_soft = clip_bound * np.tanh(z / clip_bound)
                X_trans[idx][:, raw_indices] = z_soft
                
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "task_history",
                    "history_depth_windows": int(k_task),
                    "fallback_reason": "none"
                })
            elif k_job >= k_min_windows:
                past_obs = np.concatenate([X_seq[p_idx] for p_idx in job_prior_indices], axis=0)
                med = np.median(past_obs[:, raw_indices], axis=0, keepdims=True)
                mad = np.median(np.abs(past_obs[:, raw_indices] - med), axis=0, keepdims=True) * 1.4826 + 1e-5
                scale = np.maximum(mad, mad_floor)
                
                curr_feat = X_trans[idx][:, raw_indices]
                z = (curr_feat - med) / scale
                z_soft = clip_bound * np.tanh(z / clip_bound)
                X_trans[idx][:, raw_indices] = z_soft
                
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "parent_job_history",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": f"insufficient_task_history_using_job (K_task={k_task} < {k_min_windows}, K_job={k_job})"
                })
            else:
                routing_records.append({
                    "orig_idx": int(idx),
                    "entity_id": eid,
                    "parent_job_id": jid,
                    "prediction_timestamp": float(t_pred),
                    "baseline_source": "cluster_scaler_fallback",
                    "history_depth_windows": int(k_job),
                    "fallback_reason": f"insufficient_task_and_job_history (K_task={k_task}, K_job={k_job} < {k_min_windows})"
                })
                
    routing_df = pd.DataFrame(routing_records).sort_values("orig_idx").reset_index(drop=True)
    return X_trans, routing_df
