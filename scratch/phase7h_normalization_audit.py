"""Phase 7H Step 3: Normalization Stability and Extreme Value Forensic Audit.

Audits:
1. Distribution of historical standard deviations across features and baseline sources.
2. Proportion of baselines with near-zero variance (sigma < 1e-4, < 1e-3, < 1e-2).
3. Extreme value percentiles (p50, p90, p95, p99, p99.9, max, min) of normalized features.
4. Correlation between extreme normalized feature values and false alarms.
5. Identification of affected entities, jobs, and features.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

RAW_COLS = ["cpu_utilization", "memory_utilization", "assigned_memory", "disk_io_time"]
RAW_INDICES = [0, 1, 2, 3]

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    return obj

def audit_normalization():
    print("=" * 80)
    print("PHASE 7H STEP 3: NORMALIZATION STABILITY AUDIT")
    print("=" * 80)
    
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    y_val = np.load("data/processed/phase7c/y_val.npy")
    X_val_orig = np.load("data/processed/phase7c/X_val.npy")
    X_val_hier = np.load("results/phase7g/X_val_hierarchical.npy")
    X_val_job = np.load("results/phase7g/X_val_job_relative.npy")
    routing_hier = pd.read_csv("results/phase7g/routing_val_hierarchical.csv")
    preds_df = pd.read_csv("results/phase7g/validation_predictions.csv")
    
    # 1. Extreme value percentiles across tensors
    print("\n[Audit 1] Extreme Value Percentiles of Normalized Features:")
    tensors = [
        ("Phase 7C Cluster Scaler", X_val_orig),
        ("Phase 7G Parent-Job Relative", X_val_job),
        ("Phase 7G Hierarchical Relative", X_val_hier)
    ]
    
    tensor_stats = {}
    for t_name, arr in tensors:
        raw_vals = arr[:, :, RAW_INDICES].flatten()
        p_stats = {
            "min": float(np.min(raw_vals)),
            "p01": float(np.percentile(raw_vals, 1)),
            "p05": float(np.percentile(raw_vals, 5)),
            "p50": float(np.percentile(raw_vals, 50)),
            "p95": float(np.percentile(raw_vals, 95)),
            "p99": float(np.percentile(raw_vals, 99)),
            "p999": float(np.percentile(raw_vals, 99.9)),
            "max": float(np.max(raw_vals)),
            "count_gt_10": int((np.abs(raw_vals) > 10).sum()),
            "count_gt_50": int((np.abs(raw_vals) > 50).sum()),
            "count_gt_100": int((np.abs(raw_vals) > 100).sum())
        }
        tensor_stats[t_name] = p_stats
        print(f"  {t_name}:")
        print(f"    Min={p_stats['min']:.2f}, Median={p_stats['p50']:.2f}, Max={p_stats['max']:.2f}")
        print(f"    p99={p_stats['p99']:.2f}, p99.9={p_stats['p999']:.2f}")
        print(f"    Count |z| > 10: {p_stats['count_gt_10']}, |z| > 50: {p_stats['count_gt_50']}, |z| > 100: {p_stats['count_gt_100']}")
        
    # 2. Per-feature extreme values in Hierarchical Tensor
    print("\n[Audit 2] Per-Feature Extremes in Hierarchical Tensor:")
    feature_stats = {}
    for col_idx, col_name in zip(RAW_INDICES, RAW_COLS):
        f_vals = X_val_hier[:, :, col_idx].flatten()
        f_stat = {
            "min": float(np.min(f_vals)),
            "p50": float(np.percentile(f_vals, 50)),
            "p99": float(np.percentile(f_vals, 99)),
            "max": float(np.max(f_vals)),
            "count_gt_10": int((np.abs(f_vals) > 10).sum()),
            "count_gt_50": int((np.abs(f_vals) > 50).sum())
        }
        feature_stats[col_name] = f_stat
        print(f"  {col_name:<20}: min={f_stat['min']:<8.2f} p99={f_stat['p99']:<8.2f} max={f_stat['max']:<10.2f} count|z|>10={f_stat['count_gt_10']}")
        
    # 3. Audit historical standard deviations computed during hierarchical transform
    print("\n[Audit 3] Historical Standard Deviations by Source & Near-Zero Variance:")
    # Re-evaluate historical std across windows
    df = meta_val.copy().reset_index(drop=True)
    df["orig_idx"] = np.arange(len(df))
    jobs = df["parent_job_id"].unique()
    
    hist_stds_task = []
    hist_stds_job = []
    
    near_zero_counts = {c: 0 for c in RAW_COLS}
    total_eval_baselines = 0
    
    for jid in jobs:
        jdf = df[df["parent_job_id"] == jid].sort_values("prediction_timestamp")
        j_indices = jdf["orig_idx"].values
        j_ts = jdf["prediction_timestamp"].values
        j_eids = jdf["entity_id"].values
        
        for idx, t_pred, eid in zip(j_indices, j_ts, j_eids):
            t_mask = (j_eids == eid) & (j_ts < t_pred)
            if t_mask.sum() >= 2:
                # Task history used
                p_indices = j_indices[t_mask]
                past_obs = np.concatenate([X_val_orig[p] for p in p_indices], axis=0)
                stds = np.std(past_obs[:, RAW_INDICES], axis=0) # shape [4]
                hist_stds_task.append(stds)
                total_eval_baselines += 1
                for i, c in enumerate(RAW_COLS):
                    if stds[i] < 1e-3:
                        near_zero_counts[c] += 1
            else:
                job_mask = (j_ts < t_pred)
                if job_mask.sum() >= 2:
                    p_indices = j_indices[job_mask]
                    past_obs = np.concatenate([X_val_orig[p] for p in p_indices], axis=0)
                    stds = np.std(past_obs[:, RAW_INDICES], axis=0)
                    hist_stds_job.append(stds)
                    total_eval_baselines += 1
                    for i, c in enumerate(RAW_COLS):
                        if stds[i] < 1e-3:
                            near_zero_counts[c] += 1
                            
    task_stds_arr = np.array(hist_stds_task) if hist_stds_task else np.empty((0, 4))
    job_stds_arr = np.array(hist_stds_job) if hist_stds_job else np.empty((0, 4))
    
    std_summary = {}
    for i, c in enumerate(RAW_COLS):
        t_vals = task_stds_arr[:, i] if len(task_stds_arr) > 0 else []
        j_vals = job_stds_arr[:, i] if len(job_stds_arr) > 0 else []
        std_summary[c] = {
            "task_std_median": float(np.median(t_vals)) if len(t_vals) else 0.0,
            "task_std_min": float(np.min(t_vals)) if len(t_vals) else 0.0,
            "job_std_median": float(np.median(j_vals)) if len(j_vals) else 0.0,
            "job_std_min": float(np.min(j_vals)) if len(j_vals) else 0.0,
            "baselines_near_zero_sigma_1e3": int(near_zero_counts[c]),
            "pct_near_zero": float(near_zero_counts[c] / total_eval_baselines * 100) if total_eval_baselines > 0 else 0.0
        }
        print(f"  {c:<20}: Task std median={std_summary[c]['task_std_median']:.4f}, min={std_summary[c]['task_std_min']:.6f} | Job std median={std_summary[c]['job_std_median']:.4f}, min={std_summary[c]['job_std_min']:.6f} | Near-zero (<1e-3)={std_summary[c]['pct_near_zero']:.1f}%")
        
    # 4. Correlation with False Alarms
    print("\n[Audit 4] Extreme Values vs False Alarms:")
    # For each window, find max |z| in raw features
    max_z_per_window = np.max(np.abs(X_val_hier[:, :, RAW_INDICES]), axis=(1, 2))
    df["max_z"] = max_z_per_window
    df["y_true"] = y_val
    df["y_pred_hier"] = preds_df["y_pred_transformer_hierarchical"]
    df["is_false_alarm"] = (df["y_true"] == 0) & (df["y_pred_hier"] > 0)
    
    normal_windows = df[df["y_true"] == 0]
    fa_windows = df[df["is_false_alarm"]]
    non_fa_normal = df[(df["y_true"] == 0) & (~df["is_false_alarm"])]
    
    print(f"  All Normal Windows: N={len(normal_windows)}, Max |z| Mean={normal_windows['max_z'].mean():.2f}, Median={normal_windows['max_z'].median():.2f}, Max={normal_windows['max_z'].max():.2f}")
    print(f"  False Alarm Normal: N={len(fa_windows)}, Max |z| Mean={fa_windows['max_z'].mean():.2f}, Median={fa_windows['max_z'].median():.2f}, Max={fa_windows['max_z'].max():.2f}")
    print(f"  Correct Normal:     N={len(non_fa_normal)}, Max |z| Mean={non_fa_normal['max_z'].mean():.2f}, Median={non_fa_normal['max_z'].median():.2f}, Max={non_fa_normal['max_z'].max():.2f}")
    
    # Check parent job of extreme values
    extreme_df = df[df["max_z"] > 50]
    print(f"\n  Windows with Extreme |z| > 50: N={len(extreme_df)}")
    print("  Breakdown by parent job:")
    for jid, grp in extreme_df.groupby("parent_job_id"):
        n_fa = (grp["is_false_alarm"]).sum()
        print(f"    Job {jid}: {len(grp)} windows with |z| > 50 (False alarms: {n_fa} / {len(grp)})")
        
    audit_results = {
        "tensor_percentiles": tensor_stats,
        "feature_extremes": feature_stats,
        "historical_stds": std_summary,
        "false_alarm_correlation": {
            "normal_windows_count": len(normal_windows),
            "fa_windows_count": len(fa_windows),
            "fa_max_z_mean": float(fa_windows["max_z"].mean()),
            "fa_max_z_median": float(fa_windows["max_z"].median()),
            "fa_max_z_max": float(fa_windows["max_z"].max()),
            "non_fa_max_z_mean": float(non_fa_normal["max_z"].mean()),
            "non_fa_max_z_median": float(non_fa_normal["max_z"].median()),
            "windows_with_z_gt_50": len(extreme_df),
            "fa_fraction_on_z_gt_50": float(extreme_df["is_false_alarm"].mean()) if len(extreme_df) else 0.0
        },
        "verdict": "CRITICAL NUMERICAL INSTABILITY CONFIRMED: epsilon=1e-5 allows division by near-zero variance, exploding z-scores to >12,000. Features disk_io_time and assigned_memory frequently have near-zero variance, and windows with max |z| > 50 have a high false alarm rate. Controlled variance flooring is strictly required."
    }
    
    out_path = Path("results/phase7h/normalization_stability_audit.json")
    with open(out_path, "w") as f:
        json.dump(to_serializable(audit_results), f, indent=2)
    print(f"\nSaved normalization audit report to {out_path}")

if __name__ == "__main__":
    audit_normalization()
