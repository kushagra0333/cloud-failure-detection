"""Phase 7H Step 1: Critical-Class Prediction and Error Forensics Audit.

Generates row-level audits and summary statistics for true Critical windows
across all Phase 7G validation models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

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

def run_critical_audit():
    print("=" * 80)
    print("PHASE 7H STEP 1: CRITICAL-CLASS PREDICTION AUDIT")
    print("=" * 80)
    
    preds_df = pd.read_csv("results/phase7g/validation_predictions.csv")
    routing_hier = pd.read_csv("results/phase7g/routing_val_hierarchical.csv")
    routing_job = pd.read_csv("results/phase7g/routing_val_job_relative.csv")
    
    # Calculate fail_timestamp for episode grouping
    preds_df["fail_timestamp"] = (preds_df["prediction_timestamp"] + preds_df["time_to_failure"]).round(1)
    
    models = [
        ("transformer_job_relative", "Experiment A: Parent-Job Baseline (CE)", routing_job),
        ("transformer_hierarchical", "Experiment B: Hierarchical Baseline (CE)", routing_hier),
        ("transformer_hierarchical_focal", "Experiment C: Hierarchical + Focal (gamma=1.0)", routing_hier),
        ("transformer_job_relative_focal", "Experiment C-alt: Parent-Job + Focal (gamma=1.0)", routing_job)
    ]
    
    all_crit_rows = []
    summary_results = {}
    
    crit_mask = (preds_df["y_true"] == 2)
    n_crit_total = int(crit_mask.sum())
    print(f"Total True Critical Windows in Validation: {n_crit_total}")
    
    for m_id, m_name, routing_df in models:
        pred_col = f"y_pred_{m_id}"
        p0_col = f"prob_{m_id}_c0"
        p1_col = f"prob_{m_id}_c1"
        p2_col = f"prob_{m_id}_c2"
        
        df = preds_df.copy()
        df["baseline_source"] = routing_df["baseline_source"]
        df["history_depth"] = routing_df["history_depth_windows"]
        
        # Episode detection status for each episode
        prefails = df[df["y_true"] > 0].copy()
        episodes = prefails.groupby(["entity_id", "fail_timestamp"])
        detected_episode_keys = set()
        for key, grp in episodes:
            if (grp[pred_col] > 0).any():
                detected_episode_keys.add(key)
                
        df["is_in_detected_episode"] = df.apply(
            lambda r: (r["entity_id"], r["fail_timestamp"]) in detected_episode_keys if r["y_true"] > 0 else False,
            axis=1
        )
        
        # Classify errors for true critical windows
        crit_df = df[crit_mask].copy()
        
        def get_error_type(yp):
            if yp == 2:
                return "Correct_Critical"
            elif yp == 1:
                return "Critical_to_Warning"
            elif yp == 0:
                return "Critical_to_Normal"
            return "Other"
            
        crit_df["error_type"] = crit_df[pred_col].apply(get_error_type)
        crit_df["model_id"] = m_id
        crit_df["model_name"] = m_name
        crit_df["p_normal"] = crit_df[p0_col]
        crit_df["p_warning"] = crit_df[p1_col]
        crit_df["p_critical"] = crit_df[p2_col]
        crit_df["predicted_class"] = crit_df[pred_col]
        
        # Keep selected columns for row-level output
        row_level_cols = [
            "model_id", "entity_id", "parent_job_id", "prediction_timestamp",
            "time_to_failure", "y_true", "predicted_class", "error_type",
            "p_normal", "p_warning", "p_critical", "is_in_detected_episode",
            "baseline_source", "history_depth"
        ]
        all_crit_rows.append(crit_df[row_level_cols])
        
        # Aggregate statistics
        err_counts = crit_df["error_type"].value_counts().to_dict()
        n_correct = int(err_counts.get("Correct_Critical", 0))
        n_crit_to_warn = int(err_counts.get("Critical_to_Warning", 0))
        n_crit_to_norm = int(err_counts.get("Critical_to_Normal", 0))
        
        # Episodes containing critical windows
        crit_episodes = crit_df.groupby(["entity_id", "fail_timestamp"])
        n_crit_episodes = len(crit_episodes)
        n_crit_episodes_detected = sum(1 for _, grp in crit_episodes if grp["is_in_detected_episode"].iloc[0])
        
        # Probability distributions on true Critical windows
        prob_summary = {
            "p_normal": {
                "mean": float(crit_df["p_normal"].mean()),
                "median": float(crit_df["p_normal"].median()),
                "std": float(crit_df["p_normal"].std()),
                "min": float(crit_df["p_normal"].min()),
                "max": float(crit_df["p_normal"].max())
            },
            "p_warning": {
                "mean": float(crit_df["p_warning"].mean()),
                "median": float(crit_df["p_warning"].median()),
                "std": float(crit_df["p_warning"].std()),
                "min": float(crit_df["p_warning"].min()),
                "max": float(crit_df["p_warning"].max())
            },
            "p_critical": {
                "mean": float(crit_df["p_critical"].mean()),
                "median": float(crit_df["p_critical"].median()),
                "std": float(crit_df["p_critical"].std()),
                "min": float(crit_df["p_critical"].min()),
                "max": float(crit_df["p_critical"].max())
            }
        }
        
        # Breakdown by parent job
        job_breakdown = {}
        for jid, jgrp in crit_df.groupby("parent_job_id"):
            j_errs = jgrp["error_type"].value_counts().to_dict()
            job_breakdown[str(jid)] = {
                "total_critical_windows": len(jgrp),
                "correct_critical": int(j_errs.get("Correct_Critical", 0)),
                "critical_to_warning": int(j_errs.get("Critical_to_Warning", 0)),
                "critical_to_normal": int(j_errs.get("Critical_to_Normal", 0)),
                "mean_p_critical": float(jgrp["p_critical"].mean()),
                "mean_p_warning": float(jgrp["p_warning"].mean()),
                "mean_p_normal": float(jgrp["p_normal"].mean())
            }
            
        summary_results[m_id] = {
            "model_name": m_name,
            "total_critical_windows": n_crit_total,
            "correct_critical": n_correct,
            "critical_recall": float(n_correct / n_crit_total),
            "critical_to_warning": n_crit_to_warn,
            "critical_to_warning_pct": float(n_crit_to_warn / n_crit_total * 100),
            "critical_to_normal": n_crit_to_norm,
            "critical_to_normal_pct": float(n_crit_to_norm / n_crit_total * 100),
            "failure_interception_rate_pct": float((n_correct + n_crit_to_warn) / n_crit_total * 100),
            "critical_episodes_total": n_crit_episodes,
            "critical_episodes_intercepted": n_crit_episodes_detected,
            "critical_episode_interception_pct": float(n_crit_episodes_detected / n_crit_episodes * 100) if n_crit_episodes > 0 else 0.0,
            "probability_summary": prob_summary,
            "parent_job_breakdown": job_breakdown
        }
        
        print(f"\nModel: {m_name}")
        print(f"  Correct Critical: {n_correct}/{n_crit_total} ({n_correct/n_crit_total*100:.1f}%)")
        print(f"  Critical -> Warning: {n_crit_to_warn}/{n_crit_total} ({n_crit_to_warn/n_crit_total*100:.1f}%)")
        print(f"  Critical -> Normal (Complete Miss): {n_crit_to_norm}/{n_crit_total} ({n_crit_to_norm/n_crit_total*100:.1f}%)")
        print(f"  Combined Failure Interception (Critical or Warning): {n_correct + n_crit_to_warn}/{n_crit_total} ({(n_correct+n_crit_to_warn)/n_crit_total*100:.1f}%)")
        print(f"  Mean Probabilities: P(Normal)={prob_summary['p_normal']['mean']:.4f}, P(Warning)={prob_summary['p_warning']['mean']:.4f}, P(Critical)={prob_summary['p_critical']['mean']:.4f}")
        
    out_dir = Path("results/phase7h")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined row-level CSV
    combined_rows = pd.concat(all_crit_rows, ignore_index=True)
    csv_path = out_dir / "critical_class_audit_row_level.csv"
    combined_rows.to_csv(csv_path, index=False)
    print(f"\nSaved row-level audit to {csv_path}")
    
    # Save summary JSON
    json_path = out_dir / "critical_class_audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(to_serializable(summary_results), f, indent=2)
    print(f"Saved summary JSON to {json_path}")

if __name__ == "__main__":
    run_critical_audit()
