"""Phase 7G Step 4: Decision Evaluation & Threshold Sweep.

Evaluates failure risk score S_fail = P(Warning) + P(Critical) across thresholds theta in [0.05, 0.95].
Examines whether any threshold satisfies all predeclared operational gates:
  - Gate 1: Episode Recall >= 70% AND Critical Recall >= 50%
  - Gate 2: Pure-normal Entity FAR <= 10% AND Window FPR <= 5%
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

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

def run_threshold_sweep(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    p_fail: np.ndarray,
    thresholds: np.ndarray
) -> List[Dict[str, Any]]:
    binary_true = (y_true > 0).astype(int)
    
    df = meta_df.copy().reset_index(drop=True)
    df["label"] = y_true
    prefails = df[df["label"] > 0].copy()
    prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    
    failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
    all_eids = sorted(df["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)
    
    normal_mask = (y_true == 0)
    n_normal_windows = int(normal_mask.sum())
    
    crit_mask = (y_true == 2)
    n_crit_windows = int(crit_mask.sum())
    
    results = []
    
    for theta in thresholds:
        bin_pred = (p_fail >= theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(binary_true, bin_pred, average="binary", zero_division=0.0)
        crit_rec = float(bin_pred[crit_mask].sum() / n_crit_windows) if n_crit_windows > 0 else 0.0
        
        fp_windows = int(bin_pred[normal_mask].sum())
        win_fpr = float(fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0
        
        fa_entities = sum(1 for eid in pure_normal_eids if (bin_pred[df["entity_id"] == eid] == 1).any())
        ent_far = float(fa_entities / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0
        
        det_eps = 0
        lead_times = []
        for (eid, ftime), grp in episodes:
            det_mask = (bin_pred[grp.index] == 1)
            if det_mask.any():
                det_eps += 1
                first_det_idx = grp.index[det_mask][0]
                t_pred = df.loc[first_det_idx, "prediction_timestamp"]
                lead_times.append(ftime - t_pred)
                
        ep_rec = float(det_eps / total_episodes * 100.0) if total_episodes > 0 else 0.0
        mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
        
        gate1_pass = (ep_rec >= 70.0) and (crit_rec >= 0.50)
        gate2_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        both_pass = gate1_pass and gate2_pass
        
        results.append({
            "threshold": float(theta),
            "binary_precision": float(p),
            "binary_recall": float(r),
            "binary_f1": float(f1),
            "critical_recall": float(crit_rec),
            "episode_recall_pct": float(ep_rec),
            "detected_episodes": int(det_eps),
            "total_episodes": int(total_episodes),
            "lead_time_mean_sec": float(mean_lead),
            "window_fpr_pct": float(win_fpr),
            "fp_windows": int(fp_windows),
            "entity_far_pct": float(ent_far),
            "fa_entities": int(fa_entities),
            "pure_normal_entities": int(n_pure_normal_entities),
            "gate1_pass": bool(gate1_pass),
            "gate2_pass": bool(gate2_pass),
            "both_gates_pass": bool(both_pass)
        })
        
    return results

def main():
    print("=== Phase 7G Step 4: Decision Threshold Sweep ===")
    preds_path = Path("results/phase7g/validation_predictions.csv")
    if not preds_path.exists():
        print(f"Error: {preds_path} does not exist yet.")
        return
        
    preds_df = pd.read_csv(preds_path)
    y_val = preds_df["y_true"].values
    
    model_ids = [
        ("transformer_job_relative", "Experiment A: Parent-Job Baseline (CE, gamma=0)"),
        ("transformer_hierarchical", "Experiment B: Hierarchical Causal Baseline (CE, gamma=0)"),
        ("transformer_hierarchical_focal", "Experiment C: Hierarchical Baseline + Focal Loss (gamma=1.0)"),
        ("transformer_job_relative_focal", "Experiment C-alt: Parent-Job Baseline + Focal Loss (gamma=1.0)")
    ]
    
    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_results = {}
    
    for m_id, m_name in model_ids:
        p_fail_col = f"p_fail_{m_id}"
        if p_fail_col not in preds_df.columns:
            print(f"Skipping {m_id}: column {p_fail_col} not in predictions.")
            continue
            
        p_fail = preds_df[p_fail_col].values
        sweep = run_threshold_sweep(preds_df, y_val, p_fail, thresholds)
        
        best_f1_entry = max(sweep, key=lambda x: x["binary_f1"])
        both_pass_entries = [e for e in sweep if e["both_gates_pass"]]
        g2_entries = [e for e in sweep if e["gate2_pass"]]
        best_g2 = max(g2_entries, key=lambda x: x["episode_recall_pct"]) if g2_entries else None
        g1_entries = [e for e in sweep if e["gate1_pass"]]
        best_g1 = min(g1_entries, key=lambda x: x["entity_far_pct"]) if g1_entries else None
        
        print(f"\nModel: {m_name}")
        print(f"  Best Binary F1: theta={best_f1_entry['threshold']:.2f} | F1={best_f1_entry['binary_f1']:.4f} | EpRec={best_f1_entry['episode_recall_pct']:.1f}% | CritRec={best_f1_entry['critical_recall']:.4f} | EntFAR={best_f1_entry['entity_far_pct']:.1f}% | WinFPR={best_f1_entry['window_fpr_pct']:.1f}%")
        
        if both_pass_entries:
            print(f"  >>> SUCCESS: {len(both_pass_entries)} threshold(s) passed both gates!")
        else:
            print(f"  >>> FAIL: 0 thresholds passed both operational gates simultaneously.")
            if best_g2:
                print(f"      Max EpRec under Gate 2: {best_g2['episode_recall_pct']:.1f}% at theta={best_g2['threshold']:.2f} (EntFAR={best_g2['entity_far_pct']:.1f}%, WinFPR={best_g2['window_fpr_pct']:.1f}%)")
            if best_g1:
                print(f"      Min EntFAR under Gate 1: {best_g1['entity_far_pct']:.1f}% at theta={best_g1['threshold']:.2f} (EpRec={best_g1['episode_recall_pct']:.1f}%, WinFPR={best_g1['window_fpr_pct']:.1f}%)")
                
        sweep_results[m_id] = {
            "name": m_name,
            "best_binary_f1_threshold": best_f1_entry,
            "both_gates_passed_thresholds": both_pass_entries,
            "best_gate2_compliant_threshold": best_g2,
            "best_gate1_compliant_threshold": best_g1,
            "threshold_sweep": sweep
        }
        
    out_path = Path("results/phase7g/decision_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(to_serializable(sweep_results), f, indent=2)
    print(f"\nSaved threshold sweep results to {out_path}")

if __name__ == "__main__":
    main()
