"""Phase 7F — Step 4: Decision Evaluation & Validation Predictions Export.

Evaluates failure risk score S_fail = P(Warning) + P(Critical) across thresholds theta in [0.05, 0.95].
Examines whether any threshold satisfies all predeclared operational gates:
  - Gate 1: Episode Recall >= 70% AND Critical Recall >= 50%
  - Gate 2: Pure-normal Entity FAR <= 10% AND Window FPR <= 5%
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import CompactTemporalTransformer

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

def predict_model(model: torch.nn.Module, X: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def run_threshold_sweep(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray
) -> List[Dict[str, Any]]:
    p_fail = probs[:, 1] + probs[:, 2]
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
        
        fa_entities = sum((bin_pred[df["entity_id"] == eid] == 1).any() for eid in pure_normal_eids)
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
    print("=== Phase 7F Step 4: Decision Evaluation & Prediction Export ===")
    device = torch.device("cpu")
    
    # 1. Load Data
    val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    y_val = np.load("data/processed/phase7c/y_val.npy")
    X_val_standard = np.load("data/processed/phase7c/X_val.npy")
    X_val_task_relative = np.load("results/phase7f/X_val_task_relative.npy")
    
    print(f"Validation shape: {X_val_standard.shape}, Labels: {len(y_val)}")
    
    # Models to evaluate
    model_configs = [
        {
            "id": "transformer_condC_baseline",
            "name": "Transformer Condition C (Inverse-Freq Baseline, gamma=0)",
            "checkpoint": "models/checkpoints_phase7e/transformer_condC_inverse_freq.pt",
            "X": X_val_standard
        },
        {
            "id": "transformer_focal_gamma1",
            "name": "Transformer Focal Loss (gamma=1.0, alpha=InvFreq)",
            "checkpoint": "models/checkpoints_phase7f/transformer_focal_gamma1.pt",
            "X": X_val_standard
        },
        {
            "id": "transformer_focal_gamma2",
            "name": "Transformer Focal Loss (gamma=2.0, alpha=InvFreq)",
            "checkpoint": "models/checkpoints_phase7f/transformer_focal_gamma2.pt",
            "X": X_val_standard
        },
        {
            "id": "transformer_task_relative",
            "name": "Transformer Task-Relative Normalization (gamma=0, alpha=InvFreq)",
            "checkpoint": "models/checkpoints_phase7f/transformer_task_relative.pt",
            "X": X_val_task_relative
        }
    ]
    
    # Output dataframe for predictions
    preds_df = val_meta[["entity_id", "prediction_timestamp", "label", "time_to_failure"]].copy()
    if "parent_job_id" in val_meta.columns:
        preds_df["parent_job_id"] = val_meta["parent_job_id"]
    else:
        preds_df["parent_job_id"] = val_meta["entity_id"].apply(lambda x: str(x).split("_")[0])
    preds_df["y_true"] = y_val
    
    thresholds = np.linspace(0.05, 0.95, 91)
    decision_results = {}
    
    for m in model_configs:
        m_id = m["id"]
        print(f"\nEvaluating {m['name']}...")
        model = CompactTemporalTransformer(
            input_dim=31,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            num_classes=3,
            dropout=0.1,
            max_seq_len=5
        ).to(device)
        ckpt = torch.load(m["checkpoint"], map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        
        y_pred, probs = predict_model(model, m["X"], device)
        p_fail = probs[:, 1] + probs[:, 2]
        
        preds_df[f"y_pred_{m_id}"] = y_pred
        preds_df[f"prob_{m_id}_c0"] = probs[:, 0]
        preds_df[f"prob_{m_id}_c1"] = probs[:, 1]
        preds_df[f"prob_{m_id}_c2"] = probs[:, 2]
        preds_df[f"p_fail_{m_id}"] = p_fail
        
        # Argmax multi-class metrics
        macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0.0))
        prec, rec, f1, sup = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2], zero_division=0.0)
        bal_acc = float(balanced_accuracy_score(y_val, y_pred))
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist()
        
        # Threshold sweep
        sweep = run_threshold_sweep(val_meta, y_val, probs, thresholds)
        
        # Best Binary F1 threshold
        best_f1_entry = max(sweep, key=lambda x: x["binary_f1"])
        
        # Both gates passed
        both_pass_entries = [e for e in sweep if e["both_gates_pass"]]
        
        # Gate 2 compliant entries
        g2_entries = [e for e in sweep if e["gate2_pass"]]
        best_g2 = max(g2_entries, key=lambda x: x["episode_recall_pct"]) if g2_entries else None
        
        # Gate 1 compliant entries
        g1_entries = [e for e in sweep if e["gate1_pass"]]
        best_g1 = min(g1_entries, key=lambda x: x["entity_far_pct"]) if g1_entries else None
        
        print(f"  Argmax: Macro-F1={macro_f1:.4f}, BalAcc={bal_acc:.4f}, CritRec={rec[2]:.4f}")
        print(f"  Best Binary F1: theta={best_f1_entry['threshold']:.2f} | F1={best_f1_entry['binary_f1']:.4f} | EpRec={best_f1_entry['episode_recall_pct']:.1f}% | CritRec={best_f1_entry['critical_recall']:.4f} | EntFAR={best_f1_entry['entity_far_pct']:.1f}% | WinFPR={best_f1_entry['window_fpr_pct']:.1f}%")
        
        if both_pass_entries:
            print(f"  >>> SUCCESS: {len(both_pass_entries)} threshold(s) passed both gates!")
            for b in both_pass_entries:
                print(f"      theta={b['threshold']:.2f}: EpRec={b['episode_recall_pct']:.1f}%, CritRec={b['critical_recall']:.4f}, EntFAR={b['entity_far_pct']:.1f}%, WinFPR={b['window_fpr_pct']:.1f}%")
        else:
            print("  >>> FAIL: 0 thresholds passed both operational gates simultaneously.")
            if best_g2:
                print(f"      Max EpRec under Gate 2: {best_g2['episode_recall_pct']:.1f}% at theta={best_g2['threshold']:.2f} (EntFAR={best_g2['entity_far_pct']:.1f}%, WinFPR={best_g2['window_fpr_pct']:.1f}%)")
            if best_g1:
                print(f"      Min EntFAR under Gate 1: {best_g1['entity_far_pct']:.1f}% at theta={best_g1['threshold']:.2f} (EpRec={best_g1['episode_recall_pct']:.1f}%, WinFPR={best_g1['window_fpr_pct']:.1f}%)")
                
        decision_results[m_id] = {
            "model_name": m["name"],
            "checkpoint": m["checkpoint"],
            "argmax_evaluation": {
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "precision": [float(p) for p in prec],
                "recall": [float(r) for r in rec],
                "f1": [float(f) for f in f1],
                "support": [int(s) for s in sup],
                "confusion_matrix": cm
            },
            "best_binary_f1_threshold": best_f1_entry,
            "both_gates_passed_thresholds": both_pass_entries,
            "best_gate2_compliant_threshold": best_g2,
            "best_gate1_compliant_threshold": best_g1,
            "threshold_sweep": sweep
        }
        
    # Save outputs
    Path("results/phase7f").mkdir(parents=True, exist_ok=True)
    preds_csv_path = "results/phase7f/validation_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"\nSaved validation predictions to {preds_csv_path}")
    
    eval_json_path = "results/phase7f/decision_evaluation_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(to_serializable(decision_results), f, indent=2)
    print(f"Saved decision evaluation results to {eval_json_path}")
    print("\nDecision evaluation complete.")

if __name__ == "__main__":
    main()
