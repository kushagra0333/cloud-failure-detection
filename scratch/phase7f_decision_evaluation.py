import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
y_val = np.load("data/processed/phase7c/y_val.npy")

# Load probabilities for all models
focal_res = json.loads(Path("results/phase7f/focal_loss_experiment_results.json").read_text())
task_rel_res = json.loads(Path("results/phase7f/task_relative_experiment_results.json").read_text())
p7e_val_pred = pd.read_csv("results/phase7e/validation_predictions.csv")

models_probs = {
    "transformer_condC_baseline": p7e_val_pred[[f"prob_transformer_condC_inverse_freq_c{c}" for c in [0, 1, 2]]].values,
    "transformer_focal_gamma1": np.array(focal_res["transformer_focal_gamma1"]["probabilities"]),
    "transformer_focal_gamma2": np.array(focal_res["transformer_focal_gamma2"]["probabilities"]),
    "transformer_task_relative": np.array(task_rel_res["task_relative_normalization"]["probabilities"]),
}

# Pre-index ground truth
df = meta_val.copy().reset_index(drop=True)
df["label"] = y_val
prefails = df[df["label"] > 0].copy()
prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
total_episodes = len(episodes)

failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
all_eids = sorted(df["entity_id"].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_normal_entities = len(pure_normal_eids)

normal_mask = (y_val == 0)
n_normal_windows = int(normal_mask.sum())

crit_mask = (y_val == 2)
n_crit_windows = int(crit_mask.sum())

binary_true = (y_val > 0).astype(int)
thresholds = np.linspace(0.05, 0.95, 91)

decision_eval_results = {}

print("=== STEP 4: DECISION EVALUATION ON VALIDATION SET ===")

for m_name, probs in models_probs.items():
    p_fail = probs[:, 1] + probs[:, 2]
    
    sweep = []
    both_pass = []
    gate2_compliant = []
    
    for theta in thresholds:
        bin_pred = (p_fail >= theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(binary_true, bin_pred, average="binary", zero_division=0.0)
        crit_rec = float(bin_pred[crit_mask].sum() / n_crit_windows) if n_crit_windows > 0 else 0.0
        
        fp_win = int(bin_pred[normal_mask].sum())
        win_fpr = float(fp_win / n_normal_windows * 100.0)
        
        fa_ent = sum((bin_pred[df["entity_id"] == eid] == 1).any() for eid in pure_normal_eids)
        ent_far = float(fa_ent / n_pure_normal_entities * 100.0)
        
        det_eps = sum((bin_pred[grp.index] == 1).any() for _, grp in episodes)
        ep_rec = float(det_eps / total_episodes * 100.0)
        
        g1_pass = (ep_rec >= 70.0) and (crit_rec >= 0.50)
        g2_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        
        entry = {
            "threshold": float(theta),
            "binary_precision": float(p),
            "binary_recall": float(r),
            "binary_f1": float(f1),
            "critical_recall": crit_rec,
            "episode_recall_pct": ep_rec,
            "detected_episodes": int(det_eps),
            "window_fpr_pct": win_fpr,
            "fp_windows": fp_win,
            "entity_far_pct": ent_far,
            "fa_entities": fa_ent,
            "gate_1_pass": bool(g1_pass),
            "gate_2_pass": bool(g2_pass),
            "both_pass": bool(g1_pass and g2_pass)
        }
        sweep.append(entry)
        if g1_pass and g2_pass:
            both_pass.append(entry)
        if g2_pass:
            gate2_compliant.append(entry)
            
    best_f1_entry = max(sweep, key=lambda x: x["binary_f1"])
    best_g2_entry = max(gate2_compliant, key=lambda x: x["episode_recall_pct"]) if gate2_compliant else None
    
    decision_eval_results[m_name] = {
        "model_name": m_name,
        "best_f1_entry": best_f1_entry,
        "both_gates_passed_count": len(both_pass),
        "both_gates_passed_entries": both_pass,
        "gate_2_compliant_count": len(gate2_compliant),
        "gate_2_best_recall_entry": best_g2_entry
    }
    
    print(f"\nModel: {m_name}")
    print(f"  Best F1: theta={best_f1_entry['threshold']:.2f} | F1={best_f1_entry['binary_f1']:.4f} | EpRec={best_f1_entry['episode_recall_pct']:.1f}% | CritRec={best_f1_entry['critical_recall']:.4f} | EntFAR={best_f1_entry['entity_far_pct']:.1f}% | WinFPR={best_f1_entry['window_fpr_pct']:.1f}%")
    print(f"  Both Gates Passed (EpRec>=70%, CritRec>=50%, EntFAR<=10%, WinFPR<=5%): {len(both_pass)} / 91 thresholds")
    if best_g2_entry:
        print(f"  Best under Gate 2 Constraint (EntFAR<=10%, WinFPR<=5%): theta={best_g2_entry['threshold']:.2f} | EpRec={best_g2_entry['episode_recall_pct']:.1f}% | CritRec={best_g2_entry['critical_recall']:.4f} | EntFAR={best_g2_entry['entity_far_pct']:.1f}% | WinFPR={best_g2_entry['window_fpr_pct']:.1f}%")
    else:
        print("  Best under Gate 2 Constraint: No threshold achieved Gate 2 compliance.")

def to_serializable(obj):
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

with open("results/phase7f/decision_evaluation_results.json", "w") as f:
    json.dump(to_serializable(decision_eval_results), f, indent=2)

print("\nSaved results/phase7f/decision_evaluation_results.json successfully.")
