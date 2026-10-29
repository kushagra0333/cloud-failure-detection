import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Load test predictions
pred_df = pd.read_csv("results/phase7d/test_predictions.csv")
saved_res = json.loads(open("results/phase7d/test_evaluation_results.json").read())

models = ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]

y_true = pred_df["y_true"].values
total_windows = len(y_true)

# Pure normal entities and episodes ground truth calculation
prefails = pred_df[pred_df["label"] > 0].copy()
prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
total_episodes = len(episodes)

failing_eids = set(prefails["entity_id"].unique())
all_eids = sorted(pred_df["entity_id"].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_normal_entities = len(pure_normal_eids)

normal_mask = (y_true == 0)
n_normal_windows = int(normal_mask.sum())

audit_check2 = {}

for m in models:
    y_pred = pred_df[f"y_pred_{m}"].values
    
    # Prediction counts and percentages
    counts = {}
    pcts = {}
    for c in [0, 1, 2]:
        cnt = int((y_pred == c).sum())
        counts[c] = cnt
        pcts[c] = float(cnt / total_windows * 100.0)
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    # Per-class P, R, F1, Support
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0.0)
    class_metrics = {}
    class_names = ["NORMAL", "WARNING", "CRITICAL"]
    for i, cname in enumerate(class_names):
        class_metrics[cname] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(sup[i])
        }
        
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    crit_recall = float(r[2])
    
    # Episode recall
    detected_episodes = 0
    for (eid, ftime), grp in episodes:
        grp_pred = y_pred[grp.index]
        if (grp_pred > 0).any():
            detected_episodes += 1
    ep_recall_pct = float(detected_episodes / total_episodes * 100.0)
    
    # Window FPR
    n_fp_windows = int((y_pred[normal_mask] > 0).sum())
    window_fpr_pct = float(n_fp_windows / n_normal_windows * 100.0)
    
    # Entity FAR
    entity_fa_count = sum(
        (y_pred[pred_df["entity_id"] == eid] > 0).any()
        for eid in pure_normal_eids
    )
    entity_far_pct = float(entity_fa_count / n_pure_normal_entities * 100.0)
    
    # Compare with reported in saved_res
    rep_m = saved_res["models"][m]
    rep_macro_f1 = rep_m["macro_f1"]
    rep_acc = rep_m["accuracy"]
    rep_crit_rec = rep_m["critical_recall"]
    rep_ep_rec = rep_m["episode_metrics"]["episode_recall_pct"]
    rep_win_fpr = rep_m["window_fpr"]["fpr_pct"]
    rep_ent_far = rep_m["entity_far"]["far_pct"]
    
    reproduced = (
        abs(macro_f1 - rep_macro_f1) < 1e-6 and
        abs(acc - rep_acc) < 1e-6 and
        abs(crit_recall - rep_crit_rec) < 1e-6 and
        abs(ep_recall_pct - rep_ep_rec) < 1e-6 and
        abs(window_fpr_pct - rep_win_fpr) < 1e-6 and
        abs(entity_far_pct - rep_ent_far) < 1e-6
    )
    
    audit_check2[m] = {
        "prediction_distribution": {
            "counts": counts,
            "percentages": pcts
        },
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
        "summary_metrics": {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "critical_recall": crit_recall,
            "episode_recall_pct": ep_recall_pct,
            "detected_episodes": detected_episodes,
            "total_episodes": total_episodes,
            "window_fpr_pct": window_fpr_pct,
            "fp_windows": n_fp_windows,
            "normal_windows": n_normal_windows,
            "entity_far_pct": entity_far_pct,
            "fa_entities": entity_fa_count,
            "pure_normal_entities": n_pure_normal_entities,
        },
        "independently_reproduced": reproduced
    }
    
    print(f"=== MODEL: {m} ===")
    print(f"  Prediction Counts: {counts} | Percentages: {pcts}")
    print(f"  Confusion Matrix: {cm}")
    print(f"  Macro-F1: {macro_f1:.4f} | Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
    print(f"  Critical Recall: {crit_recall:.4f} | Episode Recall: {ep_recall_pct:.2f}% ({detected_episodes}/{total_episodes})")
    print(f"  Window FPR: {window_fpr_pct:.2f}% ({n_fp_windows}/{n_normal_windows}) | Entity FAR: {entity_far_pct:.2f}% ({entity_fa_count}/{n_pure_normal_entities})")
    print(f"  Independently Reproduced from Saved Predictions: {reproduced}")

def to_serializable(val):
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_serializable(v) for v in val]
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, (np.bool_, bool)):
        return bool(val)
    return val

with open("results/phase7d_diagnostic_audit/independent_metric_reproduction.json", "w") as f:
    json.dump(to_serializable(audit_check2), f, indent=2)
