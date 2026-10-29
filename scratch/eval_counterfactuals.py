"""Compute diagnostic counterfactual metrics under alternative censoring scenarios."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load predictions
preds = pd.read_csv(PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv")

# Load protocol
with open(PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
pure_norm_eids = set(protocol["frozen_entity_ids"]["primary_normal_entities_evaluable"])

# Load raw events
event_files = sorted((PROJECT_ROOT / "data" / "raw").glob("sample_task_events*.csv.gz"))
EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]
event_dfs = []
for ep in event_files:
    df = pd.read_csv(ep, header=None, names=EVENT_COLS)
    df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
    matched = df[df["entity_id"].isin(fail_eids | pure_norm_eids)]
    if not matched.empty:
        event_dfs.append(matched)

events = pd.concat(event_dfs, ignore_index=True)
events["time_sec"] = events["timestamp"] / 1e6
fail_events = events[events["event_type"] == 3].copy()
fail_times_by_eid = fail_events.groupby("entity_id")["time_sec"].apply(list).to_dict()

# Tag each window in preds
window_tags = []
for idx, r in preds.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    label = int(r["label"])
    e_fails = sorted(fail_times_by_eid.get(eid, []))
    
    prev_fails = [ft for ft in e_fails if ft < t_pred]
    next_fails = [ft for ft in e_fails if ft > t_pred]
    
    prev_fail = prev_fails[-1] if prev_fails else None
    next_fail = next_fails[0] if next_fails else None
    
    t_start = t_pred - 1200.0
    history_crosses_fail = (prev_fail is not None and prev_fail >= t_start)
    
    if label == 0:
        if eid in pure_norm_eids:
            w_type = "PURE_NORMAL"
        else:
            if next_fail is not None:
                if history_crosses_fail:
                    w_type = "FAIL_NORM_PRE_CROSS_HIST"  # 7 windows
                else:
                    w_type = "FAIL_NORM_PRE_CLEAN"       # 140 windows
            else:
                w_type = "FAIL_NORM_POST"                # 149 windows
    elif label == 1:
        w_type = "PREFAIL_WARNING"                       # 300 windows
    elif label == 2:
        w_type = "PREFAIL_CRITICAL"                      # 400 windows
    else:
        w_type = "UNKNOWN"
    
    window_tags.append(w_type)

preds["window_type"] = window_tags

print("Window Type breakdown across all 5,511 windows:")
print(preds["window_type"].value_counts())

models = ["rf", "lstm", "transformer", "hybrid"]
model_names = {
    "rf": "Random Forest",
    "lstm": "Temporal LSTM",
    "transformer": "Compact Transformer",
    "hybrid": "Representation Hybrid",
}

def evaluate_subset(df, subset_name):
    print(f"\n=======================================================")
    print(f"EVALUATION: {subset_name} (N = {len(df)} windows)")
    print(f"=======================================================")
    norm_df = df[df["label"] == 0]
    n_norm = len(norm_df)
    print(f"Total Normal windows: {n_norm}")
    
    results = {}
    for m in models:
        pred_col = f"pred_{m}"
        
        # Confusion matrix
        y_true = df["label"].values
        y_pred = df[pred_col].values
        
        # Per class recall and precision
        # Window FPR on Normal windows (y_true == 0, y_pred > 0)
        fp_windows = (norm_df[pred_col] > 0).sum()
        fpr = fp_windows / n_norm * 100.0
        
        # Entity FAR on pure-normal entities
        pure_norm_sub = df[df["entity_id"].isin(pure_norm_eids)]
        ent_alerts = pure_norm_sub.groupby("entity_id")[pred_col].apply(lambda s: (s > 0).any())
        far_ent = ent_alerts.sum() / len(pure_norm_eids) * 100.0
        
        # Gates
        pass_far = far_ent <= 12.0
        pass_fpr = fpr <= 5.0
        pass_both = pass_far and pass_fpr
        
        # Macro F1
        f1s = []
        for c in [0, 1, 2]:
            tp = ((y_true == c) & (y_pred == c)).sum()
            fp = ((y_true != c) & (y_pred == c)).sum()
            fn = ((y_true == c) & (y_pred != c)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)
        
        results[m] = {
            "fp_windows": int(fp_windows),
            "normal_denominator": int(n_norm),
            "window_fpr_pct": round(fpr, 4),
            "pass_fpr_gate": bool(pass_fpr),
            "pure_normal_entities_evaluated": len(pure_norm_eids),
            "far_entities_count": int(ent_alerts.sum()),
            "entity_far_pct": round(far_ent, 4),
            "pass_far_gate": bool(pass_far),
            "pass_both_gates": bool(pass_both),
            "macro_f1": round(macro_f1, 4),
        }
        print(f"[{model_names[m]}]: FPR = {fpr:.2f}% ({fp_windows}/{n_norm}) [Gate: {'PASS' if pass_fpr else 'FAIL'}], FAR = {far_ent:.2f}% ({ent_alerts.sum()}/198) [Gate: {'PASS' if pass_far else 'FAIL'}], Macro-F1 = {macro_f1:.4f}, Both: {'PASS' if pass_both else 'FAIL'}")
    return results

# Scenario 0: Frozen Confirmatory (all 5,511)
s0 = evaluate_subset(preds, "Scenario 0: Frozen Confirmatory (N=5,511)")

# Scenario 1: Exclude 149 Post-Failure Windows (N=5,362)
s1_df = preds[preds["window_type"] != "FAIL_NORM_POST"]
s1 = evaluate_subset(s1_df, "Scenario 1: Censoring 149 Post-Failure Recovery Windows (N=5,362)")

# Scenario 2: Exclude all 296 Failure-Cohort Normal Windows (N=5,215)
s2_df = preds[~preds["window_type"].isin(["FAIL_NORM_POST", "FAIL_NORM_PRE_CROSS_HIST", "FAIL_NORM_PRE_CLEAN"])]
s2 = evaluate_subset(s2_df, "Scenario 2: Censoring all 296 Failure-Cohort Normal Windows (N=5,215)")

# Scenario 3: Exclude 149 Post-Failure + 7 Cross-Failure History (N=5,355)
s3_df = preds[~preds["window_type"].isin(["FAIL_NORM_POST", "FAIL_NORM_PRE_CROSS_HIST"])]
s3 = evaluate_subset(s3_df, "Scenario 3: Censoring 149 Post-Fail + 7 Cross-History Windows (N=5,355)")
