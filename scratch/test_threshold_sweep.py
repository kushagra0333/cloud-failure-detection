"""Test threshold sweep for Phase 7J across Conditions A, B, C, D."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

# Load data
df = pd.read_csv("results/phase7i/phase7i_validation_predictions.csv")
df["fail_timestamp"] = (df["prediction_timestamp"] + df["time_to_failure"]).round(1)

y_true = df["y_true"].values
n_total = len(y_true)
n_norm = int((y_true == 0).sum())
n_warn = int((y_true == 1).sum())
n_crit = int((y_true == 2).sum())

# Episode setup
prefails = df[df["y_true"] > 0]
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
total_episodes = len(episodes)

# Entity setup
failing_eids = set(prefails["entity_id"].unique())
all_eids = sorted(df["entity_id"].unique())
pure_norm_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_norm_eids = len(pure_norm_eids)

# Job setup
failing_jobs = set(df[df["y_true"] > 0]["parent_job_id"].unique())
all_jobs = sorted(df["parent_job_id"].unique())
pure_norm_jobs = sorted(list(set(all_jobs) - failing_jobs))
n_pure_norm_jobs = len(pure_norm_jobs)

print(f"Validation cohort: {n_total} windows ({n_norm} norm, {n_warn} warn, {n_crit} crit)")
print(f"Episodes: {total_episodes}, Pure-normal entities: {n_pure_norm_eids}, Pure-normal jobs: {n_pure_norm_jobs}")

conditions = ["condition_A", "condition_B", "condition_C", "condition_D"]
thresholds = np.linspace(0.0, 1.0, 201)

results_summary = []

for cid in conditions:
    p_fail = df[f"p_fail_{cid}"].values
    p1 = df[f"prob_{cid}_c1"].values
    p2 = df[f"prob_{cid}_c2"].values
    
    passing_policy_a = []
    passing_policy_b = []
    
    for th in thresholds:
        # Policy A
        alert_a = (p_fail >= th).astype(int)
        df["alert_a"] = alert_a
        
        # Episode recall
        det_eps = sum(1 for _, grp in episodes if (df.loc[grp.index, "alert_a"] > 0).any())
        ep_rec = det_eps / total_episodes * 100.0
        
        # Binary Critical Interception
        crit_inter = int(alert_a[y_true == 2].sum())
        crit_inter_pct = crit_inter / n_crit * 100.0
        
        # Entity FAR
        fa_ents = sum(1 for eid in pure_norm_eids if (df[df["entity_id"] == eid]["alert_a"] > 0).any())
        ent_far = fa_ents / n_pure_norm_eids * 100.0
        
        # Window FPR
        fp_wins = int(alert_a[y_true == 0].sum())
        win_fpr = fp_wins / n_norm * 100.0
        
        # Job FAR
        fa_jobs = sum(1 for jid in pure_norm_jobs if (df[df["parent_job_id"] == jid]["alert_a"] > 0).any())
        job_far = fa_jobs / n_pure_norm_jobs * 100.0
        
        # Gate check Policy A
        g1a_pass = (ep_rec >= 70.0) and (crit_inter_pct >= 50.0)
        g2a_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        if g1a_pass and g2a_pass:
            passing_policy_a.append({
                "threshold": float(th),
                "ep_rec": ep_rec,
                "crit_inter": crit_inter_pct,
                "ent_far": ent_far,
                "win_fpr": win_fpr
            })
            
        # Policy B
        # If p_fail < th: 0; else if p2 >= p1: 2; else: 1
        pred_b = np.zeros(n_total, dtype=int)
        mask_alert = (p_fail >= th)
        mask_crit = mask_alert & (p2 >= p1)
        mask_warn = mask_alert & (p1 > p2)
        pred_b[mask_crit] = 2
        pred_b[mask_warn] = 1
        
        crit_correct_b = int(((y_true == 2) & (pred_b == 2)).sum())
        crit_rec_b = crit_correct_b / n_crit * 100.0
        
        warn_correct_b = int(((y_true == 1) & (pred_b == 1)).sum())
        warn_rec_b = warn_correct_b / n_warn * 100.0
        
        # Gate check Policy B
        g1b_pass = (ep_rec >= 70.0) and (crit_rec_b >= 50.0)
        g2b_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        if g1b_pass and g2b_pass:
            passing_policy_b.append({
                "threshold": float(th),
                "ep_rec": ep_rec,
                "crit_rec": crit_rec_b,
                "ent_far": ent_far,
                "win_fpr": win_fpr
            })

    print(f"\nCondition {cid}:")
    print(f"  Policy A passing thresholds: {len(passing_policy_a)}")
    if passing_policy_a:
        print("    Passing A:", passing_policy_a)
    print(f"  Policy B passing thresholds: {len(passing_policy_b)}")
    if passing_policy_b:
        print("    Passing B:", passing_policy_b)

    # Let's inspect specific thresholds for Condition A to see the tradeoff
    print(f"  Condition {cid} Sample points (Policy B):")
    sample_ths = [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22] if cid == "condition_A" else \
                 [0.22, 0.25, 0.27, 0.28, 0.30, 0.32, 0.35] if cid == "condition_B" else \
                 [0.30, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45] if cid == "condition_C" else \
                 [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]
    for th in sample_ths:
        mask_alert = (p_fail >= th)
        det_eps = sum(1 for _, grp in episodes if (p_fail[grp.index] >= th).any())
        ep_rec = det_eps / total_episodes * 100.0
        crit_correct_b = int(((y_true == 2) & mask_alert & (p2 >= p1)).sum())
        crit_rec_b = crit_correct_b / n_crit * 100.0
        fa_ents = sum(1 for eid in pure_norm_eids if (p_fail[df["entity_id"] == eid] >= th).any())
        ent_far = fa_ents / n_pure_norm_eids * 100.0
        fp_wins = int(mask_alert[y_true == 0].sum())
        win_fpr = fp_wins / n_norm * 100.0
        print(f"    th={th:.3f}: EpRec={ep_rec:.1f}% ({det_eps}/20), CritRec={crit_rec_b:.1f}% ({crit_correct_b}/49), EntFAR={ent_far:.1f}% ({fa_ents}/49), WinFPR={win_fpr:.1f}% ({fp_wins}/560)")
