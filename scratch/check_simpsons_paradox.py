import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path("C:/Users/hp/Desktop/Research project")

p7q_oof = pd.read_csv(PROJECT_ROOT / "results/phase7q_signal_audit/oof_predictions.csv")
p7q_val = pd.read_csv(PROJECT_ROOT / "results/phase7q_signal_audit/validation_predictions.csv")

print("=== CHECKING PER-JOB RANKING (SIMPSON'S PARADOX CHECK) ===")

def check_per_job_auc(df, pfail_col, title=""):
    print(f"\n--- {title} ({pfail_col}) ---")
    job_aucs = []
    job_rows = []
    
    for pjob, grp in df.groupby("parent_job_id"):
        n_tot = len(grp)
        n_fail = (grp["y_true"] > 0).sum()
        n_norm = (grp["y_true"] == 0).sum()
        mean_pfail_all = grp[pfail_col].mean()
        mean_pfail_norm = grp[grp["y_true"] == 0][pfail_col].mean() if n_norm > 0 else np.nan
        mean_pfail_fail = grp[grp["y_true"] > 0][pfail_col].mean() if n_fail > 0 else np.nan
        
        auc = np.nan
        if n_fail > 0 and n_norm > 0:
            try:
                auc = roc_auc_score((grp["y_true"] > 0).astype(int), grp[pfail_col])
                job_aucs.append((pjob, auc, n_tot, n_fail, n_norm))
            except Exception:
                pass
                
        job_rows.append({
            "parent_job_id": pjob,
            "n_tot": n_tot,
            "n_fail": n_fail,
            "n_norm": n_norm,
            "prev": n_fail / n_tot,
            "auc": auc,
            "mean_pfail_all": mean_pfail_all,
            "mean_pfail_norm": mean_pfail_norm,
            "mean_pfail_fail": mean_pfail_fail
        })
        
    job_df = pd.DataFrame(job_rows)
    print(f"Total jobs: {len(job_df)}, Jobs with both fail & norm: {len(job_aucs)}")
    
    if len(job_aucs) > 0:
        valid_aucs = [a for _, a, _, _, _ in job_aucs if not np.isnan(a)]
        weights = [t for _, _, t, _, _ in job_aucs]
        fail_weights = [f for _, _, _, f, _ in job_aucs]
        print(f"Unweighted Mean Within-Job AUC: {np.mean(valid_aucs):.4f}")
        print(f"Window-Weighted Mean Within-Job AUC: {np.average(valid_aucs, weights=weights):.4f}")
        print(f"Failure-Weighted Mean Within-Job AUC: {np.average(valid_aucs, weights=fail_weights):.4f}")
        
        print("\nIndividual Failing Jobs Within-Job Ranking:")
        for pjob, auc, n_tot, n_fail, n_norm in sorted(job_aucs, key=lambda x: x[2], reverse=True)[:10]:
            sub = job_df[job_df["parent_job_id"] == pjob].iloc[0]
            print(f"  Job {pjob}: N={n_tot:4d} (norm={n_norm:3d}, fail={n_fail:3d}) -> Within-Job AUC={auc:.4f} | Mean Pfail: norm={sub['mean_pfail_norm']:.4f}, fail={sub['mean_pfail_fail']:.4f}")

    # Now let's see: what are the mean Pfail across purely normal jobs vs failing jobs?
    pure_norm = job_df[job_df["n_fail"] == 0]
    failing = job_df[job_df["n_fail"] > 0]
    print(f"\nBetween-Job Comparison:")
    print(f"  Pure Normal Jobs (N={len(pure_norm)}, wins={pure_norm['n_tot'].sum()}): Mean Pfail = {pure_norm['mean_pfail_all'].mean():.4f}")
    print(f"  Failing Jobs (N={len(failing)}, wins={failing['n_tot'].sum()}):     Mean Pfail = {failing['mean_pfail_all'].mean():.4f}")

check_per_job_auc(p7q_oof, "Q0_pfail_s42", "Train OOF Q0 Seed 42")
check_per_job_auc(p7q_val, "Q0_pfail_s42", "Validation Q0 Seed 42")
