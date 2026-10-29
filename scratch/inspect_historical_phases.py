import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("C:/Users/hp/Desktop/Research project")

print("=== CHECKING HISTORICAL PHASES ===")

# Phase 7M
p7m_val_pred = pd.read_csv(PROJECT_ROOT / "results/phase7m_multiresolution/validation_predictions.csv")
print(f"\n--- Phase 7M Val Predictions (N={len(p7m_val_pred)}) ---")
for cond in ["cond_A", "cond_B"]:
    for s in [42, 123, 456]:
        col = f"p_fail_M0_{cond}_seed{s}"
        if col in p7m_val_pred.columns:
            pfail = p7m_val_pred[col]
            y = p7m_val_pred["label"] if "label" in p7m_val_pred.columns else p7m_val_pred["y_true"]
            y_fail = (y > 0).astype(int)
            auc = roc_auc_score(y_fail, pfail)
            print(f"7M M0 {cond} s{s}: AUC={auc:.4f} | c0={pfail[y==0].mean():.4f}, c1={pfail[y==1].mean():.4f}, c2={pfail[y==2].mean():.4f}")

# Phase 7N
p7n_val_pred = pd.read_csv(PROJECT_ROOT / "results/phase7n_critical_margin/validation_predictions.csv")
print(f"\n--- Phase 7N Val Predictions (N={len(p7n_val_pred)}) ---")
y_7n = p7n_val_pred["label"]
y_fail_7n = (y_7n > 0).astype(int)
for model in ["N0", "N1", "N2"]:
    for s in [42, 123, 456]:
        col = f"p_fail_{model}_seed{s}"
        if col in p7n_val_pred.columns:
            pfail = p7n_val_pred[col]
            auc = roc_auc_score(y_fail_7n, pfail)
            print(f"7N {model} s{s}: AUC={auc:.4f} | c0={pfail[y_7n==0].mean():.4f}, c1={pfail[y_7n==1].mean():.4f}, c2={pfail[y_7n==2].mean():.4f}")

# Phase 7P
p7p_val_pred = pd.read_csv(PROJECT_ROOT / "results/phase7p_failure_calibration/validation_predictions.csv")
print(f"\n--- Phase 7P Val Predictions (N={len(p7p_val_pred)}) ---")
y_7p = p7p_val_pred["y_true"]
y_fail_7p = (y_7p > 0).astype(int)
for s in [42, 123, 456]:
    col_raw = f"raw_p_fail_seed{s}"
    col_cal = f"cal_q_fail_seed{s}"
    if col_raw in p7p_val_pred.columns:
        pfail_raw = p7p_val_pred[col_raw]
        qfail = p7p_val_pred[col_cal]
        auc_raw = roc_auc_score(y_fail_7p, pfail_raw)
        auc_cal = roc_auc_score(y_fail_7p, qfail)
        print(f"7P s{s}: Raw AUC={auc_raw:.4f}, Cal AUC={auc_cal:.4f} | Raw: c0={pfail_raw[y_7p==0].mean():.4f}, c1={pfail_raw[y_7p==1].mean():.4f}, c2={pfail_raw[y_7p==2].mean():.4f}")
