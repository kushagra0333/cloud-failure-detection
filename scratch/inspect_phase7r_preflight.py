import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path("C:/Users/hp/Desktop/Research project")

# 1. Load Phase 7c metadata and labels
p7c_tr_meta = pd.read_csv(PROJECT_ROOT / "data/processed/phase7c/train_metadata.csv")
p7c_val_meta = pd.read_csv(PROJECT_ROOT / "data/processed/phase7c/val_metadata.csv")
p7c_y_tr = np.load(PROJECT_ROOT / "data/processed/phase7c/y_train.npy")
p7c_y_val = np.load(PROJECT_ROOT / "data/processed/phase7c/y_val.npy")

print(f"Phase 7c Train: len(meta)={len(p7c_tr_meta)}, len(y)={len(p7c_y_tr)}")
print(f"Phase 7c Val:   len(meta)={len(p7c_val_meta)}, len(y)={len(p7c_y_val)}")
print(f"Phase 7c Train y counts: {pd.Series(p7c_y_tr).value_counts().to_dict()}")
print(f"Phase 7c Val y counts:   {pd.Series(p7c_y_val).value_counts().to_dict()}")

# 2. Compare with Phase 7m metadata and labels
p7m_data = PROJECT_ROOT / "results/phase7m_multiresolution/processed_data"
p7m_tr_meta = pd.read_csv(p7m_data / "train_metadata.csv")
p7m_val_meta = pd.read_csv(p7m_data / "val_metadata.csv")
p7m_y_tr = np.load(p7m_data / "y_train.npy")
p7m_y_val = np.load(p7m_data / "y_val.npy")

print(f"\np7c vs p7m Train y_true exact match: {(p7c_y_tr == p7m_y_tr).all()}")
print(f"p7c vs p7m Val y_true exact match:   {(p7c_y_val == p7m_y_val).all()}")
print(f"p7c vs p7m Train entity_id exact match: {(p7c_tr_meta['entity_id'] == p7m_tr_meta['entity_id']).all()}")
print(f"p7c vs p7m Train timestamp exact match: {(p7c_tr_meta['prediction_timestamp'] == p7m_tr_meta['prediction_timestamp']).all()}")

# 3. Check Phase 7Q OOF and Validation predictions
p7q_oof = pd.read_csv(PROJECT_ROOT / "results/phase7q_signal_audit/oof_predictions.csv")
p7q_val = pd.read_csv(PROJECT_ROOT / "results/phase7q_signal_audit/validation_predictions.csv")

print(f"\nPhase 7Q OOF len={len(p7q_oof)}, Val len={len(p7q_val)}")
print(f"p7c vs p7q OOF entity_id exact match: {(p7c_tr_meta['entity_id'] == p7q_oof['entity_id']).all()}")
print(f"p7c vs p7q OOF timestamp exact match: {(p7c_tr_meta['prediction_timestamp'] == p7q_oof['prediction_timestamp']).all()}")
print(f"p7c vs p7q OOF y_true exact match:    {(p7c_y_tr == p7q_oof['y_true']).all()}")
print(f"p7c vs p7q Val entity_id exact match: {(p7c_val_meta['entity_id'] == p7q_val['entity_id']).all()}")
print(f"p7c vs p7q Val timestamp exact match: {(p7c_val_meta['prediction_timestamp'] == p7q_val['prediction_timestamp']).all()}")
print(f"p7c vs p7q Val y_true exact match:    {(p7c_y_val == p7q_val['y_true']).all()}")

# 4. Check score distributions by class for Q0 and Q1
print("\n--- Phase 7Q OOF Score Distribution by True Class (Seed 42) ---")
for model in ["Q0", "Q1"]:
    print(f"\nModel: {model}")
    for c in [0, 1, 2]:
        sub = p7q_oof[p7q_oof["y_true"] == c]
        p0 = sub[f"{model}_prob_s42_c0"]
        p1 = sub[f"{model}_prob_s42_c1"]
        p2 = sub[f"{model}_prob_s42_c2"]
        pfail = sub[f"{model}_pfail_s42"]
        z0 = sub[f"{model}_logit_s42_c0"]
        z1 = sub[f"{model}_logit_s42_c1"]
        z2 = sub[f"{model}_logit_s42_c2"]
        print(f"Class {c} (N={len(sub)}): P0={p0.mean():.4f}, P1={p1.mean():.4f}, P2={p2.mean():.4f}, Pfail={pfail.mean():.4f} | z0={z0.mean():.4f}, z1={z1.mean():.4f}, z2={z2.mean():.4f}")

print("\n--- Phase 7Q Validation Score Distribution by True Class (Seed 42) ---")
for model in ["Q0", "Q1"]:
    print(f"\nModel: {model}")
    for c in [0, 1, 2]:
        sub = p7q_val[p7q_val["y_true"] == c]
        p0 = sub[f"{model}_prob_s42_c0"]
        p1 = sub[f"{model}_prob_s42_c1"]
        p2 = sub[f"{model}_prob_s42_c2"]
        pfail = sub[f"{model}_pfail_s42"]
        z0 = sub[f"{model}_logit_s42_c0"]
        z1 = sub[f"{model}_logit_s42_c1"]
        z2 = sub[f"{model}_logit_s42_c2"]
        print(f"Class {c} (N={len(sub)}): P0={p0.mean():.4f}, P1={p1.mean():.4f}, P2={p2.mean():.4f}, Pfail={pfail.mean():.4f} | z0={z0.mean():.4f}, z1={z1.mean():.4f}, z2={z2.mean():.4f}")

# 5. Check ROC-AUC and AP
print("\n--- AUC and Inverse AUC ---")
y_fail_oof = (p7q_oof["y_true"] > 0).astype(int)
y_fail_val = (p7q_val["y_true"] > 0).astype(int)

for model in ["Q0", "Q1"]:
    for s in [42, 123, 456]:
        pfail_oof = p7q_oof[f"{model}_pfail_s{s}"]
        p0_oof = p7q_oof[f"{model}_prob_s{s}_c0"]
        auc_oof = roc_auc_score(y_fail_oof, pfail_oof)
        auc_inv_oof = roc_auc_score(y_fail_oof, p0_oof)
        ap_oof = average_precision_score(y_fail_oof, pfail_oof)
        ap_inv_oof = average_precision_score(y_fail_oof, p0_oof)
        
        pfail_val = p7q_val[f"{model}_pfail_s{s}"]
        p0_val = p7q_val[f"{model}_prob_s{s}_c0"]
        auc_val = roc_auc_score(y_fail_val, pfail_val)
        auc_inv_val = roc_auc_score(y_fail_val, p0_val)
        ap_val = average_precision_score(y_fail_val, pfail_val)
        ap_inv_val = average_precision_score(y_fail_val, p0_val)
        print(f"{model} s{s}: OOF AUC={auc_oof:.4f} (inv={auc_inv_oof:.4f}), AP={ap_oof:.4f} (inv={ap_inv_oof:.4f}) | Val AUC={auc_val:.4f} (inv={auc_inv_val:.4f}), AP={ap_val:.4f} (inv={ap_inv_val:.4f})")
