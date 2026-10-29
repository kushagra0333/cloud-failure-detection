import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path("C:/Users/hp/Desktop/Research project")
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.multiresolution import MultiResolutionLSTMClassifier

# 1. Load Data
p7m_data = PROJECT_ROOT / "results/phase7m_multiresolution/processed_data"
X_tr_rec = np.load(p7m_data / "X_train_stream_A.npy").astype(np.float32)
X_tr_hist = np.load(p7m_data / "X_train_stream_B.npy").astype(np.float32)
y_tr = np.load(p7m_data / "y_train.npy").astype(np.int64)
tr_meta = pd.read_csv(p7m_data / "train_metadata.csv")

X_val_rec = np.load(p7m_data / "X_val_stream_A.npy").astype(np.float32)
X_val_hist = np.load(p7m_data / "X_val_stream_B.npy").astype(np.float32)
y_val = np.load(p7m_data / "y_val.npy").astype(np.int64)
val_meta = pd.read_csv(p7m_data / "val_metadata.csv")

y_fail_tr = (y_tr > 0).astype(int)
y_fail_val = (y_val > 0).astype(int)

# 2. Check saved Phase 7N checkpoints (N2 full-train models)
ckpt_dir_7n = PROJECT_ROOT / "results/phase7n_critical_margin/checkpoints"
print("=== EVALUATING PHASE 7N N2 FULL-TRAIN MODELS ON TRAIN AND VAL ===")

for s in [42, 123, 456]:
    ckpt_path = ckpt_dir_7n / f"ckpt_N2_seed{s}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        continue
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiResolutionLSTMClassifier(
        input_dim=31, hidden_dim=64, num_layers=2, num_classes=3,
        dropout=0.2, use_history=True, hist_dim=93, hist_hidden=32, hist_out=16
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        # On Train
        logits_tr = model(torch.from_numpy(X_tr_rec), torch.from_numpy(X_tr_hist))
        probs_tr = F.softmax(logits_tr, dim=-1).numpy()
        pfail_tr = probs_tr[:, 1] + probs_tr[:, 2]

        # On Val
        logits_val = model(torch.from_numpy(X_val_rec), torch.from_numpy(X_val_hist))
        probs_val = F.softmax(logits_val, dim=-1).numpy()
        pfail_val = probs_val[:, 1] + probs_val[:, 2]

    # In-sample Train AUC
    auc_tr = roc_auc_score(y_fail_tr, pfail_tr)
    ap_tr = average_precision_score(y_fail_tr, pfail_tr)
    # Val AUC
    auc_val = roc_auc_score(y_fail_val, pfail_val)
    ap_val = average_precision_score(y_fail_val, pfail_val)

    print(f"\nSeed {s}:")
    print(f"  In-Sample Train: AUC={auc_tr:.4f}, AP={ap_tr:.4f} | c0={pfail_tr[y_tr==0].mean():.4f}, c1={pfail_tr[y_tr==1].mean():.4f}, c2={pfail_tr[y_tr==2].mean():.4f}")
    print(f"  Validation:      AUC={auc_val:.4f}, AP={ap_val:.4f} | c0={pfail_val[y_val==0].mean():.4f}, c1={pfail_val[y_val==1].mean():.4f}, c2={pfail_val[y_val==2].mean():.4f}")

# 3. Check simple feature correlations with y_fail on Train and Val
print("\n=== RAW FEATURE CORRELATION WITH FAILURE (LAST TIMESTEP) ===")
# X_tr_rec is (3278, 5, 31). Step 4 is current t.
x_tr_last = X_tr_rec[:, 4, :]
corrs_tr = []
for j in range(31):
    r, _ = pearsonr(x_tr_last[:, j], y_fail_tr)
    rho, _ = spearmanr(x_tr_last[:, j], y_fail_tr)
    corrs_tr.append((j, r, rho))

corrs_tr_sorted = sorted(corrs_tr, key=lambda x: abs(x[1]), reverse=True)
print("Top 10 features by Pearson |r| with y_fail on Train:")
for j, r, rho in corrs_tr_sorted[:10]:
    print(f"  Feature {j:2d}: r={r:+.4f}, spearman={rho:+.4f}")

# 4. Check Parent Job Distribution on Train
print("\n=== PARENT JOB DISTRIBUTIONS & FAILURE RATES ON TRAIN ===")
tr_meta["y_fail"] = y_fail_tr
job_stats = tr_meta.groupby("parent_job_id").agg(
    windows=("y_fail", "count"),
    fails=("y_fail", "sum"),
    prev=("y_fail", "mean")
).reset_index().sort_values("windows", ascending=False)

print(f"Total parent jobs on Train: {len(job_stats)}")
print("Top 5 largest parent jobs on Train:")
for _, row in job_stats.head(5).iterrows():
    print(f"  Job {row['parent_job_id']}: {int(row['windows']):4d} windows ({row['windows']/len(tr_meta)*100:.1f}%), fails={int(row['fails'])}, prev={row['prev']*100:.2f}%")

print(f"Fraction of Train windows in top 3 jobs: {job_stats.head(3)['windows'].sum() / len(tr_meta) * 100:.2f}%")
print(f"Fraction of Train windows in top 5 jobs: {job_stats.head(5)['windows'].sum() / len(tr_meta) * 100:.2f}%")
