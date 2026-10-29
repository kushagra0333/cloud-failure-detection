import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/hp/Desktop/Research project")
data_dir_7l = PROJECT_ROOT / "results" / "phase7l_longer_context" / "standardized_data"
data_dir_7m = PROJECT_ROOT / "results" / "phase7m_multiresolution" / "processed_data"

X_tr_l12 = np.load(data_dir_7l / "X_train_l12.npy").astype(np.float32)
X_val_l12 = np.load(data_dir_7l / "X_val_l12.npy").astype(np.float32)
tr_meta = pd.read_csv(data_dir_7m / "train_metadata.csv")
val_meta = pd.read_csv(data_dir_7m / "val_metadata.csv")

# 1. Compute scale floors strictly on Train
hist_tr = X_tr_l12[:, :7, :]
h_med_tr = np.median(hist_tr, axis=1)
h_mad_tr = np.median(np.abs(hist_tr - h_med_tr[:, None, :]), axis=1)
h_scale_tr = 1.4826 * h_mad_tr

scale_floors = np.zeros(31, dtype=np.float32)
for j in range(31):
    pos = h_scale_tr[:, j][h_scale_tr[:, j] > 1e-6]
    scale_floors[j] = float(np.percentile(pos, 10.0)) if len(pos) > 0 else 0.01

print(f"Scale floors: min={scale_floors.min():.4f}, median={np.median(scale_floors):.4f}, max={scale_floors.max():.4f}")

def transform_s2(X_12, scale_floors):
    hist = X_12[:, :7, :]
    h_med = np.median(hist, axis=1) # (N, 31)
    h_mad = np.median(np.abs(hist - h_med[:, None, :]), axis=1) # (N, 31)
    h_scale = 1.4826 * h_mad # (N, 31)
    
    eff_scale = np.maximum(h_scale, scale_floors[None, :]) # (N, 31)
    
    raw_rel = (X_12 - h_med[:, None, :]) / eff_scale[:, None, :]
    X_rel = np.clip(raw_rel, -5.0, 5.0).astype(np.float32)
    
    # Diagnostics
    floor_hit_rate = float((h_scale < scale_floors[None, :]).mean())
    clip_hit_rate = float((np.abs(raw_rel) >= 5.0).mean())
    
    # Stream A: recent 5 steps
    X_rec = X_rel[:, 7:, :] # (N, 5, 31)
    
    # Stream B: 93 summary stats derived exclusively from 7 relative historical steps
    hist_rel = X_rel[:, :7, :] # (N, 7, 31)
    early_rel = hist_rel[:, :4, :]
    late_rel = hist_rel[:, 4:, :]
    mean_all = np.mean(hist_rel, axis=1)
    std_all = np.std(hist_rel, axis=1)
    trend_delta = np.mean(late_rel, axis=1) - np.mean(early_rel, axis=1)
    X_hist = np.concatenate([mean_all, std_all, trend_delta], axis=1).astype(np.float32)
    
    return X_rec, X_hist, floor_hit_rate, clip_hit_rate

X_tr_s2_rec, X_tr_s2_hist, fl_tr, cl_tr = transform_s2(X_tr_l12, scale_floors)
X_val_s2_rec, X_val_s2_hist, fl_val, cl_val = transform_s2(X_val_l12, scale_floors)

print(f"Train S2 Rec shape: {X_tr_s2_rec.shape}, Hist shape: {X_tr_s2_hist.shape}")
print(f"Val S2 Rec shape:   {X_val_s2_rec.shape}, Hist shape: {X_val_s2_hist.shape}")
print(f"Train S2: floor_hit_rate={fl_tr*100:.2f}%, clip_hit_rate={cl_tr*100:.2f}%")
print(f"Val S2:   floor_hit_rate={fl_val*100:.2f}%, clip_hit_rate={cl_val*100:.2f}%")
print(f"All finite: Train={np.all(np.isfinite(X_tr_s2_rec)) and np.all(np.isfinite(X_tr_s2_hist))}, Val={np.all(np.isfinite(X_val_s2_rec)) and np.all(np.isfinite(X_val_s2_hist))}")
print(f"Range S2 Rec: [{X_tr_s2_rec.min():.2f}, {X_tr_s2_rec.max():.2f}]")
print(f"Range S2 Hist: [{X_tr_s2_hist.min():.2f}, {X_tr_s2_hist.max():.2f}]")

# Check Job 6251277720 (940 normal windows) vs Job 6251995937 (failing job)
mask_940 = (tr_meta["parent_job_id"] == 6251277720)
mask_fail = (tr_meta["parent_job_id"] == 6251995937)

print("\n--- Telemetry comparison: Absolute (S0) vs Relative (S2) ---")
# In S0 (absolute), what was the mean norm of features?
X_tr_s0_rec = np.load(data_dir_7m / "X_train_stream_A.npy")
print(f"Job 6251277720 (940 wins, pure normal):")
print(f"  S0 Absolute Rec mean abs value: {np.abs(X_tr_s0_rec[mask_940]).mean():.4f}")
print(f"  S2 Relative Rec mean abs value: {np.abs(X_tr_s2_rec[mask_940]).mean():.4f}")
print(f"Job 6251995937 (283 wins, 81% fail):")
print(f"  S0 Absolute Rec mean abs value: {np.abs(X_tr_s0_rec[mask_fail]).mean():.4f}")
print(f"  S2 Relative Rec mean abs value: {np.abs(X_tr_s2_rec[mask_fail]).mean():.4f}")
