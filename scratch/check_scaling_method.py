import numpy as np

X_tr_raw = np.load("data/processed/phase7c/raw_unscaled/X_train_raw.npy")
X_tr_scaled = np.load("data/processed/phase7c/X_train.npy")

print(f"X_tr_raw col 0 mean={np.nanmean(X_tr_raw[:, :, 0]):.4f}, std={np.nanstd(X_tr_raw[:, :, 0]):.4f}")
print(f"X_tr_scaled col 0 mean={np.nanmean(X_tr_scaled[:, :, 0]):.4f}, std={np.nanstd(X_tr_scaled[:, :, 0]):.4f}")
