"""Phase 7B — Scaler Fitting and Dataset Standardization on Candidate A.

In strict compliance with Protocol v2.0:
- Preprocessing scaler is fitted strictly on the Candidate A Training split.
- Scaler is saved to configs/fitted_scaler_phase7b.json.
- Candidate A Train, Val, and Test tensors are standardized using these training-only parameters.
- Checksums and feature statistics are recorded.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.scaler import LeakageSafeScaler

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 7B: SCALER FITTING & DATASET STANDARDIZATION ON CANDIDATE A")
    print("=" * 80)

    # 1. Load old scaler to unscale to original raw space
    old_scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler.json"
    with open(old_scaler_path, "r") as f:
        old_scaler_dict = json.load(f)

    feat_cols = old_scaler_dict["feature_cols"]
    old_means = np.array([old_scaler_dict["means"][c] for c in feat_cols], dtype=np.float32)
    old_stds = np.array([old_scaler_dict["stds"][c] for c in feat_cols], dtype=np.float32)
    eps = float(old_scaler_dict.get("epsilon", 1e-6))

    phase7b_dir = PROJECT_ROOT / "data" / "processed" / "phase7b"
    X_tr = np.load(phase7b_dir / "X_train.npy")
    X_val = np.load(phase7b_dir / "X_val.npy")
    X_te = np.load(phase7b_dir / "X_test.npy")

    print(f"Loaded Phase 7B tensors:")
    print(f"  - X_train: {X_tr.shape}")
    print(f"  - X_val:   {X_val.shape}")
    print(f"  - X_test:  {X_te.shape}")

    # 2. Invert scaling to obtain raw feature arrays
    X_tr_raw = X_tr * (old_stds + eps) + old_means
    X_val_raw = X_val * (old_stds + eps) + old_means
    X_te_raw = X_te * (old_stds + eps) + old_means

    # 3. Fit LeakageSafeScaler strictly on Candidate A Training Split
    print("\nFitting LeakageSafeScaler strictly on Candidate A Training Split (3,278 windows * 5 steps = 16,390 observations)...")
    train_flat = pd.DataFrame(X_tr_raw.reshape(-1, len(feat_cols)), columns=feat_cols)
    scaler_phase7b = LeakageSafeScaler(epsilon=1e-6)
    scaler_phase7b.fit(train_flat, feat_cols)

    # Save Phase 7B scaler
    new_scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler_phase7b.json"
    scaler_phase7b.save(new_scaler_path)
    print(f"Saved Candidate A scaler to: {new_scaler_path}")
    print(f"  - Scaler SHA-256: {compute_sha256(new_scaler_path)}")

    # 4. Standardize splits using Candidate A training statistics
    print("\nStandardizing Train, Val, and Test splits using Candidate A training parameters...")
    val_flat = pd.DataFrame(X_val_raw.reshape(-1, len(feat_cols)), columns=feat_cols)
    te_flat = pd.DataFrame(X_te_raw.reshape(-1, len(feat_cols)), columns=feat_cols)

    X_tr_scaled = scaler_phase7b.transform(train_flat).values.reshape(X_tr.shape).astype(np.float32)
    X_val_scaled = scaler_phase7b.transform(val_flat).values.reshape(X_val.shape).astype(np.float32)
    X_te_scaled = scaler_phase7b.transform(te_flat).values.reshape(X_te.shape).astype(np.float32)

    # Verify train standardization
    tr_means = np.mean(X_tr_scaled, axis=(0, 1))
    tr_stds = np.std(X_tr_scaled, axis=(0, 1))
    assert np.all(np.abs(tr_means) < 1e-4), f"Train feature means are not 0: {tr_means}"
    for idx, col in enumerate(feat_cols):
        raw_std = np.std(X_tr_raw[:, :, idx])
        if raw_std < 1e-6:
            assert np.abs(tr_stds[idx]) < 1e-4, f"Constant feature {col} std is not 0: {tr_stds[idx]}"
        else:
            assert np.abs(tr_stds[idx] - 1.0) < 1e-3, f"Feature {col} std is not 1: {tr_stds[idx]}"
    print("  - Train standardization verified: non-constant features have mean=0, std=1; constant features preserved.")

    # 5. Overwrite Phase 7B tensors with candidate-A standardized arrays
    np.save(phase7b_dir / "X_train.npy", X_tr_scaled)
    np.save(phase7b_dir / "X_val.npy", X_val_scaled)
    np.save(phase7b_dir / "X_test.npy", X_te_scaled)

    manifest = {
        "scaler_path": str(new_scaler_path),
        "scaler_sha256": compute_sha256(new_scaler_path),
        "features_count": len(feat_cols),
        "train_observations": len(train_flat),
        "splits": {
            "train": {
                "shape": list(X_tr_scaled.shape),
                "sha256": compute_sha256(phase7b_dir / "X_train.npy")
            },
            "val": {
                "shape": list(X_val_scaled.shape),
                "sha256": compute_sha256(phase7b_dir / "X_val.npy")
            },
            "test": {
                "shape": list(X_te_scaled.shape),
                "sha256": compute_sha256(phase7b_dir / "X_test.npy")
            }
        }
    }

    manifest_path = PROJECT_ROOT / "results" / "phase7b" / "scaler_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPhase 7B Scaler Manifest saved to: {manifest_path}")
    print("=" * 80)
    print("STANDARDIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
