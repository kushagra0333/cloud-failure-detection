"""Phase 7H: Generate transformed feature tensors for Ablations B and C.

Ablation B: Hierarchical Causal Baseline with Variance Floor (sigma_floor = 0.05).
Ablation C: Robust Causal Scaling with Median, MAD (mad_floor = 0.05), and Soft Tanh Bounding (clip_bound = 5.0).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hierarchical_baseline import (
    compute_hierarchical_transform,
    compute_hierarchical_robust_transform
)

def main():
    print("=" * 80)
    print("PHASE 7H: GENERATING ABLATION FEATURE TENSORS")
    print("=" * 80)
    
    X_train = np.load("data/processed/phase7c/X_train.npy")
    meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
    
    X_val = np.load("data/processed/phase7c/X_val.npy")
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    
    out_dir = PROJECT_ROOT / "results" / "phase7h"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Ablation B: Variance Floored (sigma_floor = 0.05) ---
    print("\n[Ablation B] Generating Variance-Floored Hierarchical Tensors (sigma_floor=0.05)...")
    X_train_b, r_train_b = compute_hierarchical_transform(X_train, meta_train, k_min_windows=2, sigma_floor=0.05)
    X_val_b, r_val_b = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2, sigma_floor=0.05)
    
    np.save(out_dir / "X_train_hierarchical_floored.npy", X_train_b)
    np.save(out_dir / "X_val_hierarchical_floored.npy", X_val_b)
    r_train_b.to_csv(out_dir / "routing_train_hierarchical_floored.csv", index=False)
    r_val_b.to_csv(out_dir / "routing_val_hierarchical_floored.csv", index=False)
    print(f"  Val B Range: min={X_val_b.min():.2f}, max={X_val_b.max():.2f}")
    
    # --- Ablation C: Robust Scaled (MAD floor = 0.05, soft tanh clip = 5.0) ---
    print("\n[Ablation C] Generating Robust Scaled Tensors (mad_floor=0.05, clip=5.0)...")
    X_train_c, r_train_c = compute_hierarchical_robust_transform(X_train, meta_train, k_min_windows=2, mad_floor=0.05, clip_bound=5.0)
    X_val_c, r_val_c = compute_hierarchical_robust_transform(X_val, meta_val, k_min_windows=2, mad_floor=0.05, clip_bound=5.0)
    
    np.save(out_dir / "X_train_hierarchical_robust.npy", X_train_c)
    np.save(out_dir / "X_val_hierarchical_robust.npy", X_val_c)
    r_train_c.to_csv(out_dir / "routing_train_hierarchical_robust.csv", index=False)
    r_val_c.to_csv(out_dir / "routing_val_hierarchical_robust.csv", index=False)
    print(f"  Val C Range: min={X_val_c.min():.2f}, max={X_val_c.max():.2f}")
    
    print("\nAblation tensor generation complete.")

if __name__ == "__main__":
    main()
