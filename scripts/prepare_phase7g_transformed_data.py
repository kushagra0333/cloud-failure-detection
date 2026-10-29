"""Phase 7G: Transform Candidate A Train and Validation splits for Experiments A and B.

Produces:
  - Experiment A: Parent-job historical baseline (X_train_job_relative.npy, X_val_job_relative.npy)
  - Experiment B: Hierarchical causal baseline (X_train_hierarchical.npy, X_val_hierarchical.npy)
  - Corresponding routing manifests with per-window provenance.
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hierarchical_baseline import (
    compute_job_relative_transform,
    compute_hierarchical_transform
)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 7G: GENERATING CAUSAL BASELINE FEATURE MATRICES")
    print("=" * 80)
    
    # 1. Load clean Phase 7C arrays
    X_train = np.load("data/processed/phase7c/X_train.npy")
    y_train = np.load("data/processed/phase7c/y_train.npy")
    meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
    
    X_val = np.load("data/processed/phase7c/X_val.npy")
    y_val = np.load("data/processed/phase7c/y_val.npy")
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    
    print(f"Loaded: Train X={X_train.shape}, Val X={X_val.shape}")
    
    out_dir = PROJECT_ROOT / "results" / "phase7g"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Experiment A: Parent-Job Historical Baseline ---
    print("\n[Experiment A] Generating Parent-Job Baseline Tensors...")
    X_train_job, r_train_job = compute_job_relative_transform(X_train, meta_train, k_min_windows=2)
    X_val_job, r_val_job = compute_job_relative_transform(X_val, meta_val, k_min_windows=2)
    
    np.save(out_dir / "X_train_job_relative.npy", X_train_job)
    np.save(out_dir / "X_val_job_relative.npy", X_val_job)
    r_train_job.to_csv(out_dir / "routing_train_job_relative.csv", index=False)
    r_val_job.to_csv(out_dir / "routing_val_job_relative.csv", index=False)
    print("  Experiment A files saved.")
    
    # --- Experiment B: Hierarchical Causal Baseline ---
    print("\n[Experiment B] Generating Hierarchical Causal Baseline Tensors...")
    X_train_hier, r_train_hier = compute_hierarchical_transform(X_train, meta_train, k_min_windows=2)
    X_val_hier, r_val_hier = compute_hierarchical_transform(X_val, meta_val, k_min_windows=2)
    
    np.save(out_dir / "X_train_hierarchical.npy", X_train_hier)
    np.save(out_dir / "X_val_hierarchical.npy", X_val_hier)
    r_train_hier.to_csv(out_dir / "routing_train_hierarchical.csv", index=False)
    r_val_hier.to_csv(out_dir / "routing_val_hierarchical.csv", index=False)
    print("  Experiment B files saved.")
    
    print("\nVerification & Routing Breakdown:")
    for name, r_df in [("Exp A Val", r_val_job), ("Exp B Val", r_val_hier)]:
        print(f"\n{name} Source Distribution:")
        print(r_df["baseline_source"].value_counts())
        
    print("\nData preparation complete.")

if __name__ == "__main__":
    main()
