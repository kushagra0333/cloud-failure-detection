import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path(".")

# Load predictions
p7b_pred = pd.read_csv("results/phase7b/test_predictions.csv")
p7d_pred = pd.read_csv("results/phase7d/test_predictions.csv")

print("=== CHECK 3: COMPARE PREDICTIONS 7B vs 7D ===")
for m in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
    col_b = f"y_pred_{m}"
    col_d = f"y_pred_{m}"
    
    # Are discrete predictions identical?
    pred_b = p7b_pred[col_b].values
    pred_d = p7d_pred[col_d].values
    match_pct = (pred_b == pred_d).mean() * 100.0
    print(f"\nModel: {m}")
    print(f"  Discrete prediction agreement: {match_pct:.2f}% ({(pred_b == pred_d).sum()}/{len(pred_b)})")
    
    # Check probabilities
    prob_b_cols = [f"prob_{m}_c{c}" for c in [0, 1, 2]]
    prob_d_cols = [f"prob_{m}_c{c}" for c in [0, 1, 2]]
    
    probs_b = p7b_pred[prob_b_cols].values
    probs_d = p7d_pred[prob_d_cols].values
    max_prob_diff = np.max(np.abs(probs_b - probs_d))
    mean_prob_diff = np.mean(np.abs(probs_b - probs_d))
    print(f"  Max absolute probability difference: {max_prob_diff:.6e}")
    print(f"  Mean absolute probability difference: {mean_prob_diff:.6e}")
    
    # Class distribution in 7B
    b_counts = pd.Series(pred_b).value_counts().to_dict()
    d_counts = pd.Series(pred_d).value_counts().to_dict()
    print(f"  7B predicted class counts: {b_counts}")
    print(f"  7D predicted class counts: {d_counts}")

print("\n=== CHECK 3 & 4: CHECKPOINT IDENTIFIERS & WEIGHTS ===")
# Check checkpoint files
ckpt_7b_dir = root / "models" / "checkpoints_phase7b"
ckpt_7d_dir = root / "models" / "checkpoints_phase7d"

print(f"7B checkpoints exist: {ckpt_7b_dir.exists()}")
if ckpt_7b_dir.exists():
    for f in ckpt_7b_dir.glob("*"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"  7B {f.name}: {h}")

print(f"7D checkpoints exist: {ckpt_7d_dir.exists()}")
if ckpt_7d_dir.exists():
    for f in ckpt_7d_dir.glob("*"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"  7D {f.name}: {h}")
