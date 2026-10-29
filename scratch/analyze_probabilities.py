import pandas as pd
import numpy as np

df = pd.read_csv("results/phase7d/test_predictions.csv")

for m in ["temporal_lstm", "compact_transformer"]:
    p0 = df[f"prob_{m}_c0"].values
    p1 = df[f"prob_{m}_c1"].values
    p2 = df[f"prob_{m}_c2"].values
    p_fail = p1 + p2
    
    print(f"\n=== PROBABILITY ANALYSIS: {m} ===")
    print(f"P(Normal)   : min={p0.min():.4f}, max={p0.max():.4f}, mean={p0.mean():.4f}, median={np.median(p0):.4f}")
    print(f"P(Warning)  : min={p1.min():.4f}, max={p1.max():.4f}, mean={p1.mean():.4f}, median={np.median(p1):.4f}")
    print(f"P(Critical) : min={p2.min():.4f}, max={p2.max():.4f}, mean={p2.mean():.4f}, median={np.median(p2):.4f}")
    print(f"P(Failure)  : min={p_fail.min():.4f}, max={p_fail.max():.4f}, mean={p_fail.mean():.4f}, median={np.median(p_fail):.4f}")
    
    # Check on ground-truth failure windows (y_true > 0)
    fail_mask = (df["y_true"] > 0)
    print(f"On true failure windows (N={fail_mask.sum()}):")
    print(f"  P(Normal)  : mean={p0[fail_mask].mean():.4f}, median={np.median(p0[fail_mask]):.4f}")
    print(f"  P(Failure) : mean={p_fail[fail_mask].mean():.4f}, median={np.median(p_fail[fail_mask]):.4f}, max={p_fail[fail_mask].max():.4f}")
    
    # Check on normal windows (y_true == 0)
    norm_mask = (df["y_true"] == 0)
    print(f"On true normal windows (N={norm_mask.sum()}):")
    print(f"  P(Normal)  : mean={p0[norm_mask].mean():.4f}, median={np.median(p0[norm_mask]):.4f}")
    print(f"  P(Failure) : mean={p_fail[norm_mask].mean():.4f}, median={np.median(p_fail[norm_mask]):.4f}, max={p_fail[norm_mask].max():.4f}")
