"""Inspect score ranges and verify entity/episode mapping."""
import pandas as pd
import numpy as np

df_preds = pd.read_csv("results/phase7i/phase7i_validation_predictions.csv")
val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")

print("df_preds shape:", df_preds.shape)
print("val_meta shape:", val_meta.shape)

# Verify alignment
for col in ["entity_id", "prediction_timestamp", "label", "parent_job_id"]:
    assert (df_preds[col].fillna(-1) == val_meta[col].fillna(-1)).all(), f"Mismatch in {col}!"

print("Alignment assertion passed: df_preds matches val_meta exactly row-for-row.")

# Verify time_to_failure and fail_timestamp
df_preds["fail_timestamp"] = (df_preds["prediction_timestamp"] + df_preds["time_to_failure"]).round(1)
prefails = df_preds[df_preds["label"] > 0]
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
print("Total failure episodes in validation set:", len(episodes))
assert len(episodes) == 20, f"Expected 20 episodes, got {len(episodes)}"

# Check score ranges
conditions = ["condition_A", "condition_B", "condition_C", "condition_D"]
for cid in conditions:
    p_fail = df_preds[f"p_fail_{cid}"]
    p0 = df_preds[f"prob_{cid}_c0"]
    p1 = df_preds[f"prob_{cid}_c1"]
    p2 = df_preds[f"prob_{cid}_c2"]
    
    # Check finite, range [0, 1], row sum approx 1
    assert np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()
    assert (p0 >= 0).all() and (p0 <= 1).all()
    assert (p1 >= 0).all() and (p1 <= 1).all()
    assert (p2 >= 0).all() and (p2 <= 1).all()
    diff = np.abs(p0 + p1 + p2 - 1.0)
    assert (diff < 1e-5).all(), f"Row sum mismatch in {cid}: max diff {diff.max()}"
    
    print(f"\n{cid}:")
    print(f"  P_fail min: {p_fail.min():.5f}, max: {p_fail.max():.5f}, mean: {p_fail.mean():.5f}")
    print(f"  P_norm min: {p0.min():.5f}, max: {p0.max():.5f}")
    print(f"  P_warn min: {p1.min():.5f}, max: {p1.max():.5f}")
    print(f"  P_crit min: {p2.min():.5f}, max: {p2.max():.5f}")
    
    # Check if p1 == p2 ever occurs
    ties = (p1 == p2).sum()
    print(f"  Exact ties P(warn) == P(crit): {ties}")
