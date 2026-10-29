import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.features.registry import FeatureRegistry

val_pred = pd.read_csv("results/phase7e/validation_predictions.csv")
X_val = np.load("data/processed/phase7c/X_val.npy")
feature_names = FeatureRegistry().get_feature_names()

m_key = "transformer_condC_inverse_freq"

# 1. Episode Definition & Analysis
prefails = val_pred[val_pred["label"] > 0].copy()
prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
episodes = prefails.groupby(["entity_id", "fail_timestamp"])

print(f"=== STEP 1: VALIDATION ERROR ANALYSIS FOR {m_key} ===")
print(f"Total Validation Windows: {len(val_pred)}")
print(f"Total Validation Episodes: {len(episodes)}")

failing_eids = set(prefails["entity_id"].unique())
all_eids = sorted(val_pred["entity_id"].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))

# Check episode detection
ep_summary = []
for (eid, ftime), grp in episodes:
    parent_job = grp["parent_job_id"].iloc[0]
    preds = grp[f"y_pred_{m_key}"].values
    probs = grp[f"p_fail_{m_key}"].values
    detected = (preds > 0).any()
    max_prob = float(probs.max())
    mean_prob = float(probs.mean())
    n_win = len(grp)
    lead_time = float(ftime - grp[grp[f"y_pred_{m_key}"] > 0]["prediction_timestamp"].iloc[0]) if detected else None
    
    ep_summary.append({
        "entity_id": eid,
        "parent_job_id": parent_job,
        "fail_timestamp": ftime,
        "total_windows": n_win,
        "detected": bool(detected),
        "lead_time_sec": lead_time,
        "max_p_fail": max_prob,
        "mean_p_fail": mean_prob
    })

ep_df = pd.DataFrame(ep_summary)
print(f"\nEpisodes Detected: {ep_df['detected'].sum()}/{len(ep_df)} ({ep_df['detected'].mean()*100:.1f}%)")
print("Detected Episodes:")
print(ep_df[ep_df["detected"]][["entity_id", "parent_job_id", "total_windows", "lead_time_sec", "max_p_fail"]])
print("\nMissed Episodes:")
print(ep_df[~ep_df["detected"]][["entity_id", "parent_job_id", "total_windows", "max_p_fail"]])

# 2. Pure Normal Entities & False Alarms
val_pred["parent_job_id"] = val_pred["parent_job_id"].astype(str)
pn_df = val_pred[val_pred["entity_id"].isin(pure_normal_eids)].copy()
print(f"\nTotal Pure Normal Entities: {len(pure_normal_eids)} across {pn_df['parent_job_id'].nunique()} parent jobs")

fa_by_entity = []
for eid in pure_normal_eids:
    egrp = pn_df[pn_df["entity_id"] == eid]
    pjob = egrp["parent_job_id"].iloc[0]
    n_fa = int((egrp[f"y_pred_{m_key}"] > 0).sum())
    has_fa = n_fa > 0
    max_p_fail = float(egrp[f"p_fail_{m_key}"].max())
    mean_p_fail = float(egrp[f"p_fail_{m_key}"].mean())
    fa_by_entity.append({
        "entity_id": eid,
        "parent_job_id": pjob,
        "windows": len(egrp),
        "has_fa": has_fa,
        "fa_windows": n_fa,
        "max_p_fail": max_p_fail,
        "mean_p_fail": mean_p_fail
    })
fa_ent_df = pd.DataFrame(fa_by_entity)
print(f"Entities with False Alarms: {fa_ent_df['has_fa'].sum()}/{len(fa_ent_df)} ({fa_ent_df['has_fa'].mean()*100:.1f}%)")

# False alarms by parent job
fa_by_job = fa_ent_df.groupby("parent_job_id").agg(
    total_entities=("entity_id", "count"),
    fa_entities=("has_fa", "sum"),
    total_windows=("windows", "sum"),
    fa_windows=("fa_windows", "sum"),
    mean_p_fail=("mean_p_fail", "mean"),
    max_p_fail=("max_p_fail", "max")
).reset_index()
fa_by_job["entity_far_pct"] = (fa_by_job["fa_entities"] / fa_by_job["total_entities"] * 100).round(1)
fa_by_job["window_fpr_pct"] = (fa_by_job["fa_windows"] / fa_by_job["total_windows"] * 100).round(1)
print("\nFalse Alarms Grouped by Pure-Normal Parent Job:")
print(fa_by_job.to_string())

# 3. Score Distributions Comparison
print("\nScore Distributions for P(Failure) = P(Warning) + P(Critical):")
for lbl, name in [(0, "NORMAL"), (1, "WARNING"), (2, "CRITICAL")]:
    sub = val_pred[val_pred["label"] == lbl][f"p_fail_{m_key}"]
    print(f"  Class {lbl} ({name:<8}, N={len(sub):3d}): mean={sub.mean():.4f}, median={sub.median():.4f}, std={sub.std():.4f}, min={sub.min():.4f}, max={sub.max():.4f}")

# Subdivide normal into pure normal vs normal on failing tasks
p_norm_pure = val_pred[(val_pred["label"] == 0) & (val_pred["entity_id"].isin(pure_normal_eids))][f"p_fail_{m_key}"]
p_norm_fail = val_pred[(val_pred["label"] == 0) & (~val_pred["entity_id"].isin(pure_normal_eids))][f"p_fail_{m_key}"]
print(f"  Class 0 (Pure Normal Entities, N={len(p_norm_pure)}): mean={p_norm_pure.mean():.4f}, median={p_norm_pure.median():.4f}")
print(f"  Class 0 (Failing Entities Normal, N={len(p_norm_fail)}): mean={p_norm_fail.mean():.4f}, median={p_norm_fail.median():.4f}")

# 4. Feature Telemetry Breakdown: False Positives vs True Negatives in Normal Windows
norm_indices = np.where(val_pred["label"] == 0)[0]
is_fp = (val_pred.iloc[norm_indices][f"y_pred_{m_key}"] > 0).values

X_norm = X_val[norm_indices, -1, :] # Step t_pred
df_norm_features = pd.DataFrame(X_norm, columns=feature_names)
df_norm_features["is_fp"] = is_fp

diffs = []
for feat in feature_names:
    mean_fp = df_norm_features[df_norm_features["is_fp"]][feat].mean()
    mean_tn = df_norm_features[~df_norm_features["is_fp"]][feat].mean()
    diffs.append({
        "feature": feat,
        "mean_fp": mean_fp,
        "mean_tn": mean_tn,
        "abs_delta": abs(mean_fp - mean_tn),
        "signed_delta": mean_fp - mean_tn
    })
diff_df = pd.DataFrame(diffs).sort_values("abs_delta", ascending=False)
print("\nTop 10 Features with Largest Discrepancy Between False Positives and True Negatives in Normal Windows:")
print(diff_df.head(10).to_string())

# Save analysis to json
out_data = {
    "model_analyzed": m_key,
    "episodes_summary": ep_summary,
    "pure_normal_jobs": fa_by_job.to_dict(orient="records"),
    "top_discrepancy_features": diff_df.head(15).to_dict(orient="records")
}
def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    return obj

with open("results/phase7f/validation_error_analysis.json", "w") as f:
    json.dump(to_serializable(out_data), f, indent=2)

print("\nSaved results/phase7f/validation_error_analysis.json successfully.")
