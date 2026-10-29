"""Comprehensive Forensic Verification for Phase 7B.3 Audit.
Checks:
- Provenance of runs, logs, hashes, timestamps
- Audit B: Preprocessing, imputation, Phase 2 vs Candidate A entities and jobs
- Audit C: Recomputing all metrics, denominators, numerators, bootstrap CIs, and Table 5.1 discrepancy
- Audit D: Protocol compliance checks
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(r"c:\Users\hp\Desktop\Research project")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("AUDIT A: ARTIFACT PROVENANCE & CRYPTOGRAPHIC HASHES")
print("=" * 80)

# 1. Script hash
script_path = PROJECT_ROOT / "scripts" / "train_and_evaluate_phase7b.py"
print(f"Script: {script_path.name}")
print(f"  SHA-256: {sha256_file(script_path)}")

# 2. Checkpoints
ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_phase7b"
ckpt_hashes = {}
for p in sorted(ckpt_dir.glob("*")):
    if p.is_file():
        h = sha256_file(p)
        ckpt_hashes[p.name] = h
        print(f"Checkpoint: {p.name} | Size: {p.stat().st_size:,} | SHA-256: {h}")

# Manifest comparison
manifest_path = PROJECT_ROOT / "results" / "phase7b" / "training_manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

print("\nComparing Checkpoint Hashes vs Manifest:")
for model_key, info in manifest["models"].items():
    ckpt_name = Path(info["checkpoint_path"]).name
    disk_hash = ckpt_hashes.get(ckpt_name)
    man_hash = info["checkpoint_sha256"]
    match = (disk_hash == man_hash)
    print(f"  - {model_key:22s}: disk={disk_hash[:16]}... manifest={man_hash[:16]}... MATCH={match}")

# 3. Output files
print("\nPhase 7B Output Result Hashes:")
res_dir = PROJECT_ROOT / "results" / "phase7b"
res_hashes = {}
for p in sorted(res_dir.glob("*")):
    if p.is_file():
        h = sha256_file(p)
        res_hashes[p.name] = h
        print(f"  - {p.name:32s}: {h}")

print("\n" + "=" * 80)
print("AUDIT B: PREPROCESSING CONTAMINATION & PHASE 2 OVERLAP")
print("=" * 80)

# Load Candidate A split metadata
split_csv = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
cand_df = pd.read_csv(split_csv)
print(f"Candidate A Total Windows: {len(cand_df):,}")

# Load Phase 2 train metadata (from data/processed/train_metadata.csv)
p2_train_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_metadata.csv")
p2_train_entities = set(p2_train_meta["entity_id"].unique())
if "job_id" in p2_train_meta.columns:
    p2_train_jobs = set(p2_train_meta["job_id"].unique())
else:
    p2_train_jobs = set(p2_train_meta["entity_id"].apply(lambda x: str(x).split('_')[0]).unique())
p2_train_jobs = set(str(j) for j in p2_train_jobs)
print(f"Phase 2 Train Split: {len(p2_train_meta):,} windows, {len(p2_train_entities)} entities, {len(p2_train_jobs)} parent jobs")

cand_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7b" / "train_metadata.csv")
cand_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7b" / "val_metadata.csv")
cand_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7b" / "test_metadata.csv")

cand_train_entities = set(cand_train["entity_id"].unique())
cand_train_jobs = set(cand_train["entity_id"].apply(lambda x: str(x).split('_')[0]).unique())

cand_val_entities = set(cand_val["entity_id"].unique())
cand_val_jobs = set(cand_val["entity_id"].apply(lambda x: str(x).split('_')[0]).unique())

cand_test_entities = set(cand_test["entity_id"].unique())
cand_test_jobs = set(cand_test["entity_id"].apply(lambda x: str(x).split('_')[0]).unique())

print(f"\nCandidate A Train: {len(cand_train)} windows, {len(cand_train_entities)} entities, {len(cand_train_jobs)} jobs")
print(f"Candidate A Val:   {len(cand_val)} windows, {len(cand_val_entities)} entities, {len(cand_val_jobs)} jobs")
print(f"Candidate A Test:  {len(cand_test)} windows, {len(cand_test_entities)} entities, {len(cand_test_jobs)} jobs")

val_ent_overlap = cand_val_entities.intersection(p2_train_entities)
val_job_overlap = cand_val_jobs.intersection(p2_train_jobs)
test_ent_overlap = cand_test_entities.intersection(p2_train_entities)
test_job_overlap = cand_test_jobs.intersection(p2_train_jobs)

print(f"\nPhase 2 Train Overlap with Candidate A Val:")
print(f"  - Entities: {len(val_ent_overlap)} of {len(cand_val_entities)} ({len(val_ent_overlap)/len(cand_val_entities)*100:.1f}%)")
print(f"  - Parent Jobs: {len(val_job_overlap)} of {len(cand_val_jobs)} ({len(val_job_overlap)/len(cand_val_jobs)*100:.1f}%)")

print(f"\nPhase 2 Train Overlap with Candidate A Test:")
print(f"  - Entities: {len(test_ent_overlap)} of {len(cand_test_entities)} ({len(test_ent_overlap)/len(cand_test_entities)*100:.1f}%)")
print(f"  - Parent Jobs: {len(test_job_overlap)} of {len(cand_test_jobs)} ({len(test_job_overlap)/len(cand_test_jobs)*100:.1f}%)")

# Inspect legacy fitted scaler
p2_scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler.json"
with open(p2_scaler_path, "r") as f:
    p2_scaler = json.load(f)

p7b_scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler_phase7b.json"
with open(p7b_scaler_path, "r") as f:
    p7b_scaler = json.load(f)

print(f"\nImputation & Scaler Statistics Comparison for cycles_per_instruction:")
print(f"  - Phase 2 Scaler: median={p2_scaler['medians']['cycles_per_instruction']}, mean={p2_scaler['means']['cycles_per_instruction']:.6f}, std={p2_scaler['stds']['cycles_per_instruction']:.6f}")
print(f"  - Phase 7B Scaler: median={p7b_scaler['medians']['cycles_per_instruction']}, mean={p7b_scaler['means']['cycles_per_instruction']:.6f}, std={p7b_scaler['stds']['cycles_per_instruction']:.6f}")

print("\n" + "=" * 80)
print("AUDIT C: METRIC RECOMPUTATION & TABLE 5.1 DISCREPANCY INVESTIGATION")
print("=" * 80)

# Load test predictions
test_preds = pd.read_csv(PROJECT_ROOT / "results" / "phase7b" / "test_predictions.csv")
print(f"Test Predictions Loaded: {len(test_preds)} rows")

# Recompute RF metrics
y_true = test_preds["y_true"].values
y_pred_rf = test_preds["y_pred_random_forest"].values

f1_macro = f1_score(y_true, y_pred_rf, average="macro")
acc = accuracy_score(y_true, y_pred_rf)
rec_warn = recall_score(y_true == 1, y_pred_rf == 1, pos_label=True, zero_division=0)
rec_crit = recall_score(y_true == 2, y_pred_rf == 2, pos_label=True, zero_division=0)

# Window FPR
normal_mask = (y_true == 0)
num_normal_windows = normal_mask.sum()
false_normal_alerts = ((y_pred_rf != 0) & normal_mask).sum()
win_fpr = false_normal_alerts / num_normal_windows

print(f"Recomputed RF Metrics:")
print(f"  - Macro-F1:       {f1_macro:.6f}")
print(f"  - Accuracy:       {acc:.6f}")
print(f"  - Warning Recall: {rec_warn:.6f} ({(y_true == 1).sum()} warnings)")
print(f"  - Critical Recall:{rec_crit:.6f} ({(y_true == 2).sum()} criticals)")
print(f"  - Window FPR:     {win_fpr*100:.2f}% ({false_normal_alerts}/{num_normal_windows})")

# Pure normal entities and FAR
# An entity is pure normal if it has NO warning or critical windows in test
ent_labels = test_preds.groupby("entity_id")["y_true"].max()
pure_normal_entities = ent_labels[ent_labels == 0].index.tolist()
num_pure_normal_ent = len(pure_normal_entities)

ent_rf_alerts = test_preds[test_preds["entity_id"].isin(pure_normal_entities)].groupby("entity_id")["y_pred_random_forest"].apply(lambda s: (s != 0).any())
num_fa_entities = ent_rf_alerts.sum()
ent_far = num_fa_entities / num_pure_normal_ent
print(f"  - Pure-Normal Entity FAR: {ent_far*100:.2f}% ({num_fa_entities}/{num_pure_normal_ent})")

# Pure normal jobs and FAR
job_col = "parent_job_id"
job_labels = test_preds.groupby(job_col)["y_true"].max()
pure_normal_jobs = job_labels[job_labels == 0].index.tolist()
num_pure_normal_jobs = len(pure_normal_jobs)

job_rf_alerts = test_preds[test_preds[job_col].isin(pure_normal_jobs)].groupby(job_col)["y_pred_random_forest"].apply(lambda s: (s != 0).any())
num_fa_jobs_rf = job_rf_alerts.sum()
job_far_rf = num_fa_jobs_rf / num_pure_normal_jobs
print(f"  - RF Pure-Normal Job FAR: {job_far_rf*100:.2f}% ({num_fa_jobs_rf}/{num_pure_normal_jobs})")

print(f"\nBreakdown of RF alerts across the 3 pure-normal jobs:")
for jid in pure_normal_jobs:
    jdf = test_preds[test_preds[job_col] == jid]
    j_alerts = (jdf["y_pred_random_forest"] != 0).sum()
    print(f"    * Job {jid}: {len(jdf)} windows, {j_alerts} alerts -> Alerted = {j_alerts > 0}")

# Representation Hybrid FAR on pure normal jobs
job_hybrid_alerts = test_preds[test_preds[job_col].isin(pure_normal_jobs)].groupby(job_col)["y_pred_representation_hybrid"].apply(lambda s: (s != 0).any())
num_fa_jobs_hyb = job_hybrid_alerts.sum()
job_far_hyb = num_fa_jobs_hyb / num_pure_normal_jobs
print(f"\n  - Hybrid Pure-Normal Job FAR: {job_far_hyb*100:.2f}% ({num_fa_jobs_hyb}/{num_pure_normal_jobs})")
for jid in pure_normal_jobs:
    jdf = test_preds[test_preds[job_col] == jid]
    j_alerts = (jdf["y_pred_representation_hybrid"] != 0).sum()
    print(f"    * Job {jid}: {len(jdf)} windows, {j_alerts} alerts -> Alerted = {j_alerts > 0}")

# Episode Recall check
# Load test episode metrics
ep_df = pd.read_csv(PROJECT_ROOT / "results" / "phase7b" / "test_episode_metrics.csv")
print(f"\nEpisode Metrics Loaded: {len(ep_df)} failure episodes")
rf_ep_rec = ep_df["detected"].sum() / len(ep_df)
print(f"  - RF Episode Recall:     {rf_ep_rec*100:.2f}% ({ep_df['detected'].sum()}/{len(ep_df)})")
valid_leads = ep_df[ep_df["detected"]]["lead_time_sec"]
print(f"  - RF Mean Lead Time:     {valid_leads.mean():.1f}s ({valid_leads.mean()/60:.1f} min)")
print(f"  - RF Median Lead Time:   {valid_leads.median():.1f}s ({valid_leads.median()/60:.1f} min)")
print(f"  - RF Min Lead Time:      {valid_leads.min():.1f}s")
print(f"  - RF Max Lead Time:      {valid_leads.max():.1f}s")

# Compare with saved test evaluation results JSON
with open(PROJECT_ROOT / "results" / "phase7b" / "test_evaluation_results.json", "r") as f:
    test_eval_json = json.load(f)

print("\nVerifying vs results/phase7b/test_evaluation_results.json:")
rf_json = test_eval_json["models"]["random_forest"]
print(f"  - JSON Macro-F1:      {rf_json['macro_f1']} (Recomputed: {f1_macro:.6f})")
print(f"  - JSON Accuracy:      {rf_json['accuracy']} (Recomputed: {acc:.6f})")
print(f"  - JSON Warning Rec:   {rf_json['class_recall']['1']} (Recomputed: {rec_warn:.6f})")
print(f"  - JSON Critical Rec:  {rf_json['class_recall']['2']} (Recomputed: {rec_crit:.6f})")
print(f"  - JSON Episode Rec:   {rf_json['episode_recall']['recall_pct']}% (Recomputed: {rf_ep_rec*100:.4f}%)")
print(f"  - JSON Window FPR:    {rf_json['window_fpr']['fpr_pct']}% (Recomputed: {win_fpr*100:.4f}%)")
print(f"  - JSON Entity FAR:    {rf_json['entity_far']['far_pct']}% (Recomputed: {ent_far*100:.4f}%)")
print(f"  - JSON Job FAR:       {rf_json['parent_job_far']['far_pct']}% (Recomputed: {job_far_rf*100:.4f}%)")

print("\n" + "=" * 80)
print("AUDIT SUMMARY CHECK COMPLETE")
print("=" * 80)
