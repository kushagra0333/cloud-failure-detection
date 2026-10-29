"""Preflight investigation script for Phase 7U."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(".").resolve()

print("=== PHASE 7U PREFLIGHT INVESTIGATION ===")

# 1. Inspect Phase 7T artifacts
oof_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "oof_hybrid_predictions.csv"
val_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "validation_hybrid_predictions.csv"
thresh_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "threshold_selection_results.csv"
policy_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "frozen_operating_policy.json"
gate_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "operational_gate_summary.csv"
per_seed_path = PROJECT_ROOT / "results" / "phase7t_hybrid_risk" / "per_seed_metrics.csv"

oof_df = pd.read_csv(oof_path)
val_df = pd.read_csv(val_path)
thresh_df = pd.read_csv(thresh_path)
with open(policy_path, "r") as f:
    policy = json.load(f)
gate_df = pd.read_csv(gate_path)
per_seed_df = pd.read_csv(per_seed_path)

print(f"OOF shape: {oof_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Frozen Policy tau: {policy['operating_threshold']}")

# 2. Check the 0.25% Train-OOF episode recall reported at tau=0.765
row_765 = thresh_df[np.isclose(thresh_df["tau"], 0.765)].iloc[0]
print("\n--- Train-OOF Threshold Selection at tau=0.765 ---")
print(f"mean_ep_rec: {row_765['mean_ep_rec']}")
print(f"mean_crit_rec: {row_765['mean_crit_rec']}")
print(f"mean_far: {row_765['mean_far']}")
print(f"mean_fpr: {row_765['mean_fpr']}")
for s in [42, 123, 456]:
    print(f"Seed {s}: ep_rec={row_765[f'ep_rec_s{s}']}%, far={row_765[f'far_s{s}']}%, fpr={row_765[f'fpr_s{s}']}%")

# Let's count how many episodes were detected out of 131 failing entities on Train-OOF
entity_max_label = oof_df.groupby("entity_id")["y_true"].max()
failing_eids = entity_max_label[entity_max_label > 0].index.values
n_episodes = len(failing_eids)
print(f"\nTotal Train failing entities (episodes): {n_episodes}")

for s in [42, 123, 456]:
    r_hyb = oof_df[f"Hybrid_pfail_s{s}"].values
    alert = (r_hyb >= 0.765)
    det_eids = [eid for eid in failing_eids if alert[oof_df["entity_id"].values == eid].any()]
    det_count = len(det_eids)
    pct = (det_count / n_episodes) * 100.0
    print(f"Seed {s}: {det_count} / {n_episodes} episodes detected = {pct:.4f}% (Stored in csv: {row_765[f'ep_rec_s{s}']})")

# Check if 1 / 131 = 0.7634% or where 0.25% came from:
# If seed 42 had 0, seed 123 had 1, seed 456 had 0: (0 + 0.7634 + 0) / 3 = 0.2545%!
# Indeed, 0.2545% is 1 episode detected across 3 seeds! (mean of 0.0%, 0.7634%, 0.0% is 0.2545%!).
# That explains the exact arithmetic: it was 1 episode detected in seed 123 and 0 in seeds 42 and 456, so the average is 0.25%.
# It was NOT a fraction-percentage mixup (e.g. 0.25 instead of 25%), it was genuinely 1 detection out of 393 total episode evaluations!

# 3. Check Brier score relative to constant prevalence baseline:
val_y_fail = val_df["y_fail"].values
val_prev = val_y_fail.mean()
train_prev = oof_df["y_fail"].values.mean()

print(f"\nValidation failure prevalence: {val_prev:.4f} ({val_y_fail.sum()}/{len(val_y_fail)})")
print(f"Train failure prevalence: {train_prev:.4f} ({oof_df['y_fail'].sum()}/{len(oof_df)})")

# Constant prevalence predictors
brier_const_val = brier_score_loss(val_y_fail, np.full_like(val_y_fail, val_prev, dtype=float))
brier_const_train = brier_score_loss(val_y_fail, np.full_like(val_y_fail, train_prev, dtype=float))
print(f"Constant Validation Prevalence Brier: {brier_const_val:.4f}")
print(f"Constant Train Prevalence Brier: {brier_const_train:.4f}")

# Model Brier scores on Validation:
for model_col, name in [
    ("S0_pfail", "S0"),
    ("S2_pfail", "S2"),
    ("Hybrid_pfail", "Hybrid")
]:
    briers = []
    for s in [42, 123, 456]:
        b = brier_score_loss(val_y_fail, val_df[f"{model_col}_s{s}"].values)
        briers.append(b)
    print(f"{name} Validation Brier: mean={np.mean(briers):.4f}, per-seed={[round(x, 4) for x in briers]}")

# 4. Check empirical quantiles of R_rel (S2) and R_abs (S0) on Train-OOF for membership function breakpoints
print("\n--- Empirical Quantiles on Train-OOF (Pooled across 3 seeds) ---")
r_rel_pooled = np.concatenate([oof_df[f"S2_pfail_s{s}"].values for s in [42, 123, 456]])
r_abs_pooled = np.concatenate([oof_df[f"S0_pfail_s{s}"].values for s in [42, 123, 456]])

for name, arr in [("R_rel (S2)", r_rel_pooled), ("R_abs (S0)", r_abs_pooled)]:
    q20 = np.percentile(arr, 20)
    q50 = np.percentile(arr, 50)
    q80 = np.percentile(arr, 80)
    print(f"{name}: min={arr.min():.4f}, 20th={q20:.4f}, 50th={q50:.4f}, 80th={q80:.4f}, max={arr.max():.4f}")
    # Check if quantiles are strictly increasing / non-degenerate
    is_degenerate = (q20 == q50) or (q50 == q80) or (q20 == q80)
    print(f"  Degenerate quantiles? {is_degenerate}")
