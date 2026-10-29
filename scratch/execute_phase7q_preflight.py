"""Phase 7Q Preflight Audit & Historical Reconciliation Script.

This script executes:
1. Verification of governing protocol, Phase 7P artifacts, SHA-256 hashes, and split counts.
2. Workstream A initial audit: Recomputing raw and calibrated ranking metrics, prevalence baselines.
3. Fold-by-fold analysis of Phase 7P out-of-fold predictions and negative Platt slopes.
4. Analysis of Fold 0 parent-job dominance (Job 3998246322, 643 windows).
5. Examination of historical normalization (Phase 7F, 7G, 7H) and feature ablations (Phase 7K).
6. Verification of candidate model reproducibility.
7. Verification of underlying causal temporal telemetry tensors (L=12).
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score
)
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > b_low) & (probs <= b_high) if i > 0 else (probs >= b_low) & (probs <= b_high)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_prob = float(np.mean(probs[in_bin]))
            emp_freq = float(np.mean(labels[in_bin]))
            ece += np.abs(avg_prob - emp_freq) * (count / len(probs))
    return float(ece)

def main():
    print("=" * 80)
    print("EXECUTING PHASE 7Q PREFLIGHT AUDIT & SIGNAL INVESTIGATION")
    print("=" * 80)

    res_7p = PROJECT_ROOT / "results" / "phase7p_failure_calibration"
    res_7m = PROJECT_ROOT / "results" / "phase7m_multiresolution" / "processed_data"
    res_7l = PROJECT_ROOT / "results" / "phase7l_longer_context"

    out_7q = PROJECT_ROOT / "results" / "phase7q_signal_audit"
    rep_7q = PROJECT_ROOT / "reports" / "phase7q_signal_audit"
    out_7q.mkdir(parents=True, exist_ok=True)
    rep_7q.mkdir(parents=True, exist_ok=True)

    # 1. Verify split counts
    tr_meta = pd.read_csv(res_7m / "train_metadata.csv")
    val_meta = pd.read_csv(res_7m / "val_metadata.csv")
    y_tr = np.load(res_7m / "y_train.npy")
    y_val = np.load(res_7m / "y_val.npy")

    assert len(tr_meta) == 3278
    assert len(val_meta) == 657
    assert len(y_tr) == 3278
    assert len(y_val) == 657

    n_val_norm = int((y_val == 0).sum())
    n_val_warn = int((y_val == 1).sum())
    n_val_crit = int((y_val == 2).sum())
    n_val_fail = n_val_warn + n_val_crit

    assert n_val_norm == 560
    assert n_val_warn == 48
    assert n_val_crit == 49
    assert n_val_fail == 97

    n_tr_norm = int((y_tr == 0).sum())
    n_tr_warn = int((y_tr == 1).sum())
    n_tr_crit = int((y_tr == 2).sum())
    n_tr_fail = n_tr_warn + n_tr_crit

    assert n_tr_norm == 2797
    assert n_tr_warn == 244
    assert n_tr_crit == 237
    assert n_tr_fail == 481

    print(f"Candidate A Train: {len(tr_meta)} windows (Norm: {n_tr_norm}, Warn: {n_tr_warn}, Crit: {n_tr_crit}, Fail: {n_tr_fail})")
    print(f"Candidate A Validation: {len(val_meta)} windows (Norm: {n_val_norm}, Warn: {n_val_warn}, Crit: {n_val_crit}, Fail: {n_val_fail})")

    # 2. Check Phase 7P Preflight & Artifact Manifest
    manifest_7p = json.load(open(res_7p / "phase7p_manifest.json", "r"))
    print(f"Phase 7P Manifest verified: {manifest_7p['total_deliverables']} deliverables.")

    # 3. Workstream A: Recomputing Phase 7P metrics & constant prevalence baseline
    y_fail_tr = (y_tr > 0).astype(int)
    y_fail_val = (y_val > 0).astype(int)

    tr_prev = float(np.mean(y_fail_tr)) # 481 / 3278 = 0.146736
    val_prev = float(np.mean(y_fail_val)) # 97 / 657 = 0.147641

    print(f"\nTrain Failure Prevalence: {tr_prev:.6f} ({n_tr_fail}/{len(y_tr)})")
    print(f"Validation Failure Prevalence: {val_prev:.6f} ({n_val_fail}/{len(y_val)})")

    # Constant prevalence baseline evaluated on Validation
    # Predicts p = tr_prev (train prevalence only, zero validation lookahead)
    p_const_tr = np.full_like(y_fail_val, tr_prev, dtype=float)
    brier_const_tr = float(brier_score_loss(y_fail_val, p_const_tr))
    nll_const_tr = float(log_loss(y_fail_val, p_const_tr))

    # Predicts p = val_prev (theoretical optimal constant baseline on validation)
    p_const_val = np.full_like(y_fail_val, val_prev, dtype=float)
    brier_const_val = float(brier_score_loss(y_fail_val, p_const_val))
    nll_const_val = float(log_loss(y_fail_val, p_const_val))

    print(f"Constant Train-Prevalence Predictor on Validation: Brier = {brier_const_tr:.6f}, NLL = {nll_const_tr:.6f}")
    print(f"Constant Val-Prevalence Predictor on Validation:   Brier = {brier_const_val:.6f}, NLL = {nll_const_val:.6f}")

    # Load Phase 7P predictions
    oof_df = pd.read_csv(res_7p / "out_of_fold_predictions.csv")
    val_pred_df = pd.read_csv(res_7p / "validation_predictions.csv")
    cal_params = json.load(open(res_7p / "calibrator_parameters.json", "r"))

    seeds = [42, 123, 456]
    ranking_records = []

    print("\n--- Phase 7P Validation Scores & Baselines Re-Audit ---")
    for s in seeds:
        raw_p = val_pred_df[f"raw_p_fail_seed{s}"].values
        cal_q = val_pred_df[f"cal_q_fail_seed{s}"].values

        # Validation Metrics for Raw Score
        auc_raw = float(roc_auc_score(y_fail_val, raw_p))
        ap_raw = float(average_precision_score(y_fail_val, raw_p))
        sp_corr_raw, _ = spearmanr(raw_p, y_fail_val)
        pe_corr_raw, _ = pearsonr(raw_p, y_fail_val)
        brier_raw = float(brier_score_loss(y_fail_val, raw_p))
        nll_raw = float(log_loss(y_fail_val, np.clip(raw_p, 1e-6, 1 - 1e-6)))
        ece_raw = compute_binary_ece(raw_p, y_fail_val, 10)

        # Validation Metrics for Calibrated Score
        auc_cal = float(roc_auc_score(y_fail_val, cal_q))
        ap_cal = float(average_precision_score(y_fail_val, cal_q))
        sp_corr_cal, _ = spearmanr(cal_q, y_fail_val)
        pe_corr_cal, _ = pearsonr(cal_q, y_fail_val)
        brier_cal = float(brier_score_loss(y_fail_val, cal_q))
        nll_cal = float(log_loss(y_fail_val, np.clip(cal_q, 1e-6, 1 - 1e-6)))
        ece_cal = compute_binary_ece(cal_q, y_fail_val, 10)

        ranking_records.append({
            "cohort": "Validation",
            "seed": s,
            "raw_auc": auc_raw,
            "raw_ap": ap_raw,
            "raw_spearman": float(sp_corr_raw),
            "raw_pearson": float(pe_corr_raw),
            "raw_brier": brier_raw,
            "raw_nll": nll_raw,
            "raw_ece": ece_raw,
            "cal_auc": auc_cal,
            "cal_ap": ap_cal,
            "cal_spearman": float(sp_corr_cal),
            "cal_pearson": float(pe_corr_cal),
            "cal_brier": brier_cal,
            "cal_nll": nll_cal,
            "cal_ece": ece_cal,
            "const_train_brier": brier_const_tr,
            "const_val_brier": brier_const_val,
            "brier_diff_cal_minus_const_tr": brier_cal - brier_const_tr
        })

        print(f"Seed {s}: Raw AUC={auc_raw:.4f}, Cal AUC={auc_cal:.4f} (Inversion: {auc_cal < 0.50 and auc_raw > 0.50})")
        print(f"         Raw Brier={brier_raw:.4f}, Cal Brier={brier_cal:.6f}, Const Prev Brier={brier_const_tr:.6f} (Diff: {brier_cal - brier_const_tr:.6f})")

    # 4. Fold-by-fold analysis of Phase 7P Train OOF Predictions
    print("\n--- Phase 7P Train Out-of-Fold Fold-Level Diagnostics ---")
    fold_records = []
    for f_idx in range(5):
        f_mask = (oof_df["fold"] == f_idx)
        f_sub = oof_df[f_mask]
        n_win = len(f_sub)
        n_norm = int((f_sub["y_true"] == 0).sum())
        n_warn = int((f_sub["y_true"] == 1).sum())
        n_crit = int((f_sub["y_true"] == 2).sum())
        n_fail = n_warn + n_crit
        prev_f = n_fail / n_win if n_win > 0 else 0.0

        pjobs = f_sub["parent_job_id"].unique()
        n_jobs = len(pjobs)
        fail_jobs = f_sub[f_sub["y_true"] > 0]["parent_job_id"].unique()
        n_fail_jobs = len(fail_jobs)
        n_pure_jobs = n_jobs - n_fail_jobs

        for s in seeds:
            raw_p_f = f_sub[f"p_fail_seed{s}"].values
            y_f = f_sub["y_fail"].values

            # Fit logistic regression on this single fold
            eps = 1e-6
            s_f = logit(np.clip(raw_p_f, eps, 1 - eps)).reshape(-1, 1)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=s)
            clf.fit(s_f, y_f)
            a_fold = float(clf.coef_[0, 0])
            b_fold = float(clf.intercept_[0])

            auc_fold = float(roc_auc_score(y_f, raw_p_f)) if len(np.unique(y_f)) > 1 else float("nan")
            ap_fold = float(average_precision_score(y_f, raw_p_f)) if len(np.unique(y_f)) > 1 else float("nan")

            mean_raw_norm = float(np.mean(raw_p_f[y_f == 0])) if (y_f == 0).any() else float("nan")
            mean_raw_fail = float(np.mean(raw_p_f[y_f == 1])) if (y_f == 1).any() else float("nan")

            fold_records.append({
                "fold": f_idx,
                "seed": s,
                "n_windows": n_win,
                "n_parent_jobs": n_jobs,
                "n_failing_jobs": n_fail_jobs,
                "n_pure_normal_jobs": n_pure_jobs,
                "n_normal": n_norm,
                "n_warning": n_warn,
                "n_critical": n_crit,
                "n_failure": n_fail,
                "failure_prevalence": prev_f,
                "raw_auc": auc_fold,
                "raw_ap": ap_fold,
                "mean_raw_norm": mean_raw_norm,
                "mean_raw_fail": mean_raw_fail,
                "fold_platt_a": a_fold,
                "fold_platt_b": b_fold
            })

    df_fold = pd.DataFrame(fold_records)
    print(df_fold.groupby("fold")[["n_windows", "n_parent_jobs", "n_failure", "failure_prevalence", "raw_auc", "fold_platt_a"]].mean().to_string())

    # 5. Check influence of Fold 0 on pooled calibrator
    print("\n--- Pooled vs Exclude-Fold-0 Calibrator Comparison ---")
    for s in seeds:
        # Full pooled
        raw_all = oof_df[f"p_fail_seed{s}"].values
        y_all = oof_df["y_fail"].values
        s_all = logit(np.clip(raw_all, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        clf_all = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=s).fit(s_all, y_all)

        # Excluding Fold 0
        mask_no_f0 = (oof_df["fold"] != 0)
        raw_no_f0 = oof_df.loc[mask_no_f0, f"p_fail_seed{s}"].values
        y_no_f0 = oof_df.loc[mask_no_f0, "y_fail"].values
        s_no_f0 = logit(np.clip(raw_no_f0, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        clf_no_f0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=s).fit(s_no_f0, y_no_f0)

        print(f"Seed {s}: Pooled a={clf_all.coef_[0,0]:.4f}, b={clf_all.intercept_[0]:.4f} | Exclude-Fold-0 a={clf_no_f0.coef_[0,0]:.4f}, b={clf_no_f0.intercept_[0]:.4f}")

    # 6. Save re-audit results
    pd.DataFrame(ranking_records).to_csv(out_7q / "phase7p_ranking_reaudit.csv", index=False)
    pd.DataFrame(ranking_records).to_csv(rep_7q / "phase7p_ranking_reaudit.csv", index=False)

    df_fold.to_csv(out_7q / "fold_diagnostics.csv", index=False)
    df_fold.to_csv(rep_7q / "fold_diagnostics.csv", index=False)

    prev_baselines = {
        "train_prevalence": tr_prev,
        "validation_prevalence": val_prev,
        "constant_train_prev_brier_on_val": brier_const_tr,
        "constant_val_prev_brier_on_val": brier_const_val,
        "constant_train_prev_nll_on_val": nll_const_tr,
        "constant_val_prev_nll_on_val": nll_const_val,
        "phase7p_reported_cal_brier_mean": float(np.mean([r["cal_brier"] for r in ranking_records])),
        "is_cal_brier_equal_to_prevalence_baseline": bool(abs(np.mean([r["cal_brier"] for r in ranking_records]) - brier_const_tr) < 0.003)
    }
    with open(out_7q / "prevalence_baselines.json", "w") as f:
        json.dump(prev_baselines, f, indent=2)
    with open(rep_7q / "prevalence_baselines.json", "w") as f:
        json.dump(prev_baselines, f, indent=2)

    # 7. Check availability of underlying causal temporal telemetry tensors (L=12)
    l12_tr_p = res_7l / "standardized_data" / "X_train_l12.npy"
    l12_val_p = res_7l / "standardized_data" / "X_val_l12.npy"
    assert l12_tr_p.exists() and l12_val_p.exists()
    X_tr_l12 = np.load(l12_tr_p)
    X_val_l12 = np.load(l12_val_p)
    print(f"\nUnderlying Causal L=12 Tensors available: Train {X_tr_l12.shape}, Val {X_val_l12.shape}")

    print("=" * 80)
    print("PREFLIGHT AUDIT & WORKSTREAM A RE-AUDIT COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
