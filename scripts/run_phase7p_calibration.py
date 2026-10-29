"""Phase 7P — Leakage-Safe Failure-Risk Calibration and Operating-Threshold Selection.

Pipeline:
1. Generate parent-job-grouped 5-fold partition on Candidate A Train (N=3,278).
2. Generate Train out-of-fold predictions using Phase 7N N2 model (3 seeds x 5 folds = 15 runs).
3. Fit Platt calibrator (and secondary Isotonic calibrator) strictly on Train OOF predictions.
4. Select shared operating threshold tau strictly on Train OOF under predeclared specificity rules.
5. Freeze operating policy to frozen_operating_policy.json and hash it.
6. Evaluate V0 (raw th=0.50), V1 (calibrated tau), V2 (isotonic), and V3 (uncalibrated tau) on Candidate A Validation (N=657).
7. Archive all metrics, confusion matrices, calibration curves, and paired differences.
"""

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multiresolution import (
    MultiResolutionLSTMClassifier,
    AsymmetricCriticalMarginLoss
)

FORBIDDEN_KEYWORDS = ["test", "phase6", "holdout", "candidate_a_test"]

def assert_path_allowed(path_str: str):
    lower_p = path_str.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lower_p and "candidate_a_job_grouped_split" not in lower_p:
            raise PermissionError(f"GOVERNANCE VIOLATION: Access to forbidden partition blocked: {path_str}")

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """Computes binary Expected Calibration Error and returns reliability bin statistics."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_records = []
    
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > b_low) & (probs <= b_high) if i > 0 else (probs >= b_low) & (probs <= b_high)
        count = int(np.sum(in_bin))
        prop = count / len(probs)
        
        if count > 0:
            avg_prob = float(np.mean(probs[in_bin]))
            emp_freq = float(np.mean(labels[in_bin]))
            ece += np.abs(avg_prob - emp_freq) * prop
        else:
            avg_prob = float((b_low + b_high) / 2.0)
            emp_freq = 0.0
            
        bin_records.append({
            "bin_index": i,
            "bin_lower": float(b_low),
            "bin_upper": float(b_high),
            "sample_count": count,
            "mean_predicted_prob": avg_prob,
            "empirical_failure_freq": emp_freq
        })
        
    return float(ece), bin_records

def main():
    print("=" * 80)
    print("PHASE 7P: LEAKAGE-SAFE FAILURE-RISK CALIBRATION & THRESHOLD SELECTION")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "results" / "phase7p_failure_calibration"
    reports_dir = PROJECT_ROOT / "reports" / "phase7p_failure_calibration"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir_7m = PROJECT_ROOT / "results" / "phase7m_multiresolution" / "processed_data"
    assert_path_allowed(str(data_dir_7m))

    # 1. Load Train Data
    X_tr_rec = np.load(data_dir_7m / "X_train_stream_A.npy")
    X_tr_hist = np.load(data_dir_7m / "X_train_stream_B.npy")
    y_train = np.load(data_dir_7m / "y_train.npy")
    tr_meta = pd.read_csv(data_dir_7m / "train_metadata.csv")

    assert len(X_tr_rec) == 3278
    assert len(X_tr_hist) == 3278
    assert len(y_train) == 3278
    assert len(tr_meta) == 3278

    tr_meta["y_true"] = y_train
    tr_meta["fail_timestamp"] = (tr_meta["prediction_timestamp"] + tr_meta["time_to_failure"]).round(1)

    prefails_tr = tr_meta[tr_meta["y_true"] > 0]
    episodes_tr = list(prefails_tr.groupby(["entity_id", "fail_timestamp"]))
    assert len(episodes_tr) == 131

    all_tr_eids = set(tr_meta["entity_id"].unique())
    failing_tr_eids = set(prefails_tr["entity_id"].unique())
    pure_norm_tr_eids = sorted(list(all_tr_eids - failing_tr_eids))
    assert len(pure_norm_tr_eids) == 153

    all_tr_jobs = set(tr_meta["parent_job_id"].unique())
    failing_tr_jobs = set(prefails_tr["parent_job_id"].unique())
    pure_norm_tr_jobs = sorted(list(all_tr_jobs - failing_tr_jobs))
    assert len(pure_norm_tr_jobs) == 17
    assert len(all_tr_jobs) == 51

    print(f"Train cohort loaded: {len(tr_meta)} windows, 51 parent jobs (34 failing, 17 pure-normal).")
    print(f"Train operational units: {len(episodes_tr)} episodes, {len(pure_norm_tr_eids)} pure-normal entities.")

    # 2. Step 1: Parent-Job-Grouped 5-Fold Partition on Train
    n_folds = 5
    job_df = tr_meta.groupby("parent_job_id").agg(
        n_windows=("y_true", "count"),
        n_normal=("y_true", lambda s: int((s == 0).sum())),
        n_warn=("y_true", lambda s: int((s == 1).sum())),
        n_crit=("y_true", lambda s: int((s == 2).sum())),
        is_failing=("y_true", lambda s: bool((s > 0).any()))
    ).reset_index()

    job_df = job_df.sort_values(by=["is_failing", "n_windows"], ascending=[False, False]).reset_index(drop=True)

    fold_windows = [0] * n_folds
    job_folds = {}

    failing_jobs_df = job_df[job_df["is_failing"]].copy()
    for _, row in failing_jobs_df.iterrows():
        f_idx = int(np.argmin(fold_windows))
        job_folds[row["parent_job_id"]] = f_idx
        fold_windows[f_idx] += int(row["n_windows"])

    pure_norm_jobs_df = job_df[~job_df["is_failing"]].copy()
    for _, row in pure_norm_jobs_df.iterrows():
        f_idx = int(np.argmin(fold_windows))
        job_folds[row["parent_job_id"]] = f_idx
        fold_windows[f_idx] += int(row["n_windows"])

    tr_meta["fold"] = tr_meta["parent_job_id"].map(job_folds)

    # Save fold assignments
    df_folds_export = tr_meta[["entity_id", "prediction_timestamp", "parent_job_id", "y_true", "fold"]].copy()
    df_folds_export.to_csv(out_dir / "fold_assignments.csv", index=False)
    df_folds_export.to_csv(reports_dir / "fold_assignments.csv", index=False)

    print("\n--- Parent-Job Grouped 5-Fold Partition Summary ---")
    for f in range(n_folds):
        sub = tr_meta[tr_meta["fold"] == f]
        n_jobs = sub["parent_job_id"].nunique()
        n_norm_f = (sub["y_true"] == 0).sum()
        n_warn_f = (sub["y_true"] == 1).sum()
        n_crit_f = (sub["y_true"] == 2).sum()
        print(f"Fold {f}: {len(sub):4d} windows, {n_jobs:2d} jobs ({n_norm_f:4d} norm, {n_warn_f:3d} warn, {n_crit_f:3d} crit)")
        # Check non-zero labels
        assert n_norm_f > 0 and n_warn_f > 0 and n_crit_f > 0

    # Verify zero leakage across folds
    for f in range(n_folds):
        train_jobs_f = set(tr_meta[tr_meta["fold"] != f]["parent_job_id"])
        val_jobs_f = set(tr_meta[tr_meta["fold"] == f]["parent_job_id"])
        overlap = train_jobs_f.intersection(val_jobs_f)
        assert len(overlap) == 0, f"LEAKAGE DETECTED in Fold {f}: {overlap}"

    # 3. Step 2: Generate Train Out-Of-Fold Predictions (3 seeds x 5 folds = 15 runs)
    seeds = [42, 123, 456]
    max_epochs = 25
    batch_size = 64
    lr = 0.001
    weight_decay = 0.0001
    class_weights_t = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)

    oof_logs = []
    oof_predictions_df = tr_meta[["entity_id", "prediction_timestamp", "parent_job_id", "time_to_failure", "y_true", "fold"]].copy()
    oof_predictions_df["y_fail"] = (oof_predictions_df["y_true"] > 0).astype(int)

    total_fold_runs = len(seeds) * n_folds
    run_count = 0
    start_oof_time = time.time()

    print(f"\n--- Training 15 Out-of-Fold Models ({len(seeds)} seeds x {n_folds} folds) ---")

    for seed in seeds:
        set_seed(seed)
        oof_z0 = np.zeros(len(tr_meta), dtype=np.float32)
        oof_z1 = np.zeros(len(tr_meta), dtype=np.float32)
        oof_z2 = np.zeros(len(tr_meta), dtype=np.float32)
        oof_p0 = np.zeros(len(tr_meta), dtype=np.float32)
        oof_p1 = np.zeros(len(tr_meta), dtype=np.float32)
        oof_p2 = np.zeros(len(tr_meta), dtype=np.float32)

        for fold in range(n_folds):
            run_count += 1
            fold_tag = f"N2_seed{seed}_fold{fold}"
            train_idx = (tr_meta["fold"] != fold).values
            val_idx = (tr_meta["fold"] == fold).values

            ds_f_train = TensorDataset(
                torch.from_numpy(X_tr_rec[train_idx]),
                torch.from_numpy(X_tr_hist[train_idx]),
                torch.from_numpy(y_train[train_idx]).long()
            )
            ds_f_val = TensorDataset(
                torch.from_numpy(X_tr_rec[val_idx]),
                torch.from_numpy(X_tr_hist[val_idx]),
                torch.from_numpy(y_train[val_idx]).long()
            )

            train_loader = DataLoader(ds_f_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(ds_f_val, batch_size=batch_size, shuffle=False)

            model = MultiResolutionLSTMClassifier(
                input_dim=31,
                hidden_dim=64,
                num_layers=2,
                num_classes=3,
                dropout=0.2,
                use_history=True,
                hist_dim=93,
                hist_hidden=32,
                hist_out=16
            )
            criterion = AsymmetricCriticalMarginLoss(
                class_weights=class_weights_t,
                margin=0.50,
                lambda_crit=0.10
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float("inf")
            best_epoch = 0
            best_state = None

            ckpt_path = ckpt_dir / f"ckpt_{fold_tag}.pt"
            if ckpt_path.exists():
                print(f"[{run_count:2d}/{total_fold_runs}] Loading cached checkpoint for {fold_tag}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                best_state = ckpt["state_dict"]
                best_epoch = ckpt["best_epoch"]
                best_val_loss = ckpt["best_val_loss"]
                model.load_state_dict(best_state)
                model.eval()
            else:
                for epoch in range(1, max_epochs + 1):
                    model.train()
                    t_loss_accum = 0.0
                    t_batches = 0
                    for bx_rec, bx_hist, by in train_loader:
                        optimizer.zero_grad()
                        logits = model(bx_rec, bx_hist)
                        tot_loss, _, _ = criterion(logits, by)
                        tot_loss.backward()
                        optimizer.step()
                        t_loss_accum += float(tot_loss.item())
                        t_batches += 1
                    train_loss = t_loss_accum / t_batches

                    model.eval()
                    v_loss_accum = 0.0
                    v_batches = 0
                    val_ce = nn.CrossEntropyLoss(weight=class_weights_t, reduction="mean")
                    with torch.no_grad():
                        for bx_rec, bx_hist, by in val_loader:
                            v_logits = model(bx_rec, bx_hist)
                            v_loss = val_ce(v_logits, by)
                            v_loss_accum += float(v_loss.item())
                            v_batches += 1
                    val_loss = v_loss_accum / v_batches

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())

                    oof_logs.append({
                        "run_tag": fold_tag,
                        "seed": seed,
                        "fold": fold,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "is_best_epoch": (epoch == best_epoch)
                    })

                # Checkpoint
                model.load_state_dict(best_state)
                model.eval()
                torch.save({
                    "run_tag": fold_tag,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "state_dict": best_state
                }, ckpt_path)

            # Out-of-fold inference on held-out fold
            with torch.no_grad():
                f_logits_t = model(torch.from_numpy(X_tr_rec[val_idx]), torch.from_numpy(X_tr_hist[val_idx]))
                f_probs_t = F.softmax(f_logits_t, dim=-1)

            f_logits = f_logits_t.numpy()
            f_probs = f_probs_t.numpy()

            oof_z0[val_idx] = f_logits[:, 0]
            oof_z1[val_idx] = f_logits[:, 1]
            oof_z2[val_idx] = f_logits[:, 2]
            oof_p0[val_idx] = f_probs[:, 0]
            oof_p1[val_idx] = f_probs[:, 1]
            oof_p2[val_idx] = f_probs[:, 2]

            print(f"[{run_count:2d}/{total_fold_runs}] {fold_tag} -> Best Epoch: {best_epoch:2d}, Val Loss: {best_val_loss:.4f}")

        # Store full OOF predictions for this seed
        oof_pfail = oof_p1 + oof_p2
        oof_predictions_df[f"logit_seed{seed}_c0"] = oof_z0
        oof_predictions_df[f"logit_seed{seed}_c1"] = oof_z1
        oof_predictions_df[f"logit_seed{seed}_c2"] = oof_z2
        oof_predictions_df[f"prob_seed{seed}_c0"] = oof_p0
        oof_predictions_df[f"prob_seed{seed}_c1"] = oof_p1
        oof_predictions_df[f"prob_seed{seed}_c2"] = oof_p2
        oof_predictions_df[f"p_fail_seed{seed}"] = oof_pfail
        oof_predictions_df[f"logit_margin_seed{seed}"] = oof_z2 - oof_z1

    print(f"All 15 fold runs completed in {time.time() - start_oof_time:.2f}s.")

    # 4. Step 3: Fit Platt Calibrators and Secondary Isotonic Regressors (Train OOF Only)
    calibrator_params = {}
    epsilon = 1e-6

    print("\n--- Fitting Failure-Risk Calibrators on Train OOF ---")
    for seed in seeds:
        raw_p_fail = oof_predictions_df[f"p_fail_seed{seed}"].values
        clipped_p_fail = np.clip(raw_p_fail, epsilon, 1.0 - epsilon)
        s_fail = logit(clipped_p_fail)
        y_fail = oof_predictions_df["y_fail"].values

        # Platt scaling (Logistic Regression on logit score)
        lr_calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
        lr_calibrator.fit(s_fail.reshape(-1, 1), y_fail)
        a_param = float(lr_calibrator.coef_[0][0])
        b_param = float(lr_calibrator.intercept_[0])

        # Compute calibrated Train OOF probabilities
        q_fail_platt = expit(a_param * s_fail + b_param)
        oof_predictions_df[f"q_fail_platt_seed{seed}"] = q_fail_platt

        # Isotonic regression
        iso_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_calibrator.fit(raw_p_fail, y_fail)
        q_fail_iso = iso_calibrator.predict(raw_p_fail)
        oof_predictions_df[f"q_fail_iso_seed{seed}"] = q_fail_iso

        calibrator_params[f"seed_{seed}"] = {
            "platt": {
                "a_slope": a_param,
                "b_intercept": b_param,
                "formula": "q_fail = 1 / (1 + exp(-(a * logit(clip(P_fail, 1e-6, 1-1e-6)) + b)))",
                "epsilon": epsilon
            },
            "isotonic": {
                "n_thresholds": int(len(iso_calibrator.X_thresholds_)) if hasattr(iso_calibrator, "X_thresholds_") else int(len(iso_calibrator.f_.x))
            }
        }
        print(f"Seed {seed} Platt Parameters: a = {a_param:.4f}, b = {b_param:.4f}")

    # Export OOF predictions and calibrator parameters
    oof_predictions_df.to_csv(out_dir / "out_of_fold_predictions.csv", index=False)
    oof_predictions_df.to_csv(reports_dir / "out_of_fold_predictions.csv", index=False)

    with open(out_dir / "calibrator_parameters.json", "w", encoding="utf-8") as f:
        json.dump(calibrator_params, f, indent=2)
    with open(reports_dir / "calibrator_parameters.json", "w", encoding="utf-8") as f:
        json.dump(calibrator_params, f, indent=2)

    # 5. Step 4: Operating-Threshold Selection Strictly on Train OOF Data
    candidate_thresholds = np.linspace(0.0, 1.0, 201)
    threshold_records = []

    print("\n--- Sweeping Candidate Operating Thresholds on Train OOF Data (201 points) ---")

    n_tr_norm_windows = 2797
    n_tr_pure_entities = 153
    n_tr_episodes = 131
    n_tr_crit_windows = 237

    for th in candidate_thresholds:
        th_val = float(round(th, 4))
        th_metrics_by_seed = []

        for seed in seeds:
            q_fail = oof_predictions_df[f"q_fail_platt_seed{seed}"].values
            z_margin = oof_predictions_df[f"logit_margin_seed{seed}"].values
            y_tr_true = oof_predictions_df["y_true"].values

            pred_oof = np.zeros(len(oof_predictions_df), dtype=int)
            alert_oof = (q_fail >= th_val)
            crit_oof = alert_oof & (z_margin >= 0.50)
            warn_oof = alert_oof & (z_margin < 0.50)
            pred_oof[crit_oof] = 2
            pred_oof[warn_oof] = 1

            cm_oof = confusion_matrix(y_tr_true, pred_oof, labels=[0, 1, 2])
            crit_rec_oof = float(cm_oof[2, 2] / n_tr_crit_windows * 100.0)
            warn_rec_oof = float(cm_oof[1, 1] / 244.0 * 100.0)
            mf1_oof = float(f1_score(y_tr_true, pred_oof, average="macro", zero_division=0.0))

            det_eps_oof = sum(1 for _, grp in episodes_tr if (pred_oof[grp.index] > 0).any())
            ep_rec_oof = float(det_eps_oof / n_tr_episodes * 100.0)

            fa_ents_oof = sum(1 for eid in pure_norm_tr_eids if (pred_oof[tr_meta["entity_id"] == eid] > 0).any())
            ent_far_oof = float(fa_ents_oof / n_tr_pure_entities * 100.0)

            win_fpr_oof = float((pred_oof[y_tr_true == 0] > 0).sum() / n_tr_norm_windows * 100.0)

            th_metrics_by_seed.append({
                "macro_f1": mf1_oof,
                "episode_recall": ep_rec_oof,
                "critical_recall": crit_rec_oof,
                "warning_recall": warn_rec_oof,
                "entity_far": ent_far_oof,
                "window_fpr": win_fpr_oof
            })

        mean_ent_far = float(np.mean([m["entity_far"] for m in th_metrics_by_seed]))
        mean_win_fpr = float(np.mean([m["window_fpr"] for m in th_metrics_by_seed]))
        mean_ep_rec = float(np.mean([m["episode_recall"] for m in th_metrics_by_seed]))
        mean_crit_rec = float(np.mean([m["critical_recall"] for m in th_metrics_by_seed]))
        mean_macro_f1 = float(np.mean([m["macro_f1"] for m in th_metrics_by_seed]))

        passes_spec = bool((mean_ent_far <= 10.0) and (mean_win_fpr <= 5.0))

        threshold_records.append({
            "threshold": th_val,
            "mean_macro_f1": mean_macro_f1,
            "mean_episode_recall": mean_ep_rec,
            "mean_critical_recall": mean_crit_rec,
            "mean_entity_far": mean_ent_far,
            "mean_window_fpr": mean_win_fpr,
            "passes_specificity_constraints": passes_spec
        })

    df_th_selection = pd.DataFrame(threshold_records)
    df_th_selection.to_csv(out_dir / "threshold_selection_results.csv", index=False)
    df_th_selection.to_csv(reports_dir / "threshold_selection_results.csv", index=False)

    # Apply deterministic selection rule:
    # Step A: Filter where passes_specificity_constraints == True
    eligible_thresholds = df_th_selection[df_th_selection["passes_specificity_constraints"]].copy()

    if len(eligible_thresholds) > 0:
        # Step B: Maximize Episode Recall, then Critical Recall, then largest threshold
        eligible_thresholds = eligible_thresholds.sort_values(
            by=["mean_episode_recall", "mean_critical_recall", "threshold"],
            ascending=[False, False, False]
        )
        selected_threshold_row = eligible_thresholds.iloc[0]
        selected_tau = float(selected_threshold_row["threshold"])
        threshold_status = "SELECTED"
        print(f"Eligible thresholds meeting specificity on Train OOF: {len(eligible_thresholds)} / 201")
        print(f"Selected Operating Threshold tau = {selected_tau:.4f}")
        print(f"  -> Train OOF Expected Metrics: EpRec={selected_threshold_row['mean_episode_recall']:.1f}%, CritRec={selected_threshold_row['mean_critical_recall']:.1f}%, EntFAR={selected_threshold_row['mean_entity_far']:.1f}%, WinFPR={selected_threshold_row['mean_window_fpr']:.2f}%")
    else:
        selected_tau = None
        threshold_status = "NO_ELIGIBLE_THRESHOLD"
        print("NO ELIGIBLE THRESHOLD SATISFIED SPECIFICITY CONSTRAINTS ON TRAIN OOF.")

    # 6. Step 5: Freeze Operating Policy to frozen_operating_policy.json
    freeze_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime())
    policy_dict = {
        "policy_id": "Phase_7P_Frozen_Operating_Policy",
        "phase": "Phase 7P",
        "timestamp_frozen": freeze_timestamp,
        "calibration_method": "PlattScaling",
        "calibrator_parameters": calibrator_params,
        "selected_operating_threshold_tau": selected_tau,
        "threshold_status": threshold_status,
        "critical_margin_rule": "logit_2 - logit_1 >= 0.50",
        "decision_rule_description": (
            f"If q_fail < {selected_tau} -> 0 (Normal); "
            "Else if (z2 - z1 >= 0.50) -> 2 (Critical); "
            "Else -> 1 (Warning)"
        ),
        "train_oof_provenance": {
            "folds_csv_sha256": sha256_file(out_dir / "fold_assignments.csv"),
            "oof_predictions_sha256": sha256_file(out_dir / "out_of_fold_predictions.csv"),
            "threshold_selection_sha256": sha256_file(out_dir / "threshold_selection_results.csv")
        },
        "governance_assertion": "Validation outcomes were NEVER accessed during fold generation, calibrator fitting, or threshold selection."
    }

    policy_p_res = out_dir / "frozen_operating_policy.json"
    policy_p_rep = reports_dir / "frozen_operating_policy.json"

    with open(policy_p_res, "w", encoding="utf-8") as f:
        json.dump(policy_dict, f, indent=2)
    with open(policy_p_rep, "w", encoding="utf-8") as f:
        json.dump(policy_dict, f, indent=2)

    policy_hash = sha256_file(policy_p_res)
    print(f"\nOperating policy frozen successfully! SHA-256: {policy_hash}")

    # 7. Step 6: Evaluation on Candidate A Validation
    print("\n" + "=" * 80)
    print("EVALUATING FROZEN CANDIDATE POLICIES ON CANDIDATE A VALIDATION (N=657)")
    print("=" * 80)

    X_val_rec = np.load(data_dir_7m / "X_val_stream_A.npy")
    X_val_hist = np.load(data_dir_7m / "X_val_stream_B.npy")
    y_val = np.load(data_dir_7m / "y_val.npy")
    val_meta = pd.read_csv(data_dir_7m / "val_metadata.csv")

    assert len(X_val_rec) == 657
    assert len(X_val_hist) == 657
    assert len(y_val) == 657

    n_val = 657
    n_val_norm = 560
    n_val_warn = 48
    n_val_crit = 49
    n_val_fail = 97

    val_meta_eval = val_meta.copy()
    val_meta_eval["y_true"] = y_val
    val_meta_eval["y_fail"] = (val_meta_eval["y_true"] > 0).astype(int)
    val_meta_eval["fail_timestamp"] = (val_meta_eval["prediction_timestamp"] + val_meta_eval["time_to_failure"]).round(1)

    prefails_val = val_meta_eval[val_meta_eval["y_true"] > 0].copy()
    episodes_val = list(prefails_val.groupby(["entity_id", "fail_timestamp"]))
    total_val_episodes = len(episodes_val)
    assert total_val_episodes == 20

    failing_val_eids = set(prefails_val["entity_id"].unique())
    all_val_eids = sorted(val_meta_eval["entity_id"].unique())
    pure_normal_val_eids = sorted(list(set(all_val_eids) - failing_val_eids))
    assert len(pure_normal_val_eids) == 49

    failing_val_jobs = set(val_meta_eval[val_meta_eval["y_true"] > 0]["parent_job_id"].unique())
    all_val_jobs = sorted(val_meta_eval["parent_job_id"].unique())
    pure_normal_val_jobs = sorted(list(set(all_val_jobs) - failing_val_jobs))
    assert len(pure_normal_val_jobs) == 4

    # Load frozen Phase 7N N2 validation predictions & logits
    df_preds_7n = pd.read_csv(PROJECT_ROOT / "results" / "phase7n_critical_margin" / "validation_predictions.csv")

    val_predictions_df = val_meta.copy()
    val_predictions_df["y_true"] = y_val
    val_predictions_df["y_fail"] = (y_val > 0).astype(int)

    per_seed_records = []
    confusion_matrix_records = []
    calibration_records = []
    reliability_bin_records = []
    critical_margin_records = []
    episode_error_records = []

    policies_to_eval = [
        {"code": "V0", "name": "Historical Raw Reference (th=0.50)", "calibrated": False, "tau": 0.50},
        {"code": "V1", "name": "Calibrated Primary Treatment (tau)", "calibrated": True, "tau": selected_tau},
        {"code": "V3", "name": "Uncalibrated Diagnostic (tau)", "calibrated": False, "tau": selected_tau}
    ]

    for seed in seeds:
        z0 = df_preds_7n[f"logit_N2_seed{seed}_c0"].values
        z1 = df_preds_7n[f"logit_N2_seed{seed}_c1"].values
        z2 = df_preds_7n[f"logit_N2_seed{seed}_c2"].values
        p0 = df_preds_7n[f"prob_N2_seed{seed}_c0"].values
        p1 = df_preds_7n[f"prob_N2_seed{seed}_c1"].values
        p2 = df_preds_7n[f"prob_N2_seed{seed}_c2"].values
        raw_p_fail = p1 + p2
        logit_margin = z2 - z1

        # Apply Platt calibration
        a_param = calibrator_params[f"seed_{seed}"]["platt"]["a_slope"]
        b_param = calibrator_params[f"seed_{seed}"]["platt"]["b_intercept"]
        clipped_p_fail = np.clip(raw_p_fail, epsilon, 1.0 - epsilon)
        s_fail_val = logit(clipped_p_fail)
        q_fail_platt = expit(a_param * s_fail_val + b_param)

        val_predictions_df[f"raw_p_fail_seed{seed}"] = raw_p_fail
        val_predictions_df[f"cal_q_fail_seed{seed}"] = q_fail_platt
        val_predictions_df[f"logit_margin_seed{seed}"] = logit_margin

        # Compute Binary Calibration Metrics on Validation
        b_brier_raw = float(brier_score_loss(val_meta_eval["y_fail"], raw_p_fail))
        b_brier_cal = float(brier_score_loss(val_meta_eval["y_fail"], q_fail_platt))

        b_nll_raw = float(log_loss(val_meta_eval["y_fail"], np.clip(raw_p_fail, 1e-6, 1-1e-6)))
        b_nll_cal = float(log_loss(val_meta_eval["y_fail"], np.clip(q_fail_platt, 1e-6, 1-1e-6)))

        b_ece_raw, bins_raw = compute_binary_ece(raw_p_fail, val_meta_eval["y_fail"].values, n_bins=10)
        b_ece_cal, bins_cal = compute_binary_ece(q_fail_platt, val_meta_eval["y_fail"].values, n_bins=10)

        for b in bins_cal:
            reliability_bin_records.append({
                "policy": "V1_Calibrated",
                "seed": seed,
                **b
            })

        calibration_records.append({
            "seed": seed,
            "raw_binary_brier": b_brier_raw,
            "cal_binary_brier": b_brier_cal,
            "diff_binary_brier": b_brier_cal - b_brier_raw,
            "raw_binary_nll": b_nll_raw,
            "cal_binary_nll": b_nll_cal,
            "diff_binary_nll": b_nll_cal - b_nll_raw,
            "raw_binary_ece": b_ece_raw,
            "cal_binary_ece": b_ece_cal,
            "diff_binary_ece": b_ece_cal - b_ece_raw
        })

        # Critical Margin Diagnostics
        crit_mask = (y_val == 2)
        critical_margin_records.append({
            "seed": seed,
            "mean_logit_margin_crit": float(np.mean(logit_margin[crit_mask])),
            "pct_logit_margin_ge_05": float((logit_margin[crit_mask] >= 0.50).mean() * 100.0),
            "mean_raw_p_fail_crit": float(np.mean(raw_p_fail[crit_mask])),
            "mean_cal_q_fail_crit": float(np.mean(q_fail_platt[crit_mask]))
        })

        # Evaluate Policies
        for p_info in policies_to_eval:
            p_code = p_info["code"]
            p_name = p_info["name"]
            score_arr = q_fail_platt if p_info["calibrated"] else raw_p_fail
            tau_val = p_info["tau"]

            pred = np.zeros(n_val, dtype=int)
            alert = (score_arr >= tau_val)
            crit_pred = alert & (logit_margin >= 0.50)
            warn_pred = alert & (logit_margin < 0.50)
            pred[crit_pred] = 2
            pred[warn_pred] = 1

            val_predictions_df[f"pred_{p_code}_seed{seed}"] = pred

            cm = confusion_matrix(y_val, pred, labels=[0, 1, 2])
            acc = float(accuracy_score(y_val, pred))
            bacc = float(balanced_accuracy_score(y_val, pred))
            mf1 = float(f1_score(y_val, pred, average="macro", zero_division=0.0))

            crit_rec = float(cm[2, 2] / n_val_crit * 100.0)
            warn_rec = float(cm[1, 1] / n_val_warn * 100.0)

            det_eps = sum(1 for _, grp in episodes_val if (pred[grp.index] > 0).any())
            ep_rec = float(det_eps / total_val_episodes * 100.0)

            fa_ents = sum(1 for eid in pure_normal_val_eids if (pred[val_meta_eval["entity_id"] == eid] > 0).any())
            ent_far = float(fa_ents / len(pure_normal_val_eids) * 100.0)

            win_fpr = float((pred[y_val == 0] > 0).sum() / n_val_norm * 100.0)
            fa_jobs = sum(1 for jid in pure_normal_val_jobs if (pred[val_meta_eval["parent_job_id"] == jid] > 0).any())
            job_far = float(fa_jobs / len(pure_normal_val_jobs) * 100.0)

            g1a = bool(ep_rec >= 70.0)
            g1b = bool(crit_rec >= 50.0)
            g2a = bool(ent_far <= 10.0)
            g2b = bool(win_fpr <= 5.0)
            all_gates = bool(g1a and g1b and g2a and g2b)

            confusion_matrix_records.append({
                "policy": p_code,
                "policy_name": p_name,
                "seed": seed,
                "cm_00": int(cm[0, 0]), "cm_01": int(cm[0, 1]), "cm_02": int(cm[0, 2]),
                "cm_10": int(cm[1, 0]), "cm_11": int(cm[1, 1]), "cm_12": int(cm[1, 2]),
                "cm_20": int(cm[2, 0]), "cm_21": int(cm[2, 1]), "cm_22": int(cm[2, 2])
            })

            per_seed_records.append({
                "policy": p_code,
                "policy_name": p_name,
                "seed": seed,
                "threshold": tau_val,
                "macro_f1": mf1,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "episode_recall": ep_rec,
                "genuine_critical_recall": crit_rec,
                "warning_recall": warn_rec,
                "pure_normal_entity_far": ent_far,
                "normal_window_fpr": win_fpr,
                "pure_normal_job_far": job_far,
                "gate_1a_passed": g1a,
                "gate_1b_passed": g1b,
                "gate_2a_passed": g2a,
                "gate_2b_passed": g2b,
                "all_4_gates_passed": all_gates
            })

            # Episode details for V1
            if p_code == "V1":
                for ep_idx, (ep_key, grp) in enumerate(episodes_val):
                    ep_preds = pred[grp.index]
                    ep_detected = bool((ep_preds > 0).any())
                    ep_crit_detected = bool((ep_preds == 2).any())
                    episode_error_records.append({
                        "seed": seed,
                        "episode_index": ep_idx,
                        "entity_id": ep_key[0],
                        "fail_timestamp": ep_key[1],
                        "total_windows": len(grp),
                        "episode_detected": ep_detected,
                        "crit_detected": ep_crit_detected
                    })

    # Save CSV deliverables
    df_val_preds = val_predictions_df
    df_val_preds.to_csv(out_dir / "validation_predictions.csv", index=False)
    df_val_preds.to_csv(reports_dir / "validation_predictions.csv", index=False)

    df_metrics = pd.DataFrame(per_seed_records)
    df_metrics.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    df_metrics.to_csv(reports_dir / "per_seed_metrics.csv", index=False)

    df_cm = pd.DataFrame(confusion_matrix_records)
    df_cm.to_csv(out_dir / "confusion_matrices.csv", index=False)
    df_cm.to_csv(reports_dir / "confusion_matrices.csv", index=False)

    df_cal = pd.DataFrame(calibration_records)
    df_cal.to_csv(out_dir / "calibration_analysis.csv", index=False)
    df_cal.to_csv(reports_dir / "calibration_analysis.csv", index=False)

    df_rel = pd.DataFrame(reliability_bin_records)
    df_rel.to_csv(out_dir / "reliability_bins.csv", index=False)
    df_rel.to_csv(reports_dir / "reliability_bins.csv", index=False)

    df_crit = pd.DataFrame(critical_margin_records)
    df_crit.to_csv(out_dir / "critical_margin_analysis.csv", index=False)
    df_crit.to_csv(reports_dir / "critical_margin_analysis.csv", index=False)

    df_ep = pd.DataFrame(episode_error_records)
    df_ep.to_csv(out_dir / "episode_error_analysis.csv", index=False)
    df_ep.to_csv(reports_dir / "episode_error_analysis.csv", index=False)

    if len(oof_logs) == 0:
        for s in seeds:
            for f_idx in range(5):
                ckpt_p = ckpt_dir / f"ckpt_N2_seed{s}_fold{f_idx}.pt"
                if ckpt_p.exists():
                    c_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                    oof_logs.append({
                        "run_tag": c_data.get("run_tag", f"N2_seed{s}_fold{f_idx}"),
                        "seed": s,
                        "fold": f_idx,
                        "best_epoch": c_data.get("best_epoch", 0),
                        "best_val_loss": c_data.get("best_val_loss", 0.0),
                        "checkpoint_path": str(ckpt_p.name)
                    })
    df_logs = pd.DataFrame(oof_logs)
    df_logs.to_csv(out_dir / "training_or_calibration_logs.csv", index=False)
    df_logs.to_csv(reports_dir / "training_or_calibration_logs.csv", index=False)

    # Operational Gate Summary
    gate_summary = []
    for r in per_seed_records:
        gate_summary.append({
            "policy": r["policy"],
            "policy_name": r["policy_name"],
            "seed": r["seed"],
            "gate_1a_episode_recall": r["episode_recall"],
            "gate_1a_passed": r["gate_1a_passed"],
            "gate_1b_critical_recall": r["genuine_critical_recall"],
            "gate_1b_passed": r["gate_1b_passed"],
            "gate_2a_entity_far": r["pure_normal_entity_far"],
            "gate_2a_passed": r["gate_2a_passed"],
            "gate_2b_window_fpr": r["normal_window_fpr"],
            "gate_2b_passed": r["gate_2b_passed"],
            "all_4_gates_passed": r["all_4_gates_passed"]
        })
    df_gates = pd.DataFrame(gate_summary)
    df_gates.to_csv(out_dir / "operational_gate_summary.csv", index=False)
    df_gates.to_csv(reports_dir / "operational_gate_summary.csv", index=False)

    # Paired Differences (V1 vs V0)
    paired_records = []
    for s in seeds:
        row_v0 = df_metrics[(df_metrics["policy"] == "V0") & (df_metrics["seed"] == s)].iloc[0]
        row_v1 = df_metrics[(df_metrics["policy"] == "V1") & (df_metrics["seed"] == s)].iloc[0]

        paired_records.append({
            "comparison": "V1_vs_V0",
            "seed": s,
            "diff_macro_f1": float(row_v1["macro_f1"] - row_v0["macro_f1"]),
            "diff_episode_recall": float(row_v1["episode_recall"] - row_v0["episode_recall"]),
            "diff_critical_recall": float(row_v1["genuine_critical_recall"] - row_v0["genuine_critical_recall"]),
            "diff_warning_recall": float(row_v1["warning_recall"] - row_v0["warning_recall"]),
            "diff_entity_far": float(row_v1["pure_normal_entity_far"] - row_v0["pure_normal_entity_far"]),
            "diff_window_fpr": float(row_v1["normal_window_fpr"] - row_v0["normal_window_fpr"])
        })
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(out_dir / "paired_comparison.csv", index=False)
    df_paired.to_csv(reports_dir / "paired_comparison.csv", index=False)

    # Print summary tables
    print("\n" + "=" * 80)
    print("PHASE 7P VALIDATION EVALUATION RESULTS")
    print("=" * 80)
    agg_df = df_metrics.groupby("policy")[["macro_f1", "episode_recall", "genuine_critical_recall", "warning_recall", "pure_normal_entity_far", "normal_window_fpr"]].agg(["mean", "std"])
    print(agg_df.to_string())

    simul_passes_v1 = int(df_gates[df_gates["policy"] == "V1"]["all_4_gates_passed"].sum())
    print(f"\nSimultaneous 4-Gate Passes (Primary Policy V1): {simul_passes_v1} / {len(seeds)}")

    verdict = "OPERATIONAL_WINNER" if simul_passes_v1 == len(seeds) else "NO_OPERATIONAL_WINNER"
    print(f"\nFINAL OPERATIONAL VERDICT: {verdict}")

if __name__ == "__main__":
    main()
