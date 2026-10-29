"""Phase 7N — Critical-Class Separation and Asymmetric Loss Remediation.

Evaluates:
- N0: Baseline Multi-Resolution LSTM [1, 2, 2]
- N1: Critical-Weight Sensitivity [1, 2, 4]
- N2: Asymmetric Critical Margin Loss [1, 2, 2] + lambda_crit * softplus(0.5 - (z2 - z1)) on true Critical
- N3: Multiclass Focal Loss [1, 2, 2], gamma=2.0

Across seeds 42, 123, 456 on Candidate A Development Data.
STRICTLY Candidate A Train (3,278) and Validation (657) only.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multiresolution import (
    MultiResolutionLSTMClassifier,
    AsymmetricCriticalMarginLoss,
    MulticlassFocalLoss
)

# -----------------------------------------------------------------------------
# Data Governance & Security Allowlist
# -----------------------------------------------------------------------------
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

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error for multi-class classification."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def main():
    print("=" * 80)
    print("PHASE 7N: CRITICAL-CLASS SEPARATION & ASYMMETRIC LOSS REMEDIATION")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "results" / "phase7n_critical_margin"
    reports_dir = PROJECT_ROOT / "reports" / "phase7n_critical_margin"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir_7m = PROJECT_ROOT / "results" / "phase7m_multiresolution" / "processed_data"
    assert_path_allowed(str(data_dir_7m))

    # 1. Load Data
    X_tr_rec = np.load(data_dir_7m / "X_train_stream_A.npy")
    X_val_rec = np.load(data_dir_7m / "X_val_stream_A.npy")
    X_tr_hist = np.load(data_dir_7m / "X_train_stream_B.npy")
    X_val_hist = np.load(data_dir_7m / "X_val_stream_B.npy")
    y_train = np.load(data_dir_7m / "y_train.npy")
    y_val = np.load(data_dir_7m / "y_val.npy")
    val_meta = pd.read_csv(data_dir_7m / "val_metadata.csv")
    tr_meta = pd.read_csv(data_dir_7m / "train_metadata.csv")

    assert len(X_tr_rec) == 3278
    assert len(X_val_rec) == 657
    assert len(y_train) == 3278
    assert len(y_val) == 657

    n_val = 657
    n_norm = 560
    n_warn = 48
    n_crit = 49
    n_fail = 97

    val_meta_eval = val_meta.copy()
    val_meta_eval["y_true"] = y_val
    val_meta_eval["fail_timestamp"] = (val_meta_eval["prediction_timestamp"] + val_meta_eval["time_to_failure"]).round(1)

    prefails = val_meta_eval[val_meta_eval["y_true"] > 0].copy()
    episodes = list(prefails.groupby(["entity_id", "fail_timestamp"]))
    total_episodes = len(episodes)
    assert total_episodes == 20

    failing_eids = set(prefails["entity_id"].unique())
    all_eids = sorted(val_meta_eval["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    assert len(pure_normal_eids) == 49

    failing_jobs = set(val_meta_eval[val_meta_eval["y_true"] > 0]["parent_job_id"].unique())
    all_jobs = sorted(val_meta_eval["parent_job_id"].unique())
    pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
    assert len(pure_normal_jobs) == 4

    print(f"Validation cohorts verified: {n_val} windows, {total_episodes} episodes, {len(pure_normal_eids)} pure-normal entities.")

    # 2. Experimental Conditions Setup
    conditions = [
        {
            "cond_id": "N0",
            "name": "Reproduced Phase 7M Baseline",
            "loss_type": "ce",
            "weights": [1.0, 2.0, 2.0],
            "margin": 0.0,
            "lambda_crit": 0.0,
            "gamma": 0.0
        },
        {
            "cond_id": "N1",
            "name": "Critical-Weight Sensitivity",
            "loss_type": "ce",
            "weights": [1.0, 2.0, 4.0],
            "margin": 0.0,
            "lambda_crit": 0.0,
            "gamma": 0.0
        },
        {
            "cond_id": "N2",
            "name": "Asymmetric Critical Margin Loss",
            "loss_type": "margin",
            "weights": [1.0, 2.0, 2.0],
            "margin": 0.5,
            "lambda_crit": 0.10,
            "gamma": 0.0
        },
        {
            "cond_id": "N3",
            "name": "Multiclass Focal Loss",
            "loss_type": "focal",
            "weights": [1.0, 2.0, 2.0],
            "margin": 0.0,
            "lambda_crit": 0.0,
            "gamma": 2.0
        }
    ]

    seeds = [42, 123, 456]
    max_epochs = 25
    batch_size = 64
    lr = 0.001
    weight_decay = 0.0001
    thresholds = np.linspace(0.0, 1.0, 201)

    training_logs = []
    val_predictions = val_meta.copy()
    per_seed_metrics = []
    confusion_matrices_records = []
    critical_margin_records = []
    episode_error_records = []
    calibration_records = []
    threshold_sweep_records = []

    ds_train = TensorDataset(
        torch.from_numpy(X_tr_rec),
        torch.from_numpy(X_tr_hist),
        torch.from_numpy(y_train).long()
    )
    ds_val = TensorDataset(
        torch.from_numpy(X_val_rec),
        torch.from_numpy(X_val_hist),
        torch.from_numpy(y_val).long()
    )
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    run_idx = 0
    total_runs = len(conditions) * len(seeds)
    start_total_t = time.time()

    for cond in conditions:
        cid = cond["cond_id"]
        cname = cond["name"]
        ltype = cond["loss_type"]
        cweights = cond["weights"]
        weights_t = torch.tensor(cweights, dtype=torch.float32)

        for seed in seeds:
            run_idx += 1
            run_tag = f"{cid}_seed{seed}"
            print(f"\n[{run_idx}/{total_runs}] Running {run_tag}: {cname} (Seed {seed})")
            set_seed(seed)
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

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
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Initialize Loss
            if ltype == "ce":
                criterion = nn.CrossEntropyLoss(weight=weights_t, reduction="mean")
            elif ltype == "margin":
                criterion = AsymmetricCriticalMarginLoss(
                    class_weights=weights_t,
                    margin=cond["margin"],
                    lambda_crit=cond["lambda_crit"]
                )
            elif ltype == "focal":
                criterion = MulticlassFocalLoss(
                    alpha=weights_t,
                    gamma=cond["gamma"],
                    reduction="mean"
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float("inf")
            best_epoch = 0
            best_state = None

            init_weights = [p.clone().detach() for p in model.classifier.parameters()]

            for epoch in range(1, max_epochs + 1):
                model.train()
                train_loss_accum = 0.0
                ce_loss_accum = 0.0
                margin_loss_accum = 0.0
                grad_norm_accum = 0.0
                margin_grad_norm_accum = 0.0
                batches = 0

                for bx_rec, bx_hist, by in train_loader:
                    optimizer.zero_grad()
                    logits = model(bx_rec, bx_hist)

                    if ltype == "margin":
                        tot_loss, ce_loss_val, m_loss_val = criterion(logits, by)
                        ce_loss_accum += float(ce_loss_val.item())
                        margin_loss_accum += float(m_loss_val.item())
                        loss = tot_loss
                    else:
                        loss = criterion(logits, by)
                        ce_loss_accum += float(loss.item())

                    loss.backward()

                    # Record gradient norms
                    total_norm = 0.0
                    for p in model.classifier.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm_accum += np.sqrt(total_norm)

                    # Logit margin gradient check
                    if (by == 2).any():
                        crit_idx = (by == 2).nonzero(as_tuple=True)[0]
                        # Margin between class 2 and class 1
                        margin_val = (logits[crit_idx, 2] - logits[crit_idx, 1]).mean().item()
                    else:
                        margin_val = 0.0

                    optimizer.step()
                    train_loss_accum += float(loss.item())
                    batches += 1

                train_loss = train_loss_accum / batches
                avg_grad_norm = grad_norm_accum / batches
                avg_margin_loss = margin_loss_accum / batches if ltype == "margin" else 0.0

                # Validation Loss (computed under standard weighted CE for fair comparison)
                model.eval()
                val_loss_accum = 0.0
                val_batches = 0
                val_ce = nn.CrossEntropyLoss(weight=weights_t, reduction="mean")
                with torch.no_grad():
                    for bx_rec, bx_hist, by in val_loader:
                        v_logits = model(bx_rec, bx_hist)
                        v_loss = val_ce(v_logits, by)
                        val_loss_accum += float(v_loss.item())
                        val_batches += 1

                val_loss = val_loss_accum / val_batches

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())

                training_logs.append({
                    "run_tag": run_tag,
                    "condition_id": cid,
                    "condition_name": cname,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "avg_grad_norm": avg_grad_norm,
                    "margin_loss": avg_margin_loss,
                    "is_best_epoch": (epoch == best_epoch)
                })

            # Verify parameter change
            param_changed = False
            for p_init, p_final in zip(init_weights, model.classifier.parameters()):
                if not torch.equal(p_init, p_final.detach()):
                    param_changed = True
                    break
            assert param_changed, f"ERROR: Parameters did not update during training for {run_tag}"

            # Save Checkpoint
            model.load_state_dict(best_state)
            model.eval()

            ckpt_payload = {
                "run_tag": run_tag,
                "condition_id": cid,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "param_count": param_count,
                "state_dict": best_state
            }
            torch.save(ckpt_payload, ckpt_dir / f"ckpt_{run_tag}.pt")

            # 3. Validation Inference
            with torch.no_grad():
                val_logits_t = model(torch.from_numpy(X_val_rec), torch.from_numpy(X_val_hist))
                val_probs_t = F.softmax(val_logits_t, dim=-1)

            val_logits = val_logits_t.numpy()
            val_probs = val_probs_t.numpy()

            z0 = val_logits[:, 0]
            z1 = val_logits[:, 1]
            z2 = val_logits[:, 2]
            p0 = val_probs[:, 0]
            p1 = val_probs[:, 1]
            p2 = val_probs[:, 2]
            p_fail = p1 + p2
            logit_margin = z2 - z1
            prob_margin = p2 - p1

            # Store in prediction dataframe
            val_predictions[f"logit_{run_tag}_c0"] = z0
            val_predictions[f"logit_{run_tag}_c1"] = z1
            val_predictions[f"logit_{run_tag}_c2"] = z2
            val_predictions[f"prob_{run_tag}_c0"] = p0
            val_predictions[f"prob_{run_tag}_c1"] = p1
            val_predictions[f"prob_{run_tag}_c2"] = p2
            val_predictions[f"p_fail_{run_tag}"] = p_fail
            val_predictions[f"logit_margin_{run_tag}"] = logit_margin
            val_predictions[f"prob_margin_{run_tag}"] = prob_margin

            # Probability validity checks
            assert np.all(p0 >= 0.0) and np.all(p0 <= 1.0)
            assert np.all(p1 >= 0.0) and np.all(p1 <= 1.0)
            assert np.all(p2 >= 0.0) and np.all(p2 <= 1.0)
            assert np.allclose(p0 + p1 + p2, 1.0, atol=1e-5)

            # 4. Critical Margin Diagnostics on True Critical Windows
            crit_mask = (y_val == 2)
            warn_mask = (y_val == 1)
            norm_mask = (y_val == 0)

            crit_z_margin = logit_margin[crit_mask]
            crit_p_margin = prob_margin[crit_mask]
            crit_pfail = p_fail[crit_mask]
            crit_p2 = p2[crit_mask]
            crit_p1 = p1[crit_mask]

            critical_margin_records.append({
                "run_tag": run_tag,
                "condition_id": cid,
                "seed": seed,
                "mean_logit_margin_crit": float(np.mean(crit_z_margin)),
                "median_logit_margin_crit": float(np.median(crit_z_margin)),
                "q25_logit_margin_crit": float(np.percentile(crit_z_margin, 25)),
                "q75_logit_margin_crit": float(np.percentile(crit_z_margin, 75)),
                "count_logit_margin_ge_05": int((crit_z_margin >= 0.50).sum()),
                "pct_logit_margin_ge_05": float((crit_z_margin >= 0.50).mean() * 100.0),
                "count_p2_gt_p1_crit": int((crit_p_margin > 0.0).sum()),
                "pct_p2_gt_p1_crit": float((crit_p_margin > 0.0).mean() * 100.0),
                "mean_p_fail_crit": float(np.mean(crit_pfail)),
                "mean_p2_crit": float(np.mean(crit_p2)),
                "mean_p1_crit": float(np.mean(crit_p1)),
                "mean_logit_margin_warn": float(np.mean(logit_margin[warn_mask])),
                "mean_logit_margin_norm": float(np.mean(logit_margin[norm_mask]))
            })

            # 5. Calibration Metrics
            y_one_hot = np.zeros((n_val, 3), dtype=np.float32)
            y_one_hot[np.arange(n_val), y_val] = 1.0
            brier_score = float(np.mean(np.sum((val_probs - y_one_hot) ** 2, axis=1)))
            ece_score = compute_ece(val_probs, y_val, n_bins=10)
            nll_score = float(log_loss(y_val, val_probs))

            calibration_records.append({
                "run_tag": run_tag,
                "condition_id": cid,
                "seed": seed,
                "brier_score": brier_score,
                "ece": ece_score,
                "nll": nll_score
            })

            # 6. Evaluate Predeclared Primary Decision Policy:
            # Rule: P_fail < 0.50 -> 0; else if (logit_2 - logit_1 >= 0.50) -> 2; else -> 1
            pred_primary = np.zeros(n_val, dtype=int)
            alert_primary = (p_fail >= 0.50)
            crit_primary = alert_primary & (logit_margin >= 0.50)
            warn_primary = alert_primary & (logit_margin < 0.50)
            pred_primary[crit_primary] = 2
            pred_primary[warn_primary] = 1

            cm_pri = confusion_matrix(y_val, pred_primary, labels=[0, 1, 2])
            acc_pri = float(accuracy_score(y_val, pred_primary))
            bacc_pri = float(balanced_accuracy_score(y_val, pred_primary))
            mf1_pri = float(f1_score(y_val, pred_primary, average="macro", zero_division=0.0))

            crit_rec_pri = float(cm_pri[2, 2] / n_crit * 100.0)
            warn_rec_pri = float(cm_pri[1, 1] / n_warn * 100.0)
            crit_to_warn_pri = int(cm_pri[2, 1])
            crit_to_norm_pri = int(cm_pri[2, 0])
            norm_to_crit_fp_pri = int(cm_pri[0, 2])
            warn_to_crit_fp_pri = int(cm_pri[1, 2])

            det_eps_pri = sum(1 for _, grp in episodes if (pred_primary[grp.index] > 0).any())
            ep_rec_pri = float(det_eps_pri / 20.0 * 100.0)

            fa_ents_pri = sum(1 for eid in pure_normal_eids if (pred_primary[val_meta_eval["entity_id"] == eid] > 0).any())
            ent_far_pri = float(fa_ents_pri / 49.0 * 100.0)

            win_fpr_pri = float((pred_primary[y_val == 0] > 0).sum() / 560.0 * 100.0)

            fa_jobs_pri = sum(1 for jid in pure_normal_jobs if (pred_primary[val_meta_eval["parent_job_id"] == jid] > 0).any())
            job_far_pri = float(fa_jobs_pri / 4.0 * 100.0)

            g1a_pri = bool(ep_rec_pri >= 70.0)
            g1b_pri = bool(crit_rec_pri >= 50.0)
            g2a_pri = bool(ent_far_pri <= 10.0)
            g2b_pri = bool(win_fpr_pri <= 5.0)
            all_gates_pri = bool(g1a_pri and g1b_pri and g2a_pri and g2b_pri)

            # Store confusion matrix
            confusion_matrices_records.append({
                "run_tag": run_tag,
                "policy": "Predeclared_Primary",
                "condition_id": cid,
                "seed": seed,
                "cm_00": int(cm_pri[0, 0]), "cm_01": int(cm_pri[0, 1]), "cm_02": int(cm_pri[0, 2]),
                "cm_10": int(cm_pri[1, 0]), "cm_11": int(cm_pri[1, 1]), "cm_12": int(cm_pri[1, 2]),
                "cm_20": int(cm_pri[2, 0]), "cm_21": int(cm_pri[2, 1]), "cm_22": int(cm_pri[2, 2])
            })

            # 7. Evaluate Secondary Argmax Policy
            pred_argmax = np.argmax(val_probs, axis=1)
            cm_arg = confusion_matrix(y_val, pred_argmax, labels=[0, 1, 2])
            acc_arg = float(accuracy_score(y_val, pred_argmax))
            bacc_arg = float(balanced_accuracy_score(y_val, pred_argmax))
            mf1_arg = float(f1_score(y_val, pred_argmax, average="macro", zero_division=0.0))
            crit_rec_arg = float(cm_arg[2, 2] / n_crit * 100.0)
            warn_rec_arg = float(cm_arg[1, 1] / n_warn * 100.0)
            ep_rec_arg = float(sum(1 for _, grp in episodes if (pred_argmax[grp.index] > 0).any()) / 20.0 * 100.0)
            ent_far_arg = float(sum(1 for eid in pure_normal_eids if (pred_argmax[val_meta_eval["entity_id"] == eid] > 0).any()) / 49.0 * 100.0)
            win_fpr_arg = float((pred_argmax[y_val == 0] > 0).sum() / 560.0 * 100.0)

            confusion_matrices_records.append({
                "run_tag": run_tag,
                "policy": "Three_Class_Argmax",
                "condition_id": cid,
                "seed": seed,
                "cm_00": int(cm_arg[0, 0]), "cm_01": int(cm_arg[0, 1]), "cm_02": int(cm_arg[0, 2]),
                "cm_10": int(cm_arg[1, 0]), "cm_11": int(cm_arg[1, 1]), "cm_12": int(cm_arg[1, 2]),
                "cm_20": int(cm_arg[2, 0]), "cm_21": int(cm_arg[2, 1]), "cm_22": int(cm_arg[2, 2])
            })

            # 8. Record Per-Seed Metrics
            per_seed_metrics.append({
                "run_tag": run_tag,
                "condition_id": cid,
                "condition_name": cname,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "param_count": param_count,
                # Primary Predeclared Policy Metrics
                "pri_macro_f1": mf1_pri,
                "pri_accuracy": acc_pri,
                "pri_balanced_accuracy": bacc_pri,
                "pri_episode_recall": ep_rec_pri,
                "pri_genuine_critical_recall": crit_rec_pri,
                "pri_warning_recall": warn_rec_pri,
                "pri_pure_normal_entity_far": ent_far_pri,
                "pri_normal_window_fpr": win_fpr_pri,
                "pri_pure_normal_job_far": job_far_pri,
                "pri_gate_1a_passed": g1a_pri,
                "pri_gate_1b_passed": g1b_pri,
                "pri_gate_2a_passed": g2a_pri,
                "pri_gate_2b_passed": g2b_pri,
                "pri_all_gates_passed": all_gates_pri,
                "crit_to_warn_errors": crit_to_warn_pri,
                "crit_to_norm_errors": crit_to_norm_pri,
                "norm_to_crit_fp": norm_to_crit_fp_pri,
                "warn_to_crit_fp": warn_to_crit_fp_pri,
                # Secondary Argmax Metrics
                "argmax_macro_f1": mf1_arg,
                "argmax_balanced_accuracy": bacc_arg,
                "argmax_episode_recall": ep_rec_arg,
                "argmax_critical_recall": crit_rec_arg,
                "argmax_warning_recall": warn_rec_arg,
                "argmax_entity_far": ent_far_arg,
                "argmax_window_fpr": win_fpr_arg,
                # Calibration
                "brier_score": brier_score,
                "ece": ece_score,
                "nll": nll_score,
                "mean_logit_margin_crit": float(np.mean(crit_z_margin)),
                "pct_logit_margin_ge_05": float((crit_z_margin >= 0.50).mean() * 100.0)
            })

            # 9. Episode Interception Forensics
            for ep_idx, (ep_key, grp) in enumerate(episodes):
                ep_preds = pred_primary[grp.index]
                ep_y = y_val[grp.index]
                ep_detected = bool((ep_preds > 0).any())
                ep_crit_detected = bool((ep_preds == 2).any())
                ep_crit_windows = int((ep_y == 2).sum())
                ep_warn_windows = int((ep_y == 1).sum())

                episode_error_records.append({
                    "run_tag": run_tag,
                    "condition_id": cid,
                    "seed": seed,
                    "episode_index": ep_idx,
                    "entity_id": ep_key[0],
                    "fail_timestamp": ep_key[1],
                    "total_windows": len(grp),
                    "crit_windows": ep_crit_windows,
                    "warn_windows": ep_warn_windows,
                    "episode_detected": ep_detected,
                    "crit_detected": ep_crit_detected
                })

            # 10. Exploratory Threshold Sweep (201 points)
            for th in thresholds:
                th_val = float(round(th, 4))
                pred_sw = np.zeros(n_val, dtype=int)
                alert_sw = (p_fail >= th_val)
                # Primary logit margin rule at arbitrary failure threshold th
                crit_sw = alert_sw & (logit_margin >= 0.50)
                warn_sw = alert_sw & (logit_margin < 0.50)
                pred_sw[crit_sw] = 2
                pred_sw[warn_sw] = 1

                cm_sw = confusion_matrix(y_val, pred_sw, labels=[0, 1, 2])
                crit_rec_sw = float(cm_sw[2, 2] / n_crit * 100.0)
                warn_rec_sw = float(cm_sw[1, 1] / n_warn * 100.0)
                mf1_sw = float(f1_score(y_val, pred_sw, average="macro", zero_division=0.0))
                bacc_sw = float(balanced_accuracy_score(y_val, pred_sw))

                det_eps_sw = sum(1 for _, grp in episodes if (pred_sw[grp.index] > 0).any())
                ep_rec_sw = float(det_eps_sw / 20.0 * 100.0)

                fa_ents_sw = sum(1 for eid in pure_normal_eids if (pred_sw[val_meta_eval["entity_id"] == eid] > 0).any())
                ent_far_sw = float(fa_ents_sw / 49.0 * 100.0)
                win_fpr_sw = float((pred_sw[y_val == 0] > 0).sum() / 560.0 * 100.0)

                g1a_sw = bool(ep_rec_sw >= 70.0)
                g1b_sw = bool(crit_rec_sw >= 50.0)
                g2a_sw = bool(ent_far_sw <= 10.0)
                g2b_sw = bool(win_fpr_sw <= 5.0)
                all_gates_sw = bool(g1a_sw and g1b_sw and g2a_sw and g2b_sw)

                threshold_sweep_records.append({
                    "run_tag": run_tag,
                    "condition_id": cid,
                    "seed": seed,
                    "threshold": th_val,
                    "macro_f1": mf1_sw,
                    "balanced_accuracy": bacc_sw,
                    "episode_recall": ep_rec_sw,
                    "genuine_critical_recall": crit_rec_sw,
                    "warning_recall": warn_rec_sw,
                    "pure_normal_entity_far": ent_far_sw,
                    "normal_window_fpr": win_fpr_sw,
                    "gate_1a_passed": g1a_sw,
                    "gate_1b_passed": g1b_sw,
                    "gate_2a_passed": g2a_sw,
                    "gate_2b_passed": g2b_sw,
                    "all_gates_passed": all_gates_sw
                })

            print(f"  -> Best Epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
            print(f"  -> Primary Policy (th=0.50, margin>=0.5): MacroF1={mf1_pri:.4f}, EpRec={ep_rec_pri:.1f}%, CritRec={crit_rec_pri:.1f}%, EntFAR={ent_far_pri:.1f}%, WinFPR={win_fpr_pri:.1f}%")
            print(f"  -> True Critical Windows: mean(z2 - z1)={np.mean(crit_z_margin):+.4f}, % margin >= 0.5: {(crit_z_margin >= 0.50).mean()*100:.1f}%")
            print(f"  -> All 4 Gates Passed? {all_gates_pri}")

    print(f"\nAll {total_runs} runs completed in {time.time() - start_total_t:.2f}s")

    # 11. Build Paired Comparison (Treatment vs N0 Baseline)
    df_metrics = pd.DataFrame(per_seed_metrics)
    df_metrics.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    df_metrics.to_csv(reports_dir / "per_seed_metrics.csv", index=False)

    paired_records = []
    for treat_cid in ["N1", "N2", "N3"]:
        for s in seeds:
            base_row = df_metrics[(df_metrics["condition_id"] == "N0") & (df_metrics["seed"] == s)].iloc[0]
            treat_row = df_metrics[(df_metrics["condition_id"] == treat_cid) & (df_metrics["seed"] == s)].iloc[0]

            paired_records.append({
                "comparison": f"{treat_cid}_vs_N0",
                "condition_id": treat_cid,
                "seed": s,
                "diff_pri_macro_f1": float(treat_row["pri_macro_f1"] - base_row["pri_macro_f1"]),
                "diff_pri_episode_recall": float(treat_row["pri_episode_recall"] - base_row["pri_episode_recall"]),
                "diff_pri_critical_recall": float(treat_row["pri_genuine_critical_recall"] - base_row["pri_genuine_critical_recall"]),
                "diff_pri_warning_recall": float(treat_row["pri_warning_recall"] - base_row["pri_warning_recall"]),
                "diff_pri_entity_far": float(treat_row["pri_pure_normal_entity_far"] - base_row["pri_pure_normal_entity_far"]),
                "diff_pri_window_fpr": float(treat_row["pri_normal_window_fpr"] - base_row["pri_normal_window_fpr"]),
                "diff_brier_score": float(treat_row["brier_score"] - base_row["brier_score"]),
                "diff_ece": float(treat_row["ece"] - base_row["ece"]),
                "diff_mean_logit_margin_crit": float(treat_row["mean_logit_margin_crit"] - base_row["mean_logit_margin_crit"]),
                "diff_pct_margin_ge_05": float(treat_row["pct_logit_margin_ge_05"] - base_row["pct_logit_margin_ge_05"])
            })

    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(out_dir / "paired_comparison.csv", index=False)
    df_paired.to_csv(reports_dir / "paired_comparison.csv", index=False)

    # 12. Save Remaining Artifacts
    pd.DataFrame(training_logs).to_csv(out_dir / "training_logs.csv", index=False)
    pd.DataFrame(training_logs).to_csv(reports_dir / "training_logs.csv", index=False)

    val_predictions.to_csv(out_dir / "validation_predictions.csv", index=False)
    val_predictions.to_csv(reports_dir / "validation_predictions.csv", index=False)

    pd.DataFrame(confusion_matrices_records).to_csv(out_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(confusion_matrices_records).to_csv(reports_dir / "confusion_matrices.csv", index=False)

    pd.DataFrame(critical_margin_records).to_csv(out_dir / "critical_margin_analysis.csv", index=False)
    pd.DataFrame(critical_margin_records).to_csv(reports_dir / "critical_margin_analysis.csv", index=False)

    pd.DataFrame(episode_error_records).to_csv(out_dir / "episode_error_analysis.csv", index=False)
    pd.DataFrame(episode_error_records).to_csv(reports_dir / "episode_error_analysis.csv", index=False)

    pd.DataFrame(calibration_records).to_csv(out_dir / "calibration_analysis.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(reports_dir / "calibration_analysis.csv", index=False)

    df_sweep = pd.DataFrame(threshold_sweep_records)
    df_sweep.to_csv(out_dir / "exploratory_threshold_sweep.csv", index=False)
    df_sweep.to_csv(reports_dir / "exploratory_threshold_sweep.csv", index=False)

    # Operational Gate Summary
    gate_summary = []
    for r in per_seed_metrics:
        gate_summary.append({
            "run_tag": r["run_tag"],
            "condition_id": r["condition_id"],
            "seed": r["seed"],
            "policy": "Predeclared_Primary",
            "gate_1a_episode_recall": r["pri_episode_recall"],
            "gate_1a_passed": r["pri_gate_1a_passed"],
            "gate_1b_critical_recall": r["pri_genuine_critical_recall"],
            "gate_1b_passed": r["pri_gate_1b_passed"],
            "gate_2a_entity_far": r["pri_pure_normal_entity_far"],
            "gate_2a_passed": r["pri_gate_2a_passed"],
            "gate_2b_window_fpr": r["pri_normal_window_fpr"],
            "gate_2b_passed": r["pri_gate_2b_passed"],
            "all_4_gates_passed": r["pri_all_gates_passed"]
        })
    df_gates = pd.DataFrame(gate_summary)
    df_gates.to_csv(out_dir / "operational_gate_summary.csv", index=False)
    df_gates.to_csv(reports_dir / "operational_gate_summary.csv", index=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 7N EXPERIMENTAL RESULTS SUMMARY (PREDECLARED PRIMARY POLICY)")
    print("=" * 80)
    summary_cols = ["condition_id", "pri_macro_f1", "pri_episode_recall", "pri_genuine_critical_recall", "pri_warning_recall", "pri_pure_normal_entity_far", "pri_normal_window_fpr", "brier_score", "ece", "mean_logit_margin_crit"]
    agg_df = df_metrics.groupby("condition_id")[summary_cols[1:]].agg(["mean", "std"])
    print(agg_df.to_string())

    simultaneous_passes = int(df_gates["all_4_gates_passed"].sum())
    sweep_passes = int(df_sweep["all_gates_passed"].sum())
    print(f"\nSimultaneous 4-Gate Passes (Primary Policy): {simultaneous_passes} / {len(df_gates)}")
    print(f"Simultaneous 4-Gate Passes (Full 201-Threshold Sweep): {sweep_passes} / {len(df_sweep)}")

    verdict = "OPERATIONAL_WINNER" if simultaneous_passes > 0 else "NO_OPERATIONAL_WINNER"
    print(f"\nFINAL OPERATIONAL VERDICT: {verdict}")

if __name__ == "__main__":
    main()
