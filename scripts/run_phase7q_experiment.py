"""Phase 7Q — Cross-Job Predictive Signal Audit and Causal Workload-Relative Features.

Pipeline:
1. Balanced parent-job-grouped 5-fold partition on Candidate A Train (10-11 jobs/fold).
2. Compute Train-derived scale floors and causal relative deltas for Stream C.
3. Train Q0 (Matched N2 Baseline) and Q1 (Causal Relative Treatment) across 3 seeds x 5 folds = 30 runs.
4. Train full-Train Q0 and Q1 across 3 seeds = 6 runs.
5. Generate comprehensive OOF ranking metrics (AUC, AP, Spearman, Pearson, Macro-F1, Margin).
6. Evaluate Candidate A Validation under fixed operational policy (P_fail >= 0.50, margin >= 0.50) and argmax.
7. Export all 22 required Phase 7Q deliverables and logs.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
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

class MultiResolutionRelativeLSTMClassifier(nn.Module):
    """Q1 Treatment: Dual-stream multi-resolution backbone + causal relative deviation MLP."""
    def __init__(
        self,
        input_dim: int = 31,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        hist_dim: int = 93,
        hist_hidden: int = 32,
        hist_out: int = 16,
        rel_dim: int = 31,
        rel_hidden: int = 32,
        rel_out: int = 16
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_dim, hist_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hist_hidden, hist_out),
            nn.ReLU()
        )
        self.rel_encoder = nn.Sequential(
            nn.Linear(rel_dim, rel_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, rel_out),
            nn.ReLU()
        )
        fused_dim = hidden_dim + hist_out + rel_out
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_recent: torch.Tensor, x_hist: torch.Tensor, x_rel: torch.Tensor) -> torch.Tensor:
        out_rec, _ = self.lstm(x_recent)
        feat_rec = out_rec[:, -1, :]
        feat_hist = self.hist_encoder(x_hist)
        feat_rel = self.rel_encoder(x_rel)
        fused = torch.cat([feat_rec, feat_hist, feat_rel], dim=-1)
        return self.classifier(fused)

def compute_causal_relative_deltas(
    X_12: np.ndarray,
    scale_floors: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Computes causal recent-versus-historical relative deviation features.
    
    Historical interval: steps 0..6 (t-55m to t-25m, 7 steps).
    Recent interval: steps 7..11 (t-20m to t, 5 steps).
    Formula: delta_j = clip((recent_median_j - hist_median_j) / max(1.4826 * MAD_hist_j, floor_j), -5.0, +5.0)
    """
    hist = X_12[:, :7, :]
    recent = X_12[:, 7:, :]
    
    h_med = np.median(hist, axis=1) # (N, 31)
    r_med = np.median(recent, axis=1) # (N, 31)
    
    h_mad = np.median(np.abs(hist - h_med[:, None, :]), axis=1) # (N, 31)
    h_scale = 1.4826 * h_mad # (N, 31)
    
    effective_scale = np.maximum(h_scale, scale_floors[None, :])
    raw_delta = (r_med - h_med) / effective_scale
    clipped_delta = np.clip(raw_delta, -5.0, 5.0).astype(np.float32)
    
    floor_hit_rate = float((h_scale < scale_floors[None, :]).mean())
    clip_hit_rate = float((np.abs(raw_delta) >= 5.0).mean())
    
    return clipped_delta, floor_hit_rate, clip_hit_rate

def main():
    print("=" * 80)
    print("PHASE 7Q: CROSS-JOB PREDICTIVE SIGNAL AUDIT & CAUSAL WORKLOAD-RELATIVE FEATURES")
    print("=" * 80)

    start_total_time = time.time()
    out_dir = PROJECT_ROOT / "results" / "phase7q_signal_audit"
    rep_dir = PROJECT_ROOT / "reports" / "phase7q_signal_audit"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir_7m = PROJECT_ROOT / "results" / "phase7m_multiresolution" / "processed_data"
    data_dir_7l = PROJECT_ROOT / "results" / "phase7l_longer_context" / "standardized_data"
    assert_path_allowed(str(data_dir_7m))
    assert_path_allowed(str(data_dir_7l))

    # 1. Load Data
    print("\n--- 1. Loading Authorized Data Partitions ---")
    X_tr_rec = np.load(data_dir_7m / "X_train_stream_A.npy").astype(np.float32)
    X_tr_hist = np.load(data_dir_7m / "X_train_stream_B.npy").astype(np.float32)
    y_tr = np.load(data_dir_7m / "y_train.npy").astype(np.int64)
    tr_meta = pd.read_csv(data_dir_7m / "train_metadata.csv")

    X_val_rec = np.load(data_dir_7m / "X_val_stream_A.npy").astype(np.float32)
    X_val_hist = np.load(data_dir_7m / "X_val_stream_B.npy").astype(np.float32)
    y_val = np.load(data_dir_7m / "y_val.npy").astype(np.int64)
    val_meta = pd.read_csv(data_dir_7m / "val_metadata.csv")

    X_tr_l12 = np.load(data_dir_7l / "X_train_l12.npy").astype(np.float32)
    X_val_l12 = np.load(data_dir_7l / "X_val_l12.npy").astype(np.float32)

    assert len(X_tr_rec) == 3278
    assert len(X_val_rec) == 657
    assert len(X_tr_l12) == 3278
    assert len(X_val_l12) == 657

    # 2. Derive Scale Floors strictly from Train and compute Stream C
    print("\n--- 2. Computing Causal Workload-Relative Features (Stream C) ---")
    hist_tr = X_tr_l12[:, :7, :]
    h_med_tr = np.median(hist_tr, axis=1)
    h_mad_tr = np.median(np.abs(hist_tr - h_med_tr[:, None, :]), axis=1)
    h_scale_tr = 1.4826 * h_mad_tr

    scale_floors = np.zeros(31, dtype=np.float32)
    for j in range(31):
        pos_scales = h_scale_tr[:, j][h_scale_tr[:, j] > 1e-6]
        scale_floors[j] = float(np.percentile(pos_scales, 10.0)) if len(pos_scales) > 0 else 0.01

    X_tr_rel, fl_tr, cl_tr = compute_causal_relative_deltas(X_tr_l12, scale_floors)
    X_val_rel, fl_val, cl_val = compute_causal_relative_deltas(X_val_l12, scale_floors)

    assert np.all(np.isfinite(X_tr_rel))
    assert np.all(np.isfinite(X_val_rel))
    print(f"Scale floors computed: min={scale_floors.min():.4f}, median={np.median(scale_floors):.4f}, max={scale_floors.max():.4f}")
    print(f"Train Stream C: Floor hit rate = {fl_tr*100:.2f}%, Clip rate = {cl_tr*100:.2f}%")
    print(f"Val Stream C:   Floor hit rate = {fl_val*100:.2f}%, Clip rate = {cl_val*100:.2f}%")

    # 3. Construct Balanced Parent-Job-Grouped 5-Fold Partition
    print("\n--- 3. Constructing Balanced Parent-Job-Grouped 5-Fold Partition ---")
    tr_meta_eval = tr_meta.copy()
    tr_meta_eval["y_true"] = y_tr
    tr_meta_eval["fail_timestamp"] = (tr_meta_eval["prediction_timestamp"] + tr_meta_eval["time_to_failure"]).round(1)

    job_df = tr_meta_eval.groupby("parent_job_id").agg(
        windows=("y_true", "count"),
        normal=("y_true", lambda s: int((s == 0).sum())),
        warn=("y_true", lambda s: int((s == 1).sum())),
        crit=("y_true", lambda s: int((s == 2).sum())),
        fails=("y_true", lambda s: int((s > 0).sum())),
        is_fail=("y_true", lambda s: bool((s > 0).any()))
    ).reset_index()

    failing_jobs = job_df[job_df["is_fail"]].sort_values("windows", ascending=False).reset_index(drop=True)
    pure_jobs = job_df[~job_df["is_fail"]].sort_values("windows", ascending=False).reset_index(drop=True)

    # 34 failing jobs -> 7, 7, 7, 7, 6
    # 17 pure jobs -> 3, 3, 3, 4, 4
    # Total jobs per fold: 10, 10, 10, 11, 10
    folds_jobs = [[] for _ in range(5)]
    fold_wins = [0] * 5

    for _, row in failing_jobs.iterrows():
        candidates = [f for f in range(5) if len([j for j in folds_jobs[f] if j["is_fail"]]) < (7 if f < 4 else 6)]
        best_f = min(candidates, key=lambda f: fold_wins[f])
        folds_jobs[best_f].append(row)
        fold_wins[best_f] += int(row["windows"])

    for _, row in pure_jobs.iterrows():
        candidates = [f for f in range(5) if len([j for j in folds_jobs[f] if not j["is_fail"]]) < (4 if f in [3, 4] else 3)]
        best_f = min(candidates, key=lambda f: fold_wins[f])
        folds_jobs[best_f].append(row)
        fold_wins[best_f] += int(row["windows"])

    job_to_fold = {}
    for f in range(5):
        for j in folds_jobs[f]:
            job_to_fold[j["parent_job_id"]] = f

    tr_meta_eval["fold"] = tr_meta_eval["parent_job_id"].map(job_to_fold)
    tr_meta_eval["y_fail"] = (tr_meta_eval["y_true"] > 0).astype(int)

    # Export fold assignments
    fold_df_out = tr_meta_eval[["entity_id", "prediction_timestamp", "parent_job_id", "time_to_failure", "y_true", "fold", "y_fail"]].copy()
    fold_df_out.to_csv(out_dir / "fold_assignments.csv", index=False)
    fold_df_out.to_csv(rep_dir / "fold_assignments.csv", index=False)

    for f in range(5):
        sub = fold_df_out[fold_df_out["fold"] == f]
        n_pjobs = sub["parent_job_id"].nunique()
        print(f"Fold {f}: {len(sub):4d} windows, {n_pjobs:2d} jobs (Norm: {(sub['y_true']==0).sum():4d}, Warn: {(sub['y_true']==1).sum():3d}, Crit: {(sub['y_true']==2).sum():3d}, Prev: {sub['y_fail'].mean()*100:5.2f}%)")

    # Check zero job overlap
    job_fold_counts = fold_df_out.groupby("parent_job_id")["fold"].nunique()
    assert (job_fold_counts == 1).all()
    print("Fold integrity verified: 0 parent jobs cross fold boundaries.")

    # 4. Training Setup
    seeds = [42, 123, 456]
    class_weights_t = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)
    batch_size = 64
    lr = 0.001
    weight_decay = 0.0001
    max_epochs = 25

    training_logs = []
    oof_predictions = tr_meta_eval[["entity_id", "prediction_timestamp", "parent_job_id", "time_to_failure", "y_true", "fold", "y_fail"]].copy()

    # Training OOF models
    print("\n--- 4. Training OOF Models: Q0 (Baseline) vs Q1 (Treatment) ---")
    start_train_time = time.time()
    total_runs = len(seeds) * 5 * 2
    run_idx = 0

    for model_type in ["Q0", "Q1"]:
        for seed in seeds:
            set_seed(seed)
            oof_z0 = np.zeros(len(X_tr_rec), dtype=np.float32)
            oof_z1 = np.zeros(len(X_tr_rec), dtype=np.float32)
            oof_z2 = np.zeros(len(X_tr_rec), dtype=np.float32)
            oof_p0 = np.zeros(len(X_tr_rec), dtype=np.float32)
            oof_p1 = np.zeros(len(X_tr_rec), dtype=np.float32)
            oof_p2 = np.zeros(len(X_tr_rec), dtype=np.float32)

            for fold in range(5):
                run_idx += 1
                train_idx = np.where(tr_meta_eval["fold"] != fold)[0]
                val_idx = np.where(tr_meta_eval["fold"] == fold)[0]

                if model_type == "Q0":
                    ds_train = TensorDataset(
                        torch.from_numpy(X_tr_rec[train_idx]),
                        torch.from_numpy(X_tr_hist[train_idx]),
                        torch.from_numpy(y_tr[train_idx])
                    )
                    ds_val = TensorDataset(
                        torch.from_numpy(X_tr_rec[val_idx]),
                        torch.from_numpy(X_tr_hist[val_idx]),
                        torch.from_numpy(y_tr[val_idx])
                    )
                    model = MultiResolutionLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, use_history=True, hist_dim=93, hist_hidden=32, hist_out=16)
                else:
                    ds_train = TensorDataset(
                        torch.from_numpy(X_tr_rec[train_idx]),
                        torch.from_numpy(X_tr_hist[train_idx]),
                        torch.from_numpy(X_tr_rel[train_idx]),
                        torch.from_numpy(y_tr[train_idx])
                    )
                    ds_val = TensorDataset(
                        torch.from_numpy(X_tr_rec[val_idx]),
                        torch.from_numpy(X_tr_hist[val_idx]),
                        torch.from_numpy(X_tr_rel[val_idx]),
                        torch.from_numpy(y_tr[val_idx])
                    )
                    model = MultiResolutionRelativeLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, hist_dim=93, hist_hidden=32, hist_out=16, rel_dim=31, rel_hidden=32, rel_out=16)

                train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

                criterion = AsymmetricCriticalMarginLoss(class_weights=class_weights_t, margin=0.50, lambda_crit=0.10)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                best_val_loss = float("inf")
                best_epoch = 0
                best_state = None

                ckpt_tag = f"ckpt_{model_type}_seed{seed}_fold{fold}.pt"
                ckpt_path = ckpt_dir / ckpt_tag
                if ckpt_path.exists():
                    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    best_state = ckpt_data["state_dict"]
                    best_epoch = ckpt_data["best_epoch"]
                    best_val_loss = ckpt_data["best_val_loss"]
                    model.load_state_dict(best_state)
                    model.eval()
                else:
                    for epoch in range(1, max_epochs + 1):
                        model.train()
                        t_loss_accum = 0.0
                        t_batches = 0
                        for batch in train_loader:
                            optimizer.zero_grad()
                            if model_type == "Q0":
                                bx_rec, bx_hist, by = batch
                                logits = model(bx_rec, bx_hist)
                            else:
                                bx_rec, bx_hist, bx_rel, by = batch
                                logits = model(bx_rec, bx_hist, bx_rel)
                            loss, _, _ = criterion(logits, by)
                            loss.backward()
                            optimizer.step()
                            t_loss_accum += float(loss.item())
                            t_batches += 1
                        t_loss = t_loss_accum / t_batches

                        model.eval()
                        v_loss_accum = 0.0
                        v_batches = 0
                        val_ce = nn.CrossEntropyLoss(weight=class_weights_t, reduction="mean")
                        with torch.no_grad():
                            for batch in val_loader:
                                if model_type == "Q0":
                                    bx_rec, bx_hist, by = batch
                                    v_logits = model(bx_rec, bx_hist)
                                else:
                                    bx_rec, bx_hist, bx_rel, by = batch
                                    v_logits = model(bx_rec, bx_hist, bx_rel)
                                v_loss = val_ce(v_logits, by)
                                v_loss_accum += float(v_loss.item())
                                v_batches += 1
                        v_loss = v_loss_accum / v_batches

                        if v_loss < best_val_loss:
                            best_val_loss = v_loss
                            best_epoch = epoch
                            best_state = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_state)
                    model.eval()
                    torch.save({"model_type": model_type, "seed": seed, "fold": fold, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "state_dict": best_state}, ckpt_path)

                training_logs.append({
                    "model": model_type,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "checkpoint": ckpt_tag
                })

                with torch.no_grad():
                    if model_type == "Q0":
                        f_logits_t = model(torch.from_numpy(X_tr_rec[val_idx]), torch.from_numpy(X_tr_hist[val_idx]))
                    else:
                        f_logits_t = model(torch.from_numpy(X_tr_rec[val_idx]), torch.from_numpy(X_tr_hist[val_idx]), torch.from_numpy(X_tr_rel[val_idx]))
                    f_probs_t = F.softmax(f_logits_t, dim=-1)

                f_logits = f_logits_t.numpy()
                f_probs = f_probs_t.numpy()

                oof_z0[val_idx] = f_logits[:, 0]
                oof_z1[val_idx] = f_logits[:, 1]
                oof_z2[val_idx] = f_logits[:, 2]
                oof_p0[val_idx] = f_probs[:, 0]
                oof_p1[val_idx] = f_probs[:, 1]
                oof_p2[val_idx] = f_probs[:, 2]

                print(f"[{run_idx:2d}/{total_runs}] {model_type} seed={seed} fold={fold} -> Best Epoch: {best_epoch:2d}, Val Loss: {best_val_loss:.4f}")

            # Store OOF predictions
            oof_predictions[f"{model_type}_logit_s{seed}_c0"] = oof_z0
            oof_predictions[f"{model_type}_logit_s{seed}_c1"] = oof_z1
            oof_predictions[f"{model_type}_logit_s{seed}_c2"] = oof_z2
            oof_predictions[f"{model_type}_prob_s{seed}_c0"] = oof_p0
            oof_predictions[f"{model_type}_prob_s{seed}_c1"] = oof_p1
            oof_predictions[f"{model_type}_prob_s{seed}_c2"] = oof_p2
            oof_predictions[f"{model_type}_pfail_s{seed}"] = oof_p1 + oof_p2
            oof_predictions[f"{model_type}_margin_s{seed}"] = oof_z2 - oof_z1

    print(f"All 30 OOF models trained in {time.time() - start_train_time:.2f}s.")
    oof_predictions.to_csv(out_dir / "oof_predictions.csv", index=False)
    oof_predictions.to_csv(rep_dir / "oof_predictions.csv", index=False)

    # 5. Train Full-Train Models and Evaluate on Candidate A Validation
    print("\n--- 5. Training Full-Train Models & Evaluating on Candidate A Validation ---")
    val_predictions = val_meta[["entity_id", "prediction_timestamp", "parent_job_id", "time_to_failure", "assigned_split"]].copy()
    val_predictions["y_true"] = y_val
    val_predictions["y_fail"] = (y_val > 0).astype(int)

    for model_type in ["Q0", "Q1"]:
        for seed in seeds:
            set_seed(seed)
            if model_type == "Q0":
                ds_full = TensorDataset(torch.from_numpy(X_tr_rec), torch.from_numpy(X_tr_hist), torch.from_numpy(y_tr))
                model = MultiResolutionLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, use_history=True, hist_dim=93, hist_hidden=32, hist_out=16)
            else:
                ds_full = TensorDataset(torch.from_numpy(X_tr_rec), torch.from_numpy(X_tr_hist), torch.from_numpy(X_tr_rel), torch.from_numpy(y_tr))
                model = MultiResolutionRelativeLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, hist_dim=93, hist_hidden=32, hist_out=16, rel_dim=31, rel_hidden=32, rel_out=16)

            full_loader = DataLoader(ds_full, batch_size=batch_size, shuffle=True)
            criterion = AsymmetricCriticalMarginLoss(class_weights=class_weights_t, margin=0.50, lambda_crit=0.10)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            ckpt_tag = f"ckpt_{model_type}_seed{seed}_full.pt"
            ckpt_path = ckpt_dir / ckpt_tag
            if ckpt_path.exists():
                ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt_data["state_dict"])
            else:
                for epoch in range(1, max_epochs + 1):
                    model.train()
                    for batch in full_loader:
                        optimizer.zero_grad()
                        if model_type == "Q0":
                            bx_rec, bx_hist, by = batch
                            logits = model(bx_rec, bx_hist)
                        else:
                            bx_rec, bx_hist, bx_rel, by = batch
                            logits = model(bx_rec, bx_hist, bx_rel)
                        loss, _, _ = criterion(logits, by)
                        loss.backward()
                        optimizer.step()

                torch.save({"model_type": model_type, "seed": seed, "state_dict": model.state_dict()}, ckpt_path)
            training_logs.append({"model": model_type, "seed": seed, "fold": "full", "best_epoch": max_epochs, "best_val_loss": 0.0, "checkpoint": ckpt_tag})

            model.eval()
            with torch.no_grad():
                if model_type == "Q0":
                    v_logits_t = model(torch.from_numpy(X_val_rec), torch.from_numpy(X_val_hist))
                else:
                    v_logits_t = model(torch.from_numpy(X_val_rec), torch.from_numpy(X_val_hist), torch.from_numpy(X_val_rel))
                v_probs_t = F.softmax(v_logits_t, dim=-1)

            v_logits = v_logits_t.numpy()
            v_probs = v_probs_t.numpy()

            val_predictions[f"{model_type}_logit_s{seed}_c0"] = v_logits[:, 0]
            val_predictions[f"{model_type}_logit_s{seed}_c1"] = v_logits[:, 1]
            val_predictions[f"{model_type}_logit_s{seed}_c2"] = v_logits[:, 2]
            val_predictions[f"{model_type}_prob_s{seed}_c0"] = v_probs[:, 0]
            val_predictions[f"{model_type}_prob_s{seed}_c1"] = v_probs[:, 1]
            val_predictions[f"{model_type}_prob_s{seed}_c2"] = v_probs[:, 2]

            v_pfail = v_probs[:, 1] + v_probs[:, 2]
            v_margin = v_logits[:, 2] - v_logits[:, 1]
            val_predictions[f"{model_type}_pfail_s{seed}"] = v_pfail
            val_predictions[f"{model_type}_margin_s{seed}"] = v_margin

            # Primary Fixed Decision Rule: P_fail >= 0.50, Critical if z2 - z1 >= 0.50 else Warning
            pred_fixed = np.zeros(len(v_pfail), dtype=int)
            alert = (v_pfail >= 0.50)
            pred_fixed[alert & (v_margin >= 0.50)] = 2
            pred_fixed[alert & (v_margin < 0.50)] = 1
            val_predictions[f"pred_{model_type}_fixed_s{seed}"] = pred_fixed

            # Diagnostic Argmax Rule
            pred_argmax = np.argmax(v_probs, axis=-1)
            val_predictions[f"pred_{model_type}_argmax_s{seed}"] = pred_argmax

            print(f"Full-Train {model_type} seed={seed} evaluated on Validation.")

    val_predictions.to_csv(out_dir / "validation_predictions.csv", index=False)
    val_predictions.to_csv(rep_dir / "validation_predictions.csv", index=False)

    df_logs = pd.DataFrame(training_logs)
    df_logs.to_csv(out_dir / "training_logs.csv", index=False)
    df_logs.to_csv(rep_dir / "training_logs.csv", index=False)

    # 6. Primary Ranking & Classification Metrics on Train OOF
    print("\n--- 6. Computing Primary Ranking & Classification Metrics on Train OOF ---")
    per_seed_records = []
    per_fold_records = []
    confusion_records = []
    crit_margin_records = []

    y_fail_val = (y_val > 0).astype(int)
    prefails_val = val_predictions[val_predictions["y_true"] > 0].copy()
    prefails_val["fail_timestamp"] = (prefails_val["prediction_timestamp"] + prefails_val["time_to_failure"]).round(1)
    episodes_val = list(prefails_val.groupby(["entity_id", "fail_timestamp"]))
    assert len(episodes_val) == 20

    failing_val_eids = set(prefails_val["entity_id"].unique())
    all_val_eids = sorted(val_predictions["entity_id"].unique())
    pure_normal_val_eids = sorted(list(set(all_val_eids) - failing_val_eids))
    assert len(pure_normal_val_eids) == 49

    for model_type in ["Q0", "Q1"]:
        for seed in seeds:
            # --- OOF Metrics ---
            pfail_oof = oof_predictions[f"{model_type}_pfail_s{seed}"].values
            margin_oof = oof_predictions[f"{model_type}_margin_s{seed}"].values
            y_fail_oof = oof_predictions["y_fail"].values
            y_true_oof = oof_predictions["y_true"].values

            oof_auc = float(roc_auc_score(y_fail_oof, pfail_oof))
            oof_ap = float(average_precision_score(y_fail_oof, pfail_oof))
            oof_spearman, _ = spearmanr(pfail_oof, y_fail_oof)
            oof_pearson, _ = pearsonr(pfail_oof, y_fail_oof)

            pred_oof_fixed = np.zeros(len(pfail_oof), dtype=int)
            alert_oof = (pfail_oof >= 0.50)
            pred_oof_fixed[alert_oof & (margin_oof >= 0.50)] = 2
            pred_oof_fixed[alert_oof & (margin_oof < 0.50)] = 1

            oof_mf1 = float(f1_score(y_true_oof, pred_oof_fixed, average="macro", zero_division=0.0))
            oof_acc = float(accuracy_score(y_true_oof, pred_oof_fixed))
            oof_bal_acc = float(balanced_accuracy_score(y_true_oof, pred_oof_fixed))

            cm_oof = confusion_matrix(y_true_oof, pred_oof_fixed, labels=[0, 1, 2])
            crit_mask_oof = (y_true_oof == 2)
            mean_margin_oof = float(np.mean(margin_oof[crit_mask_oof]))
            pct_margin_oof = float((margin_oof[crit_mask_oof] >= 0.50).mean() * 100.0)

            # --- Validation Metrics under Fixed Policy ---
            pred_val_fixed = val_predictions[f"pred_{model_type}_fixed_s{seed}"].values
            pfail_val = val_predictions[f"{model_type}_pfail_s{seed}"].values
            margin_val = val_predictions[f"{model_type}_margin_s{seed}"].values

            val_auc = float(roc_auc_score(y_fail_val, pfail_val))
            val_ap = float(average_precision_score(y_fail_val, pfail_val))
            val_spearman, _ = spearmanr(pfail_val, y_fail_val)
            val_pearson, _ = pearsonr(pfail_val, y_fail_val)

            val_mf1 = float(f1_score(y_val, pred_val_fixed, average="macro", zero_division=0.0))
            val_acc = float(accuracy_score(y_val, pred_val_fixed))
            val_bal_acc = float(balanced_accuracy_score(y_val, pred_val_fixed))

            cm_val = confusion_matrix(y_val, pred_val_fixed, labels=[0, 1, 2])
            val_crit_rec = float(cm_val[2, 2] / 49.0 * 100.0)
            val_warn_rec = float(cm_val[1, 1] / 48.0 * 100.0)

            det_eps = sum(1 for _, grp in episodes_val if (pred_val_fixed[grp.index] > 0).any())
            val_ep_rec = float(det_eps / 20.0 * 100.0)

            fa_ents = sum(1 for eid in pure_normal_val_eids if (pred_val_fixed[val_predictions["entity_id"] == eid] > 0).any())
            val_ent_far = float(fa_ents / 49.0 * 100.0)

            val_win_fpr = float((pred_val_fixed[y_val == 0] > 0).sum() / 560.0 * 100.0)

            crit_mask_val = (y_val == 2)
            mean_margin_val = float(np.mean(margin_val[crit_mask_val]))
            pct_margin_val = float((margin_val[crit_mask_val] >= 0.50).mean() * 100.0)

            gate_1a = (val_ep_rec >= 70.0)
            gate_1b = (val_crit_rec >= 50.0)
            gate_2a = (val_ent_far <= 10.0)
            gate_2b = (val_win_fpr <= 5.0)
            all_gates = bool(gate_1a and gate_1b and gate_2a and gate_2b)

            per_seed_records.append({
                "model": model_type,
                "seed": seed,
                "oof_auc": oof_auc,
                "oof_ap": oof_ap,
                "oof_spearman": float(oof_spearman),
                "oof_pearson": float(oof_pearson),
                "oof_macro_f1": oof_mf1,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_spearman": float(val_spearman),
                "val_pearson": float(val_pearson),
                "val_macro_f1": val_mf1,
                "val_accuracy": val_acc,
                "val_balanced_acc": val_bal_acc,
                "val_episode_recall": val_ep_rec,
                "val_critical_recall": val_crit_rec,
                "val_warning_recall": val_warn_rec,
                "val_entity_far": val_ent_far,
                "val_window_fpr": val_win_fpr,
                "gate_1a_passed": gate_1a,
                "gate_1b_passed": gate_1b,
                "gate_2a_passed": gate_2a,
                "gate_2b_passed": gate_2b,
                "all_4_gates_passed": all_gates
            })

            crit_margin_records.append({
                "model": model_type,
                "seed": seed,
                "cohort": "Train_OOF",
                "mean_logit_margin": mean_margin_oof,
                "pct_margin_ge_05": pct_margin_oof
            })
            crit_margin_records.append({
                "model": model_type,
                "seed": seed,
                "cohort": "Validation",
                "mean_logit_margin": mean_margin_val,
                "pct_margin_ge_05": pct_margin_val
            })

            confusion_records.append({
                "model": model_type,
                "seed": seed,
                "cohort": "Validation_FixedPolicy",
                "cm_00": int(cm_val[0, 0]),
                "cm_01": int(cm_val[0, 1]),
                "cm_02": int(cm_val[0, 2]),
                "cm_10": int(cm_val[1, 0]),
                "cm_11": int(cm_val[1, 1]),
                "cm_12": int(cm_val[1, 2]),
                "cm_20": int(cm_val[2, 0]),
                "cm_21": int(cm_val[2, 1]),
                "cm_22": int(cm_val[2, 2]),
                "crit_to_warn_confusion": int(cm_val[2, 1]),
                "crit_to_norm_confusion": int(cm_val[2, 0])
            })

            # Diagnostic Argmax Confusion Matrix
            pred_val_arg = val_predictions[f"pred_{model_type}_argmax_s{seed}"].values
            cm_arg = confusion_matrix(y_val, pred_val_arg, labels=[0, 1, 2])
            confusion_records.append({
                "model": model_type,
                "seed": seed,
                "cohort": "Validation_ArgmaxDiagnostic",
                "cm_00": int(cm_arg[0, 0]),
                "cm_01": int(cm_arg[0, 1]),
                "cm_02": int(cm_arg[0, 2]),
                "cm_10": int(cm_arg[1, 0]),
                "cm_11": int(cm_arg[1, 1]),
                "cm_12": int(cm_arg[1, 2]),
                "cm_20": int(cm_arg[2, 0]),
                "cm_21": int(cm_arg[2, 1]),
                "cm_22": int(cm_arg[2, 2]),
                "crit_to_warn_confusion": int(cm_arg[2, 1]),
                "crit_to_norm_confusion": int(cm_arg[2, 0])
            })

            # Fold-level ranking metrics
            for f in range(5):
                f_mask = (oof_predictions["fold"] == f)
                f_pfail = pfail_oof[f_mask]
                f_yfail = y_fail_oof[f_mask]
                f_ytrue = y_true_oof[f_mask]
                f_pred = pred_oof_fixed[f_mask]

                f_auc = float(roc_auc_score(f_yfail, f_pfail)) if len(np.unique(f_yfail)) > 1 else float("nan")
                f_ap = float(average_precision_score(f_yfail, f_pfail)) if len(np.unique(f_yfail)) > 1 else float("nan")
                f_mf1 = float(f1_score(f_ytrue, f_pred, average="macro", zero_division=0.0))

                per_fold_records.append({
                    "model": model_type,
                    "seed": seed,
                    "fold": f,
                    "auc": f_auc,
                    "ap": f_ap,
                    "macro_f1": f_mf1,
                    "windows": int(f_mask.sum()),
                    "failures": int(f_yfail.sum())
                })

    df_per_seed = pd.DataFrame(per_seed_records)
    df_per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    df_per_seed.to_csv(rep_dir / "per_seed_metrics.csv", index=False)

    df_per_fold = pd.DataFrame(per_fold_records)
    df_per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    df_per_fold.to_csv(rep_dir / "per_fold_metrics.csv", index=False)

    df_cm = pd.DataFrame(confusion_records)
    df_cm.to_csv(out_dir / "confusion_matrices.csv", index=False)
    df_cm.to_csv(rep_dir / "confusion_matrices.csv", index=False)

    df_crit = pd.DataFrame(crit_margin_records)
    df_crit.to_csv(out_dir / "critical_margin_analysis.csv", index=False)
    df_crit.to_csv(rep_dir / "critical_margin_analysis.csv", index=False)

    # Operational Gate Summary
    gate_summary = []
    for r in per_seed_records:
        gate_summary.append({
            "model": r["model"],
            "seed": r["seed"],
            "episode_recall": r["val_episode_recall"],
            "gate_1a_passed": r["gate_1a_passed"],
            "critical_recall": r["val_critical_recall"],
            "gate_1b_passed": r["gate_1b_passed"],
            "entity_far": r["val_entity_far"],
            "gate_2a_passed": r["gate_2a_passed"],
            "window_fpr": r["val_window_fpr"],
            "gate_2b_passed": r["gate_2b_passed"],
            "all_4_gates_passed": r["all_4_gates_passed"]
        })
    df_gates = pd.DataFrame(gate_summary)
    df_gates.to_csv(out_dir / "operational_gate_summary.csv", index=False)
    df_gates.to_csv(rep_dir / "operational_gate_summary.csv", index=False)

    # Paired Comparison (Q1 minus Q0)
    paired_records = []
    for s in seeds:
        row_q0 = df_per_seed[(df_per_seed["model"] == "Q0") & (df_per_seed["seed"] == s)].iloc[0]
        row_q1 = df_per_seed[(df_per_seed["model"] == "Q1") & (df_per_seed["seed"] == s)].iloc[0]

        paired_records.append({
            "comparison": "Q1_minus_Q0",
            "seed": s,
            "diff_oof_auc": float(row_q1["oof_auc"] - row_q0["oof_auc"]),
            "diff_oof_ap": float(row_q1["oof_ap"] - row_q0["oof_ap"]),
            "diff_oof_spearman": float(row_q1["oof_spearman"] - row_q0["oof_spearman"]),
            "diff_oof_macro_f1": float(row_q1["oof_macro_f1"] - row_q0["oof_macro_f1"]),
            "diff_val_auc": float(row_q1["val_auc"] - row_q0["val_auc"]),
            "diff_val_ap": float(row_q1["val_ap"] - row_q0["val_ap"]),
            "diff_val_spearman": float(row_q1["val_spearman"] - row_q0["val_spearman"]),
            "diff_val_macro_f1": float(row_q1["val_macro_f1"] - row_q0["val_macro_f1"]),
            "diff_val_ep_recall": float(row_q1["val_episode_recall"] - row_q0["val_episode_recall"]),
            "diff_val_crit_recall": float(row_q1["val_critical_recall"] - row_q0["val_critical_recall"]),
            "diff_val_entity_far": float(row_q1["val_entity_far"] - row_q0["val_entity_far"]),
            "diff_val_window_fpr": float(row_q1["val_window_fpr"] - row_q0["val_window_fpr"])
        })
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(out_dir / "paired_comparison.csv", index=False)
    df_paired.to_csv(rep_dir / "paired_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("PHASE 7Q RESULTS SUMMARY: MATCHED Q0 BASELINE VS Q1 TREATMENT")
    print("=" * 80)
    print("\n--- Train Out-of-Fold Ranking (Q0 vs Q1) ---")
    print(df_per_seed.groupby("model")[["oof_auc", "oof_ap", "oof_spearman", "oof_macro_f1"]].agg(["mean", "std"]).to_string())

    print("\n--- Candidate A Validation Ranking & Operations (Fixed Policy) ---")
    print(df_per_seed.groupby("model")[["val_auc", "val_ap", "val_macro_f1", "val_episode_recall", "val_critical_recall", "val_entity_far", "val_window_fpr"]].agg(["mean", "std"]).to_string())

    print("\n--- Paired Q1-minus-Q0 Differences Across Seeds ---")
    print(df_paired.to_string())

    simul_passes_q0 = int(df_gates[df_gates["model"] == "Q0"]["all_4_gates_passed"].sum())
    simul_passes_q1 = int(df_gates[df_gates["model"] == "Q1"]["all_4_gates_passed"].sum())
    print(f"\nSimultaneous 4-Gate Passes: Q0 = {simul_passes_q0}/3, Q1 = {simul_passes_q1}/3")

    verdict = "OPERATIONAL_WINNER" if simul_passes_q1 == len(seeds) else "NO_OPERATIONAL_WINNER"
    print(f"\nFINAL OPERATIONAL VERDICT: {verdict}")
    print(f"Total execution runtime: {time.time() - start_total_time:.2f}s.")

if __name__ == "__main__":
    main()
