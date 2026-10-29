"""Phase 7M — Multi-Resolution Temporal Redesign and Critical-Failure Detection.

Trains M1 (Recent-only control) and M2 (Multi-resolution treatment) across seeds 42, 123, 456
under Condition B [1, 2, 2] (primary) and Condition A [1, 1, 1] (reference).
Evaluates M0 (baseline L=5), M1, and M2 on Candidate A Validation (N=657).

STRICTLY CANDIDATE A TRAIN AND VALIDATION ONLY.
Zero access to Candidate A Test or Phase 6 Holdout.
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

from src.models.lstm import TemporalLSTMClassifier

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

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

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

# -----------------------------------------------------------------------------
# Architecture Definitions
# -----------------------------------------------------------------------------
class StreamBEncoder(nn.Module):
    """Encodes 93 historical trend summary features into compact representation."""
    def __init__(self, in_dim: int = 93, hidden_dim: int = 32, out_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiResolutionLSTMClassifier(nn.Module):
    """Multi-resolution temporal classifier combining Stream A and Stream B."""
    def __init__(
        self,
        input_dim: int = 31,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        use_history: bool = True,
        hist_dim: int = 93,
        hist_hidden: int = 32,
        hist_out: int = 16
    ):
        super().__init__()
        self.use_history = use_history
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        if use_history:
            self.hist_encoder = StreamBEncoder(
                in_dim=hist_dim,
                hidden_dim=hist_hidden,
                out_dim=hist_out,
                dropout=dropout
            )
            head_in = hidden_dim + hist_out
        else:
            self.hist_encoder = None
            head_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_recent: torch.Tensor, x_hist: torch.Tensor = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(x_recent)
        z_rec = lstm_out[:, -1, :]
        if self.use_history and x_hist is not None:
            z_hist = self.hist_encoder(x_hist)
            z = torch.cat([z_rec, z_hist], dim=-1)
        else:
            z = z_rec
        logits = self.classifier(z)
        return logits

# -----------------------------------------------------------------------------
# Paths & Data Setup
# -----------------------------------------------------------------------------
out_dir = PROJECT_ROOT / "results" / "phase7m_multiresolution"
reports_dir = PROJECT_ROOT / "reports" / "phase7m_multiresolution"
ckpt_dir = out_dir / "checkpoints"
processed_dir = out_dir / "processed_data"
results_7l_dir = PROJECT_ROOT / "results" / "phase7l_longer_context"

out_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Load data
X_tr_rec = np.load(processed_dir / "X_train_stream_A.npy")
X_val_rec = np.load(processed_dir / "X_val_stream_A.npy")
X_tr_hist = np.load(processed_dir / "X_train_stream_B.npy")
X_val_hist = np.load(processed_dir / "X_val_stream_B.npy")
y_train = np.load(processed_dir / "y_train.npy")
y_val = np.load(processed_dir / "y_val.npy")
val_meta = pd.read_csv(processed_dir / "val_metadata.csv")
train_meta = pd.read_csv(processed_dir / "train_metadata.csv")

assert len(X_tr_rec) == 3278
assert len(X_val_rec) == 657
assert len(y_train) == 3278
assert len(y_val) == 657

# Validation Cohort structure
n_val = len(y_val)
n_norm = int((y_val == 0).sum())
n_warn = int((y_val == 1).sum())
n_crit = int((y_val == 2).sum())
n_fail = n_warn + n_crit

assert n_norm == 560
assert n_warn == 48
assert n_crit == 49
assert n_fail == 97

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
n_pure_normal_entities = len(pure_normal_eids)
assert n_pure_normal_entities == 49

failing_jobs = set(val_meta_eval[val_meta_eval["y_true"] > 0]["parent_job_id"].unique())
all_jobs = sorted(val_meta_eval["parent_job_id"].unique())
pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
n_pure_normal_jobs = len(pure_normal_jobs)
assert n_pure_normal_jobs == 4

print(f"Validation cohort verified: {n_val} windows (560 Norm, 48 Warn, 49 Crit)")
print(f"Operational groups: {total_episodes} episodes, {n_pure_normal_entities} pure-normal entities, {n_pure_normal_jobs} pure-normal jobs.")

# -----------------------------------------------------------------------------
# Experimental Setup
# -----------------------------------------------------------------------------
models_config = [
    {"model_id": "M1", "name": "Recent_Only_Control", "use_history": False},
    {"model_id": "M2", "name": "Multi_Resolution_Treatment", "use_history": True}
]

conditions = [
    {"cond_id": "cond_B", "name": "Moderate [1,2,2] (Primary Baseline)", "weights": [1.0, 2.0, 2.0]},
    {"cond_id": "cond_A", "name": "Unweighted [1,1,1] (Reference Baseline)", "weights": [1.0, 1.0, 1.0]}
]

seeds = [42, 123, 456]
max_epochs = 25
batch_size = 64
lr = 0.001
weight_decay = 0.0001
thresholds = np.linspace(0.0, 1.0, 201)
margin_grid = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]

training_log_records = []
val_predictions_dict = {}
threshold_metrics_records = []
per_seed_metric_records = []

# Load M0 baseline predictions from Phase 7L
preds_7l = pd.read_csv(results_7l_dir / "validation_predictions.csv")
for cid in ["cond_B", "cond_A"]:
    for s in seeds:
        tag_m0 = f"L5_{cid}_seed{s}"
        val_predictions_dict[f"prob_M0_{cid}_seed{s}_c0"] = preds_7l[f"prob_{tag_m0}_c0"].values
        val_predictions_dict[f"prob_M0_{cid}_seed{s}_c1"] = preds_7l[f"prob_{tag_m0}_c1"].values
        val_predictions_dict[f"prob_M0_{cid}_seed{s}_c2"] = preds_7l[f"prob_{tag_m0}_c2"].values
        val_predictions_dict[f"p_fail_M0_{cid}_seed{s}"] = preds_7l[f"p_fail_{tag_m0}"].values

print(f"\n{'='*80}")
print("STARTING PHASE 7M TRAINING & EVALUATION (12 RUNS)")
print(f"{'='*80}")

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

total_runs = len(models_config) * len(conditions) * len(seeds)
run_idx = 0
start_time_all = time.time()

for m_cfg in models_config:
    mid = m_cfg["model_id"]
    mname = m_cfg["name"]
    use_hist = m_cfg["use_history"]

    for cond in conditions:
        cid = cond["cond_id"]
        cname = cond["name"]
        cweights = cond["weights"]
        class_weights_t = torch.tensor(cweights, dtype=torch.float32)

        for seed in seeds:
            run_idx += 1
            run_tag = f"{mid}_{cid}_seed{seed}"
            run_start_t = time.time()

            set_seed(seed)
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

            model = MultiResolutionLSTMClassifier(
                input_dim=31,
                hidden_dim=64,
                num_layers=2,
                num_classes=3,
                dropout=0.2,
                use_history=use_hist,
                hist_dim=93,
                hist_hidden=32,
                hist_out=16
            )
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float("inf")
            best_epoch = 0
            best_model_state = None

            for epoch in range(1, max_epochs + 1):
                model.train()
                train_loss_accum = 0.0
                train_batches = 0
                for bx_rec, bx_hist, by in train_loader:
                    optimizer.zero_grad()
                    logits = model(bx_rec, bx_hist if use_hist else None)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    train_loss_accum += float(loss.item())
                    train_batches += 1

                train_loss = train_loss_accum / train_batches

                model.eval()
                val_loss_accum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for bx_rec, bx_hist, by in val_loader:
                        logits = model(bx_rec, bx_hist if use_hist else None)
                        loss = criterion(logits, by)
                        val_loss_accum += float(loss.item())
                        val_batches += 1

                val_loss = val_loss_accum / val_batches

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())

                training_log_records.append({
                    "run_tag": run_tag,
                    "model_id": mid,
                    "condition_id": cid,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "is_best_epoch": (epoch == best_epoch)
                })

            model.load_state_dict(best_model_state)
            model.eval()

            ckpt_path = ckpt_dir / f"ckpt_{run_tag}.pt"
            torch.save({
                "run_tag": run_tag,
                "model_id": mid,
                "condition_id": cid,
                "seed": seed,
                "param_count": param_count,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "state_dict": best_model_state
            }, ckpt_path)

            with torch.no_grad():
                val_logits = model(
                    torch.from_numpy(X_val_rec),
                    torch.from_numpy(X_val_hist) if use_hist else None
                )
                val_probs = torch.softmax(val_logits, dim=-1).numpy()

            p0 = val_probs[:, 0]
            p1 = val_probs[:, 1]
            p2 = val_probs[:, 2]
            p_fail = p1 + p2

            val_predictions_dict[f"prob_{run_tag}_c0"] = p0
            val_predictions_dict[f"prob_{run_tag}_c1"] = p1
            val_predictions_dict[f"prob_{run_tag}_c2"] = p2
            val_predictions_dict[f"p_fail_{run_tag}"] = p_fail

            # Calibration metrics
            y_one_hot = np.eye(3)[y_val]
            brier_val = float(np.mean(np.sum((val_probs - y_one_hot) ** 2, axis=1)))
            nll_val = float(log_loss(y_val, val_probs, labels=[0, 1, 2]))
            ece_val = compute_ece(val_probs, y_val)

            elapsed_run = time.time() - run_start_t
            print(f"[{run_idx:2d}/{total_runs}] {run_tag:20s} | Best Ep: {best_epoch:2d} (Loss: {best_val_loss:.4f}) | Brier: {brier_val:.4f} | ECE: {ece_val:.4f} | Params: {param_count} | Time: {elapsed_run:.1f}s")

print(f"\nAll 12 training runs finished in {(time.time() - start_time_all)/60:.2f} min!")

# -----------------------------------------------------------------------------
# Save Predictions & Training Logs
# -----------------------------------------------------------------------------
df_train_logs = pd.DataFrame(training_log_records)
df_train_logs.to_csv(out_dir / "training_logs.csv", index=False)
df_train_logs.to_csv(reports_dir / "training_logs.csv", index=False)

df_val_preds = val_meta.copy()
df_val_preds["y_true"] = y_val
for col, arr in val_predictions_dict.items():
    df_val_preds[col] = arr
df_val_preds.to_csv(out_dir / "validation_predictions.csv", index=False)
df_val_preds.to_csv(reports_dir / "validation_predictions.csv", index=False)

# -----------------------------------------------------------------------------
# Comprehensive Evaluation Engine (M0, M1, M2)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EVALUATING M0, M1, M2 ACROSS THRESHOLDS AND POLICIES")
print("="*80)

eval_model_tags = []
for mid in ["M0", "M1", "M2"]:
    for cid in ["cond_B", "cond_A"]:
        for s in seeds:
            eval_model_tags.append((mid, cid, s, f"{mid}_{cid}_seed{s}"))

threshold_sweep_records = []
gate_summary_records = []
per_seed_records = []
confusion_matrix_records = []

for mid, cid, s, run_tag in eval_model_tags:
    p0 = df_val_preds[f"prob_{run_tag}_c0"].values
    p1 = df_val_preds[f"prob_{run_tag}_c1"].values
    p2 = df_val_preds[f"prob_{run_tag}_c2"].values
    p_fail = df_val_preds[f"p_fail_{run_tag}"].values
    probs = np.column_stack([p0, p1, p2])

    y_one_hot = np.eye(3)[y_val]
    brier_val = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
    nll_val = float(log_loss(y_val, probs, labels=[0, 1, 2]))
    ece_val = compute_ece(probs, y_val)

    # 1. Unthresholded Standard 3-Class Argmax
    pred_argmax = np.argmax(probs, axis=1)
    macro_f1_argmax = float(f1_score(y_val, pred_argmax, average="macro", zero_division=0.0))
    bal_acc_argmax = float(balanced_accuracy_score(y_val, pred_argmax))
    acc_argmax = float(accuracy_score(y_val, pred_argmax))
    cm_argmax = confusion_matrix(y_val, pred_argmax, labels=[0, 1, 2])
    crit_rec_argmax = float(cm_argmax[2, 2] / n_crit * 100.0)
    warn_rec_argmax = float(cm_argmax[1, 1] / n_warn * 100.0)
    norm_rec_argmax = float(cm_argmax[0, 0] / n_norm * 100.0)

    confusion_matrix_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        "eval_type": "standard_argmax", "threshold": "N/A", "margin": "N/A",
        "macro_f1": macro_f1_argmax, "bal_acc": bal_acc_argmax, "acc": acc_argmax,
        "crit_rec_pct": crit_rec_argmax, "warn_rec_pct": warn_rec_argmax, "norm_rec_pct": norm_rec_argmax,
        "cm_00": int(cm_argmax[0, 0]), "cm_01": int(cm_argmax[0, 1]), "cm_02": int(cm_argmax[0, 2]),
        "cm_10": int(cm_argmax[1, 0]), "cm_11": int(cm_argmax[1, 1]), "cm_12": int(cm_argmax[1, 2]),
        "cm_20": int(cm_argmax[2, 0]), "cm_21": int(cm_argmax[2, 1]), "cm_22": int(cm_argmax[2, 2]),
    })

    # 2. Sweep over 201 Thresholds under Policy A and Policy B
    run_sweep_pts = []
    pt_th050 = None
    passing_pts_b = []

    for th in thresholds:
        th_val = float(round(th, 4))

        # Policy A: Binary Alert
        alert_a = (p_fail >= th_val).astype(int)
        det_eps_a = sum(1 for _, grp in episodes if (alert_a[grp.index] > 0).any())
        ep_rec_a = det_eps_a / total_episodes * 100.0
        crit_inter_a = int(alert_a[y_val == 2].sum()) / n_crit * 100.0
        fa_ents_a = sum(1 for eid in pure_normal_eids if (alert_a[val_meta_eval["entity_id"] == eid] > 0).any()) / n_pure_normal_entities * 100.0
        win_fpr_a = int(alert_a[y_val == 0].sum()) / n_norm * 100.0

        # Policy B: Cascaded 3-Class Decision
        pred_b = np.zeros(n_val, dtype=int)
        mask_alert = (p_fail >= th_val)
        mask_crit = mask_alert & (p2 >= p1)
        mask_warn = mask_alert & (p1 > p2)
        pred_b[mask_crit] = 2
        pred_b[mask_warn] = 1

        det_eps_b = sum(1 for _, grp in episodes if (pred_b[grp.index] > 0).any())
        ep_rec_b = det_eps_b / total_episodes * 100.0
        cm_b = confusion_matrix(y_val, pred_b, labels=[0, 1, 2])
        crit_rec_b = float(cm_b[2, 2] / n_crit * 100.0)
        warn_rec_b = float(cm_b[1, 1] / n_warn * 100.0)
        norm_rec_b = float(cm_b[0, 0] / n_norm * 100.0)

        fa_ents_b = sum(1 for eid in pure_normal_eids if (pred_b[val_meta_eval["entity_id"] == eid] > 0).any()) / n_pure_normal_entities * 100.0
        win_fpr_b = int((pred_b[y_val == 0] > 0).sum()) / n_norm * 100.0
        fa_jobs_b = sum(1 for jid in pure_normal_jobs if (pred_b[val_meta_eval["parent_job_id"] == jid] > 0).any()) / n_pure_normal_jobs * 100.0

        macro_f1_b = float(f1_score(y_val, pred_b, average="macro", zero_division=0.0))
        bal_acc_b = float(balanced_accuracy_score(y_val, pred_b))
        acc_b = float(accuracy_score(y_val, pred_b))

        g1_pass = bool(ep_rec_b >= 70.0 and crit_rec_b >= 50.0)
        g2_pass = bool(fa_ents_b <= 10.0 and win_fpr_b <= 5.0)
        all_pass = bool(g1_pass and g2_pass)
        if all_pass:
            passing_pts_b.append(th_val)

        row_th = {
            "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s, "threshold": th_val,
            "polA_ep_rec": ep_rec_a, "polA_crit_inter": crit_inter_a, "polA_ent_far": fa_ents_a, "polA_win_fpr": win_fpr_a,
            "polB_macro_f1": macro_f1_b, "polB_bal_acc": bal_acc_b, "polB_acc": acc_b,
            "polB_ep_rec": ep_rec_b, "polB_crit_rec": crit_rec_b, "polB_warn_rec": warn_rec_b, "polB_norm_rec": norm_rec_b,
            "polB_ent_far": fa_ents_b, "polB_win_fpr": win_fpr_b, "polB_job_far": fa_jobs_b,
            "gate1_pass": g1_pass, "gate2_pass": g2_pass, "all_gates_pass": all_pass,
            "cm_00": int(cm_b[0, 0]), "cm_01": int(cm_b[0, 1]), "cm_02": int(cm_b[0, 2]),
            "cm_10": int(cm_b[1, 0]), "cm_11": int(cm_b[1, 1]), "cm_12": int(cm_b[1, 2]),
            "cm_20": int(cm_b[2, 0]), "cm_21": int(cm_b[2, 1]), "cm_22": int(cm_b[2, 2]),
        }
        threshold_sweep_records.append(row_th)
        run_sweep_pts.append(row_th)

        if np.isclose(th_val, 0.50):
            pt_th050 = row_th

    # Peak Macro-F1 point
    peak_f1_pt = max(run_sweep_pts, key=lambda x: x["polB_macro_f1"])

    # Record peak confusion matrix
    confusion_matrix_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        "eval_type": "peak_macro_f1", "threshold": str(peak_f1_pt["threshold"]), "margin": "N/A",
        "macro_f1": peak_f1_pt["polB_macro_f1"], "bal_acc": peak_f1_pt["polB_bal_acc"], "acc": peak_f1_pt["polB_acc"],
        "crit_rec_pct": peak_f1_pt["polB_crit_rec"], "warn_rec_pct": peak_f1_pt["polB_warn_rec"], "norm_rec_pct": peak_f1_pt["polB_norm_rec"],
        "cm_00": peak_f1_pt["cm_00"], "cm_01": peak_f1_pt["cm_01"], "cm_02": peak_f1_pt["cm_02"],
        "cm_10": peak_f1_pt["cm_10"], "cm_11": peak_f1_pt["cm_11"], "cm_12": peak_f1_pt["cm_12"],
        "cm_20": peak_f1_pt["cm_20"], "cm_21": peak_f1_pt["cm_21"], "cm_22": peak_f1_pt["cm_22"],
    })

    # Record predeclared th=0.50 confusion matrix
    confusion_matrix_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        "eval_type": "predeclared_th_050", "threshold": "0.50", "margin": "N/A",
        "macro_f1": pt_th050["polB_macro_f1"], "bal_acc": pt_th050["polB_bal_acc"], "acc": pt_th050["polB_acc"],
        "crit_rec_pct": pt_th050["polB_crit_rec"], "warn_rec_pct": pt_th050["polB_warn_rec"], "norm_rec_pct": pt_th050["polB_norm_rec"],
        "cm_00": pt_th050["cm_00"], "cm_01": pt_th050["cm_01"], "cm_02": pt_th050["cm_02"],
        "cm_10": pt_th050["cm_10"], "cm_11": pt_th050["cm_11"], "cm_12": pt_th050["cm_12"],
        "cm_20": pt_th050["cm_20"], "cm_21": pt_th050["cm_21"], "cm_22": pt_th050["cm_22"],
    })

    per_seed_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        # Predeclared th=0.50 Primary Metrics
        "th050_macro_f1": pt_th050["polB_macro_f1"],
        "th050_bal_acc": pt_th050["polB_bal_acc"],
        "th050_acc": pt_th050["polB_acc"],
        "th050_ep_rec": pt_th050["polB_ep_rec"],
        "th050_crit_rec": pt_th050["polB_crit_rec"],
        "th050_warn_rec": pt_th050["polB_warn_rec"],
        "th050_ent_far": pt_th050["polB_ent_far"],
        "th050_win_fpr": pt_th050["polB_win_fpr"],
        "th050_job_far": pt_th050["polB_job_far"],
        "th050_all_pass": pt_th050["all_gates_pass"],
        # Peak F1 Metrics (Optimistic Sweep Ceiling)
        "peak_th": peak_f1_pt["threshold"],
        "peak_macro_f1": peak_f1_pt["polB_macro_f1"],
        "peak_ep_rec": peak_f1_pt["polB_ep_rec"],
        "peak_crit_rec": peak_f1_pt["polB_crit_rec"],
        "peak_warn_rec": peak_f1_pt["polB_warn_rec"],
        "peak_ent_far": peak_f1_pt["polB_ent_far"],
        "peak_win_fpr": peak_f1_pt["polB_win_fpr"],
        # Calibration
        "brier_score": brier_val,
        "nll": nll_val,
        "ece": ece_val,
        # Gate Summary
        "passing_thresholds_count": len(passing_pts_b),
        "any_threshold_passes_all_gates": len(passing_pts_b) > 0
    })

    gate_summary_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        "predeclared_th": 0.50,
        "th050_all_pass": pt_th050["all_gates_pass"],
        "peak_f1_th": peak_f1_pt["threshold"],
        "peak_f1": peak_f1_pt["polB_macro_f1"],
        "passing_threshold_count": len(passing_pts_b),
        "operational_winner_verdict": "OPERATIONAL_WINNER" if len(passing_pts_b) > 0 else "NO_OPERATIONAL_WINNER"
    })

# Save Threshold Sweep & Gate Summary
df_th_sweep = pd.DataFrame(threshold_metrics_records)
df_th_sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
df_th_sweep.to_csv(reports_dir / "threshold_sweep.csv", index=False)

df_per_seed = pd.DataFrame(per_seed_records)
df_per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)
df_per_seed.to_csv(reports_dir / "per_seed_metrics.csv", index=False)

df_gate_summary = pd.DataFrame(gate_summary_records)
df_gate_summary.to_csv(out_dir / "operational_gate_summary.csv", index=False)
df_gate_summary.to_csv(reports_dir / "operational_gate_summary.csv", index=False)

df_cm = pd.DataFrame(confusion_matrix_records)
df_cm.to_csv(out_dir / "confusion_matrices.csv", index=False)
df_cm.to_csv(reports_dir / "confusion_matrices.csv", index=False)

# -----------------------------------------------------------------------------
# Paired Comparisons: (M2 - M1) and (M2 - M0)
# -----------------------------------------------------------------------------
paired_records = []
for cid in ["cond_B", "cond_A"]:
    for s in seeds:
        r_m0 = df_per_seed[(df_per_seed["model_id"] == "M0") & (df_per_seed["condition_id"] == cid) & (df_per_seed["seed"] == s)].iloc[0]
        r_m1 = df_per_seed[(df_per_seed["model_id"] == "M1") & (df_per_seed["condition_id"] == cid) & (df_per_seed["seed"] == s)].iloc[0]
        r_m2 = df_per_seed[(df_per_seed["model_id"] == "M2") & (df_per_seed["condition_id"] == cid) & (df_per_seed["seed"] == s)].iloc[0]

        # M2 vs M1 (Treatment vs Matched Control)
        paired_records.append({
            "comparison": "M2_vs_M1", "condition_id": cid, "seed": s,
            "th050_macro_f1_base": r_m1["th050_macro_f1"], "th050_macro_f1_treat": r_m2["th050_macro_f1"],
            "delta_th050_macro_f1": r_m2["th050_macro_f1"] - r_m1["th050_macro_f1"],
            "th050_ep_rec_base": r_m1["th050_ep_rec"], "th050_ep_rec_treat": r_m2["th050_ep_rec"],
            "delta_th050_ep_rec": r_m2["th050_ep_rec"] - r_m1["th050_ep_rec"],
            "th050_crit_rec_base": r_m1["th050_crit_rec"], "th050_crit_rec_treat": r_m2["th050_crit_rec"],
            "delta_th050_crit_rec": r_m2["th050_crit_rec"] - r_m1["th050_crit_rec"],
            "th050_ent_far_base": r_m1["th050_ent_far"], "th050_ent_far_treat": r_m2["th050_ent_far"],
            "delta_th050_ent_far": r_m2["th050_ent_far"] - r_m1["th050_ent_far"],
            "th050_win_fpr_base": r_m1["th050_win_fpr"], "th050_win_fpr_treat": r_m2["th050_win_fpr"],
            "delta_th050_win_fpr": r_m2["th050_win_fpr"] - r_m1["th050_win_fpr"],
            "peak_f1_base": r_m1["peak_macro_f1"], "peak_f1_treat": r_m2["peak_macro_f1"],
            "delta_peak_f1": r_m2["peak_macro_f1"] - r_m1["peak_macro_f1"],
            "brier_base": r_m1["brier_score"], "brier_treat": r_m2["brier_score"],
            "delta_brier": r_m2["brier_score"] - r_m1["brier_score"],
            "ece_base": r_m1["ece"], "ece_treat": r_m2["ece"],
            "delta_ece": r_m2["ece"] - r_m1["ece"]
        })

        # M2 vs M0 (Treatment vs Phase 7L Baseline)
        paired_records.append({
            "comparison": "M2_vs_M0", "condition_id": cid, "seed": s,
            "th050_macro_f1_base": r_m0["th050_macro_f1"], "th050_macro_f1_treat": r_m2["th050_macro_f1"],
            "delta_th050_macro_f1": r_m2["th050_macro_f1"] - r_m0["th050_macro_f1"],
            "th050_ep_rec_base": r_m0["th050_ep_rec"], "th050_ep_rec_treat": r_m2["th050_ep_rec"],
            "delta_th050_ep_rec": r_m2["th050_ep_rec"] - r_m0["th050_ep_rec"],
            "th050_crit_rec_base": r_m0["th050_crit_rec"], "th050_crit_rec_treat": r_m2["th050_crit_rec"],
            "delta_th050_crit_rec": r_m2["th050_crit_rec"] - r_m0["th050_crit_rec"],
            "th050_ent_far_base": r_m0["th050_ent_far"], "th050_ent_far_treat": r_m2["th050_ent_far"],
            "delta_th050_ent_far": r_m2["th050_ent_far"] - r_m0["th050_ent_far"],
            "th050_win_fpr_base": r_m0["th050_win_fpr"], "th050_win_fpr_treat": r_m2["th050_win_fpr"],
            "delta_th050_win_fpr": r_m2["th050_win_fpr"] - r_m0["th050_win_fpr"],
            "peak_f1_base": r_m0["peak_macro_f1"], "peak_f1_treat": r_m2["peak_macro_f1"],
            "delta_peak_f1": r_m2["peak_macro_f1"] - r_m0["peak_macro_f1"],
            "brier_base": r_m0["brier_score"], "brier_treat": r_m2["brier_score"],
            "delta_brier": r_m2["brier_score"] - r_m0["brier_score"],
            "ece_base": r_m0["ece"], "ece_treat": r_m2["ece"],
            "delta_ece": r_m2["ece"] - r_m0["ece"]
        })

df_paired = pd.DataFrame(paired_records)
df_paired.to_csv(out_dir / "paired_comparison.csv", index=False)
df_paired.to_csv(reports_dir / "paired_comparison.csv", index=False)

# -----------------------------------------------------------------------------
# Forensic Probability & Critical Diagnostics
# -----------------------------------------------------------------------------
crit_prob_records = []
for mid, cid, s, run_tag in eval_model_tags:
    p0 = df_val_preds[f"prob_{run_tag}_c0"].values
    p1 = df_val_preds[f"prob_{run_tag}_c1"].values
    p2 = df_val_preds[f"prob_{run_tag}_c2"].values
    p_fail = df_val_preds[f"p_fail_{run_tag}"].values

    for c in [0, 1, 2]:
        mask_c = (y_val == c)
        c_name = ["Normal", "Warning", "Critical"][c]
        p0_c = p0[mask_c]
        p1_c = p1[mask_c]
        p2_c = p2[mask_c]
        pf_c = p_fail[mask_c]
        margin_2_1 = p2_c - p1_c
        ptrue = [p0_c, p1_c, p2_c][c]

        crit_prob_records.append({
            "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
            "true_class": c, "class_name": c_name, "count": int(mask_c.sum()),
            "mean_p0": float(np.mean(p0_c)), "mean_p1": float(np.mean(p1_c)),
            "mean_p2": float(np.mean(p2_c)), "mean_pfail": float(np.mean(pf_c)),
            "median_p2": float(np.median(p2_c)),
            "mean_margin_p2_minus_p1": float(np.mean(margin_2_1)),
            "p25_margin": float(np.percentile(margin_2_1, 25)),
            "p75_margin": float(np.percentile(margin_2_1, 75)),
            "fraction_p2_ge_p1": float(np.mean(p2_c >= p1_c)),
            "fraction_p2_argmax": float(np.mean((p2_c >= p1_c) & (p2_c >= p0_c))),
            "mean_true_class_prob": float(np.mean(ptrue))
        })

df_crit_prob = pd.DataFrame(crit_prob_records)
df_crit_prob.to_csv(out_dir / "critical_probability_analysis.csv", index=False)
df_crit_prob.to_csv(reports_dir / "critical_probability_analysis.csv", index=False)

# -----------------------------------------------------------------------------
# Episode-Level Error Analysis
# -----------------------------------------------------------------------------
episode_records = []
for ep_idx, ((eid, fail_t), grp) in enumerate(episodes):
    n_crit_wins = int((grp["y_true"] == 2).sum())
    n_warn_wins = int((grp["y_true"] == 1).sum())
    parent_job = grp["parent_job_id"].iloc[0]

    # Evaluate detection at predeclared th=0.50 under Cond B across seeds for M0, M1, M2
    det_m0_seeds = []
    det_m1_seeds = []
    det_m2_seeds = []
    crit_m0_seeds = []
    crit_m1_seeds = []
    crit_m2_seeds = []
    first_t_m0 = []
    first_t_m1 = []
    first_t_m2 = []

    for s in seeds:
        for mid, det_list, crit_list, t_list in [
            ("M0", det_m0_seeds, crit_m0_seeds, first_t_m0),
            ("M1", det_m1_seeds, crit_m1_seeds, first_t_m1),
            ("M2", det_m2_seeds, crit_m2_seeds, first_t_m2)
        ]:
            tag = f"{mid}_cond_B_seed{s}"
            pf = df_val_preds.loc[grp.index, f"p_fail_{tag}"].values
            p1_ep = df_val_preds.loc[grp.index, f"prob_{tag}_c1"].values
            p2_ep = df_val_preds.loc[grp.index, f"prob_{tag}_c2"].values
            pred_times = grp["prediction_timestamp"].values

            # Policy B at th=0.50
            alerted = (pf >= 0.50)
            det_list.append(bool(alerted.any()))
            
            # Critical classified as critical
            crit_detected = (alerted & (p2_ep >= p1_ep) & (grp["y_true"] == 2)).any()
            crit_list.append(bool(crit_detected))

            t_alert = float(pred_times[alerted][0]) if alerted.any() else None
            t_list.append(t_alert)

    episode_records.append({
        "episode_id": ep_idx + 1, "entity_id": eid, "parent_job_id": parent_job,
        "fail_timestamp": fail_t, "total_windows": len(grp),
        "warning_windows": n_warn_wins, "critical_windows": n_crit_wins,
        "M0_detected_seeds_count": sum(det_m0_seeds),
        "M1_detected_seeds_count": sum(det_m1_seeds),
        "M2_detected_seeds_count": sum(det_m2_seeds),
        "M0_crit_correct_seeds_count": sum(crit_m0_seeds),
        "M1_crit_correct_seeds_count": sum(crit_m1_seeds),
        "M2_crit_correct_seeds_count": sum(crit_m2_seeds),
        "M2_vs_M1_status": "M2_IMPROVED" if sum(det_m2_seeds) > sum(det_m1_seeds) else ("M2_DEGRADED" if sum(det_m2_seeds) < sum(det_m1_seeds) else "UNCHANGED")
    })

df_episodes = pd.DataFrame(episode_records)
df_episodes.to_csv(out_dir / "episode_error_analysis.csv", index=False)
df_episodes.to_csv(reports_dir / "episode_error_analysis.csv", index=False)

# -----------------------------------------------------------------------------
# Calibration Analysis Table
# -----------------------------------------------------------------------------
cal_records = []
for mid, cid, s, run_tag in eval_model_tags:
    p0 = df_val_preds[f"prob_{run_tag}_c0"].values
    p1 = df_val_preds[f"prob_{run_tag}_c1"].values
    p2 = df_val_preds[f"prob_{run_tag}_c2"].values
    probs = np.column_stack([p0, p1, p2])

    y_one_hot = np.eye(3)[y_val]
    brier_val = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
    nll_val = float(log_loss(y_val, probs, labels=[0, 1, 2]))
    ece_val = compute_ece(probs, y_val)

    cal_records.append({
        "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s,
        "brier_score": brier_val, "nll": nll_val, "ece": ece_val
    })

df_cal = pd.DataFrame(cal_records)
df_cal.to_csv(out_dir / "calibration_analysis.csv", index=False)
df_cal.to_csv(reports_dir / "calibration_analysis.csv", index=False)

print("\n" + "="*80)
print("PHASE 7M RESULTS SUMMARY TABLE (PREDECLARED THRESHOLD TH=0.50):")
print("="*80)
for cid in ["cond_B", "cond_A"]:
    print(f"\n--- Condition: {cid} ---")
    for mid in ["M0", "M1", "M2"]:
        sub = df_per_seed[(df_per_seed["model_id"] == mid) & (df_per_seed["condition_id"] == cid)]
        f1_m, f1_s = sub["th050_macro_f1"].mean(), sub["th050_macro_f1"].std()
        ep_m, ep_s = sub["th050_ep_rec"].mean(), sub["th050_ep_rec"].std()
        cr_m, cr_s = sub["th050_crit_rec"].mean(), sub["th050_crit_rec"].std()
        far_m, far_s = sub["th050_ent_far"].mean(), sub["th050_ent_far"].std()
        fpr_m, fpr_s = sub["th050_win_fpr"].mean(), sub["th050_win_fpr"].std()
        br_m, br_s = sub["brier_score"].mean(), sub["brier_score"].std()
        ece_m, ece_s = sub["ece"].mean(), sub["ece"].std()
        print(f"{mid:3s}: F1@0.50 = {f1_m:.4f} +/- {f1_s:.4f} | EpRec = {ep_m:5.1f}% | CritRec = {cr_m:5.1f}% | EntFAR = {far_m:5.1f}% | WinFPR = {fpr_m:5.1f}% | Brier = {br_m:.4f} | ECE = {ece_m:.4f}")

print("\n" + "="*80)
print("PEAK MACRO-F1 SWEEP CEILING (EXPLORATORY):")
print("="*80)
for cid in ["cond_B", "cond_A"]:
    print(f"\n--- Condition: {cid} ---")
    for mid in ["M0", "M1", "M2"]:
        sub = df_per_seed[(df_per_seed["model_id"] == mid) & (df_per_seed["condition_id"] == cid)]
        f1_m, f1_s = sub["peak_macro_f1"].mean(), sub["peak_macro_f1"].std()
        ep_m, ep_s = sub["peak_ep_rec"].mean(), sub["peak_ep_rec"].std()
        cr_m, cr_s = sub["peak_crit_rec"].mean(), sub["peak_crit_rec"].std()
        far_m, far_s = sub["peak_ent_far"].mean(), sub["peak_ent_far"].std()
        fpr_m, fpr_s = sub["peak_win_fpr"].mean(), sub["peak_win_fpr"].std()
        print(f"{mid:3s}: Peak F1 = {f1_m:.4f} +/- {f1_s:.4f} | EpRec = {ep_m:5.1f}% | CritRec = {cr_m:5.1f}% | EntFAR = {far_m:5.1f}% | WinFPR = {fpr_m:5.1f}%")

print("\nPhase 7M Experiment execution completed successfully!")
