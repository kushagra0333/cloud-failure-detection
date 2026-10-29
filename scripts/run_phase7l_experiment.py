"""Phase 7L — Controlled Longer-Temporal-Context Experiment.

Evaluates whether increasing causal sequence receptive field from L=5 (25 minutes)
to L=12 (60 minutes) improves failure prediction performance on Candidate A Development Data.

Conditions:
- Context Lengths: L=5 (Baseline), L=12 (Treatment)
- Weighting Conditions:
  * cond_B: [1.0, 2.0, 2.0] (Predeclared Primary Comparison)
  * cond_A: [1.0, 1.0, 1.0] (Unweighted Reference)
- Seeds: 42, 123, 456
- Total runs: 2 x 2 x 3 = 12 runs.

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
    brier_score_loss,
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
# Setup Paths & Directories
# -----------------------------------------------------------------------------
out_dir = PROJECT_ROOT / "results" / "phase7l_longer_context"
ckpt_dir = out_dir / "checkpoints"
data_dir = out_dir / "standardized_data"
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Load standardized arrays and metadata
X_tr_l5 = np.load(data_dir / "X_train_l5.npy")
X_val_l5 = np.load(data_dir / "X_val_l5.npy")
X_tr_l12 = np.load(data_dir / "X_train_l12.npy")
X_val_l12 = np.load(data_dir / "X_val_l12.npy")

y_train = np.load(data_dir / "y_train.npy")
y_val = np.load(data_dir / "y_val.npy")

val_meta = pd.read_csv(data_dir / "val_metadata.csv")
train_meta = pd.read_csv(data_dir / "train_metadata.csv")

assert len(train_meta) == 3278
assert len(val_meta) == 657
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
print(f"Operational entities: {total_episodes} episodes across {len(failing_eids)} failing entities; {n_pure_normal_entities} pure-normal entities; {n_pure_normal_jobs} pure-normal jobs.")

# -----------------------------------------------------------------------------
# Experimental Grid
# -----------------------------------------------------------------------------
contexts = [
    {"ctx_id": "L5", "name": "Baseline_L5", "seq_len": 5, "X_tr": X_tr_l5, "X_val": X_val_l5},
    {"ctx_id": "L12", "name": "Treatment_L12", "seq_len": 12, "X_tr": X_tr_l12, "X_val": X_val_l12}
]

conditions = [
    {"cond_id": "cond_B", "name": "Moderate [1,2,2] (Primary Baseline)", "weights": [1.0, 2.0, 2.0]},
    {"cond_id": "cond_A", "name": "Unweighted [1,1,1] (Reference Baseline)", "weights": [1.0, 1.0, 1.0]},
]

seeds = [42, 123, 456]
max_epochs = 25
batch_size = 64
lr = 0.001
weight_decay = 0.0001
thresholds = np.linspace(0.0, 1.0, 201)

training_log_records = []
val_predictions_dict = {}
threshold_metrics_records = []
gate_summary_records = []

total_runs = len(contexts) * len(conditions) * len(seeds)
run_idx = 0

print(f"\n{'='*80}")
print(f"STARTING PHASE 7L CONTROLLED LONGER-CONTEXT EXPERIMENT ({total_runs} RUNS)")
print(f"{'='*80}")
start_time_all = time.time()

for ctx in contexts:
    ctx_id = ctx["ctx_id"]
    ctx_name = ctx["name"]
    seq_len = ctx["seq_len"]
    X_tr = ctx["X_tr"]
    X_v = ctx["X_val"]
    dim = 31

    ds_train = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train).long())
    ds_val = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_val).long())
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    for cond in conditions:
        cid = cond["cond_id"]
        cname = cond["name"]
        cweights = cond["weights"]
        class_weights_t = torch.tensor(cweights, dtype=torch.float32)

        for seed in seeds:
            run_idx += 1
            run_tag = f"{ctx_id}_{cid}_seed{seed}"
            run_start_t = time.time()

            set_seed(seed)
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

            model = TemporalLSTMClassifier(
                input_dim=dim,
                hidden_dim=64,
                num_layers=2,
                num_classes=3,
                dropout=0.2,
                bidirectional=False
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float("inf")
            best_epoch = 0
            best_model_state = None

            for epoch in range(1, max_epochs + 1):
                model.train()
                train_loss_accum = 0.0
                train_batches = 0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    logits = model(bx)
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
                    for bx, by in val_loader:
                        logits = model(bx)
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
                    "context_id": ctx_id,
                    "seq_len": seq_len,
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
                "context_id": ctx_id,
                "seq_len": seq_len,
                "dim": dim,
                "condition_id": cid,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "state_dict": best_model_state
            }, ckpt_path)

            with torch.no_grad():
                val_logits = model(torch.from_numpy(X_v))
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

            df_eval_run = val_meta_eval.copy()
            df_eval_run["p_fail"] = p_fail
            df_eval_run["p1"] = p1
            df_eval_run["p2"] = p2

            passing_thresholds_a = []
            passing_thresholds_b = []
            run_threshold_rows = []

            for th in thresholds:
                th_val = float(round(th, 4))

                # Policy A: Binary Alert
                alert_a = (p_fail >= th).astype(int)
                df_eval_run["alert_a"] = alert_a

                det_eps_a = sum(1 for _, grp in episodes if (df_eval_run.loc[grp.index, "alert_a"] > 0).any())
                ep_rec_a = det_eps_a / total_episodes * 100.0

                crit_inter = int(alert_a[y_val == 2].sum())
                crit_inter_pct = crit_inter / n_crit * 100.0

                fa_ents_a = sum(1 for eid in pure_normal_eids if (df_eval_run[df_eval_run["entity_id"] == eid]["alert_a"] > 0).any())
                ent_far_a = fa_ents_a / n_pure_normal_entities * 100.0

                fp_wins_a = int(alert_a[y_val == 0].sum())
                win_fpr_a = fp_wins_a / n_norm * 100.0

                fa_jobs_a = sum(1 for jid in pure_normal_jobs if (df_eval_run[df_eval_run["parent_job_id"] == jid]["alert_a"] > 0).any())
                job_far_a = fa_jobs_a / n_pure_normal_jobs * 100.0

                tp_a = int((alert_a[y_val > 0] == 1).sum())
                bin_prec = tp_a / (tp_a + fp_wins_a) if (tp_a + fp_wins_a) > 0 else 0.0
                bin_rec = tp_a / n_fail if n_fail > 0 else 0.0
                bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0.0

                g1a_pass = bool((ep_rec_a >= 70.0) and (crit_inter_pct >= 50.0))
                g2a_pass = bool((ent_far_a <= 10.0) and (win_fpr_a <= 5.0))
                all_a_pass = bool(g1a_pass and g2a_pass)
                if all_a_pass:
                    passing_thresholds_a.append(th_val)

                # Policy B: Cascaded Three-Class Decision
                pred_b = np.zeros(n_val, dtype=int)
                mask_alert = (p_fail >= th)
                mask_crit = mask_alert & (p2 >= p1)
                mask_warn = mask_alert & (p1 > p2)
                pred_b[mask_crit] = 2
                pred_b[mask_warn] = 1
                df_eval_run["pred_b"] = pred_b

                det_eps_b = sum(1 for _, grp in episodes if (df_eval_run.loc[grp.index, "pred_b"] > 0).any())
                ep_rec_b = det_eps_b / total_episodes * 100.0

                crit_correct_b = int(((y_val == 2) & (pred_b == 2)).sum())
                crit_rec_b = crit_correct_b / n_crit * 100.0

                warn_correct_b = int(((y_val == 1) & (pred_b == 1)).sum())
                warn_rec_b = warn_correct_b / n_warn * 100.0

                norm_correct_b = int(((y_val == 0) & (pred_b == 0)).sum())
                norm_rec_b = norm_correct_b / n_norm * 100.0

                fa_ents_b = sum(1 for eid in pure_normal_eids if (df_eval_run[df_eval_run["entity_id"] == eid]["pred_b"] > 0).any())
                ent_far_b = fa_ents_b / n_pure_normal_entities * 100.0

                fp_wins_b = int((pred_b[y_val == 0] > 0).sum())
                win_fpr_b = fp_wins_b / n_norm * 100.0

                fa_jobs_b = sum(1 for jid in pure_normal_jobs if (df_eval_run[df_eval_run["parent_job_id"] == jid]["pred_b"] > 0).any())
                job_far_b = fa_jobs_b / n_pure_normal_jobs * 100.0

                macro_f1_b = float(f1_score(y_val, pred_b, average="macro", zero_division=0.0))
                bal_acc_b = float(balanced_accuracy_score(y_val, pred_b))
                acc_b = float(accuracy_score(y_val, pred_b))
                cm_b = confusion_matrix(y_val, pred_b, labels=[0, 1, 2]).tolist()

                g1b_pass = bool((ep_rec_b >= 70.0) and (crit_rec_b >= 50.0))
                g2b_pass = bool((ent_far_b <= 10.0) and (win_fpr_b <= 5.0))
                all_b_pass = bool(g1b_pass and g2b_pass)
                if all_b_pass:
                    passing_thresholds_b.append(th_val)

                row_metrics = {
                    "run_tag": run_tag,
                    "context_id": ctx_id,
                    "seq_len": seq_len,
                    "condition_id": cid,
                    "seed": seed,
                    "threshold": th_val,
                    # Policy A
                    "polA_ep_recall_pct": ep_rec_a,
                    "polA_crit_inter_pct": crit_inter_pct,
                    "polA_ent_far_pct": ent_far_a,
                    "polA_win_fpr_pct": win_fpr_a,
                    "polA_job_far_pct": job_far_a,
                    "polA_prec": bin_prec,
                    "polA_rec": bin_rec,
                    "polA_f1": bin_f1,
                    "polA_all_pass": all_a_pass,
                    # Policy B
                    "polB_macro_f1": macro_f1_b,
                    "polB_bal_acc": bal_acc_b,
                    "polB_acc": acc_b,
                    "polB_norm_rec_pct": norm_rec_b,
                    "polB_warn_rec_pct": warn_rec_b,
                    "polB_crit_rec_pct": crit_rec_b,
                    "polB_ep_recall_pct": ep_rec_b,
                    "polB_ent_far_pct": ent_far_b,
                    "polB_win_fpr_pct": win_fpr_b,
                    "polB_job_far_pct": job_far_b,
                    "polB_gate1_pass": g1b_pass,
                    "polB_gate2_pass": g2b_pass,
                    "polB_all_pass": all_b_pass,
                    "cm_00": cm_b[0][0], "cm_01": cm_b[0][1], "cm_02": cm_b[0][2],
                    "cm_10": cm_b[1][0], "cm_11": cm_b[1][1], "cm_12": cm_b[1][2],
                    "cm_20": cm_b[2][0], "cm_21": cm_b[2][1], "cm_22": cm_b[2][2]
                }
                threshold_metrics_records.append(row_metrics)
                run_threshold_rows.append(row_metrics)

            best_f1_pt = max(run_threshold_rows, key=lambda x: x["polB_macro_f1"])
            g2_pass_pts = [r for r in run_threshold_rows if r["polB_gate2_pass"]]
            best_det_when_g2 = max(g2_pass_pts, key=lambda x: (x["polB_ep_recall_pct"], x["polB_crit_rec_pct"])) if g2_pass_pts else None

            gate_summary_records.append({
                "run_tag": run_tag,
                "context_id": ctx_id,
                "context_name": ctx_name,
                "seq_len": seq_len,
                "condition_id": cid,
                "condition_name": cname,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "brier_score": brier_val,
                "nll": nll_val,
                "ece": ece_val,
                "policyA_passing_count": len(passing_thresholds_a),
                "policyB_passing_count": len(passing_thresholds_b),
                "policyB_all_pass_any": len(passing_thresholds_b) > 0,
                "max_macro_f1_th": best_f1_pt["threshold"],
                "max_macro_f1": best_f1_pt["polB_macro_f1"],
                "max_f1_ep_recall": best_f1_pt["polB_ep_recall_pct"],
                "max_f1_crit_recall": best_f1_pt["polB_crit_rec_pct"],
                "max_f1_ent_far": best_f1_pt["polB_ent_far_pct"],
                "max_f1_win_fpr": best_f1_pt["polB_win_fpr_pct"],
                "g2_best_th": best_det_when_g2["threshold"] if best_det_when_g2 else None,
                "g2_best_ep_recall": best_det_when_g2["polB_ep_recall_pct"] if best_det_when_g2 else 0.0,
                "g2_best_crit_recall": best_det_when_g2["polB_crit_rec_pct"] if best_det_when_g2 else 0.0,
                "g2_best_ent_far": best_det_when_g2["polB_ent_far_pct"] if best_det_when_g2 else 0.0,
                "g2_best_win_fpr": best_det_when_g2["polB_win_fpr_pct"] if best_det_when_g2 else 0.0,
            })

            elapsed_run = time.time() - run_start_t
            print(f"[{run_idx:2d}/{total_runs}] {run_tag:20s} | Best Ep: {best_epoch:2d} (Loss: {best_val_loss:.4f}) | Peak F1: {best_f1_pt['polB_macro_f1']:.4f} (th={best_f1_pt['threshold']:.3f}) | Passing Gates: {len(passing_thresholds_b)} | Time: {elapsed_run:.1f}s")

print(f"\nTraining & Evaluation complete! Total elapsed time: {(time.time() - start_time_all)/60:.2f} min")

# -----------------------------------------------------------------------------
# Save Artifacts
# -----------------------------------------------------------------------------
# 1. Training logs CSV
df_train_logs = pd.DataFrame(training_log_records)
train_logs_p = out_dir / "training_logs.csv"
df_train_logs.to_csv(train_logs_p, index=False)
print(f"Saved training logs to {train_logs_p}")

# 2. Validation predictions CSV
df_val_preds = val_meta.copy()
df_val_preds["y_true"] = y_val
for col_name, arr in val_predictions_dict.items():
    df_val_preds[col_name] = arr
val_preds_p = out_dir / "validation_predictions.csv"
df_val_preds.to_csv(val_preds_p, index=False)
print(f"Saved validation predictions to {val_preds_p} ({df_val_preds.shape})")

# 3. Threshold metrics CSV
df_th_metrics = pd.DataFrame(threshold_metrics_records)
th_metrics_p = out_dir / "threshold_sweep_metrics.csv"
df_th_metrics.to_csv(th_metrics_p, index=False)
print(f"Saved threshold sweep metrics to {th_metrics_p} ({len(df_th_metrics)} rows)")

# 4. Gate summary CSV
df_gate_summary = pd.DataFrame(gate_summary_records)
gate_summary_p = out_dir / "gate_summary.csv"
df_gate_summary.to_csv(gate_summary_p, index=False)
print(f"Saved gate summary to {gate_summary_p}")

# 5. Cross-seed aggregate
agg_rows = []
for (ctx_id, cid), grp in df_gate_summary.groupby(["context_id", "condition_id"]):
    cname = grp["condition_name"].iloc[0]
    seq_len = grp["seq_len"].iloc[0]
    
    mean_peak_f1 = grp["max_macro_f1"].mean()
    std_peak_f1 = grp["max_macro_f1"].std()
    
    mean_ep_rec = grp["max_f1_ep_recall"].mean()
    std_ep_rec = grp["max_f1_ep_recall"].std()
    
    mean_crit_rec = grp["max_f1_crit_recall"].mean()
    std_crit_rec = grp["max_f1_crit_recall"].std()
    
    mean_ent_far = grp["max_f1_ent_far"].mean()
    std_ent_far = grp["max_f1_ent_far"].std()
    
    mean_win_fpr = grp["max_f1_win_fpr"].mean()
    std_win_fpr = grp["max_f1_win_fpr"].std()
    
    mean_brier = grp["brier_score"].mean()
    std_brier = grp["brier_score"].std()
    
    mean_nll = grp["nll"].mean()
    std_nll = grp["nll"].std()
    
    mean_ece = grp["ece"].mean()
    std_ece = grp["ece"].std()
    
    total_passing_runs = int(grp["policyB_all_pass_any"].sum())
    
    agg_rows.append({
        "context_id": ctx_id,
        "seq_len": seq_len,
        "condition_id": cid,
        "condition_name": cname,
        "runs_count": len(grp),
        "all_gates_pass_runs": total_passing_runs,
        "mean_peak_f1": mean_peak_f1,
        "std_peak_f1": std_peak_f1,
        "mean_ep_rec_at_peak_f1": mean_ep_rec,
        "std_ep_rec_at_peak_f1": std_ep_rec,
        "mean_crit_rec_at_peak_f1": mean_crit_rec,
        "std_crit_rec_at_peak_f1": std_crit_rec,
        "mean_ent_far_at_peak_f1": mean_ent_far,
        "std_ent_far_at_peak_f1": std_ent_far,
        "mean_win_fpr_at_peak_f1": mean_win_fpr,
        "std_win_fpr_at_peak_f1": std_win_fpr,
        "mean_brier": mean_brier,
        "std_brier": std_brier,
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "mean_ece": mean_ece,
        "std_ece": std_ece,
    })

df_agg = pd.DataFrame(agg_rows)
agg_p = out_dir / "cross_seed_aggregate.csv"
df_agg.to_csv(agg_p, index=False)
print(f"Saved cross-seed aggregate summary to {agg_p}")

print("\n" + "="*80)
print("PHASE 7L AGGREGATE RESULTS SUMMARY:")
print("="*80)
for _, r in df_agg.iterrows():
    print(f"{r['context_id']:4s} ({r['condition_id']:6s}, L={r['seq_len']:2d}): Peak F1 = {r['mean_peak_f1']:.4f} +/- {r['std_peak_f1']:.4f} | EpRec = {r['mean_ep_rec_at_peak_f1']:.1f}% | CritRec = {r['mean_crit_rec_at_peak_f1']:.1f}% | EntFAR = {r['mean_ent_far_at_peak_f1']:.1f}% | WinFPR = {r['mean_win_fpr_at_peak_f1']:.1f}% | Brier = {r['mean_brier']:.4f} | ECE = {r['mean_ece']:.4f} | Passing Runs = {r['all_gates_pass_runs']}/{r['runs_count']}")
