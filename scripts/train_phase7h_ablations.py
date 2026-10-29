"""Phase 7H: Controlled Normalization Ablations Training and Evaluation Pipeline.

Models:
  1. Ablation A: Existing Phase 7G Hierarchical Baseline (unfloored, eps=1e-5)
  2. Ablation B: Hierarchical Causal Baseline with Variance Floor (sigma_floor = 0.05)
  3. Ablation C: Robust Causal Scaling with Median, MAD (mad_floor = 0.05), and Soft Tanh Bounding (clip=5.0)

Evaluates:
  - Macro-F1, Balanced Accuracy, Accuracy, Per-class Precision, Recall, F1, Confusion Matrix
  - Episode Recall, Critical Recall, Warning Recall, Mean Lead Time
  - Pure-Normal Entity FAR, Normal-Window FPR, Parent-Job FAR & Affected Jobs
  - Stratification by baseline source & history depth
  - Probability distributions on true Critical windows and pure-normal windows
  - Bootstrap 95% Confidence Intervals
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import CompactTemporalTransformer
from src.training.trainer import compute_training_class_weights

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

def to_serializable(obj: Any) -> Any:
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

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int = 40,
    patience: int = 10,
    checkpoint_path: Path = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, int, Dict[str, Any]]:
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    return model, best_epoch, {"history": history, "best_loss": best_loss}

def evaluate_validation(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    labels = [0, 1, 2]
    prec, rec, f1, sup = precision_recall_fscore_support(y_val, preds, labels=labels, zero_division=0.0)
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_val, preds))
    bal_acc = float(balanced_accuracy_score(y_val, preds))
    cm = confusion_matrix(y_val, preds, labels=labels).tolist()

    df = meta_df.copy().reset_index(drop=True)
    df["y_true"] = y_val
    df["y_pred"] = preds
    df["p_c0"] = probs[:, 0]
    df["p_c1"] = probs[:, 1]
    df["p_c2"] = probs[:, 2]
    df["p_fail"] = probs[:, 1] + probs[:, 2]
    df["baseline_source"] = routing_df["baseline_source"]
    df["history_depth"] = routing_df["history_depth_windows"]
    df["fail_timestamp"] = (df["prediction_timestamp"] + df["time_to_failure"]).round(1)

    # 1. Episode-level recall
    prefails = df[df["y_true"] > 0].copy()
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    detected_episodes = sum(1 for _, grp in episodes if (grp["y_pred"] > 0).any())
    episode_recall_pct = (detected_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0

    lead_times = []
    for _, grp in episodes:
        det = grp[grp["y_pred"] > 0]
        if not det.empty:
            lead_times.append(grp["fail_timestamp"].iloc[0] - det["prediction_timestamp"].iloc[0])
    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    # 2. Pure-normal Entity FAR
    failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
    all_eids = sorted(df["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)

    fa_entities = sum(1 for eid in pure_normal_eids if (df[df["entity_id"] == eid]["y_pred"] > 0).any())
    entity_far_pct = (fa_entities / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0

    # 3. Normal-window FPR
    normal_mask = (y_val == 0)
    n_normal_windows = int(normal_mask.sum())
    fp_windows = int((preds[normal_mask] > 0).sum())
    window_fpr_pct = (fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0

    # 4. Parent-Job FAR & Affected Jobs
    failing_jobs = set(df[df["y_true"] > 0]["parent_job_id"].unique())
    all_jobs = sorted(df["parent_job_id"].unique())
    pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
    n_pure_normal_jobs = len(pure_normal_jobs)

    job_fa_counts = {}
    for jid in all_jobs:
        job_df = df[df["parent_job_id"] == jid]
        job_fp_win = int((job_df[job_df["y_true"] == 0]["y_pred"] > 0).sum())
        job_total_norm = int((job_df["y_true"] == 0).sum())
        job_fa_counts[str(jid)] = {
            "pure_normal_job": bool(jid in pure_normal_jobs),
            "false_alarm_windows": job_fp_win,
            "normal_windows": job_total_norm,
            "window_fpr_pct": float(job_fp_win / job_total_norm * 100.0) if job_total_norm > 0 else 0.0,
            "entity_count": int(job_df["entity_id"].nunique()),
            "fa_entities": int(sum(1 for eid in job_df["entity_id"].unique() if (job_df[job_df["entity_id"] == eid]["y_pred"] > 0).any()))
        }
    pure_normal_jobs_with_fa = sum(1 for jid in pure_normal_jobs if job_fa_counts[str(jid)]["false_alarm_windows"] > 0)
    job_far_pct = (pure_normal_jobs_with_fa / n_pure_normal_jobs * 100.0) if n_pure_normal_jobs > 0 else 0.0

    # 5. Stratification by baseline source
    source_strat = {}
    for src in df["baseline_source"].unique():
        sdf = df[df["baseline_source"] == src]
        s_norm = (sdf["y_true"] == 0)
        s_crit = (sdf["y_true"] == 2)
        s_warn = (sdf["y_true"] == 1)
        
        s_fp = int((sdf.loc[s_norm, "y_pred"] > 0).sum())
        s_tot_norm = int(s_norm.sum())
        s_win_fpr = float(s_fp / s_tot_norm * 100.0) if s_tot_norm > 0 else 0.0
        
        s_crit_det = int((sdf.loc[s_crit, "y_pred"] == 2).sum())
        s_tot_crit = int(s_crit.sum())
        s_crit_rec = float(s_crit_det / s_tot_crit) if s_tot_crit > 0 else 0.0
        
        s_warn_det = int((sdf.loc[s_warn, "y_pred"] == 1).sum())
        s_tot_warn = int(s_warn.sum())
        s_warn_rec = float(s_warn_det / s_tot_warn) if s_tot_warn > 0 else 0.0
        
        source_strat[src] = {
            "total_windows": len(sdf),
            "normal_windows": s_tot_norm,
            "fp_windows": s_fp,
            "window_fpr_pct": s_win_fpr,
            "warning_windows": s_tot_warn,
            "warning_recall": s_warn_rec,
            "critical_windows": s_tot_crit,
            "critical_recall": s_crit_rec
        }

    # 6. Stratification by history depth brackets
    depth_bins = [(-1, 0, "K=0 (No History)"), (0, 2, "K=1-2 (Minimal)"), (2, 5, "K=3-5 (Moderate)"), (5, 1000, "K>5 (Rich)")]
    depth_strat = {}
    for low, high, bname in depth_bins:
        bdf = df[(df["history_depth"] > low) & (df["history_depth"] <= high)]
        b_norm = (bdf["y_true"] == 0)
        b_crit = (bdf["y_true"] == 2)
        b_warn = (bdf["y_true"] == 1)
        
        b_fp = int((bdf.loc[b_norm, "y_pred"] > 0).sum())
        b_tot_norm = int(b_norm.sum())
        b_fpr = float(b_fp / b_tot_norm * 100.0) if b_tot_norm > 0 else 0.0
        
        b_crit_det = int((bdf.loc[b_crit, "y_pred"] == 2).sum())
        b_tot_crit = int(b_crit.sum())
        b_crit_rec = float(b_crit_det / b_tot_crit) if b_tot_crit > 0 else 0.0
        
        depth_strat[bname] = {
            "total_windows": len(bdf),
            "normal_windows": b_tot_norm,
            "fp_windows": b_fp,
            "window_fpr_pct": b_fpr,
            "critical_windows": b_tot_crit,
            "critical_recall": b_crit_rec
        }

    # 7. Probability distributions on true Critical vs Pure-Normal windows
    crit_df = df[df["y_true"] == 2]
    pure_normal_df = df[df["entity_id"].isin(pure_normal_eids) & (df["y_true"] == 0)]
    
    prob_dist = {
        "critical_windows": {
            "count": len(crit_df),
            "p_normal_mean": float(crit_df["p_c0"].mean()),
            "p_normal_median": float(crit_df["p_c0"].median()),
            "p_warning_mean": float(crit_df["p_c1"].mean()),
            "p_warning_median": float(crit_df["p_c1"].median()),
            "p_critical_mean": float(crit_df["p_c2"].mean()),
            "p_critical_median": float(crit_df["p_c2"].median())
        },
        "pure_normal_windows": {
            "count": len(pure_normal_df),
            "p_normal_mean": float(pure_normal_df["p_c0"].mean()),
            "p_normal_median": float(pure_normal_df["p_c0"].median()),
            "p_warning_mean": float(pure_normal_df["p_c1"].mean()),
            "p_warning_median": float(pure_normal_df["p_c1"].median()),
            "p_critical_mean": float(pure_normal_df["p_c2"].mean()),
            "p_critical_median": float(pure_normal_df["p_c2"].median())
        }
    }

    # 8. Fast Cluster Bootstrap (pre-indexed by entity)
    np.random.seed(42)
    B = 1000
    boot_ep_recs = []
    boot_ent_fars = []
    boot_win_fprs = []
    
    ent_dfs = {e: df[df["entity_id"] == e] for e in all_eids}
    unique_eids = np.array(all_eids)
    
    for _ in range(B):
        sample_eids = np.random.choice(unique_eids, size=len(unique_eids), replace=True)
        boot_dfs = [ent_dfs[e].assign(inst_id=i) for i, e in enumerate(sample_eids)]
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        
        # Win FPR
        b_norm = (boot_df["y_true"] == 0)
        if b_norm.sum() > 0:
            boot_win_fprs.append(float((boot_df.loc[b_norm, "y_pred"] > 0).sum() / b_norm.sum() * 100.0))
            
        # Ent FAR
        b_pure = boot_df[boot_df["entity_id"].isin(pure_normal_eids)]
        if not b_pure.empty:
            b_fa = sum(1 for _, grp in b_pure.groupby("inst_id") if (grp["y_pred"] > 0).any())
            boot_ent_fars.append(float(b_fa / b_pure["inst_id"].nunique() * 100.0))
            
        # Ep Rec
        b_pre = boot_df[boot_df["y_true"] > 0]
        if not b_pre.empty:
            b_episodes = b_pre.groupby(["inst_id", "fail_timestamp"])
            b_det = sum(1 for _, grp in b_episodes if (grp["y_pred"] > 0).any())
            boot_ep_recs.append(float(b_det / len(b_episodes) * 100.0))

    ci_ep_rec = [float(np.percentile(boot_ep_recs, 2.5)), float(np.percentile(boot_ep_recs, 97.5))] if boot_ep_recs else [0.0, 0.0]
    ci_ent_far = [float(np.percentile(boot_ent_fars, 2.5)), float(np.percentile(boot_ent_fars, 97.5))] if boot_ent_fars else [0.0, 0.0]
    ci_win_fpr = [float(np.percentile(boot_win_fprs, 2.5)), float(np.percentile(boot_win_fprs, 97.5))] if boot_win_fprs else [0.0, 0.0]

    gate1_pass = (episode_recall_pct >= 70.0) and (rec[2] >= 0.50)
    gate2_pass = (entity_far_pct <= 10.0) and (window_fpr_pct <= 5.0)
    both_pass = gate1_pass and gate2_pass

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1],
        "support": [int(s) for s in sup],
        "confusion_matrix": cm,
        "warning_recall": float(rec[1]),
        "critical_recall": float(rec[2]),
        "episode_metrics": {
            "total_episodes": total_episodes,
            "detected_episodes": detected_episodes,
            "episode_recall_pct": episode_recall_pct,
            "episode_recall_ci95": ci_ep_rec,
            "lead_time_mean_sec": mean_lead_time
        },
        "false_alarm_metrics": {
            "pure_normal_entities": n_pure_normal_entities,
            "fa_entities": fa_entities,
            "entity_far_pct": entity_far_pct,
            "entity_far_ci95": ci_ent_far,
            "normal_windows": n_normal_windows,
            "fp_windows": fp_windows,
            "window_fpr_pct": window_fpr_pct,
            "window_fpr_ci95": ci_win_fpr,
            "pure_normal_jobs": n_pure_normal_jobs,
            "fa_jobs": pure_normal_jobs_with_fa,
            "job_far_pct": job_far_pct,
            "job_breakdown": job_fa_counts
        },
        "stratification_by_baseline_source": source_strat,
        "stratification_by_history_depth": depth_strat,
        "probability_distributions": prob_dist,
        "operational_gates": {
            "gate_1_pass": bool(gate1_pass),
            "gate_2_pass": bool(gate2_pass),
            "both_gates_pass": bool(both_pass)
        },
        "probabilities": probs,
        "predictions": preds
    }

def main():
    print("=" * 80)
    print("PHASE 7H: CONTROLLED NORMALIZATION ABLATIONS")
    print("=" * 80)
    
    device = torch.device("cpu")
    set_seed(42)
    
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_phase7h"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = PROJECT_ROOT / "results" / "phase7h"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    y_train = np.load("data/processed/phase7c/y_train.npy").astype(np.int64)
    y_val = np.load("data/processed/phase7c/y_val.npy").astype(np.int64)
    meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
    meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    
    raw_w = compute_training_class_weights(y_train).to(device)
    weights_C = (raw_w / raw_w[0]).to(device)
    print(f"Training-only inverse-frequency weights: {weights_C.tolist()}")
    
    ablations = [
        {
            "id": "transformer_hierarchical_baseline",
            "name": "Ablation A: Existing Hierarchical Baseline (unfloored)",
            "train_X": "results/phase7g/X_train_hierarchical.npy",
            "val_X": "results/phase7g/X_val_hierarchical.npy",
            "routing_val": "results/phase7g/routing_val_hierarchical.csv",
            "ckpt": "models/checkpoints_phase7g/transformer_hierarchical.pt",
            "pretrained": True
        },
        {
            "id": "transformer_hierarchical_floored",
            "name": "Ablation B: Hierarchical with Variance Floor (sigma_floor=0.05)",
            "train_X": "results/phase7h/X_train_hierarchical_floored.npy",
            "val_X": "results/phase7h/X_val_hierarchical_floored.npy",
            "routing_val": "results/phase7h/routing_val_hierarchical_floored.csv",
            "ckpt": ckpt_dir / "transformer_hierarchical_floored.pt",
            "pretrained": False
        },
        {
            "id": "transformer_hierarchical_robust",
            "name": "Ablation C: Robust Scaling (Median, MAD, Soft Tanh)",
            "train_X": "results/phase7h/X_train_hierarchical_robust.npy",
            "val_X": "results/phase7h/X_val_hierarchical_robust.npy",
            "routing_val": "results/phase7h/routing_val_hierarchical_robust.csv",
            "ckpt": ckpt_dir / "transformer_hierarchical_robust.pt",
            "pretrained": False
        }
    ]
    
    ablation_results = {}
    preds_df = meta_val[["entity_id", "prediction_timestamp", "label", "time_to_failure", "parent_job_id"]].copy()
    preds_df["y_true"] = y_val
    
    for ab in ablations:
        ab_id = ab["id"]
        ab_name = ab["name"]
        print(f"\n{'='*70}")
        print(f"EVALUATING: {ab_name}")
        print(f"{'='*70}")
        
        X_val_arr = np.load(ab["val_X"]).astype(np.float32)
        routing_val = pd.read_csv(ab["routing_val"])
        
        model = CompactTemporalTransformer(
            input_dim=31,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            num_classes=3,
            dropout=0.1,
            max_seq_len=5
        ).to(device)
        
        if ab["pretrained"]:
            print(f"  Loading pre-trained checkpoint from {ab['ckpt']}...")
            ckpt = torch.load(ab["ckpt"], map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            best_epoch = ckpt.get("epoch", 2)
            train_time = 0.0
        else:
            print(f"  Training {ab_name}...")
            t0 = time.time()
            set_seed(42)
            X_train_arr = np.load(ab["train_X"]).astype(np.float32)
            
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train_arr), torch.from_numpy(y_train)),
                batch_size=64,
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_arr), torch.from_numpy(y_val)),
                batch_size=64,
                shuffle=False
            )
            
            criterion = nn.CrossEntropyLoss(weight=weights_C)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            
            model, best_epoch, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=40,
                patience=10,
                checkpoint_path=ab["ckpt"],
                device=device
            )
            train_time = time.time() - t0
            print(f"  Training finished in {train_time:.1f}s (Best Epoch: {best_epoch})")
            
        eval_res = evaluate_validation(
            model=model,
            X_val=X_val_arr,
            y_val=y_val,
            meta_df=meta_val,
            routing_df=routing_val,
            device=device
        )
        
        probs = eval_res["probabilities"]
        preds = eval_res["predictions"]
        preds_df[f"y_pred_{ab_id}"] = preds
        preds_df[f"prob_{ab_id}_c0"] = probs[:, 0]
        preds_df[f"prob_{ab_id}_c1"] = probs[:, 1]
        preds_df[f"prob_{ab_id}_c2"] = probs[:, 2]
        preds_df[f"p_fail_{ab_id}"] = probs[:, 1] + probs[:, 2]
        
        clean_eval = {k: v for k, v in eval_res.items() if k not in ["probabilities", "predictions"]}
        ablation_results[ab_id] = {
            "name": ab_name,
            "checkpoint_path": str(ab["ckpt"]),
            "best_epoch": best_epoch,
            "training_time_sec": float(train_time),
            "metrics": clean_eval
        }
        
        m = eval_res
        g1 = "PASS" if m["operational_gates"]["gate_1_pass"] else "FAIL"
        g2 = "PASS" if m["operational_gates"]["gate_2_pass"] else "FAIL"
        both = "PASS" if m["operational_gates"]["both_gates_pass"] else "FAIL"
        print(f"  Macro-F1: {m['macro_f1']:.4f} | BalAcc: {m['balanced_accuracy']:.4f}")
        print(f"  Episode Recall: {m['episode_metrics']['episode_recall_pct']:.2f}% (CI: {m['episode_metrics']['episode_recall_ci95']})")
        print(f"  Critical Recall: {m['critical_recall']:.4f} | Warning Recall: {m['warning_recall']:.4f}")
        print(f"  Entity FAR: {m['false_alarm_metrics']['entity_far_pct']:.2f}% (CI: {m['false_alarm_metrics']['entity_far_ci95']})")
        print(f"  Window FPR: {m['false_alarm_metrics']['window_fpr_pct']:.2f}% (CI: {m['false_alarm_metrics']['window_fpr_ci95']})")
        print(f"  Parent-Job FAR: {m['false_alarm_metrics']['job_far_pct']:.2f}%")
        print(f"  Operational Gates: Gate 1={g1}, Gate 2={g2} => Overall={both}")
        
    preds_path = res_dir / "validation_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved validation predictions to {preds_path}")
    
    results_path = res_dir / "phase7h_ablation_metrics.json"
    with open(results_path, "w") as f:
        json.dump(to_serializable(ablation_results), f, indent=2)
    print(f"Saved ablation metrics to {results_path}")
    
    print("\n" + "=" * 125)
    print("PHASE 7H ABLATION COMPARISON MATRIX")
    print("=" * 125)
    print(f"{'Ablation':<45} {'Macro-F1':<10} {'BalAcc':<8} {'Ep-Rec %':<10} {'Crit-Rec':<10} {'Entity-FAR %':<14} {'Win-FPR %':<10} {'Gate 1':<8} {'Gate 2':<8}")
    print("-" * 125)
    for ab in ablations:
        m = ablation_results[ab["id"]]["metrics"]
        g1 = "PASS" if m["operational_gates"]["gate_1_pass"] else "FAIL"
        g2 = "PASS" if m["operational_gates"]["gate_2_pass"] else "FAIL"
        print(f"{ab['name']:<45} {m['macro_f1']:<10.4f} {m['balanced_accuracy']:<8.4f} {m['episode_metrics']['episode_recall_pct']:<10.2f} {m['critical_recall']:<10.4f} {m['false_alarm_metrics']['entity_far_pct']:<14.2f} {m['false_alarm_metrics']['window_fpr_pct']:<10.2f} {g1:<8} {g2:<8}")
    print("=" * 125)

if __name__ == "__main__":
    main()
