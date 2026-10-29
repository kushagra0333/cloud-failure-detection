"""Phase 7E — Class Weighting and Threshold Calibration Development Experiment.

Objective: Determine whether neural-model failure detection improves on Candidate A
validation data when class imbalance is handled correctly via inverse-frequency weighting,
and evaluate operational thresholding on failure-risk score P(Warning) + P(Critical).

Strict Restrictions:
1. Candidate A Train (N=3,278) and Validation (N=657) ONLY.
2. ZERO access to Candidate A Test set (completely excluded from loading/evaluating).
3. ZERO access to quarantined Phase 6 confirmatory holdout.
4. Existing Phase 7B and Phase 7D artifacts remain untouched.
"""

import sys
import os
import json
import time
import hashlib
import random
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    average_precision_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.training.trainer import LSTMTrainer, compute_training_class_weights
from src.evaluation.metrics import CLASS_NAMES

CANDIDATE_A_SHA256 = "f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946"
SCALER_PHASE7C_SHA256 = "1170ef8689ec64594874f502d7b7ff5c3f195e188244d21cc8f975e31a0db5fb"

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

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

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

def compute_window_metrics_val(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    labels = [0, 1, 2]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    class_metrics = {}
    for i, c in enumerate(labels):
        cname = CLASS_NAMES[c]
        class_metrics[cname] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }

    pr_auc_metrics = {}
    if y_prob is not None and y_prob.ndim == 2:
        for c in [1, 2]:
            cname = CLASS_NAMES[c]
            y_bin = (y_true == c).astype(int)
            prob_c = y_prob[:, c] if y_prob.shape[1] > c else np.zeros(len(y_true))
            if np.all(prob_c == prob_c[0]) or len(np.unique(y_bin)) < 2:
                pr_auc_metrics[f"pr_auc_{cname.lower()}"] = float(np.mean(y_bin))
            else:
                pr_auc_metrics[f"pr_auc_{cname.lower()}"] = float(average_precision_score(y_bin, prob_c))

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "class_metrics": class_metrics,
        "warning_recall": float(recalls[1]),
        "critical_recall": float(recalls[2]),
        "confusion_matrix": cm,
        "pr_auc": pr_auc_metrics,
        "total_windows": len(y_true),
    }

def compute_episode_and_fa_metrics_val(metadata_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    df = metadata_df.copy().reset_index(drop=True)
    df["label"] = y_true
    df["y_pred"] = y_pred

    # 1. Episode Detection
    prefails = df[df["label"] > 0].copy()
    if prefails.empty:
        ep_metrics = {
            "total_episodes": 0,
            "detected_episodes": 0,
            "episode_recall_pct": 0.0,
            "lead_time_mean_sec": 0.0,
            "lead_time_mean_min": 0.0,
            "episodes_detail": []
        }
    else:
        prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
        episodes = prefails.groupby(["entity_id", "fail_timestamp"])
        total_episodes = len(episodes)
        detected_episodes = 0
        lead_times_sec = []
        ep_detail = []

        for (eid, ftime), grp in episodes:
            grp_sorted = grp.sort_values("prediction_timestamp")
            alert_windows = grp_sorted[grp_sorted["y_pred"] > 0]
            detected = not alert_windows.empty
            if detected:
                detected_episodes += 1
                earliest_alert_time = float(alert_windows["prediction_timestamp"].iloc[0])
                lead_time = float(ftime - earliest_alert_time)
                lead_times_sec.append(lead_time)
            else:
                lead_time = None

            ep_detail.append({
                "entity_id": eid,
                "fail_timestamp": ftime,
                "total_windows": len(grp),
                "detected": detected,
                "lead_time_sec": lead_time
            })

        ep_metrics = {
            "total_episodes": total_episodes,
            "detected_episodes": detected_episodes,
            "episode_recall_pct": (detected_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0,
            "lead_time_mean_sec": float(np.mean(lead_times_sec)) if lead_times_sec else 0.0,
            "lead_time_mean_min": (float(np.mean(lead_times_sec)) / 60.0) if lead_times_sec else 0.0,
            "episodes_detail": ep_detail
        }

    # 2. Window FPR
    normal_window_mask = (y_true == 0)
    n_normal_windows = int(normal_window_mask.sum())
    n_fp_windows = int((y_pred[normal_window_mask] > 0).sum())
    window_fpr_pct = (n_fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0

    # 3. Pure-Normal Entity FAR
    failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
    all_eids = sorted(df["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)
    entity_fa_count = sum(
        (df[df["entity_id"] == eid]["y_pred"] > 0).any()
        for eid in pure_normal_eids
    )
    entity_far_pct = (entity_fa_count / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0

    # 4. Pure-Normal Parent-Job FAR (Secondary Diagnostic Metric)
    if "parent_job_id" not in df.columns:
        df["parent_job_id"] = df["entity_id"].apply(lambda x: str(x).split("_")[0])
    
    failing_jobs = set(df[df["label"] > 0]["parent_job_id"].unique())
    all_jobs = sorted(df["parent_job_id"].unique())
    pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
    n_pure_normal_jobs = len(pure_normal_jobs)
    job_fa_count = sum(
        (df[df["parent_job_id"] == jid]["y_pred"] > 0).any()
        for jid in pure_normal_jobs
    )
    job_far_pct = (job_fa_count / n_pure_normal_jobs * 100.0) if n_pure_normal_jobs > 0 else 0.0

    return {
        "episode_metrics": ep_metrics,
        "window_fpr": {
            "false_positive_windows": n_fp_windows,
            "denominator_windows": n_normal_windows,
            "fpr_pct": window_fpr_pct
        },
        "entity_far": {
            "false_alarm_entities": entity_fa_count,
            "denominator_entities": n_pure_normal_entities,
            "far_pct": entity_far_pct
        },
        "parent_job_far": {
            "false_alarm_jobs": job_fa_count,
            "denominator_jobs": n_pure_normal_jobs,
            "far_pct": job_far_pct
        }
    }

def evaluate_extended_val(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    win = compute_window_metrics_val(y_true, y_pred, y_prob)
    ep_fa = compute_episode_and_fa_metrics_val(meta_df, y_true, y_pred)
    return {**win, **ep_fa}

def run_threshold_sweep(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray
) -> List[Dict[str, Any]]:
    """Sweeps failure threshold theta on failure risk score S_fail = P(Warning) + P(Critical)."""
    p_fail = probs[:, 1] + probs[:, 2]
    binary_true = (y_true > 0).astype(int)
    
    # Pre-index episodes and pure normal entities
    df = meta_df.copy().reset_index(drop=True)
    df["label"] = y_true
    prefails = df[df["label"] > 0].copy()
    prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    
    failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
    all_eids = sorted(df["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)
    
    normal_mask = (y_true == 0)
    n_normal_windows = int(normal_mask.sum())
    
    crit_mask = (y_true == 2)
    n_crit_windows = int(crit_mask.sum())
    
    results = []
    
    for theta in thresholds:
        bin_pred = (p_fail >= theta).astype(int)
        
        # Binary precision, recall, F1
        p, r, f1, _ = precision_recall_fscore_support(binary_true, bin_pred, average="binary", zero_division=0.0)
        
        # Critical recall at this threshold
        crit_rec = float(bin_pred[crit_mask].sum() / n_crit_windows) if n_crit_windows > 0 else 0.0
        
        # Window FPR
        fp_windows = int(bin_pred[normal_mask].sum())
        win_fpr = float(fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0
        
        # Entity FAR
        fa_entities = sum((bin_pred[df["entity_id"] == eid] == 1).any() for eid in pure_normal_eids)
        ent_far = float(fa_entities / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0
        
        # Episode recall
        det_eps = 0
        for (eid, ftime), grp in episodes:
            if (bin_pred[grp.index] == 1).any():
                det_eps += 1
        ep_rec = float(det_eps / total_episodes * 100.0) if total_episodes > 0 else 0.0
        
        gate1_pass = (ep_rec >= 70.0) and (crit_rec >= 0.50)
        gate2_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        
        results.append({
            "threshold": float(theta),
            "binary_precision": float(p),
            "binary_recall": float(r),
            "binary_f1": float(f1),
            "critical_recall": crit_rec,
            "episode_recall_pct": ep_rec,
            "detected_episodes": det_eps,
            "total_episodes": total_episodes,
            "window_fpr_pct": win_fpr,
            "fp_windows": fp_windows,
            "entity_far_pct": ent_far,
            "fa_entities": fa_entities,
            "gate_1_pass": bool(gate1_pass),
            "gate_2_pass": bool(gate2_pass),
            "both_pass": bool(gate1_pass and gate2_pass)
        })
        
    return results

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80)
    print("PHASE 7E: NEURAL CLASS-WEIGHTING & VALIDATION THRESHOLD EXPERIMENT")
    print("=" * 80)
    t_start = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Random Seed: 42")
    
    # Pre-execution Invariant Checks
    print("\n[Step 1] Verifying Pre-Execution Invariants and Split Checksums...")
    cand_csv = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
    assert cand_csv.exists(), "Missing Candidate A CSV"
    h_cand = compute_sha256(cand_csv)
    assert h_cand == CANDIDATE_A_SHA256, f"Corrupted Candidate A CSV: {h_cand}"
    print(f"  - Candidate A CSV verified: {h_cand[:16]}...")
    
    scaler_json = PROJECT_ROOT / "configs" / "phase7c" / "fitted_scaler_phase7c.json"
    assert scaler_json.exists(), "Missing Phase 7C scaler"
    h_scaler = compute_sha256(scaler_json)
    assert h_scaler == SCALER_PHASE7C_SHA256, f"Corrupted scaler: {h_scaler}"
    print(f"  - Preprocessing scaler verified: {h_scaler[:16]}...")
    
    # Load Train and Validation Splits ONLY
    phase7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    X_train_seq = np.load(phase7c_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(phase7c_dir / "y_train.npy").astype(np.int64)
    meta_train = pd.read_csv(phase7c_dir / "train_metadata.csv")
    
    X_val_seq = np.load(phase7c_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(phase7c_dir / "y_val.npy").astype(np.int64)
    meta_val = pd.read_csv(phase7c_dir / "val_metadata.csv")
    
    assert X_train_seq.shape == (3278, 5, 31) and len(y_train) == 3278
    assert X_val_seq.shape == (657, 5, 31) and len(y_val) == 657
    print("  - Partition Shapes Verified: Train=(3278, 5, 31), Val=(657, 5, 31)")
    print("  - STRICT PROTOCOL ENFORCEMENT: Candidate A Test split and Phase 6 Holdout NOT LOADED.")
    
    # Check Class Distribution & Compute Weights
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    train_dist = dict(zip([int(c) for c in train_classes], [int(cnt) for cnt in train_counts]))
    print(f"\n[Step 2] Training Set Class Distribution: {train_dist}")
    
    # Condition A: Unweighted [1.0, 1.0, 1.0]
    weights_A = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    
    # Condition B: Protocol v2.0 Ad-hoc Weights [1.0, 1.5, 2.0]
    weights_B = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32, device=device)
    
    # Condition C: Inverse-Frequency Balanced Weights
    # Formula: w_c = N_train / (n_classes * N_train_c)
    raw_w_C = compute_training_class_weights(y_train).to(device)
    # Unit-normalized so w_0 = 1.0:
    weights_C = (raw_w_C / raw_w_C[0]).to(device)
    
    print("Class Weight Configurations:")
    print(f"  - Condition A (Unweighted):         {weights_A.tolist()}")
    print(f"  - Condition B (Ad-hoc v2.0):        {weights_B.tolist()}")
    print(f"  - Condition C (Raw Inverse-Freq):   {raw_w_C.tolist()}")
    print(f"  - Condition C (Unit-Normalized w0): {weights_C.tolist()}")
    
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_phase7e"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val)),
        batch_size=64,
        shuffle=False
    )
    
    conditions = [
        ("condA_unweighted", weights_A, "Condition A: Unweighted [1.0, 1.0, 1.0]"),
        ("condB_adhoc", weights_B, "Condition B: Protocol v2.0 Ad-hoc [1.0, 1.5, 2.0]"),
        ("condC_inverse_freq", weights_C, "Condition C: Inverse-Frequency Balanced [1.0, 11.46, 11.80]"),
    ]
    
    manifest = {
        "experiment": "Phase 7E Class Weighting & Validation Threshold Experiment",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits_used": {
            "train": "data/processed/phase7c/X_train.npy (N=3,278)",
            "val": "data/processed/phase7c/X_val.npy (N=657)",
            "test_status": "STRICTLY_BLINDED_AND_EXCLUDED"
        },
        "class_distributions": {
            "train": train_dist,
            "val": dict(zip([int(c) for c in np.unique(y_val)], [int(cnt) for cnt in np.bincount(y_val)]))
        },
        "weight_configurations": {
            "condA": weights_A.tolist(),
            "condB": weights_B.tolist(),
            "condC_raw": raw_w_C.tolist(),
            "condC_unit_normalized": weights_C.tolist()
        },
        "models": {}
    }
    
    val_predictions = {}
    val_probabilities = {}
    val_comparison_metrics = {}
    threshold_tuning_results = {}
    
    # -------------------------------------------------------------------------
    # STAGE 1: Training under Conditions A, B, C
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING TEMPORAL LSTM ACROSS THREE WEIGHTING CONDITIONS")
    print("=" * 80)
    
    for cond_key, w_tensor, cond_desc in conditions:
        model_id = f"lstm_{cond_key}"
        print(f"\n>>> Training {model_id} ({cond_desc})...")
        t0 = time.time()
        set_seed(42)
        
        lstm_model = TemporalLSTMClassifier(
            input_dim=31,
            hidden_dim=64,
            num_layers=2,
            num_classes=3,
            dropout=0.2
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )
        ckpt_path = ckpt_dir / f"{model_id}.pt"
        
        trainer = LSTMTrainer(
            model=lstm_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            monitor_metric="val_macro_f1",
            metric_mode="max"
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=35,
            patience=10,
            checkpoint_path=ckpt_path
        )
        t_dur = time.time() - t0
        print(f"  Finished in {t_dur:.2f}s | Best Epoch: {history['best_epoch']} | Best Val Macro-F1: {history['best_metric']:.4f}")
        
        # Load best checkpoint and predict
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lstm_model.load_state_dict(ckpt["model_state_dict"])
        lstm_model.eval()
        
        with torch.no_grad():
            logits = lstm_model(torch.from_numpy(X_val_seq).to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
        val_probabilities[model_id] = probs
        val_predictions[model_id] = preds
        
        metrics = evaluate_extended_val(meta_val, y_val, preds, probs)
        val_comparison_metrics[model_id] = metrics
        
        manifest["models"][model_id] = {
            "architecture": "TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, dropout=0.2)",
            "condition": cond_desc,
            "weights": w_tensor.tolist(),
            "best_epoch": history["best_epoch"],
            "best_val_macro_f1": float(history["best_metric"]),
            "training_duration_seconds": float(t_dur),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": compute_sha256(ckpt_path),
            "epoch_history": history["history"]
        }
        
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING COMPACT TRANSFORMER ACROSS THREE WEIGHTING CONDITIONS")
    print("=" * 80)
    
    for cond_key, w_tensor, cond_desc in conditions:
        model_id = f"transformer_{cond_key}"
        print(f"\n>>> Training {model_id} ({cond_desc})...")
        t0 = time.time()
        set_seed(42)
        
        tf_model = CompactTemporalTransformer(
            input_dim=31,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            num_classes=3,
            dropout=0.1,
            max_seq_len=5
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = torch.optim.AdamW(tf_model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )
        ckpt_path = ckpt_dir / f"{model_id}.pt"
        
        trainer = LSTMTrainer(
            model=tf_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            monitor_metric="val_macro_f1",
            metric_mode="max"
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=35,
            patience=10,
            checkpoint_path=ckpt_path
        )
        t_dur = time.time() - t0
        print(f"  Finished in {t_dur:.2f}s | Best Epoch: {history['best_epoch']} | Best Val Macro-F1: {history['best_metric']:.4f}")
        
        # Load best checkpoint and predict
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tf_model.load_state_dict(ckpt["model_state_dict"])
        tf_model.eval()
        
        with torch.no_grad():
            logits = tf_model(torch.from_numpy(X_val_seq).to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
        val_probabilities[model_id] = probs
        val_predictions[model_id] = preds
        
        metrics = evaluate_extended_val(meta_val, y_val, preds, probs)
        val_comparison_metrics[model_id] = metrics
        
        manifest["models"][model_id] = {
            "architecture": "CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, max_len=5)",
            "condition": cond_desc,
            "weights": w_tensor.tolist(),
            "best_epoch": history["best_epoch"],
            "best_val_macro_f1": float(history["best_metric"]),
            "training_duration_seconds": float(t_dur),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": compute_sha256(ckpt_path),
            "epoch_history": history["history"]
        }
        
    # -------------------------------------------------------------------------
    # STAGE 3: Comparative Validation Analysis (Argmax Decision Rule)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 3: VALIDATION PERFORMANCE COMPARISON UNDER STANDARD ARGMAX DECISION RULE")
    print("=" * 115)
    print(f"{'Model Identifier':<30} {'Macro-F1':<10} {'Acc':<8} {'BalAcc':<8} {'Ep-Rec %':<10} {'Crit-Rec':<10} {'Entity-FAR %':<14} {'Win-FPR %':<10} {'Gate 1':<8} {'Gate 2':<8}")
    print("-" * 115)
    
    for m_id, m_metrics in val_comparison_metrics.items():
        ep_rec = m_metrics["episode_metrics"]["episode_recall_pct"]
        crit_rec = m_metrics["critical_recall"]
        ent_far = m_metrics["entity_far"]["far_pct"]
        win_fpr = m_metrics["window_fpr"]["fpr_pct"]
        
        gate1_pass = (ep_rec >= 70.0) and (crit_rec >= 0.50)
        gate2_pass = (ent_far <= 10.0) and (win_fpr <= 5.0)
        
        g1_str = "PASS" if gate1_pass else "FAIL"
        g2_str = "PASS" if gate2_pass else "FAIL"
        
        print(f"{m_id:<30} {m_metrics['macro_f1']:<10.4f} {m_metrics['accuracy']:<8.4f} {m_metrics['balanced_accuracy']:<8.4f} {ep_rec:<10.2f} {crit_rec:<10.4f} {ent_far:<14.2f} {win_fpr:<10.2f} {g1_str:<8} {g2_str:<8}")
        
    print("-" * 115)
    
    # -------------------------------------------------------------------------
    # STAGE 4: Operational Failure-Risk Thresholding Sweep P(Warning) + P(Critical) >= theta
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 4: OPERATIONAL FAILURE-RISK THRESHOLD SWEEP (P(Warning) + P(Critical) >= theta)")
    print("=" * 115)
    
    thresholds = np.linspace(0.05, 0.95, 91)
    
    for m_id in val_comparison_metrics:
        probs = val_probabilities[m_id]
        sweep_res = run_threshold_sweep(meta_val, y_val, probs, thresholds)
        
        # Analyze sweep:
        # 1. Best Binary F1
        best_f1_entry = max(sweep_res, key=lambda x: x["binary_f1"])
        
        # 2. Gate-passing thresholds (both Gate 1 and Gate 2)
        both_pass_entries = [x for x in sweep_res if x["both_pass"]]
        
        # 3. Gate 2 compliant thresholds (Entity FAR <= 10%, Window FPR <= 5%)
        gate2_entries = [x for x in sweep_res if x["gate_2_pass"]]
        
        threshold_tuning_results[m_id] = {
            "model_id": m_id,
            "best_f1_threshold": best_f1_entry,
            "both_gates_passed_count": len(both_pass_entries),
            "both_gates_passed_thresholds": both_pass_entries,
            "gate_2_compliant_count": len(gate2_entries),
            "gate_2_best_recall_entry": max(gate2_entries, key=lambda x: x["episode_recall_pct"]) if gate2_entries else None,
            "full_sweep": sweep_res
        }
        
        print(f"\nModel: {m_id}")
        print(f"  Best Binary F1: theta={best_f1_entry['threshold']:.2f} | F1={best_f1_entry['binary_f1']:.4f} | EpRec={best_f1_entry['episode_recall_pct']:.1f}% | CritRec={best_f1_entry['critical_recall']:.4f} | EntFAR={best_f1_entry['entity_far_pct']:.1f}% | WinFPR={best_f1_entry['window_fpr_pct']:.1f}%")
        if both_pass_entries:
            print(f"  >>> OPERATIONAL GATE SUCCESS: {len(both_pass_entries)} threshold(s) passed both gates!")
            for bentry in both_pass_entries[:3]:
                print(f"      theta={bentry['threshold']:.2f}: EpRec={bentry['episode_recall_pct']:.1f}%, CritRec={bentry['critical_recall']:.4f}, EntFAR={bentry['entity_far_pct']:.1f}%, WinFPR={bentry['window_fpr_pct']:.1f}%")
        else:
            print(f"  >>> NO THRESHOLD PASSED BOTH GATES.")
            if gate2_entries:
                best_g2 = max(gate2_entries, key=lambda x: x["episode_recall_pct"])
                print(f"      Max Episode Recall under Gate 2 constraint: {best_g2['episode_recall_pct']:.1f}% (at theta={best_g2['threshold']:.2f}, CritRec={best_g2['critical_recall']:.4f}, EntFAR={best_g2['entity_far_pct']:.1f}%, WinFPR={best_g2['window_fpr_pct']:.1f}%)")
            else:
                print("      No threshold achieved Gate 2 compliance.")
                
    # -------------------------------------------------------------------------
    # STAGE 5: Protocol Amendment & Deliverables Serialization
    # -------------------------------------------------------------------------
    print("\n[Step 5] Serializing all Phase 7E deliverables...")
    res_dir = PROJECT_ROOT / "results" / "phase7e"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "training_manifest.json", "w") as f:
        json.dump(to_serializable(manifest), f, indent=2)
        
    with open(res_dir / "validation_comparison_metrics.json", "w") as f:
        json.dump(to_serializable(val_comparison_metrics), f, indent=2)
        
    with open(res_dir / "threshold_tuning_validation.json", "w") as f:
        json.dump(to_serializable(threshold_tuning_results), f, indent=2)
        
    # Serialize validation predictions CSV
    pred_val_df = meta_val.copy()
    pred_val_df["y_true"] = y_val
    for m_id in val_predictions:
        pred_val_df[f"y_pred_{m_id}"] = val_predictions[m_id]
        for c in [0, 1, 2]:
            pred_val_df[f"prob_{m_id}_c{c}"] = val_probabilities[m_id][:, c]
        pred_val_df[f"p_fail_{m_id}"] = val_probabilities[m_id][:, 1] + val_probabilities[m_id][:, 2]
    pred_val_df.to_csv(res_dir / "validation_predictions.csv", index=False)
    print(f"  - Serialized validation predictions: {res_dir / 'validation_predictions.csv'}")
    
    # Protocol Amendment Document
    amendment = {
        "amendment_id": "PROTOCOL-AMENDMENT-PHASE7E-01",
        "title": "Protocol Amendment: Inverse-Frequency Loss Weighting and Validation-Calibrated Alert Thresholding",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rationale": "Audit of Phase 7D revealed that neural networks suffered majority-class mode collapse under hardcoded weights [1.0, 1.5, 2.0] and uniform argmax. This amendment formally pre-registers inverse-frequency loss weighting w = [1.0, 11.46, 11.80] and validation-only calibration of decision threshold theta on failure-risk score S_fail = P(Warning) + P(Critical).",
        "scope": "Candidate A Training and Validation splits ONLY. Test split remains strictly blinded and unaccessed for confirmation.",
        "threshold_tuning_policy": "Threshold theta is selected strictly on Candidate A Validation split to maximize Episode Recall subject to Gate 2 (Entity FAR <= 10%, Window FPR <= 5%). The selected theta is permanently frozen prior to any future independent test scoring."
    }
    with open(res_dir / "phase7e_protocol_amendment.json", "w") as f:
        json.dump(to_serializable(amendment), f, indent=2)
        
    print(f"  - Serialized protocol amendment: {res_dir / 'phase7e_protocol_amendment.json'}")
    print("\n" + "=" * 80)
    print(f"PHASE 7E EXPERIMENT COMPLETE | Total Duration: {time.time() - t_start:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
