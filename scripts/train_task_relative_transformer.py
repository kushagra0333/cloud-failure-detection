"""Phase 7F Step 3 — Task-Relative Normalization Experiment on Compact Temporal Transformer.

Baseline: Phase 7E Condition C (Original cluster-scaler preprocessing, inverse-frequency weighted cross-entropy).
Experimental Model: Compact Temporal Transformer trained on causally task-relative normalized telemetry with minimum history fallback.
Strict Invariants:
1. Candidate A Train and Validation ONLY.
2. ZERO access to Test split or Confirmatory Holdout.
3. Loss function, weights, seed, architecture, and hyperparameters identical to baseline.
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import CompactTemporalTransformer
from src.training.trainer import LSTMTrainer, compute_training_class_weights
from src.evaluation.metrics import CLASS_NAMES

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

def compute_metrics_val(metadata_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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

    # Episode Metrics
    df = metadata_df.copy().reset_index(drop=True)
    df["label"] = y_true
    df["y_pred"] = y_pred

    prefails = df[df["label"] > 0].copy()
    prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    detected_episodes = 0
    lead_times_sec = []

    for (eid, ftime), grp in episodes:
        alert_windows = grp[grp["y_pred"] > 0]
        if not alert_windows.empty:
            detected_episodes += 1
            earliest_alert_time = float(alert_windows.sort_values("prediction_timestamp")["prediction_timestamp"].iloc[0])
            lead_times_sec.append(float(ftime - earliest_alert_time))

    ep_metrics = {
        "total_episodes": total_episodes,
        "detected_episodes": detected_episodes,
        "episode_recall_pct": (detected_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0,
        "lead_time_mean_sec": float(np.mean(lead_times_sec)) if lead_times_sec else 0.0,
    }

    # Window FPR
    normal_mask = (y_true == 0)
    n_normal_windows = int(normal_mask.sum())
    n_fp_windows = int((y_pred[normal_mask] > 0).sum())
    window_fpr_pct = (n_fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0

    # Entity FAR
    failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
    all_eids = sorted(df["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)
    entity_fa_count = sum(
        (df[df["entity_id"] == eid]["y_pred"] > 0).any()
        for eid in pure_normal_eids
    )
    entity_far_pct = (entity_fa_count / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0

    # Parent-Job FAR
    if "parent_job_id" not in df.columns:
        df["parent_job_id"] = df["entity_id"].apply(lambda x: str(x).split("_")[0])
    failing_jobs = set(df[df["label"] > 0]["parent_job_id"].unique())
    all_jobs = sorted(df["parent_job_id"].unique())
    pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
    n_pure_normal_jobs = len(pure_normal_jobs)
    job_fa_count = sum((df[df["parent_job_id"] == jid]["y_pred"] > 0).any() for jid in pure_normal_jobs)
    job_far_pct = (job_fa_count / n_pure_normal_jobs * 100.0) if n_pure_normal_jobs > 0 else 0.0

    gate1_pass = (ep_metrics["episode_recall_pct"] >= 70.0) and (recalls[2] >= 0.50)
    gate2_pass = (entity_far_pct <= 10.0) and (window_fpr_pct <= 5.0)

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "class_metrics": class_metrics,
        "warning_recall": float(recalls[1]),
        "critical_recall": float(recalls[2]),
        "confusion_matrix": cm,
        "episode_recall_pct": ep_metrics["episode_recall_pct"],
        "detected_episodes": detected_episodes,
        "total_episodes": total_episodes,
        "lead_time_mean_sec": ep_metrics["lead_time_mean_sec"],
        "window_fpr_pct": window_fpr_pct,
        "fp_windows": n_fp_windows,
        "normal_windows": n_normal_windows,
        "entity_far_pct": entity_far_pct,
        "fa_entities": entity_fa_count,
        "pure_normal_entities": n_pure_normal_entities,
        "parent_job_far_pct": job_far_pct,
        "fa_jobs": job_fa_count,
        "pure_normal_jobs": n_pure_normal_jobs,
        "gate_1_pass": bool(gate1_pass),
        "gate_2_pass": bool(gate2_pass),
        "both_gates_pass": bool(gate1_pass and gate2_pass),
    }

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80)
    print("PHASE 7F STEP 3: TASK-RELATIVE NORMALIZATION EXPERIMENT")
    print("=" * 80)
    t_start = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: 42")

    # Load Transformed Tensors
    out_dir = PROJECT_ROOT / "results" / "phase7f"
    X_train_rel = np.load(out_dir / "X_train_task_relative.npy").astype(np.float32)
    X_val_rel = np.load(out_dir / "X_val_task_relative.npy").astype(np.float32)

    y_train = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_train.npy").astype(np.int64)
    meta_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv")

    y_val = np.load(PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy").astype(np.int64)
    meta_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv")

    assert X_train_rel.shape == (3278, 5, 31) and len(y_train) == 3278
    assert X_val_rel.shape == (657, 5, 31) and len(y_val) == 657
    print("  - Partition Shapes: Train=(3278, 5, 31), Val=(657, 5, 31)")
    print("  - STRICT PROTOCOL ENFORCEMENT: Candidate A Test split and Phase 6 Holdout NOT LOADED.")

    raw_w = compute_training_class_weights(y_train).to(device)
    weights_C = (raw_w / raw_w[0]).to(device)
    print(f"  - Fitted Alpha Weights from Training: {weights_C.tolist()}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_rel), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_rel), torch.from_numpy(y_val)),
        batch_size=64,
        shuffle=False
    )

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_phase7f"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "transformer_task_relative.pt"

    print("\n>>> Training Compact Temporal Transformer on Task-Relative Features (Condition C CE Loss)...")
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

    criterion = nn.CrossEntropyLoss(weight=weights_C)
    optimizer = torch.optim.AdamW(tf_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

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

    # Load best checkpoint and evaluate
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tf_model.load_state_dict(ckpt["model_state_dict"])
    tf_model.eval()

    with torch.no_grad():
        logits = tf_model(torch.from_numpy(X_val_rel).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    val_metrics = compute_metrics_val(meta_val, y_val, preds)

    # Load Phase 7E baseline metrics for comparison
    p7e_metrics = json.loads((PROJECT_ROOT / "results" / "phase7e" / "validation_comparison_metrics.json").read_text())
    b_m = p7e_metrics["transformer_condC_inverse_freq"]

    print("\n" + "=" * 115)
    print("STEP 3 COMPARISON: TASK-RELATIVE NORMALIZATION VS ORIGINAL PREPROCESSING BASELINE")
    print("=" * 115)
    print(f"{'Condition':<35} {'Macro-F1':<10} {'Acc':<8} {'BalAcc':<8} {'Ep-Rec %':<10} {'Crit-Rec':<10} {'Entity-FAR %':<14} {'Win-FPR %':<10} {'Gate 1':<8} {'Gate 2':<8}")
    print("-" * 115)

    g1_b = "PASS" if (b_m["episode_metrics"]["episode_recall_pct"] >= 70.0 and b_m["critical_recall"] >= 0.50) else "FAIL"
    g2_b = "PASS" if (b_m["entity_far"]["far_pct"] <= 10.0 and b_m["window_fpr"]["fpr_pct"] <= 5.0) else "FAIL"
    print(f"{'Baseline: Original Preprocessing':<35} {b_m['macro_f1']:<10.4f} {b_m['accuracy']:<8.4f} {b_m['balanced_accuracy']:<8.4f} {b_m['episode_metrics']['episode_recall_pct']:<10.2f} {b_m['critical_recall']:<10.4f} {b_m['entity_far']['far_pct']:<14.2f} {b_m['window_fpr']['fpr_pct']:<10.2f} {g1_b:<8} {g2_b:<8}")

    m = val_metrics
    g1 = "PASS" if m["gate_1_pass"] else "FAIL"
    g2 = "PASS" if m["gate_2_pass"] else "FAIL"
    print(f"{'Task-Relative Normalization':<35} {m['macro_f1']:<10.4f} {m['accuracy']:<8.4f} {m['balanced_accuracy']:<8.4f} {m['episode_recall_pct']:<10.2f} {m['critical_recall']:<10.4f} {m['entity_far_pct']:<14.2f} {m['window_fpr_pct']:<10.2f} {g1:<8} {g2:<8}")
    print("-" * 115)

    results = {
        "baseline_preprocessing": {
            "description": "Original Phase 7C Cluster-Fitted Scaler Preprocessing",
            "metrics": b_m
        },
        "task_relative_normalization": {
            "description": "Causal Task-Relative Normalization with Minimum History Fallback (K_min=2 windows)",
            "best_epoch": history["best_epoch"],
            "best_val_macro_f1": float(history["best_metric"]),
            "training_duration_seconds": float(t_dur),
            "checkpoint_path": str(ckpt_path),
            "metrics": val_metrics,
            "probabilities": probs,
            "predictions": preds
        }
    }

    out_file = PROJECT_ROOT / "results" / "phase7f" / "task_relative_experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"\nSaved results/phase7f/task_relative_experiment_results.json successfully.")

if __name__ == "__main__":
    main()
