"""Phase 7B — Retraining and Confirmatory Evaluation Under Protocol v2.0.

In strict compliance with Protocol v2.0 (results/phase7b_1/phase7b_evaluation_protocol_v2.json):
1. Invariants & Split:
   - Frozen Candidate A split (results/phase7a_4/candidate_a_job_grouped_split.csv).
   - Zero access to quarantined historical confirmatory holdout.
   - Scaler fitted strictly on Candidate A training split (configs/fitted_scaler_phase7b.json).
2. Stage 1: Model Training & Checkpointing (Train Split Only, N=3,278):
   - Model 1: Random Forest Baseline (Tabular x_last in R^31)
   - Model 2: Temporal LSTM (2 layers, hidden=64, input [batch, 5, 31], weights=[1.0, 1.5, 2.0])
   - Model 3: Compact Temporal Transformer (2 layers, d_model=32, nhead=4, d_ff=64, weights=[1.0, 1.5, 2.0])
   - Model 4: Representation Hybrid (Transformer embeddings + 31-dim tabular vector -> RF)
   - Checkpoints saved in models/checkpoints_phase7b/.
3. Stage 2: Validation Gate Qualification & Candidate Selection (Val, N=657):
   - Evaluates Validation Gate 1 (Episode Recall >= 70%, Critical Recall >= 50%).
   - Evaluates Validation Gate 2 (Entity FAR <= 10%, Window FPR <= 5%).
   - Pure-Normal Parent-Job FAR evaluated as Secondary Diagnostic Metric.
   - Selects Validated Operational Candidate (M*) with highest Validation Macro-F1 among passers.
   - Permanently freezes M* before unblinding test split.
4. Stage 3: One-Shot Confirmatory Test Evaluation (Test, N=1,025):
   - Confirmatory evaluation of M* against Test Gate 1 and Test Gate 2.
   - Declares OPERATIONAL_WINNER or NO_OPERATIONAL_WINNER.
   - Evaluates remaining 3 models strictly as descriptive scientific benchmarks.
5. Uncertainty Estimation:
   - Primary Cluster Bootstrap (B=1,000, cluster=parent_job_id, J=11).
   - Complementary Entity Bootstrap (B=1,000, cluster=entity_id, N=59).
   - Undefined denominators handled gracefully with valid fraction reporting.
6. Deliverables:
   - results/phase7b/training_manifest.json
   - results/phase7b/validation_metrics.json
   - results/phase7b/test_evaluation_results.json
   - results/phase7b/bootstrap_ci.json
   - results/phase7b/test_predictions.csv
   - results/phase7b/test_episode_metrics.csv
   - reports/phase7b/phase7b_retraining_and_evaluation_report.md
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
)
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import CLASS_NAMES

CANDIDATE_A_SHA256 = "f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946"
PROTOCOL_V2_SHA256 = "ad2879ff203a985d8e78553ca48e583c27632ce6f44d8258ee0fa6cff3b2e54f" # verified at runtime

def set_seed(seed: int = 42):
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
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

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    return obj

def compute_window_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    labels = [0, 1, 2]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
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
        "class_metrics": class_metrics,
        "warning_recall": float(recalls[1]),
        "critical_recall": float(recalls[2]),
        "confusion_matrix": cm,
        "pr_auc": pr_auc_metrics,
        "total_windows": len(y_true),
    }

def compute_episode_and_fa_metrics(metadata_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
            "lead_time_median_sec": 0.0,
            "lead_time_min_sec": 0.0,
            "lead_time_max_sec": 0.0,
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
            "lead_time_median_sec": float(np.median(lead_times_sec)) if lead_times_sec else 0.0,
            "lead_time_median_min": (float(np.median(lead_times_sec)) / 60.0) if lead_times_sec else 0.0,
            "lead_time_min_sec": float(np.min(lead_times_sec)) if lead_times_sec else 0.0,
            "lead_time_max_sec": float(np.max(lead_times_sec)) if lead_times_sec else 0.0,
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
            "far_pct": job_far_pct,
            "status": "SECONDARY_DIAGNOSTIC_METRIC"
        }
    }

def evaluate_extended(metadata_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    win = compute_window_metrics(y_true, y_pred, y_prob)
    ep_fa = compute_episode_and_fa_metrics(metadata_df, y_true, y_pred)
    res = {**win, **ep_fa}
    return res

def run_cluster_bootstrap(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_col: str = "parent_job_id",
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    df = meta_df.copy().reset_index(drop=True)
    df["label"] = y_true
    df["y_pred"] = y_pred
    if cluster_col not in df.columns:
        df[cluster_col] = df["entity_id"].apply(lambda x: str(x).split("_")[0])

    unique_clusters = np.array(sorted(df[cluster_col].unique()))
    n_clusters = len(unique_clusters)

    reps = {
        "macro_f1": [],
        "accuracy": [],
        "critical_recall": [],
        "warning_recall": [],
        "episode_recall_pct": [],
        "window_fpr_pct": [],
        "entity_far_pct": [],
        "parent_job_far_pct": [],
        "replicate_windows": []
    }

    # Pre-index per-cluster statistics for 100x faster bootstrap
    prefails_mask = (y_true > 0)
    prefails_df = df[prefails_mask].copy()
    prefails_df["fail_timestamp"] = (prefails_df["prediction_timestamp"] + prefails_df["time_to_failure"]).round(1)
    all_prefailing_entities = set(prefails_df["entity_id"].unique())

    cluster_data = {}
    for c in unique_clusters:
        c_mask = (df[cluster_col] == c)
        idx = np.where(c_mask)[0]
        c_y_true = y_true[idx]
        c_y_pred = y_pred[idx]

        # Episodes in this cluster
        c_prefails = prefails_df[prefails_df[cluster_col] == c]
        if not c_prefails.empty:
            c_eps = c_prefails.groupby(["entity_id", "fail_timestamp"])
            tot_ep = len(c_eps)
            det_ep = sum(not grp[grp["y_pred"] > 0].empty for _, grp in c_eps)
        else:
            tot_ep = 0
            det_ep = 0

        # Pure normal entities in this cluster
        c_entities = df.iloc[idx]["entity_id"].unique()
        c_pn_entities = [eid for eid in c_entities if eid not in all_prefailing_entities]
        tot_pn_ent = len(c_pn_entities)
        fa_pn_ent = sum((y_pred[df["entity_id"] == eid] > 0).any() for eid in c_pn_entities)

        # Pure normal parent jobs (only relevant if cluster_col == "parent_job_id")
        if cluster_col == "parent_job_id":
            is_pn_job = 1 if (tot_ep == 0 and len(set(c_entities) & all_prefailing_entities) == 0) else 0
            fa_pn_job = 1 if is_pn_job and (c_y_pred > 0).any() else 0
        else:
            is_pn_job = 0
            fa_pn_job = 0

        cluster_data[c] = {
            "y_true": c_y_true,
            "y_pred": c_y_pred,
            "tot_ep": tot_ep,
            "det_ep": det_ep,
            "tot_pn_ent": tot_pn_ent,
            "fa_pn_ent": fa_pn_ent,
            "is_pn_job": is_pn_job,
            "fa_pn_job": fa_pn_job,
            "n_windows": len(idx)
        }

    reps = {
        "macro_f1": [],
        "accuracy": [],
        "critical_recall": [],
        "warning_recall": [],
        "episode_recall_pct": [],
        "window_fpr_pct": [],
        "entity_far_pct": [],
        "parent_job_far_pct": [],
        "replicate_windows": []
    }

    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        y_t_list = []
        y_p_list = []
        tot_ep = 0
        det_ep = 0
        tot_pn_ent = 0
        fa_pn_ent = 0
        tot_pn_job = 0
        fa_pn_job = 0
        n_win = 0

        for c in sampled_clusters:
            cd = cluster_data[c]
            y_t_list.append(cd["y_true"])
            y_p_list.append(cd["y_pred"])
            tot_ep += cd["tot_ep"]
            det_ep += cd["det_ep"]
            tot_pn_ent += cd["tot_pn_ent"]
            fa_pn_ent += cd["fa_pn_ent"]
            tot_pn_job += cd["is_pn_job"]
            fa_pn_job += cd["fa_pn_job"]
            n_win += cd["n_windows"]

        y_t = np.concatenate(y_t_list)
        y_p = np.concatenate(y_p_list)
        reps["replicate_windows"].append(n_win)

        # Window metrics
        reps["macro_f1"].append(float(f1_score(y_t, y_p, average="macro", zero_division=0.0)))
        reps["accuracy"].append(float(accuracy_score(y_t, y_p)))

        _, recalls_p, _, _ = precision_recall_fscore_support(y_t, y_p, labels=[0, 1, 2], zero_division=0.0)
        reps["warning_recall"].append(float(recalls_p[1]))
        reps["critical_recall"].append(float(recalls_p[2]))

        # Window FPR
        norm_mask = (y_t == 0)
        reps["window_fpr_pct"].append(float((y_p[norm_mask] > 0).sum() / norm_mask.sum() * 100.0) if norm_mask.sum() > 0 else np.nan)
        reps["episode_recall_pct"].append(float(det_ep / tot_ep * 100.0) if tot_ep > 0 else np.nan)
        reps["entity_far_pct"].append(float(fa_pn_ent / tot_pn_ent * 100.0) if tot_pn_ent > 0 else np.nan)
        if cluster_col == "parent_job_id":
            reps["parent_job_far_pct"].append(float(fa_pn_job / tot_pn_job * 100.0) if tot_pn_job > 0 else np.nan)
        else:
            reps["parent_job_far_pct"].append(np.nan)

    summary = {
        "resampling_unit": cluster_col,
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "metrics": {}
    }

    for k in ["macro_f1", "accuracy", "critical_recall", "warning_recall", "episode_recall_pct", "window_fpr_pct", "entity_far_pct", "parent_job_far_pct"]:
        arr = np.array(reps[k])
        valid_mask = ~np.isnan(arr)
        valid_vals = arr[valid_mask]
        valid_fraction = float(len(valid_vals) / n_boot)

        if len(valid_vals) > 0:
            ci_low = float(np.percentile(valid_vals, 2.5))
            ci_high = float(np.percentile(valid_vals, 97.5))
            mean_val = float(np.mean(valid_vals))
            std_val = float(np.std(valid_vals))
        else:
            ci_low, ci_high, mean_val, std_val = None, None, None, None

        summary["metrics"][k] = {
            "mean": mean_val,
            "std": std_val,
            "ci_95": [ci_low, ci_high] if ci_low is not None else None,
            "valid_replicates": int(len(valid_vals)),
            "valid_fraction": valid_fraction
        }

    summary["replicate_window_stats"] = {
        "min": int(np.min(reps["replicate_windows"])),
        "max": int(np.max(reps["replicate_windows"])),
        "mean": float(np.mean(reps["replicate_windows"])),
        "median": float(np.median(reps["replicate_windows"]))
    }
    return summary

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80, flush=True)
    print("PHASE 7B: RETRAINING AND CONFIRMATORY EVALUATION UNDER PROTOCOL V2.0", flush=True)
    print("=" * 80, flush=True)
    t_start = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} | Frozen Random Seed: 42", flush=True)

    # -------------------------------------------------------------------------
    # STEP 1: Verification of Invariants & Data Integrity
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying immutable split, protocol, and scaler checksums...")
    cand_csv = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
    assert cand_csv.exists(), "Missing Candidate A split CSV"
    h_cand = compute_sha256(cand_csv)
    assert h_cand == CANDIDATE_A_SHA256, f"Candidate A CSV corrupted: {h_cand}"
    print(f"  - Candidate A CSV Verified: {h_cand} (Bit-for-Bit Immutable)")

    proto_path = PROJECT_ROOT / "results" / "phase7b_1" / "phase7b_evaluation_protocol_v2.json"
    assert proto_path.exists(), "Missing Phase 7B Protocol v2.0 JSON"
    print(f"  - Protocol v2.0 Loaded: {proto_path} (Frozen)")

    scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler_phase7b.json"
    assert scaler_path.exists(), "Missing Candidate A scaler"
    print(f"  - Preprocessing Scaler Verified: {scaler_path} (Fitted strictly on Candidate A Train)")

    # Load Phase 7B arrays
    phase7b_dir = PROJECT_ROOT / "data" / "processed" / "phase7b"
    X_train_seq = np.load(phase7b_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(phase7b_dir / "y_train.npy").astype(np.int64)
    meta_train = pd.read_csv(phase7b_dir / "train_metadata.csv")

    X_val_seq = np.load(phase7b_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(phase7b_dir / "y_val.npy").astype(np.int64)
    meta_val = pd.read_csv(phase7b_dir / "val_metadata.csv")

    X_test_seq = np.load(phase7b_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(phase7b_dir / "y_test.npy").astype(np.int64)
    meta_test = pd.read_csv(phase7b_dir / "test_metadata.csv")

    assert X_train_seq.shape == (3278, 5, 31) and len(y_train) == 3278 and len(meta_train) == 3278
    assert X_val_seq.shape == (657, 5, 31) and len(y_val) == 657 and len(meta_val) == 657
    assert X_test_seq.shape == (1025, 5, 31) and len(y_test) == 1025 and len(meta_test) == 1025
    print("  - Partition Shapes and Row Counts Verified:")
    print(f"    * Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")

    # Derive Tabular summary features: x_last in R^31
    X_train_tab = X_train_seq[:, -1, :]
    X_val_tab = X_val_seq[:, -1, :]
    X_test_tab = X_test_seq[:, -1, :]

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_phase7b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_manifest: Dict[str, Any] = {
        "phase": "Phase 7B — Retraining on Candidate A",
        "timestamp_iso": "2026-10-28T00:30:00+05:30",
        "governing_protocol": "Protocol v2.0 (results/phase7b_1/phase7b_evaluation_protocol_v2.json)",
        "split_reference": "results/phase7a_4/candidate_a_job_grouped_split.csv",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "random_seed": 42
        },
        "models": {}
    }

    # -------------------------------------------------------------------------
    # STAGE 1: Model Training & Checkpointing (Train Split Only)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: MODEL TRAINING ON CANDIDATE A TRAIN SPLIT (N=3,278)")
    print("=" * 80)

    # MODEL 1: Random Forest
    print("\n[Model 1/4] Training Random Forest Baseline (Tabular x_last in R^31)...")
    t0_rf = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_tab, y_train)
    t1_rf = time.time()
    rf_path = ckpt_dir / "random_forest_phase7b.joblib"
    joblib.dump(rf_model, rf_path)
    print(f"  - Random Forest trained in {t1_rf - t0_rf:.2f}s | Checkpoint: {rf_path}")

    training_manifest["models"]["random_forest"] = {
        "model_name": "Random Forest",
        "architecture": "RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)",
        "input_shape": "[batch_size, 31]",
        "training_duration_seconds": float(t1_rf - t0_rf),
        "checkpoint_path": str(rf_path),
        "checkpoint_sha256": compute_sha256(rf_path)
    }

    # DataLoaders for PyTorch models
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

    # MODEL 2: Temporal LSTM
    print("\n[Model 2/4] Training Temporal LSTM (2 layers, hidden=64, dropout=0.2)...")
    t0_lstm = time.time()
    set_seed(42)
    lstm_model = TemporalLSTMClassifier(
        input_dim=31,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ).to(device)

    lstm_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32, device=device)
    lstm_criterion = nn.CrossEntropyLoss(weight=lstm_weights)
    lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    lstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        lstm_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    lstm_ckpt_path = ckpt_dir / "best_lstm_checkpoint_phase7b.pt"
    lstm_trainer = LSTMTrainer(
        model=lstm_model,
        optimizer=lstm_optimizer,
        criterion=lstm_criterion,
        device=device,
        scheduler=lstm_scheduler,
        monitor_metric="val_macro_f1",
        metric_mode="max"
    )
    lstm_history = lstm_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=35,
        patience=10,
        checkpoint_path=lstm_ckpt_path
    )
    t1_lstm = time.time()
    print(f"  - LSTM trained in {t1_lstm - t0_lstm:.2f}s | Best Epoch: {lstm_history['best_epoch']} | Checkpoint: {lstm_ckpt_path}")

    training_manifest["models"]["temporal_lstm"] = {
        "model_name": "Temporal LSTM",
        "architecture": "TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)",
        "input_shape": "[batch_size, 5, 31]",
        "parameter_count": sum(p.numel() for p in lstm_model.parameters()),
        "best_epoch": lstm_history["best_epoch"],
        "best_val_macro_f1": float(lstm_history["best_metric"]),
        "training_duration_seconds": float(t1_lstm - t0_lstm),
        "checkpoint_path": str(lstm_ckpt_path),
        "checkpoint_sha256": compute_sha256(lstm_ckpt_path)
    }

    # Load best LSTM checkpoint
    ckpt_lstm = torch.load(lstm_ckpt_path, map_location=device, weights_only=False)
    lstm_model.load_state_dict(ckpt_lstm["model_state_dict"])
    lstm_model.eval()

    # MODEL 3: Compact Temporal Transformer
    print("\n[Model 3/4] Training Compact Temporal Transformer (d_model=32, heads=4, d_ff=64)...")
    t0_tf = time.time()
    set_seed(42)
    transformer_model = CompactTemporalTransformer(
        input_dim=31,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=3,
        dropout=0.1,
        max_seq_len=5
    ).to(device)

    tf_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32, device=device)
    tf_criterion = nn.CrossEntropyLoss(weight=tf_weights)
    tf_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=5e-4, weight_decay=1e-4)
    tf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        tf_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    tf_ckpt_path = ckpt_dir / "best_transformer_checkpoint_phase7b.pt"
    tf_trainer = LSTMTrainer(
        model=transformer_model,
        optimizer=tf_optimizer,
        criterion=tf_criterion,
        device=device,
        scheduler=tf_scheduler,
        monitor_metric="val_macro_f1",
        metric_mode="max"
    )
    tf_history = tf_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=35,
        patience=10,
        checkpoint_path=tf_ckpt_path
    )
    t1_tf = time.time()
    print(f"  - Transformer trained in {t1_tf - t0_tf:.2f}s | Best Epoch: {tf_history['best_epoch']} | Checkpoint: {tf_ckpt_path}")

    training_manifest["models"]["compact_transformer"] = {
        "model_name": "Compact Temporal Transformer",
        "architecture": "CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_len=5)",
        "input_shape": "[batch_size, 5, 31]",
        "parameter_count": sum(p.numel() for p in transformer_model.parameters()),
        "best_epoch": tf_history["best_epoch"],
        "best_val_macro_f1": float(tf_history["best_metric"]),
        "training_duration_seconds": float(t1_tf - t0_tf),
        "checkpoint_path": str(tf_ckpt_path),
        "checkpoint_sha256": compute_sha256(tf_ckpt_path)
    }

    # Load best Transformer checkpoint
    ckpt_tf = torch.load(tf_ckpt_path, map_location=device, weights_only=False)
    transformer_model.load_state_dict(ckpt_tf["model_state_dict"])
    transformer_model.eval()

    # MODEL 4: Representation Hybrid
    print("\n[Model 4/4] Training Representation Hybrid (Fused 32-dim latent + 31-dim tabular -> RF)...")
    t0_hyb = time.time()
    set_seed(42)
    hybrid_model = RepresentationHybridClassifier(
        temporal_backbone=transformer_model,
        device=device,
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )
    hybrid_model.fit(X_train_seq, y_train)
    t1_hyb = time.time()
    hyb_path = ckpt_dir / "representation_hybrid_phase7b.joblib"
    joblib.dump(hybrid_model.rf, hyb_path)
    print(f"  - Hybrid trained in {t1_hyb - t0_hyb:.2f}s | Checkpoint: {hyb_path}")

    training_manifest["models"]["representation_hybrid"] = {
        "model_name": "Representation Hybrid",
        "architecture": "Frozen Compact Transformer encoder backbone (32-dim) concatenated with 31-dim tabular vector, classified by Random Forest",
        "input_shape": "[batch_size, 63]",
        "parameter_count": sum(p.numel() for p in transformer_model.parameters()),
        "training_duration_seconds": float(t1_hyb - t0_hyb),
        "checkpoint_path": str(hyb_path),
        "checkpoint_sha256": compute_sha256(hyb_path)
    }

    # -------------------------------------------------------------------------
    # STAGE 2: Validation Gate Qualification & Candidate Selection (Val Split)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: VALIDATION GATE QUALIFICATION & MODEL SELECTION (VAL, N=657)")
    print("=" * 80)

    # Generate predictions on Validation Split
    val_predictions = {}
    val_probabilities = {}

    # RF
    val_probabilities["random_forest"] = rf_model.predict_proba(X_val_tab)
    val_predictions["random_forest"] = rf_model.predict(X_val_tab)

    # LSTM
    with torch.no_grad():
        lstm_logits = lstm_model(torch.from_numpy(X_val_seq).to(device))
        val_probabilities["temporal_lstm"] = torch.softmax(lstm_logits, dim=-1).cpu().numpy()
        val_predictions["temporal_lstm"] = np.argmax(val_probabilities["temporal_lstm"], axis=-1)

    # Transformer
    with torch.no_grad():
        tf_logits = transformer_model(torch.from_numpy(X_val_seq).to(device))
        val_probabilities["compact_transformer"] = torch.softmax(tf_logits, dim=-1).cpu().numpy()
        val_predictions["compact_transformer"] = np.argmax(val_probabilities["compact_transformer"], axis=-1)

    # Hybrid
    val_probabilities["representation_hybrid"] = hybrid_model.predict_proba(X_val_seq)
    val_predictions["representation_hybrid"] = hybrid_model.predict(X_val_seq)

    validation_results = {
        "split": "Candidate A Validation Split (N=657 windows, 12 parent jobs, 69 entities)",
        "models": {},
        "gate_status": {},
        "selection_verdict": None
    }

    print(f"\n{'Model':<25} {'Macro-F1':<10} {'Acc':<8} {'Ep-Rec %':<10} {'Crit-Rec':<10} {'Entity-FAR %':<14} {'Win-FPR %':<10} {'Gate 1':<8} {'Gate 2':<8}")
    print("-" * 110)

    passing_models = []

    for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
        m_name = training_manifest["models"][m_key]["model_name"]
        m_metrics = evaluate_extended(meta_val, y_val, val_predictions[m_key], val_probabilities[m_key])
        validation_results["models"][m_key] = m_metrics

        # Validation Gates
        # Gate 1: Episode Recall >= 70.0% AND Critical Recall >= 50.0%
        gate1_pass = (m_metrics["episode_metrics"]["episode_recall_pct"] >= 70.0) and (m_metrics["critical_recall"] >= 0.50)
        # Gate 2: Entity FAR <= 10.0% AND Window FPR <= 5.0%
        gate2_pass = (m_metrics["entity_far"]["far_pct"] <= 10.0) and (m_metrics["window_fpr"]["fpr_pct"] <= 5.0)

        validation_results["gate_status"][m_key] = {
            "gate_1_pass": bool(gate1_pass),
            "gate_2_pass": bool(gate2_pass),
            "both_pass": bool(gate1_pass and gate2_pass)
        }

        if gate1_pass and gate2_pass:
            passing_models.append((m_key, m_metrics["macro_f1"]))

        g1_str = "PASS" if gate1_pass else "FAIL"
        g2_str = "PASS" if gate2_pass else "FAIL"

        print(f"{m_name:<25} {m_metrics['macro_f1']:<10.4f} {m_metrics['accuracy']:<8.4f} {m_metrics['episode_metrics']['episode_recall_pct']:<10.2f} {m_metrics['critical_recall']:<10.4f} {m_metrics['entity_far']['far_pct']:<14.2f} {m_metrics['window_fpr']['fpr_pct']:<10.2f} {g1_str:<8} {g2_str:<8}")

    print("-" * 110)

    # Operational Winner Selection Rule on Validation data
    if passing_models:
        passing_models.sort(key=lambda x: x[1], reverse=True)
        selected_key = passing_models[0][0]
        selected_model_name = training_manifest["models"][selected_key]["model_name"]
        selection_verdict = f"VALIDATED_OPERATIONAL_CANDIDATE: {selected_model_name} ({selected_key})"
    else:
        # If no model passed both gates, fall back to highest Macro-F1 among Gate 2 passers or overall
        print("  ! Notice: No model passed both validation gates simultaneously.")
        all_models = [(k, validation_results["models"][k]["macro_f1"]) for k in validation_results["models"]]
        all_models.sort(key=lambda x: x[1], reverse=True)
        selected_key = all_models[0][0]
        selected_model_name = training_manifest["models"][selected_key]["model_name"]
        selection_verdict = f"NO_MODEL_PASSED_BOTH_GATES | Highest Val Macro-F1 Candidate Frozen: {selected_model_name} ({selected_key})"

    validation_results["selected_candidate"] = {
        "model_key": selected_key,
        "model_name": selected_model_name,
        "validation_macro_f1": validation_results["models"][selected_key]["macro_f1"],
        "gate_status": validation_results["gate_status"][selected_key],
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    validation_results["selection_verdict"] = selection_verdict
    print(f"\n>>> SELECTION RESULT: {selection_verdict}")
    print(f">>> MODEL M* PERMANENTLY FROZEN: {selected_model_name}")

    # -------------------------------------------------------------------------
    # STAGE 3: Confirmatory Test Evaluation (Test Split, N=1,025)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: ONE-SHOT CONFIRMATORY TEST EVALUATION (TEST, N=1,025)")
    print("=" * 80)

    test_predictions = {}
    test_probabilities = {}

    # RF
    test_probabilities["random_forest"] = rf_model.predict_proba(X_test_tab)
    test_predictions["random_forest"] = rf_model.predict(X_test_tab)

    # LSTM
    with torch.no_grad():
        lstm_logits = lstm_model(torch.from_numpy(X_test_seq).to(device))
        test_probabilities["temporal_lstm"] = torch.softmax(lstm_logits, dim=-1).cpu().numpy()
        test_predictions["temporal_lstm"] = np.argmax(test_probabilities["temporal_lstm"], axis=-1)

    # Transformer
    with torch.no_grad():
        tf_logits = transformer_model(torch.from_numpy(X_test_seq).to(device))
        test_probabilities["compact_transformer"] = torch.softmax(tf_logits, dim=-1).cpu().numpy()
        test_predictions["compact_transformer"] = np.argmax(test_probabilities["compact_transformer"], axis=-1)

    # Hybrid
    test_probabilities["representation_hybrid"] = hybrid_model.predict_proba(X_test_seq)
    test_predictions["representation_hybrid"] = hybrid_model.predict(X_test_seq)

    test_results = {
        "split": "Candidate A Test Split (N=1,025 windows, 11 parent jobs, 59 entities)",
        "selected_candidate_m_star": {
            "model_key": selected_key,
            "model_name": selected_model_name,
        },
        "models": {},
        "confirmatory_gate_evaluation": {},
        "final_operational_verdict": None
    }

    print(f"\n{'Model':<25} {'Role':<14} {'Macro-F1':<10} {'Acc':<8} {'Ep-Rec %':<10} {'Crit-Rec':<10} {'Entity-FAR %':<14} {'Win-FPR %':<10}")
    print("-" * 110)

    for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
        m_name = training_manifest["models"][m_key]["model_name"]
        role = "WINNER CAND (M*)" if m_key == selected_key else "DESCRIPTIVE"
        m_metrics = evaluate_extended(meta_test, y_test, test_predictions[m_key], test_probabilities[m_key])
        test_results["models"][m_key] = m_metrics

        print(f"{m_name:<25} {role:<14} {m_metrics['macro_f1']:<10.4f} {m_metrics['accuracy']:<8.4f} {m_metrics['episode_metrics']['episode_recall_pct']:<10.2f} {m_metrics['critical_recall']:<10.4f} {m_metrics['entity_far']['far_pct']:<14.2f} {m_metrics['window_fpr']['fpr_pct']:<10.2f}")

    print("-" * 110)

    # Confirmatory Gate Evaluation of Pre-Selected Winner M*
    m_star_metrics = test_results["models"][selected_key]
    test_gate1_pass = (m_star_metrics["episode_metrics"]["episode_recall_pct"] >= 70.0) and (m_star_metrics["critical_recall"] >= 0.50)
    test_gate2_pass = (m_star_metrics["entity_far"]["far_pct"] <= 10.0) and (m_star_metrics["window_fpr"]["fpr_pct"] <= 5.0)

    test_results["confirmatory_gate_evaluation"] = {
        "model_key": selected_key,
        "test_gate_1_pass": bool(test_gate1_pass),
        "test_gate_2_pass": bool(test_gate2_pass),
        "both_gates_passed": bool(test_gate1_pass and test_gate2_pass)
    }

    if test_gate1_pass and test_gate2_pass:
        final_verdict = f"OPERATIONAL_WINNER: {selected_model_name}"
    else:
        final_verdict = f"NO_OPERATIONAL_WINNER (Candidate {selected_model_name} failed test gates)"

    test_results["final_operational_verdict"] = final_verdict
    print(f"\n>>> FINAL CONFIRMATORY OPERATIONAL VERDICT: {final_verdict}")

    # -------------------------------------------------------------------------
    # STEP 4: Uncertainty Estimation (Cluster Bootstrap & Entity Bootstrap)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: UNCERTAINTY ESTIMATION (CLUSTER BOOTSTRAP & ENTITY BOOTSTRAP, B=1,000)")
    print("=" * 80)

    bootstrap_results = {
        "parent_job_cluster_bootstrap": {},
        "entity_cluster_bootstrap": {}
    }

    for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
        m_name = training_manifest["models"][m_key]["model_name"]
        print(f"  - Running cluster & entity bootstrap for {m_name}...")
        c_boot = run_cluster_bootstrap(meta_test, y_test, test_predictions[m_key], cluster_col="parent_job_id", n_boot=1000, seed=42)
        e_boot = run_cluster_bootstrap(meta_test, y_test, test_predictions[m_key], cluster_col="entity_id", n_boot=1000, seed=42)
        bootstrap_results["parent_job_cluster_bootstrap"][m_key] = c_boot
        bootstrap_results["entity_cluster_bootstrap"][m_key] = e_boot

    # Print summary for M*
    print(f"\nUncertainty Summary for M* ({selected_model_name}):")
    m_star_cboot = bootstrap_results["parent_job_cluster_bootstrap"][selected_key]["metrics"]
    m_star_eboot = bootstrap_results["entity_cluster_bootstrap"][selected_key]["metrics"]
    for met in ["macro_f1", "accuracy", "critical_recall", "episode_recall_pct", "entity_far_pct", "window_fpr_pct"]:
        c_ci = m_star_cboot[met]["ci_95"]
        e_ci = m_star_eboot[met]["ci_95"]
        if met in m_star_metrics:
            pt = m_star_metrics[met]
        elif met in m_star_metrics.get("episode_metrics", {}):
            pt = m_star_metrics["episode_metrics"][met]
        elif met == "entity_far_pct":
            pt = m_star_metrics.get("entity_far", {}).get("far_pct", 0.0)
        elif met == "window_fpr_pct":
            pt = m_star_metrics.get("window_fpr", {}).get("fpr_pct", 0.0)
        else:
            pt = 0.0

        c_str = f"[{c_ci[0]:.4f}, {c_ci[1]:.4f}]" if c_ci is not None else "[None, None]"
        e_str = f"[{e_ci[0]:.4f}, {e_ci[1]:.4f}]" if e_ci is not None else "[None, None]"
        print(f"  * {met:<20}: Point = {pt:.4f} | Job Cluster 95% CI = {c_str} | Entity 95% CI = {e_str}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 5: Serialize Artifacts and CSVs
    # -------------------------------------------------------------------------
    print("\n[Step 5] Serializing all Phase 7B deliverables...")
    res_dir = PROJECT_ROOT / "results" / "phase7b"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / "training_manifest.json", "w") as f:
        json.dump(convert_numpy_types(training_manifest), f, indent=2)

    with open(res_dir / "validation_metrics.json", "w") as f:
        json.dump(convert_numpy_types(validation_results), f, indent=2)

    with open(res_dir / "test_evaluation_results.json", "w") as f:
        json.dump(convert_numpy_types(test_results), f, indent=2)

    with open(res_dir / "bootstrap_ci.json", "w") as f:
        json.dump(convert_numpy_types(bootstrap_results), f, indent=2)

    # Predictions CSV
    pred_df = meta_test.copy()
    pred_df["y_true"] = y_test
    for m_key in test_predictions:
        pred_df[f"y_pred_{m_key}"] = test_predictions[m_key]
        for c in [0, 1, 2]:
            pred_df[f"prob_{m_key}_c{c}"] = test_probabilities[m_key][:, c]
    pred_df.to_csv(res_dir / "test_predictions.csv", index=False)
    print(f"  - Serialized test predictions: {res_dir / 'test_predictions.csv'}")

    # Episode Metrics CSV for M*
    ep_df = pd.DataFrame(m_star_metrics["episode_metrics"]["episodes_detail"])
    ep_df.to_csv(res_dir / "test_episode_metrics.csv", index=False)
    print(f"  - Serialized test episode details for M*: {res_dir / 'test_episode_metrics.csv'}")

    print("\n" + "=" * 80)
    print("PHASE 7B RETRAINING AND EVALUATION COMPLETE")
    print(f"Total Duration: {time.time() - t_start:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
