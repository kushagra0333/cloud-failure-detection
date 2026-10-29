"""Phase 6.3 — L=5 Model Training Under Ratified Protocol Amendment 2.1.

Trains four early failure prediction models from scratch on L=5 sequences:
1. Random Forest baseline (31-dim tabular features x_last)
2. Temporal LSTM (2 layers, hidden=64, input [batch, 5, 31], weights=[1.0, 1.5, 2.0])
3. Compact Temporal Transformer (2 layers, d_model=32, nhead=4, d_ff=64, input [batch, 5, 31], weights=[1.0, 1.5, 2.0])
4. Representation Hybrid (Frozen Transformer 32-dim embeddings + 31-dim tabular vector -> RF)

Strict Scientific Invariants:
- Preserves results/phase6/protocol_l5.json and protocol_l5_amendment_2_1.json unmodified.
- Confirmatory holdout data is strictly sealed and never accessed or scored.
- Preprocessing fitted strictly on training data (configs/fitted_scaler.json).
- Checkpoints selected exclusively on development validation split.
- Checkpoints saved in models/checkpoints_l5/.
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
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import (
    compute_window_metrics,
    compute_episode_metrics,
)
from src.evaluation.confirmatory_evaluator_l5 import load_ratified_l5_protocol

ORIGINAL_PROTOCOL_SHA256 = "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"


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
    """Computes SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_numpy_types(obj: Any) -> Any:
    """Recursively converts NumPy types to native Python types for JSON serialization."""
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


def compute_extended_validation_metrics(
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Computes window-level, episode-level, and operational false-alarm metrics on validation data."""
    win_metrics = compute_window_metrics(y_true, y_pred, y_prob)
    ep_metrics = compute_episode_metrics(metadata_df, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    normal_window_mask = (y_true == 0)
    n_normal_windows = int(normal_window_mask.sum())
    n_fp_windows = int((y_pred[normal_window_mask] > 0).sum())
    window_fpr_pct = (n_fp_windows / n_normal_windows * 100.0) if n_normal_windows > 0 else 0.0

    all_eids = sorted(metadata_df["entity_id"].unique())
    prefails = metadata_df[metadata_df["label"] > 0]
    failing_eids = set(prefails["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal = len(pure_normal_eids)

    pure_normal_fa = sum(
        (y_pred[metadata_df["entity_id"] == eid] > 0).any()
        for eid in pure_normal_eids
    )
    pure_normal_far_pct = (pure_normal_fa / n_pure_normal * 100.0) if n_pure_normal > 0 else 0.0

    return {
        "macro_f1": float(win_metrics["macro_f1"]),
        "accuracy": float(win_metrics["accuracy"]),
        "warning_recall": float(win_metrics["warning_recall"]),
        "critical_recall": float(win_metrics["critical_recall"]),
        "class_metrics": win_metrics["class_metrics"],
        "confusion_matrix": cm,
        "pr_auc": win_metrics["pr_auc"],
        "episode_metrics": ep_metrics,
        "window_fpr": {
            "false_positive_windows": n_fp_windows,
            "denominator_windows": n_normal_windows,
            "fpr_pct": window_fpr_pct,
        },
        "entity_far": {
            "false_alarm_entities": pure_normal_fa,
            "denominator_entities": n_pure_normal,
            "far_pct": pure_normal_far_pct,
        },
        "total_evaluated_windows": len(y_true),
    }


def train_l5_models():
    print("=" * 80)
    print("PHASE 6.3 — L=5 MODEL TRAINING UNDER RATIFIED PROTOCOL AMENDMENT 2.1")
    print("=" * 80)

    t_start = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} | Random Seed: 42 (Frozen)")

    # 1. Pre-Training Invariant Verification
    print("\n[Step 1] Verifying protocol artifacts and holdout quarantine integrity...")
    orig_proto_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5.json"
    assert orig_proto_path.exists(), "Original protocol_l5.json does not exist"
    orig_hash = compute_sha256(orig_proto_path)
    assert orig_hash == ORIGINAL_PROTOCOL_SHA256, f"Original protocol was modified! SHA256: {orig_hash}"
    print(f"  - Original Protocol 2.0 SHA-256 Verified: {orig_hash[:16]}... (Immutable)")

    amend_proto_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json"
    protocol = load_ratified_l5_protocol(str(amend_proto_path))
    print(f"  - Amended Protocol Version: {protocol['amendment_version']} ({protocol['status']})")
    print(f"  - Ratified Denominators: 5,511 windows (400 Critical, 300 Warning, 4,811 Normal)")

    # Verify holdout entities isolation
    f_eids_holdout = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
    n_eids_holdout = set(protocol["frozen_entity_ids"]["primary_normal_entities_sampled"])
    all_holdout_eids = f_eids_holdout | n_eids_holdout

    # 2. Load Development Datasets and Construct L=5 Tensors
    print("\n[Step 2] Loading development datasets and slicing L=5 sequences...")
    data_dir = PROJECT_ROOT / "data" / "processed"
    X_train_raw = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    meta_train = pd.read_csv(data_dir / "train_metadata.csv")

    X_val_raw = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    meta_val = pd.read_csv(data_dir / "val_metadata.csv")

    X_test_raw = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "y_test.npy").astype(np.int64)
    meta_test = pd.read_csv(data_dir / "test_metadata.csv")

    dev_eids = set(meta_train["entity_id"]) | set(meta_val["entity_id"]) | set(meta_test["entity_id"])
    print(f"  - Development unique entities: {len(dev_eids)}")
    assert len(dev_eids & all_holdout_eids) == 0, "FATAL: Entity overlap detected between development and holdout!"
    print(f"  - Zero Entity Leakage: 0 development entities overlap with the 337 frozen confirmatory entities.")

    # Slice L=5 sequences (trailing 5 steps ending at prediction anchor)
    X_train_l5 = X_train_raw[:, -5:, :]
    X_val_l5 = X_val_raw[:, -5:, :]
    X_test_l5 = X_test_raw[:, -5:, :]

    assert X_train_l5.shape == (len(y_train), 5, 31)
    assert X_val_l5.shape == (len(y_val), 5, 31)
    assert X_test_l5.shape == (len(y_test), 5, 31)
    print(f"  - L=5 Sequence Tensors constructed successfully:")
    print(f"    * X_train_l5 : {X_train_l5.shape}")
    print(f"    * X_val_l5   : {X_val_l5.shape}")
    print(f"    * X_test_l5  : {X_test_l5.shape}")

    # Tabular instantaneous feature representation: x_last in R^31
    X_train_tab = X_train_l5[:, -1, :]
    X_val_tab = X_val_l5[:, -1, :]
    X_test_tab = X_test_l5[:, -1, :]

    # Save L=5 processed development artifacts
    l5_data_dir = data_dir / "l5"
    l5_data_dir.mkdir(parents=True, exist_ok=True)
    np.save(l5_data_dir / "X_train.npy", X_train_l5)
    np.save(l5_data_dir / "y_train.npy", y_train)
    np.save(l5_data_dir / "X_val.npy", X_val_l5)
    np.save(l5_data_dir / "y_val.npy", y_val)
    np.save(l5_data_dir / "X_test.npy", X_test_l5)
    np.save(l5_data_dir / "y_test.npy", y_test)
    meta_train.to_csv(l5_data_dir / "train_metadata.csv", index=False)
    meta_val.to_csv(l5_data_dir / "val_metadata.csv", index=False)
    meta_test.to_csv(l5_data_dir / "test_metadata.csv", index=False)
    print(f"  - Serialized L=5 development tensors to: {l5_data_dir}")

    # Directory for L=5 checkpoints
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_l5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "phase": "Phase 6.3 — L=5 Model Training",
        "date": "2026-10-27",
        "random_seed": 42,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
        },
        "predecessor_protocol_amendment": {
            "protocol_path": "results/phase6/protocol_l5_amendment_2_1.json",
            "amendment_version": "2.1",
            "status": "RATIFIED_PRE_REGISTERED_FROZEN",
        },
        "development_splits": {
            "train": {"windows": len(X_train_l5), "entities": meta_train["entity_id"].nunique(), "labels": pd.Series(y_train).value_counts().to_dict()},
            "val": {"windows": len(X_val_l5), "entities": meta_val["entity_id"].nunique(), "labels": pd.Series(y_val).value_counts().to_dict()},
            "test": {"windows": len(X_test_l5), "entities": meta_test["entity_id"].nunique(), "labels": pd.Series(y_test).value_counts().to_dict()},
        },
        "models": {},
    }

    validation_results: Dict[str, Any] = {
        "evaluation_partition": "Development Validation Set (m_val, N=1,015)",
        "disclaimer": "Development-validation results only. These do NOT represent confirmatory holdout performance.",
        "date_evaluated": "2026-10-27",
        "models": {},
    }

    # =========================================================================
    # MODEL 1: Random Forest Baseline
    # =========================================================================
    print("\n" + "=" * 78)
    print("[Model 1/4] Training Class-Weighted Random Forest (Tabular x_last in R^31)...")
    print("=" * 78)
    t0_rf = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_tab, y_train)
    t1_rf = time.time()
    rf_duration = t1_rf - t0_rf

    # Save RF checkpoint
    import joblib
    rf_ckpt_path = ckpt_dir / "random_forest_l5.joblib"
    joblib.dump(rf_model, rf_ckpt_path)
    print(f"  - RF training completed in {rf_duration:.2f}s | Checkpoint saved: {rf_ckpt_path}")

    # Evaluate RF on Validation
    rf_val_probs = rf_model.predict_proba(X_val_tab)
    rf_val_preds = rf_model.predict(X_val_tab)
    rf_val_metrics = compute_extended_validation_metrics(meta_val, y_val, rf_val_preds, rf_val_probs)
    print(f"  - Validation Macro-F1: {rf_val_metrics['macro_f1']:.4f} | Accuracy: {rf_val_metrics['accuracy']:.4f}")
    print(f"  - Validation FAR: {rf_val_metrics['entity_far']['far_pct']:.2f}% | FPR: {rf_val_metrics['window_fpr']['fpr_pct']:.2f}%")

    training_manifest["models"]["random_forest"] = {
        "model_name": "Random Forest",
        "architecture": "RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)",
        "input_shape": "[batch_size, 31]",
        "parameter_count": None,  # Non-parametric tree ensemble
        "training_duration_seconds": rf_duration,
        "checkpoint_path": str(rf_ckpt_path),
        "checkpoint_sha256": compute_sha256(rf_ckpt_path),
    }
    validation_results["models"]["random_forest"] = rf_val_metrics

    # =========================================================================
    # MODEL 2: Temporal LSTM
    # =========================================================================
    print("\n" + "=" * 78)
    print("[Model 2/4] Training Temporal LSTM (input: [N, 5, 31], hidden=64, layers=2)...")
    print("=" * 78)
    t0_lstm = time.time()
    set_seed(42)
    lstm_model = TemporalLSTMClassifier(
        input_dim=31,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2,
    ).to(device)

    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  - LSTM Parameter Count: {lstm_params:,}")

    # Frozen Class-Weighted Loss
    lstm_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32, device=device)
    lstm_criterion = nn.CrossEntropyLoss(weight=lstm_weights)
    lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    lstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        lstm_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_l5), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_l5), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    lstm_ckpt_path = ckpt_dir / "best_lstm_checkpoint_l5.pt"
    lstm_trainer = LSTMTrainer(
        model=lstm_model,
        optimizer=lstm_optimizer,
        criterion=lstm_criterion,
        device=device,
        scheduler=lstm_scheduler,
        monitor_metric="val_macro_f1",
        metric_mode="max",
    )
    lstm_history = lstm_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=35,
        patience=10,
        checkpoint_path=lstm_ckpt_path,
    )
    t1_lstm = time.time()
    lstm_duration = t1_lstm - t0_lstm

    # Load best checkpoint for validation evaluation
    ckpt_lstm = torch.load(lstm_ckpt_path, map_location=device, weights_only=False)
    lstm_model.load_state_dict(ckpt_lstm["model_state_dict"])
    lstm_model.eval()
    with torch.no_grad():
        lstm_val_logits = lstm_model(torch.from_numpy(X_val_l5).to(device))
        lstm_val_probs = torch.softmax(lstm_val_logits, dim=-1).cpu().numpy()
        lstm_val_preds = np.argmax(lstm_val_probs, axis=-1)

    lstm_val_metrics = compute_extended_validation_metrics(meta_val, y_val, lstm_val_preds, lstm_val_probs)
    print(f"  - LSTM training completed in {lstm_duration:.2f}s | Best Epoch: {lstm_history['best_epoch']}")
    print(f"  - Validation Macro-F1: {lstm_val_metrics['macro_f1']:.4f} | Accuracy: {lstm_val_metrics['accuracy']:.4f}")
    print(f"  - Validation FAR: {lstm_val_metrics['entity_far']['far_pct']:.2f}% | FPR: {lstm_val_metrics['window_fpr']['fpr_pct']:.2f}%")

    training_manifest["models"]["temporal_lstm"] = {
        "model_name": "Temporal LSTM",
        "architecture": "TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)",
        "input_shape": "[batch_size, 5, 31]",
        "parameter_count": lstm_params,
        "loss_function": "Weighted CrossEntropyLoss(weights=[1.0, 1.5, 2.0])",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=5, min_lr=1e-5)",
        "max_epochs": 35,
        "early_stopping_patience": 10,
        "best_epoch": lstm_history["best_epoch"],
        "best_val_macro_f1": float(lstm_history["best_metric"]),
        "training_duration_seconds": lstm_duration,
        "checkpoint_path": str(lstm_ckpt_path),
        "checkpoint_sha256": compute_sha256(lstm_ckpt_path),
        "history": lstm_history["history"],
    }
    validation_results["models"]["temporal_lstm"] = lstm_val_metrics

    # =========================================================================
    # MODEL 3: Compact Temporal Transformer
    # =========================================================================
    print("\n" + "=" * 78)
    print("[Model 3/4] Training Compact Temporal Transformer (input: [N, 5, 31], d_model=32, heads=4)...")
    print("=" * 78)
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
        max_seq_len=5,
    ).to(device)

    tf_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"  - Transformer Parameter Count: {tf_params:,}")

    tf_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32, device=device)
    tf_criterion = nn.CrossEntropyLoss(weight=tf_weights)
    tf_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=5e-4, weight_decay=1e-4)
    tf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        tf_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    tf_ckpt_path = ckpt_dir / "best_transformer_checkpoint_l5.pt"
    tf_trainer = LSTMTrainer(
        model=transformer_model,
        optimizer=tf_optimizer,
        criterion=tf_criterion,
        device=device,
        scheduler=tf_scheduler,
        monitor_metric="val_macro_f1",
        metric_mode="max",
    )
    tf_history = tf_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=35,
        patience=10,
        checkpoint_path=tf_ckpt_path,
    )
    t1_tf = time.time()
    tf_duration = t1_tf - t0_tf

    # Load best Transformer checkpoint
    ckpt_tf = torch.load(tf_ckpt_path, map_location=device, weights_only=False)
    transformer_model.load_state_dict(ckpt_tf["model_state_dict"])
    transformer_model.eval()
    with torch.no_grad():
        tf_val_logits = transformer_model(torch.from_numpy(X_val_l5).to(device))
        tf_val_probs = torch.softmax(tf_val_logits, dim=-1).cpu().numpy()
        tf_val_preds = np.argmax(tf_val_probs, axis=-1)

    tf_val_metrics = compute_extended_validation_metrics(meta_val, y_val, tf_val_preds, tf_val_probs)
    print(f"  - Transformer training completed in {tf_duration:.2f}s | Best Epoch: {tf_history['best_epoch']}")
    print(f"  - Validation Macro-F1: {tf_val_metrics['macro_f1']:.4f} | Accuracy: {tf_val_metrics['accuracy']:.4f}")
    print(f"  - Validation FAR: {tf_val_metrics['entity_far']['far_pct']:.2f}% | FPR: {tf_val_metrics['window_fpr']['fpr_pct']:.2f}%")

    training_manifest["models"]["compact_transformer"] = {
        "model_name": "Compact Temporal Transformer",
        "architecture": "CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_len=5)",
        "input_shape": "[batch_size, 5, 31]",
        "parameter_count": tf_params,
        "loss_function": "Weighted CrossEntropyLoss(weights=[1.0, 1.5, 2.0])",
        "optimizer": "AdamW(lr=5e-4, weight_decay=1e-4)",
        "scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=5, min_lr=1e-5)",
        "max_epochs": 35,
        "early_stopping_patience": 10,
        "best_epoch": tf_history["best_epoch"],
        "best_val_macro_f1": float(tf_history["best_metric"]),
        "training_duration_seconds": tf_duration,
        "checkpoint_path": str(tf_ckpt_path),
        "checkpoint_sha256": compute_sha256(tf_ckpt_path),
        "history": tf_history["history"],
    }
    validation_results["models"]["compact_transformer"] = tf_val_metrics

    # =========================================================================
    # MODEL 4: Representation-Fused Hybrid
    # =========================================================================
    print("\n" + "=" * 78)
    print("[Model 4/4] Training Representation Hybrid (Fused 32-dim latent + 31-dim tabular -> RF)...")
    print("=" * 78)
    t0_hyb = time.time()
    set_seed(42)
    hybrid_model = RepresentationHybridClassifier(
        temporal_backbone=transformer_model,
        device=device,
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
    )
    # Fit strictly on in-sample training features
    hybrid_model.fit(X_train_l5, y_train)
    t1_hyb = time.time()
    hyb_duration = t1_hyb - t0_hyb

    hyb_ckpt_path = ckpt_dir / "representation_hybrid_l5.joblib"
    joblib.dump(hybrid_model.rf, hyb_ckpt_path)
    print(f"  - Hybrid training completed in {hyb_duration:.2f}s | Checkpoint saved: {hyb_ckpt_path}")

    # Evaluate Hybrid on Validation
    hyb_val_probs = hybrid_model.predict_proba(X_val_l5)
    hyb_val_preds = hybrid_model.predict(X_val_l5)
    hyb_val_metrics = compute_extended_validation_metrics(meta_val, y_val, hyb_val_preds, hyb_val_probs)
    print(f"  - Validation Macro-F1: {hyb_val_metrics['macro_f1']:.4f} | Accuracy: {hyb_val_metrics['accuracy']:.4f}")
    print(f"  - Validation FAR: {hyb_val_metrics['entity_far']['far_pct']:.2f}% | FPR: {hyb_val_metrics['window_fpr']['fpr_pct']:.2f}%")

    training_manifest["models"]["representation_hybrid"] = {
        "model_name": "Representation Hybrid",
        "architecture": "Frozen Compact Transformer encoder backbone (32-dim latent embedding) concatenated with 31-dim tabular vector, classified by Random Forest",
        "input_shape": "[batch_size, 63]",
        "parameter_count": tf_params,  # Transformer backbone params + RF ensemble
        "temporal_backbone_checkpoint": str(tf_ckpt_path),
        "embedding_dimension": 32,
        "tabular_dimension": 31,
        "fused_dimension": 63,
        "training_duration_seconds": hyb_duration,
        "checkpoint_path": str(hyb_ckpt_path),
        "checkpoint_sha256": compute_sha256(hyb_ckpt_path),
    }
    validation_results["models"]["representation_hybrid"] = hyb_val_metrics

    # Total duration
    t_total = time.time() - t_start
    training_manifest["total_training_duration_seconds"] = t_total
    training_manifest["final_verdict"] = "TRAINING_COMPLETE_READY_FOR_CONFIRMATORY_EVALUATION"

    # Save manifest and validation metrics JSON files
    clean_manifest = convert_numpy_types(training_manifest)
    clean_val_results = convert_numpy_types(validation_results)

    manifest_path = PROJECT_ROOT / "results" / "phase6" / "training_manifest_l5.json"
    with open(manifest_path, "w") as f:
        json.dump(clean_manifest, f, indent=2)

    val_metrics_path = PROJECT_ROOT / "results" / "phase6" / "validation_metrics_l5.json"
    with open(val_metrics_path, "w") as f:
        json.dump(clean_val_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 6.3 TRAINING SUMMARY TABLE (DEVELOPMENT VALIDATION SET)")
    print("=" * 80)
    print(f"{'Model Name':<30} {'Macro-F1':<10} {'Accuracy':<10} {'Warn-Rec':<10} {'Crit-Rec':<10} {'FAR %':<8} {'FPR %':<8}")
    print("-" * 80)
    for m_key, m_res in validation_results["models"].items():
        m_name = training_manifest["models"][m_key]["model_name"]
        print(f"{m_name:<30} {m_res['macro_f1']:<10.4f} {m_res['accuracy']:<10.4f} {m_res['warning_recall']:<10.4f} {m_res['critical_recall']:<10.4f} {m_res['entity_far']['far_pct']:<8.2f} {m_res['window_fpr']['fpr_pct']:<8.2f}")
    print("-" * 80)

    print(f"\nSaved artifacts:")
    print(f"  - Training manifest: {manifest_path}")
    print(f"  - Validation metrics: {val_metrics_path}")
    print(f"  - Checkpoints: {ckpt_dir}")
    print(f"Final Verdict: {training_manifest['final_verdict']}")

    return clean_manifest, clean_val_results


if __name__ == "__main__":
    train_l5_models()
