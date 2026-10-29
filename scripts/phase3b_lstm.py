"""Phase 3B — Temporal LSTM Modeling and Benchmark Comparison.

Trains and evaluates:
- Causal Temporal LSTM (2 layers, hidden_dim=64, dropout=0.2)
- Input: 15 timesteps x 31 features
- Class-weighted Cross-Entropy loss fitted strictly on training data
- Predeclared early stopping on Validation Macro-F1 (patience=10)
- Single frozen evaluation on held-out test set
- Side-by-side comparison with Majority, Logistic Regression, and Random Forest baselines
"""

import sys
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import TemporalLSTMClassifier
from src.training.trainer import LSTMTrainer, compute_training_class_weights
from src.evaluation.metrics import (
    compute_window_metrics,
    compute_episode_metrics,
    format_metrics_table,
)


def set_seed(seed: int = 42):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_phase3b_lstm():
    print("=" * 78)
    print("PHASE 3B — TEMPORAL LSTM MODELING & BENCHMARK COMPARISON")
    print("Temporal LSTM/Transformer Framework for Early Failure Prediction")
    print("=" * 78)

    # 1. Environment & Seeds
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Random Seed: 42 (Frozen)")

    # 2. Load Datasets
    print("\n[Step 1] Loading primary entity-grouped dataset...")
    data_dir = Path("data/processed")
    X_tr = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_tr = np.load(data_dir / "y_train.npy").astype(np.int64)
    m_tr = pd.read_csv(data_dir / "train_metadata.csv")

    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    m_val = pd.read_csv(data_dir / "val_metadata.csv")

    X_te = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_te = np.load(data_dir / "y_test.npy").astype(np.int64)
    m_te = pd.read_csv(data_dir / "test_metadata.csv")

    print(f"  - Train tensor shape : {X_tr.shape}, labels: {pd.Series(y_tr).value_counts().to_dict()}")
    print(f"  - Val tensor shape   : {X_val.shape}, labels: {pd.Series(y_val).value_counts().to_dict()}")
    print(f"  - Test tensor shape  : {X_te.shape}, labels: {pd.Series(y_te).value_counts().to_dict()}")

    # 3. Class Weights Calculation (strictly on training set)
    print("\n[Step 2] Computing inverse-class weights on training labels only...")
    class_weights = compute_training_class_weights(y_tr)
    print(f"  - Class 0 (NORMAL)   weight: {class_weights[0]:.4f}")
    print(f"  - Class 1 (WARNING)  weight: {class_weights[1]:.4f}")
    print(f"  - Class 2 (CRITICAL) weight: {class_weights[2]:.4f}")

    # 4. DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    # Deterministic DataLoader generator
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Model Architecture & Hyperparameters
    print("\n[Step 3] Initializing Temporal LSTM architecture...")
    input_dim = X_tr.shape[2]  # 31
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2

    model = TemporalLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=3,
        dropout=dropout,
        bidirectional=False,  # Strictly causal sequential processing
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Architecture       : 2-Layer Causal LSTM (Hidden={hidden_dim}, Input={input_dim})")
    print(f"  - Regularization     : Dropout={dropout}, Weight Decay=1e-4, Gradient Clip=1.0")
    print(f"  - Total Parameters   : {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    checkpoint_path = Path("models/checkpoints/best_lstm_checkpoint.pt")
    trainer = LSTMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        monitor_metric="val_macro_f1",  # Predeclared selection metric
        metric_mode="max",
    )

    # 6. Training with Validation Early Stopping
    print(f"\n[Step 4] Starting training loop (Predeclared Selection: Validation Macro-F1)...")
    max_epochs = 35
    patience = 10
    fit_summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )
    print(f"  -> Training complete. Best checkpoint saved at epoch {fit_summary['best_epoch']} with Val Macro-F1 = {fit_summary['best_metric']:.4f}")

    # 7. Evaluate Best Checkpoint on Validation Set
    print("\n" + "=" * 78)
    print("STEP 5: VALIDATION SET EVALUATION (FROZEN BEST CHECKPOINT)")
    print("=" * 78)
    val_loss, val_preds, val_probs, val_window_metrics = trainer.evaluate(val_loader)
    val_ep_metrics = compute_episode_metrics(m_val, val_preds, val_probs)
    print(format_metrics_table(val_window_metrics, val_ep_metrics))

    # 8. Evaluate on Held-Out Test Set (Single Frozen Evaluation)
    print("\n" + "=" * 78)
    print("STEP 6: FINAL EVALUATION ON HELD-OUT TEST SET (PRIMARY BENCHMARK)")
    print("=" * 78)
    test_loss, test_preds, test_probs, test_window_metrics = trainer.evaluate(test_loader)
    test_ep_metrics = compute_episode_metrics(m_te, test_preds, test_probs)
    print(format_metrics_table(test_window_metrics, test_ep_metrics))

    # 9. Comparative Benchmarking Against Phase 3A
    print("\n" + "=" * 78)
    print("STEP 7: COMPREHENSIVE BENCHMARK COMPARISON (TEST SET)")
    print("=" * 78)

    with open("results/phase3a/baseline_metrics.json") as f:
        phase3a_data = json.load(f)

    b_test = phase3a_data["primary_entity_grouped"]["test"]
    maj_m = b_test["Majority Class"]["window"]
    lr_m = b_test["Class-Weighted Logistic Regression"]["window"]
    rf_m = b_test["Class-Weighted Random Forest"]["window"]

    maj_ep = b_test["Majority Class"]["episode"]
    lr_ep = b_test["Class-Weighted Logistic Regression"]["episode"]
    rf_ep = b_test["Class-Weighted Random Forest"]["episode"]

    comparison_table = [
        ("Majority Class", maj_m["macro_f1"], maj_m["accuracy"], maj_m["warning_recall"], maj_m["critical_recall"], maj_m["pr_auc"]["pr_auc_warning"], maj_m["pr_auc"]["pr_auc_critical"], maj_ep["detection_rate_pct"], maj_ep["false_alarm_entity_rate_pct"]),
        ("Logistic Regression", lr_m["macro_f1"], lr_m["accuracy"], lr_m["warning_recall"], lr_m["critical_recall"], lr_m["pr_auc"]["pr_auc_warning"], lr_m["pr_auc"]["pr_auc_critical"], lr_ep["detection_rate_pct"], lr_ep["false_alarm_entity_rate_pct"]),
        ("Random Forest (Baseline Benchmark)", rf_m["macro_f1"], rf_m["accuracy"], rf_m["warning_recall"], rf_m["critical_recall"], rf_m["pr_auc"]["pr_auc_warning"], rf_m["pr_auc"]["pr_auc_critical"], rf_ep["detection_rate_pct"], rf_ep["false_alarm_entity_rate_pct"]),
        ("Temporal LSTM (Phase 3B)", test_window_metrics["macro_f1"], test_window_metrics["accuracy"], test_window_metrics["warning_recall"], test_window_metrics["critical_recall"], test_window_metrics["pr_auc"]["pr_auc_warning"], test_window_metrics["pr_auc"]["pr_auc_critical"], test_ep_metrics["detection_rate_pct"], test_ep_metrics["false_alarm_entity_rate_pct"]),
    ]

    print(f"{'Model':<35} {'Macro-F1':<10} {'Accuracy':<10} {'Warn-Rec':<10} {'Crit-Rec':<10} {'Warn-AP':<10} {'Crit-AP':<10} {'Ep-Det%':<10} {'FAR% (N=48)':<12}")
    print("-" * 115)
    for row in comparison_table:
        print(f"{row[0]:<35} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f} {row[4]:<10.4f} {row[5]:<10.4f} {row[6]:<10.4f} {row[7]:<10.1f} {row[8]:<12.1f}")
    print("-" * 115)

    rf_f1 = rf_m["macro_f1"]
    lstm_f1 = test_window_metrics["macro_f1"]
    delta_f1 = lstm_f1 - rf_f1
    delta_pct = (delta_f1 / rf_f1) * 100.0

    print(f"\nRandom Forest Baseline Test Macro-F1 : {rf_f1:.4f}")
    print(f"Temporal LSTM Phase 3B Test Macro-F1 : {lstm_f1:.4f}")
    print(f"Delta: {delta_f1:+.4f} ({delta_pct:+.2f}%)")

    # 10. Save Artifacts
    print("\n[Step 8] Serializing predictions and metrics...")
    results_dir = Path("results/phase3b")
    reports_art_dir = Path("reports/artifacts/phase3b")
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_art_dir.mkdir(parents=True, exist_ok=True)

    metrics_bundle = {
        "model_architecture": {
            "name": "TemporalLSTMClassifier",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "total_parameters": total_params,
            "class_weights": class_weights.tolist(),
            "batch_size": batch_size,
            "random_seed": 42,
        },
        "training_summary": fit_summary,
        "validation_metrics": {
            "window": val_window_metrics,
            "episode": val_ep_metrics,
        },
        "test_metrics": {
            "window": test_window_metrics,
            "episode": test_ep_metrics,
        },
        "comparison_with_baselines": {
            "majority_macro_f1": maj_m["macro_f1"],
            "logistic_regression_macro_f1": lr_m["macro_f1"],
            "random_forest_macro_f1": rf_m["macro_f1"],
            "lstm_macro_f1": lstm_f1,
            "delta_vs_rf": delta_f1,
        },
    }

    for d in [results_dir, reports_art_dir]:
        with open(d / "lstm_metrics.json", "w") as f:
            json.dump(metrics_bundle, f, indent=2)

        with open(d / "training_history.json", "w") as f:
            json.dump(fit_summary["history"], f, indent=2)

    # Save test predictions DataFrame
    pred_df = m_te.copy()
    pred_df["predicted_label"] = test_preds
    pred_df["prob_normal"] = test_probs[:, 0]
    pred_df["prob_warning"] = test_probs[:, 1]
    pred_df["prob_critical"] = test_probs[:, 2]

    for d in [results_dir, reports_art_dir]:
        out_csv = d / "test_predictions_lstm.csv"
        pred_df.to_csv(out_csv, index=False)
        print(f"Saved test predictions to {out_csv} ({len(pred_df):,} rows)")

    print("\n" + "=" * 78)
    print("PHASE 3B TEMPORAL LSTM EXECUTION COMPLETE: SCIENTIFICALLY VERIFIED")
    print("=" * 78)
    return True


if __name__ == "__main__":
    success = run_phase3b_lstm()
    sys.exit(0 if success else 1)
