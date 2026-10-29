"""Validation-only LSTM experiments for Phase 3C.

Evaluates a predeclared, controlled set of architectural and loss variants
strictly on Training and Validation sets:
1. Exp 1: 2-layer LSTM, hidden_dim=64, balanced weights (Baseline Phase 3B)
2. Exp 2: 1-layer LSTM, hidden_dim=64, balanced weights (Reduced depth)
3. Exp 3: 2-layer LSTM, hidden_dim=32, balanced weights (Reduced width)
4. Exp 4: 1-layer LSTM, hidden_dim=32, balanced weights (Minimal capacity)
5. Exp 5: 2-layer LSTM, hidden_dim=64, critical weight boosted 1.5x

CRITICAL RULE: The test set is NOT evaluated during these experiments.
All model selection is governed exclusively by validation metrics.
"""

import sys
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
from src.evaluation.metrics import compute_episode_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_validation_experiments():
    print("=" * 78)
    print("PHASE 3C: VALIDATION-ONLY LSTM EXPERIMENTS")
    print("=" * 78)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Random Seed: 42 (Frozen)")

    # 1. Load Training and Validation Sets Only
    data_dir = Path("data/processed")
    X_tr = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_tr = np.load(data_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    m_val = pd.read_csv(data_dir / "val_metadata.csv")

    print(f"Loaded Train: {X_tr.shape}, Val: {X_val.shape}")

    batch_size = 64
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    base_weights = compute_training_class_weights(y_tr)
    crit_boost_weights = base_weights.clone()
    crit_boost_weights[2] *= 1.5  # Boost critical penalty

    # Predeclared experiment matrix
    experiments = [
        {
            "id": "exp1_lstm_2l_h64_balanced",
            "name": "2-Layer LSTM, Hidden=64, Balanced Weights (Phase 3B Baseline)",
            "num_layers": 2,
            "hidden_dim": 64,
            "dropout": 0.2,
            "weights": base_weights,
        },
        {
            "id": "exp2_lstm_1l_h64_balanced",
            "name": "1-Layer LSTM, Hidden=64, Balanced Weights (Reduced Depth)",
            "num_layers": 1,
            "hidden_dim": 64,
            "dropout": 0.2,
            "weights": base_weights,
        },
        {
            "id": "exp3_lstm_2l_h32_balanced",
            "name": "2-Layer LSTM, Hidden=32, Balanced Weights (Reduced Width)",
            "num_layers": 2,
            "hidden_dim": 32,
            "dropout": 0.2,
            "weights": base_weights,
        },
        {
            "id": "exp4_lstm_1l_h32_balanced",
            "name": "1-Layer LSTM, Hidden=32, Balanced Weights (Minimal Capacity)",
            "num_layers": 1,
            "hidden_dim": 32,
            "dropout": 0.0,
            "weights": base_weights,
        },
        {
            "id": "exp5_lstm_2l_h64_critboost",
            "name": "2-Layer LSTM, Hidden=64, Critical Boost (1.5x Critical Weight)",
            "num_layers": 2,
            "hidden_dim": 64,
            "dropout": 0.2,
            "weights": crit_boost_weights,
        },
    ]

    exp_results = []

    for exp in experiments:
        set_seed(42)
        print(f"\nRunning {exp['id']}: {exp['name']}...")
        model = TemporalLSTMClassifier(
            input_dim=31,
            hidden_dim=exp["hidden_dim"],
            num_layers=exp["num_layers"],
            num_classes=3,
            dropout=exp["dropout"],
            bidirectional=False,
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        criterion = nn.CrossEntropyLoss(weight=exp["weights"].to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        trainer = LSTMTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            monitor_metric="val_macro_f1",
            metric_mode="max",
        )

        ckpt_temp = Path(f"models/checkpoints/{exp['id']}.pt")
        fit_res = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=25,
            patience=8,
            checkpoint_path=ckpt_temp,
        )

        # Evaluate best checkpoint on validation set
        val_loss, val_preds, val_probs, val_metrics = trainer.evaluate(val_loader)
        val_ep_metrics = compute_episode_metrics(m_val, val_preds, val_probs)

        record = {
            "experiment_id": exp["id"],
            "name": exp["name"],
            "num_layers": exp["num_layers"],
            "hidden_dim": exp["hidden_dim"],
            "dropout": exp["dropout"],
            "total_parameters": total_params,
            "best_epoch": fit_res["best_epoch"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_normal_f1": val_metrics["class_metrics"]["NORMAL"]["f1"],
            "val_warning_recall": val_metrics["warning_recall"],
            "val_warning_f1": val_metrics["class_metrics"]["WARNING"]["f1"],
            "val_critical_recall": val_metrics["critical_recall"],
            "val_critical_f1": val_metrics["class_metrics"]["CRITICAL"]["f1"],
            "val_warning_pr_auc": val_metrics["pr_auc"]["pr_auc_warning"],
            "val_critical_pr_auc": val_metrics["pr_auc"]["pr_auc_critical"],
            "val_episode_detection_pct": val_ep_metrics["detection_rate_pct"],
            "val_lead_time_mean_min": val_ep_metrics["lead_time_mean_min"],
            "val_false_alarm_entity_pct": val_ep_metrics["false_alarm_entity_rate_pct"],
        }
        exp_results.append(record)

        print(
            f"  -> Best Epoch {fit_res['best_epoch']}: "
            f"Val Macro-F1 = {val_metrics['macro_f1']:.4f} | "
            f"Val Warn Rec = {val_metrics['warning_recall']:.4f} | "
            f"Val Crit Rec = {val_metrics['critical_recall']:.4f} | "
            f"Val Ep Det = {val_ep_metrics['detection_rate_pct']:.1f}%"
        )

    # Summary table
    print("\n" + "=" * 78)
    print("VALIDATION EXPERIMENT COMPARISON SUMMARY (TRAIN/VAL ONLY)")
    print("=" * 78)
    df_res = pd.DataFrame(exp_results)
    cols_to_print = [
        "experiment_id", "total_parameters", "best_epoch", "val_macro_f1",
        "val_warning_recall", "val_critical_recall", "val_episode_detection_pct",
        "val_false_alarm_entity_pct"
    ]
    print(df_res[cols_to_print].to_string(index=False))

    # Save machine-readable results
    out_dir = Path("results/phase3c")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation_experiments.json", "w") as f:
        json.dump(exp_results, f, indent=2)
    print(f"\nSaved validation experiment results to {out_dir / 'validation_experiments.json'}")

    return exp_results


if __name__ == "__main__":
    run_validation_experiments()
