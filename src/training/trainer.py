"""Training engine, early stopping, and checkpoint management for PyTorch deep models."""

from typing import Dict, Any, Optional, Tuple, List
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_window_metrics


def compute_training_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Computes balanced inverse-class weights strictly from training labels.
    
    Formula: w_c = N_train / (n_classes * N_train_c)
    """
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)
    weights = np.zeros(n_classes, dtype=np.float32)

    for c, count in zip(classes, counts):
        weights[c] = n_samples / (n_classes * float(count))

    return torch.tensor(weights, dtype=torch.float32)


class LSTMTrainer:
    """Manages training loop, validation-driven early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        monitor_metric: str = "val_macro_f1",
        metric_mode: str = "max",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.metric_mode = metric_mode

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Trains the model for one full epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Evaluates model on dataloader.
        
        Returns:
            avg_loss: Average cross-entropy loss
            all_preds: 1D array of predicted classes
            all_probs: 2D array of predicted class probabilities
            metrics: Dictionary of window metrics (macro_f1, per-class P/R/F1, PR-AUC)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        preds_list = []
        probs_list = []
        targets_list = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += float(loss.item())
                n_batches += 1

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                preds_list.append(preds.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
                targets_list.append(y_batch.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = np.concatenate(preds_list, axis=0)
        all_probs = np.concatenate(probs_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        metrics = compute_window_metrics(all_targets, all_preds, all_probs)
        return avg_loss, all_preds, all_probs, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 35,
        patience: int = 10,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Trains model with early stopping on validation metric.
        
        Saves best checkpoint to disk and returns training history.
        """
        best_metric = -float("inf") if self.metric_mode == "max" else float("inf")
        best_state = None
        best_epoch = -1
        patience_counter = 0

        history: List[Dict[str, Any]] = []

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_preds, val_probs, val_metrics = self.evaluate(val_loader)
            val_macro_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "val_warning_recall": val_metrics["warning_recall"],
                "val_critical_recall": val_metrics["critical_recall"],
                "lr": current_lr,
            }
            history.append(epoch_record)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.monitor_metric == "val_loss" else val_macro_f1)
                else:
                    self.scheduler.step()

            # Early stopping check
            current_metric = val_macro_f1 if self.monitor_metric == "val_macro_f1" else val_loss
            is_better = (
                (current_metric > best_metric) if self.metric_mode == "max"
                else (current_metric < best_metric)
            )

            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())

                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "best_metric": best_metric,
                        "monitor_metric": self.monitor_metric,
                        "val_metrics": val_metrics,
                    }, checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or is_better:
                marker = "*" if is_better else " "
                print(
                    f"Epoch {epoch:2d}/{max_epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Macro-F1: {val_macro_f1:.4f} {marker} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.2e}",
                    flush=True,
                )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best {self.monitor_metric}: {best_metric:.4f})")
                break

        # Load best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model checkpoint from epoch {best_epoch} ({self.monitor_metric} = {best_metric:.4f})")

        return {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "monitor_metric": self.monitor_metric,
            "history": history,
        }
