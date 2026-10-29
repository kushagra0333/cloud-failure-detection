"""Representation-Fused Hybrid Classifier for Early Failure Prediction.

Implements Phase 4 Hybrid Architecture:
- Fuses dense temporal representations (d=64) from trained temporal encoders with 
  instantaneous tabular summary features (d=31) to produce a 95-dimensional fused feature vector.
- Fits a Class-Weighted Random Forest directly on the fused representation strictly on training data.
- Avoids validation-gating leakage: no validation data is used to fit fusion parameters or thresholds.
- Combines sequential representation learning with tree-based orthogonal decision boundaries.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class RepresentationHybridClassifier:
    """Hybrid classifier fusing deep temporal embeddings with tabular decision boundaries."""

    def __init__(
        self,
        temporal_backbone: nn.Module,
        device: torch.device,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.temporal_backbone = temporal_backbone.to(device)
        self.device = device
        self.temporal_backbone.eval()

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def extract_fused_features(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Extracts and concatenates temporal embeddings with instantaneous features.
        
        Args:
            X: Input array of shape (N, seq_len=15, input_dim=31)
            batch_size: Batch size for neural inference
            
        Returns:
            Fused feature array of shape (N, 64 + 31 = 95)
        """
        self.temporal_backbone.eval()
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
                if hasattr(self.temporal_backbone, "get_embedding"):
                    emb = self.temporal_backbone.get_embedding(batch_x)
                elif hasattr(self.temporal_backbone, "lstm"):
                    lstm_out, _ = self.temporal_backbone.lstm(batch_x)
                    emb = lstm_out[:, -1, :]
                else:
                    raise AttributeError("Backbone does not support embedding extraction.")
                embeddings_list.append(emb.cpu().numpy())

        H_temp = np.concatenate(embeddings_list, axis=0)  # Shape: (N, 64)
        X_tabular = X[:, -1, :]  # Instantaneous features at prediction anchor t_pred (N, 31)

        Z = np.concatenate([H_temp, X_tabular], axis=1)  # Shape: (N, 95)
        return Z

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RepresentationHybridClassifier":
        """Fits Random Forest on fused features extracted from training data."""
        Z_train = self.extract_fused_features(X_train)
        self.rf.fit(Z_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class probabilities."""
        assert self.is_fitted, "Hybrid model must be fitted before predict_proba."
        Z = self.extract_fused_features(X)
        return self.rf.predict_proba(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels."""
        assert self.is_fitted, "Hybrid model must be fitted before predict."
        Z = self.extract_fused_features(X)
        return self.rf.predict(Z)
