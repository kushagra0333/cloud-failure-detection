"""Temporal LSTM Classifier for cloud task failure prediction.

Implements Phase 3B Recurrent Deep Architecture:
- Multi-layer Temporal LSTM consuming (N, L=15, F=31) tensors.
- Causal sequential hidden-state accumulation without manual feature flattening.
- Dropout regularization and linear multi-class projection head.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTMClassifier(nn.Module):
    """Causal multi-layer LSTM classifier for multi-class early failure prediction."""

    def __init__(
        self,
        input_dim: int = 31,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=15, input_dim=31)
            
        Returns:
            Logits of shape (batch_size, num_classes=3)
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)

        # Use the final hidden state of the sequence (causal summary at t_pred)
        # lstm_out[:, -1, :] corresponds to timestep L=15
        final_state = lstm_out[:, -1, :]
        final_state = self.dropout(final_state)

        logits = self.classifier(final_state)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for input tensor."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
