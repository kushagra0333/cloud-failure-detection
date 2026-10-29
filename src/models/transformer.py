"""Compact Temporal Transformer Encoder for cloud task failure prediction.

Implements Phase 4 Temporal Attention Architecture:
- Consumes sequential windows of shape (N, L=15, F=31).
- Linear input projection to compact latent dimension d_model=64.
- Documented sinusoidal / learnable positional encoding over historical steps.
- 2-layer TransformerEncoder with multi-head self-attention (nhead=4, d_ff=128, dropout=0.2).
- Causal temporal pooling at prediction anchor (t_pred = step 15).
- Linear multi-class classification projection head (NORMAL, WARNING, CRITICAL).
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length sequence models."""

    def __init__(self, d_model: int, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input tensor.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CompactTemporalTransformer(nn.Module):
    """Compact Transformer Encoder for early failure anticipation in cloud traces."""

    def __init__(
        self,
        input_dim: int = 31,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
        max_seq_len: int = 15,
        pool_mode: str = "last",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.pool_mode = pool_mode

        # 1. Linear input projection: F -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Normalization and Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 5. Multi-Class Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense pooled temporal representation from input sequences.
        
        Args:
            x: Tensor of shape (batch_size, seq_len=15, input_dim=31)
            
        Returns:
            Representation tensor of shape (batch_size, d_model=64)
        """
        # Linear projection: (B, L, F) -> (B, L, d_model)
        h = self.input_proj(x)

        # Add positional encoding
        h = self.pos_encoder(h)

        # Pass through Transformer encoder: (B, L, d_model)
        h = self.transformer_encoder(h)
        h = self.norm(h)

        # Pool temporal representation
        if self.pool_mode == "last":
            # Causal anchor: last historical observation at prediction time (t_pred)
            pooled = h[:, -1, :]
        elif self.pool_mode == "mean":
            pooled = h.mean(dim=1)
        elif self.pool_mode == "max":
            pooled, _ = h.max(dim=1)
        else:
            pooled = h[:, -1, :]

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=15, input_dim=31)
            
        Returns:
            Logits of shape (batch_size, num_classes=3)
        """
        pooled = self.get_embedding(x)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
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
