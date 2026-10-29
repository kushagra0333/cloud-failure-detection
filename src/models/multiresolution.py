"""Multi-Resolution Temporal Sequence Models for Failure Prediction.

Implements the audited dual-stream architecture from Phase 7M:
- Stream A: Compact 2-layer LSTM encoding recent high-resolution telemetry (L=5, F=31).
- Stream B: Multi-layer Perceptron encoding causal historical trend summaries (F=93).
- Fused Head: Concatenates representations and projects to 3 target classes.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamBEncoder(nn.Module):
    """Encodes 93 historical trend summary features into compact representation."""
    def __init__(self, in_dim: int = 93, hidden_dim: int = 32, out_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiResolutionLSTMClassifier(nn.Module):
    """Multi-resolution temporal classifier combining Stream A and Stream B."""
    def __init__(
        self,
        input_dim: int = 31,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        use_history: bool = True,
        hist_dim: int = 93,
        hist_hidden: int = 32,
        hist_out: int = 16
    ):
        super().__init__()
        self.use_history = use_history
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        if use_history:
            self.hist_encoder = StreamBEncoder(
                in_dim=hist_dim,
                hidden_dim=hist_hidden,
                out_dim=hist_out,
                dropout=dropout
            )
            head_in = hidden_dim + hist_out
        else:
            self.hist_encoder = None
            head_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_recent: torch.Tensor, x_hist: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(x_recent)
        z_rec = lstm_out[:, -1, :]
        if self.use_history and x_hist is not None:
            z_hist = self.hist_encoder(x_hist)
            z = torch.cat([z_rec, z_hist], dim=-1)
        else:
            z = z_rec
        logits = self.classifier(z)
        return logits


class AsymmetricCriticalMarginLoss(nn.Module):
    """Asymmetric Critical Margin Loss for Phase 7N.
    
    Combines weighted multiclass cross-entropy with a softplus margin penalty
    on true Critical training samples (target == 2):
        L_margin = mean_{i: y_i=2} (softplus(margin - (logit_2 - logit_1)))
        L_total = L_ce + lambda_crit * L_margin
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, margin: float = 0.5, lambda_crit: float = 0.10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        self.margin = margin
        self.lambda_crit = lambda_crit

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self.ce(logits, targets)
        crit_mask = (targets == 2)
        if crit_mask.any():
            crit_logits = logits[crit_mask]
            margin_diff = self.margin - (crit_logits[:, 2] - crit_logits[:, 1])
            margin_loss = F.softplus(margin_diff).mean()
        else:
            margin_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        total_loss = ce_loss + self.lambda_crit * margin_loss
        return total_loss, ce_loss, margin_loss


class MulticlassFocalLoss(nn.Module):
    """Multiclass Focal Loss with class weighting."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - target_probs) ** self.gamma
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            focal_loss = -alpha_weights * focal_weight * target_log_probs
        else:
            focal_loss = -focal_weight * target_log_probs

        if self.reduction == "mean":
            if self.alpha is not None:
                return focal_loss.sum() / self.alpha[targets].sum()
            return focal_loss.mean()
        return focal_loss
