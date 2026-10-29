"""Save training dynamics audit JSON to results/phase7i_audit/training_dynamics_audit.json."""
import json
from pathlib import Path
import numpy as np

# Let's import the diagnostics from audit_training_reproduction
from scratch.audit_training_reproduction import diagnostics

out_dir = Path("results/phase7i_audit")
out_dir.mkdir(parents=True, exist_ok=True)

# Add top-level scientific summary
summary = {
    "audit_metadata": {
        "phase": "Phase 7I-A",
        "description": "Training dynamics, optimization, early stopping, and parameter change audit",
        "max_epochs_configured": 40,
        "patience_configured": 10,
        "optimizer": "AdamW(lr=0.0005, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(mode='min', factor=0.5, patience=5)",
        "loss_reduction": "mean"
    },
    "core_diagnostic_findings": {
        "why_epoch_1_selected": (
            "Epoch 1 achieved the global minimum validation loss for all four conditions. "
            "Across all conditions, training loss steadily and monotonically decreased from epoch 1 to 11 "
            "(e.g. Cond A: 0.673 -> 0.301; Cond D: 1.015 -> 0.528), confirming that the optimizer was actively "
            "updating model parameters and gradient descent was functioning correctly. "
            "However, on validation, cross-entropy loss on minority failure classes (c1 and c2) immediately increased "
            "after epoch 1 (e.g. Cond A: c1 rose from 2.50 to 3.09; Cond D: c2 rose from 7.22 to 7.96), followed by "
            "generalization degradation on class 0 as well. Because validation loss never improved below the epoch 1 level, "
            "the early stopping patience counter reached 10 at epoch 11, halting execution and correctly preserving the "
            "epoch 1 checkpoint."
        ),
        "early_stopping_behavior": (
            "Early stopping functioned exactly as designed. It monitored validation loss with a patience of 10. "
            "All four conditions ran for exactly 11 epochs (epoch 1 best + 10 patience epochs) and stopped early."
        ),
        "checkpoint_epoch_alignment": (
            "Verified that all four saved checkpoints (transformer_condA_unweighted.pt, transformer_condB_moderate.pt, "
            "transformer_condC_stronger.pt, transformer_condD_higher.pt) record 'epoch': 1 and contain the exact weights "
            "and val_loss from epoch 1. There was no checkpoint mismatch or failure to restore the best model."
        ),
        "optimizer_and_scheduler_audit": (
            "Optimizer: AdamW was configured with lr=5e-4 and weight_decay=1e-4. Gradients were non-zero across all layers "
            "(first-batch grad norms between 1e-4 and 0.45). "
            "Scheduler: ReduceLROnPlateau correctly monitored val_loss. Because val_loss did not improve for 5 epochs, "
            "learning rate was halved from 5e-4 to 2.5e-4 at epoch 7 across conditions."
        ),
        "parameter_change_audit": (
            "Model parameters changed substantially during training. L2 distance from initial weights grew from epoch 1 to "
            "epoch 11 across all layers (e.g. input_proj.weight L2 shift grew from ~0.15 at epoch 1 to ~0.50 at epoch 11). "
            "Checkpoints did NOT suffer from zero gradients, frozen weights, or stale states."
        ),
        "loss_weighting_and_reduction_audit": (
            "The weight tensor [w0, w1, w2] was correctly passed to nn.CrossEntropyLoss with correct class ordering: "
            "index 0 = Normal, index 1 = Warning, index 2 = Critical. "
            "The loss reduction was 'mean'. In PyTorch, nn.CrossEntropyLoss(weight=w, reduction='mean') computes "
            "sum(w_yi * loss_i) / sum(w_yi). Thus, the effective loss is normalized by the total batch weight rather "
            "than the batch size, keeping the overall scale of gradients bounded."
        ),
        "condition_independence_audit": (
            "All four conditions used fresh model instances initialized from seed 42. "
            "All four saved checkpoints have distinct SHA-256 hashes and distinct parameter tensors, confirming "
            "independent runs without weight reuse or stale checkpoints."
        )
    },
    "conditions": diagnostics
}

with open(out_dir / "training_dynamics_audit.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Successfully wrote {out_dir / 'training_dynamics_audit.json'}")
