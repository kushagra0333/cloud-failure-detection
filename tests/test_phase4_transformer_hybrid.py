"""Unit tests for Phase 4 Compact Temporal Transformer and Hybrid Classifier."""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn

from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier


def test_transformer_forward_shape_and_classes():
    """Verifies that CompactTemporalTransformer produces valid logits, probabilities, and class indices."""
    model = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2, num_classes=3)
    batch_size = 8
    seq_len = 15
    input_dim = 31

    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    assert logits.shape == (batch_size, 3)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size, 3)
    # Check probabilities sum to 1
    prob_sums = probs.sum(dim=-1).numpy()
    np.testing.assert_allclose(prob_sums, np.ones(batch_size), atol=1e-5)

    preds = model.predict(x)
    assert preds.shape == (batch_size,)
    assert set(preds.numpy()).issubset({0, 1, 2})


def test_transformer_embedding_extraction():
    """Verifies that get_embedding extracts the 64-dimensional causal temporal vector."""
    model = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(5, 15, 31)

    emb = model.get_embedding(x)
    assert emb.shape == (5, 64)
    assert torch.isfinite(emb).all()


def test_transformer_finite_loss_and_gradients():
    """Verifies finite cross-entropy loss and gradient propagation to all parameters."""
    model = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(16, 15, 31)
    y = torch.randint(0, 3, (16,))

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"


def test_transformer_checkpoint_loading():
    """Verifies that the best Transformer checkpoint exists and can be loaded cleanly."""
    ckpt_path = Path("models/checkpoints/best_transformer_checkpoint.pt")
    if not ckpt_path.exists():
        pytest.skip("Transformer checkpoint not yet saved.")

    model = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "best_metric" in ckpt

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_x = torch.randn(2, 15, 31)
    with torch.no_grad():
        out = model(dummy_x)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_representation_hybrid_feature_fusion_and_fit():
    """Verifies that RepresentationHybridClassifier extracts fused (N, 95) features and fits properly."""
    device = torch.device("cpu")
    backbone = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2)
    hybrid = RepresentationHybridClassifier(
        temporal_backbone=backbone,
        device=device,
        n_estimators=10,
        max_depth=5,
    )

    X_sample = np.random.randn(25, 15, 31).astype(np.float32)
    y_sample = np.random.choice([0, 1, 2], size=25)

    # Check fusion shape: 64 (embedding) + 31 (tabular features) = 95
    Z = hybrid.extract_fused_features(X_sample)
    assert Z.shape == (25, 95)
    assert np.isfinite(Z).all()

    # Check fit and predict
    hybrid.fit(X_sample, y_sample)
    preds = hybrid.predict(X_sample)
    probs = hybrid.predict_proba(X_sample)

    assert preds.shape == (25,)
    assert probs.shape == (25, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_confirmatory_protocol_invariants():
    """Verifies that confirmatory protocol is frozen, pre-registered, and strictly isolated."""
    proto_path = Path("results/phase4/confirmatory_protocol.json")
    assert proto_path.exists(), "confirmatory_protocol.json not found"

    with open(proto_path) as f:
        proto = json.load(f)

    assert proto["status"] == "FROZEN_PRE_REGISTERED_UNTOUCHED"
    assert proto["random_seed"] == 42
    assert proto["quarantine_policy"]["phase4_evaluation"] is False
    assert proto["quarantine_policy"]["model_selection_use"] is False

    fails = proto["frozen_entity_ids"]["failure_entities"]
    normals = proto["frozen_entity_ids"]["normal_entities"]

    assert len(fails) == 100
    assert len(normals) == 200
    assert len(set(fails) & set(normals)) == 0

    # Verify zero overlap with development entities
    train_meta = pd.read_csv("data/processed/train_metadata.csv")
    val_meta = pd.read_csv("data/processed/val_metadata.csv")
    test_meta = pd.read_csv("data/processed/test_metadata.csv")

    dev_eids = set(train_meta["entity_id"]) | set(val_meta["entity_id"]) | set(test_meta["entity_id"])
    assert len(set(fails) & dev_eids) == 0, "Confirmatory failure entities overlap with development data!"
    assert len(set(normals) & dev_eids) == 0, "Confirmatory normal entities overlap with development data!"


def test_model_comparison_artifact_completeness():
    """Verifies that model_comparison.json contains all required models and metrics."""
    comp_path = Path("results/phase4/model_comparison.json")
    assert comp_path.exists(), "model_comparison.json not found"

    with open(comp_path) as f:
        data = json.load(f)

    assert "models" in data
    models = data["models"]

    expected_models = [
        "Random Forest",
        "Temporal LSTM",
        "Compact Temporal Transformer",
        "Representation Hybrid",
    ]
    for m in expected_models:
        assert m in models, f"Missing model: {m}"
        model_data = models[m]
        assert "parameter_count" in model_data
        assert "training_runtime_seconds" in model_data
        assert "metrics" in model_data

        metrics = model_data["metrics"]
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "class_metrics" in metrics
        assert "confusion_matrix" in metrics
        assert "episode_detection" in metrics
        assert "lead_time_distribution_minutes" in metrics
        assert "false_alarm_metrics" in metrics
