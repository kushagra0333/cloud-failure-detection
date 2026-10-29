"""Unit tests for Phase 3B Temporal LSTM architecture, checkpointing, and prediction alignment."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from src.models.lstm import TemporalLSTMClassifier
from src.training.trainer import compute_training_class_weights, LSTMTrainer


def test_lstm_forward_shape():
    """Verifies that the LSTM forward pass produces logits of shape (batch, 3)."""
    batch_size = 8
    seq_len = 15
    input_dim = 31
    num_classes = 3

    model = TemporalLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.1,
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)

    assert logits.shape == (batch_size, num_classes)
    assert probs.shape == (batch_size, num_classes)
    assert preds.shape == (batch_size,)
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), np.ones(batch_size), atol=1e-5)


def test_class_weights_calculation():
    """Verifies that class weights are computed strictly inverse to class frequencies."""
    y = np.array([0] * 80 + [1] * 10 + [2] * 10)
    weights = compute_training_class_weights(y)

    assert len(weights) == 3
    # Class 0 has 80% prevalence -> weight should be smaller
    # Class 1 and 2 have 10% prevalence -> weights should be 8x higher
    assert weights[0] < weights[1]
    assert np.isclose(float(weights[1]), float(weights[2]))
    assert np.isclose(float(weights[1] / weights[0]), 8.0)


def test_lstm_checkpoint_loading():
    """Verifies that the trained LSTM checkpoint loads properly and reproduces predictions."""
    checkpoint_path = Path("models/checkpoints/best_lstm_checkpoint.pt")
    if not checkpoint_path.exists():
        pytest.skip("Checkpoint not yet saved.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "val_metrics" in checkpoint
    assert "best_metric" in checkpoint

    model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Pass dummy input
    dummy_x = torch.randn(2, 15, 31)
    probs = model.predict_proba(dummy_x)
    assert probs.shape == (2, 3)


def test_lstm_test_prediction_alignment():
    """Verifies that saved LSTM test predictions match test_metadata index-for-index."""
    pred_path = Path("results/phase3b/test_predictions_lstm.csv")
    meta_path = Path("data/processed/test_metadata.csv")

    if not pred_path.exists() or not meta_path.exists():
        pytest.skip("Predictions or metadata file not found.")

    pred_df = pd.read_csv(pred_path)
    meta_df = pd.read_csv(meta_path)

    assert len(pred_df) == len(meta_df) == 1003
    np.testing.assert_array_equal(pred_df["entity_id"].values, meta_df["entity_id"].values)
    np.testing.assert_array_equal(pred_df["prediction_timestamp"].values, meta_df["prediction_timestamp"].values)
    np.testing.assert_array_equal(pred_df["label"].values, meta_df["label"].values)

    assert "predicted_label" in pred_df.columns
    assert "prob_normal" in pred_df.columns
    assert "prob_warning" in pred_df.columns
    assert "prob_critical" in pred_df.columns

    # Probability simplex check
    prob_sums = pred_df["prob_normal"] + pred_df["prob_warning"] + pred_df["prob_critical"]
    np.testing.assert_allclose(prob_sums.values, np.ones(1003), atol=1e-4)
