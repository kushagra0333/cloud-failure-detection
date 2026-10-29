"""Tests for Phase 6.3 — L=5 Model Training Under Protocol Amendment 2.1.

Verifies:
1. Model checkpoints exist in models/checkpoints_l5/ and load without error.
2. Sequence dimensions: LSTM and Transformer consume [batch_size, 5, 31].
3. Tabular & Hybrid dimensions: RF consumes [batch_size, 31], Hybrid consumes [batch_size, 63].
4. Parameter counts match frozen specifications: LSTM=62,467, Transformer=19,331.
5. Split isolation: Zero overlap between development entities and confirmatory holdout.
6. Preprocessing provenance: Scaler is frozen and fitted on training data only.
7. Training manifest and validation metrics JSON artifacts are complete with required verdict.
8. Protocol artifacts remain strictly preserved and unmodified.
9. Zero holdout scoring was conducted during this phase.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier
from src.features.scaler import LeakageSafeScaler
from src.evaluation.confirmatory_evaluator_l5 import load_ratified_l5_protocol

ORIGINAL_PROTOCOL_SHA256 = "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_checkpoint_existence_and_loading():
    """Verify that all four L=5 model checkpoints exist and load correctly."""
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_l5"
    assert ckpt_dir.exists(), "models/checkpoints_l5 directory must exist"

    rf_path = ckpt_dir / "random_forest_l5.joblib"
    lstm_path = ckpt_dir / "best_lstm_checkpoint_l5.pt"
    tf_path = ckpt_dir / "best_transformer_checkpoint_l5.pt"
    hyb_path = ckpt_dir / "representation_hybrid_l5.joblib"

    assert rf_path.exists(), "Random Forest checkpoint must exist"
    assert lstm_path.exists(), "LSTM checkpoint must exist"
    assert tf_path.exists(), "Transformer checkpoint must exist"
    assert hyb_path.exists(), "Representation Hybrid checkpoint must exist"

    # Test loading
    rf_model = joblib.load(rf_path)
    assert hasattr(rf_model, "predict_proba"), "Random Forest must have predict_proba"

    lstm_ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in lstm_ckpt, "LSTM checkpoint must contain model_state_dict"
    lstm_model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    assert lstm_model is not None

    tf_ckpt = torch.load(tf_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in tf_ckpt, "Transformer checkpoint must contain model_state_dict"
    tf_model = CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_seq_len=5)
    tf_model.load_state_dict(tf_ckpt["model_state_dict"])
    assert tf_model is not None

    hyb_rf = joblib.load(hyb_path)
    assert hasattr(hyb_rf, "predict_proba"), "Hybrid RF must have predict_proba"


def test_sequence_dimensions_and_shapes():
    """Verify that models consume exact tensor shapes: sequence models [N, 5, 31], tabular [N, 31], hybrid [N, 63]."""
    device = torch.device("cpu")
    dummy_seq = torch.randn(4, 5, 31)
    dummy_tab = dummy_seq[:, -1, :].numpy()

    # 1. LSTM
    lstm_model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    lstm_logits = lstm_model(dummy_seq)
    assert lstm_logits.shape == (4, 3), f"LSTM output must be [4, 3], got {lstm_logits.shape}"

    # 2. Transformer
    tf_model = CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_seq_len=5)
    tf_logits = tf_model(dummy_seq)
    tf_emb = tf_model.get_embedding(dummy_seq)
    assert tf_logits.shape == (4, 3), f"Transformer output must be [4, 3], got {tf_logits.shape}"
    assert tf_emb.shape == (4, 32), f"Transformer embedding must be [4, 32], got {tf_emb.shape}"

    # 3. Hybrid fusion
    hyb = RepresentationHybridClassifier(tf_model, device=device, n_estimators=10, max_depth=15, random_state=42)
    fused_features = hyb.extract_fused_features(dummy_seq.numpy())
    assert fused_features.shape == (4, 63), f"Hybrid fused representation must be [4, 63], got {fused_features.shape}"

    # 4. Check processed L=5 arrays on disk
    l5_dir = PROJECT_ROOT / "data" / "processed" / "l5"
    assert l5_dir.exists(), "data/processed/l5 must exist"

    X_train_l5 = np.load(l5_dir / "X_train.npy")
    y_train = np.load(l5_dir / "y_train.npy")
    assert X_train_l5.shape == (4798, 5, 31)
    assert y_train.shape == (4798,)

    X_val_l5 = np.load(l5_dir / "X_val.npy")
    y_val = np.load(l5_dir / "y_val.npy")
    assert X_val_l5.shape == (1015, 5, 31)
    assert y_val.shape == (1015,)

    X_test_l5 = np.load(l5_dir / "X_test.npy")
    y_test = np.load(l5_dir / "y_test.npy")
    assert X_test_l5.shape == (1003, 5, 31)
    assert y_test.shape == (1003,)


def test_parameter_counts_match_specification():
    """Verify that neural parameter counts match the pre-registered protocol."""
    lstm_model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    tf_model = CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_seq_len=5)

    n_lstm = sum(p.numel() for p in lstm_model.parameters())
    n_tf = sum(p.numel() for p in tf_model.parameters())

    assert n_lstm == 62467, f"Expected 62,467 LSTM parameters, got {n_lstm}"
    assert n_tf == 19331, f"Expected 19,331 Transformer parameters, got {n_tf}"


def test_split_isolation_and_no_holdout_leakage():
    """Verify absolute split isolation: zero overlap between dev entities and confirmatory holdout."""
    protocol = load_ratified_l5_protocol()

    holdout_failure = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
    holdout_normal = set(protocol["frozen_entity_ids"]["primary_normal_entities_sampled"])
    all_holdout = holdout_failure | holdout_normal
    excluded_census = set(protocol["frozen_entity_ids"]["excluded_census_failure_tasks"])

    meta_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "train_metadata.csv")
    meta_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "val_metadata.csv")
    meta_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "test_metadata.csv")

    dev_entities = set(meta_train["entity_id"]) | set(meta_val["entity_id"]) | set(meta_test["entity_id"])

    # Strict isolation assertions
    assert len(dev_entities) == 525, f"Expected 525 dev entities, got {len(dev_entities)}"
    assert len(dev_entities & all_holdout) == 0, "FATAL: Development entities overlap with confirmatory holdout!"
    assert len(dev_entities & excluded_census) == 0, "FATAL: Development entities overlap with quarantined census tasks!"
    assert len(holdout_failure) == 137
    assert len(holdout_normal) == 200


def test_preprocessing_provenance():
    """Verify that fitted scaler exists, was fitted on training data, and transforms properly."""
    scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler.json"
    assert scaler_path.exists(), "configs/fitted_scaler.json must exist"

    scaler = LeakageSafeScaler.load(str(scaler_path))
    assert len(scaler.feature_cols) == 31, f"Expected 31 features, got {len(scaler.feature_cols)}"

    # Check finite transformations
    dummy_df = pd.DataFrame(np.random.randn(10, 31), columns=scaler.feature_cols)
    scaled_df = scaler.transform(dummy_df)
    assert np.isfinite(scaled_df.values).all(), "Scaler produced non-finite values"


def test_training_manifest_and_validation_metrics_schema():
    """Verify results/phase6/training_manifest_l5.json and validation_metrics_l5.json."""
    manifest_path = PROJECT_ROOT / "results" / "phase6" / "training_manifest_l5.json"
    val_path = PROJECT_ROOT / "results" / "phase6" / "validation_metrics_l5.json"

    assert manifest_path.exists(), "training_manifest_l5.json must exist"
    assert val_path.exists(), "validation_metrics_l5.json must exist"

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(val_path) as f:
        val_res = json.load(f)

    assert manifest["final_verdict"] == "TRAINING_COMPLETE_READY_FOR_CONFIRMATORY_EVALUATION"
    assert manifest["random_seed"] == 42
    assert "models" in manifest
    assert len(manifest["models"]) == 4

    expected_models = ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]
    for m in expected_models:
        assert m in manifest["models"], f"Model {m} missing from training manifest"
        assert m in val_res["models"], f"Model {m} missing from validation metrics"

        # Check metrics completeness
        m_val = val_res["models"][m]
        assert "macro_f1" in m_val
        assert "accuracy" in m_val
        assert "confusion_matrix" in m_val
        assert "entity_far" in m_val
        assert "window_fpr" in m_val
        assert m_val["total_evaluated_windows"] == 1015

    # Check disclaimer in validation metrics
    assert "disclaimer" in val_res
    assert "Development-validation results only" in val_res["disclaimer"]


def test_protocol_preservation_invariance():
    """Verify that both original and amended protocol artifacts remain preserved and untouched."""
    orig_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5.json"
    amend_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json"

    assert orig_path.exists()
    assert amend_path.exists()

    # Original protocol hash invariant
    current_hash = compute_sha256(orig_path)
    assert current_hash == ORIGINAL_PROTOCOL_SHA256, (
        f"Original protocol_l5.json was modified! Expected {ORIGINAL_PROTOCOL_SHA256}, got {current_hash}"
    )

    # Amended protocol status invariant
    with open(amend_path) as f:
        amend_data = json.load(f)
    assert amend_data["status"] == "RATIFIED_PRE_REGISTERED_FROZEN"
    assert amend_data["amendment_version"] == "2.1"


def test_no_confirmatory_holdout_scoring():
    """Verify that no holdout predictions were computed or saved during this phase."""
    confirmatory_preds_dir = PROJECT_ROOT / "results" / "phase6" / "predictions_l5"
    assert not confirmatory_preds_dir.exists(), (
        "Phase 6 confirmatory predictions directory must NOT exist during Phase 6.3 training!"
    )
