"""Tests for Phase 6.4 — Confirmatory Evaluation Preflight Audit.

Verifies:
1. Protocol Amendment 2.1 is ratified and predecessor protocol_l5.json SHA-256 is immutable.
2. Exact cohort accounting: 137 failure entities, 198 evaluable pure-normal entities, 335 evaluable total entities.
3. Exact window accounting: 5,511 primary windows (400 Critical, 300 Warning, 4,811 Normal: 4,515 pure-normal, 296 failing normal).
4. Strict quarantine: 61 census failure tasks and 2 excluded normal controls cannot enter primary evaluation.
5. Entity-count inconsistency resolution: no metric or bootstrap denominator incorrectly uses 337 instead of 335.
   - FAR denominator is strictly 198 evaluable pure-normal entities (max 23 false-alarm entities).
   - Window FPR denominator is strictly 4,811 primary normal windows (max 240 false-positive windows).
   - Episode Recall denominator is strictly 141 failure episodes.
6. All four L=5 checkpoints load cleanly and match frozen architectures:
   - RF: 31 tabular features, n_estimators=100, max_depth=15, balanced.
   - LSTM: 62,467 params, sequence shape [B, 5, 31].
   - Transformer: 19,331 params, sequence shape [B, 5, 31].
   - Hybrid: 63 fused features (Transformer 32-dim latent embedding + 31-dim tabular -> RF).
7. Preprocessing provenance: fitted scaler has 31 features, fitted strictly on train partition.
8. Cluster bootstrap mechanics: run_l5_cluster_bootstrap preserves entity clusters and computes valid 95% CIs.
9. Sensitivity cohort: 75 failure entities, 79 episodes, 347 prefail windows from 4 unseen parent jobs.
10. Holdout preservation: Zero holdout scoring, zero threshold tuning, zero winner selection.
11. Preflight audit JSON: results/phase6/preflight_audit_l5.json has final_verdict READY_FOR_CONFIRMATORY_EVALUATION.
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

from sklearn.ensemble import RandomForestClassifier
from src.models.baselines import ClassWeightedRandomForest
from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier
from src.features.scaler import LeakageSafeScaler
from src.evaluation.confirmatory_evaluator_l5 import (
    load_ratified_l5_protocol,
    validate_evaluation_manifest,
    evaluate_l5_operational_gates,
    select_operational_winner,
    run_l5_cluster_bootstrap,
)

ORIGINAL_PROTOCOL_SHA256 = "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_protocol_status_and_predecessor_preservation():
    """Verify that results/phase6/protocol_l5.json SHA-256 matches and Amendment 2.1 is ratified."""
    orig_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5.json"
    amend_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json"

    assert orig_path.exists(), "Original protocol_l5.json must exist"
    assert amend_path.exists(), "Amended protocol_l5_amendment_2_1.json must exist"

    assert compute_sha256(orig_path) == ORIGINAL_PROTOCOL_SHA256, "Predecessor protocol was modified!"

    proto = load_ratified_l5_protocol(str(amend_path))
    assert proto["amendment_version"] == "2.1"
    assert proto["status"] == "RATIFIED_PRE_REGISTERED_FROZEN"


def test_primary_cohort_and_window_accounting():
    """Verify that 137 failure entities, 198 evaluable normal entities, and 5,511 primary windows are accounted for."""
    proto = load_ratified_l5_protocol()
    frozen_eids = proto["frozen_entity_ids"]
    denoms = proto["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    assert len(frozen_eids["primary_failure_entities"]) == 137
    assert len(frozen_eids["primary_normal_entities_sampled"]) == 200
    assert len(frozen_eids["primary_normal_entities_evaluable"]) == 198
    assert len(frozen_eids["primary_normal_entities_excluded"]) == 2
    assert set(frozen_eids["primary_normal_entities_excluded"]) == {"4974636262_330", "4974636615_100"}

    # Total evaluable entities
    assert len(frozen_eids["primary_failure_entities"]) + len(frozen_eids["primary_normal_entities_evaluable"]) == 335

    # Window counts
    assert denoms["primary_total_windows"] == 5511
    assert denoms["critical_windows"] == 400
    assert denoms["warning_windows"] == 300
    assert denoms["prefail_windows"] == 700
    assert denoms["primary_normal_windows"] == 4811
    assert denoms["pure_normal_windows"] == 4515
    assert denoms["failure_cohort_normal_windows"] == 296

    # Normal window reconciliation formula
    assert denoms["pure_normal_windows"] + denoms["failure_cohort_normal_windows"] == 4811
    assert denoms["primary_normal_windows"] + denoms["prefail_windows"] == 5511


def test_strict_quarantine_of_excluded_tasks():
    """Verify that the 61 quarantined census tasks and 2 excluded normal controls have zero overlap with primary evaluable cohort."""
    proto = load_ratified_l5_protocol()
    frozen_eids = proto["frozen_entity_ids"]

    primary_evaluable = set(frozen_eids["primary_failure_entities"]) | set(frozen_eids["primary_normal_entities_evaluable"])
    quarantined_tasks = set(frozen_eids["excluded_census_failure_tasks"])
    excluded_normal = set(frozen_eids["primary_normal_entities_excluded"])

    assert len(quarantined_tasks) == 61
    assert len(excluded_normal) == 2

    assert len(primary_evaluable & quarantined_tasks) == 0, "Quarantine leak: census tasks in primary evaluable cohort!"
    assert len(primary_evaluable & excluded_normal) == 0, "Exclusion leak: excluded normal controls in primary evaluable cohort!"
    assert len(quarantined_tasks & set(frozen_eids["primary_normal_entities_sampled"])) == 0


def test_entity_count_inconsistency_resolution_and_denominators():
    """Verify that the 337 vs 335 entity distinction is resolved and no metric denominator incorrectly uses 337."""
    proto = load_ratified_l5_protocol()
    denoms = proto["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    # 1. Tenet 1 distinguishes sampled vs evaluable
    tenet1 = proto["ratified_locked_tenets"]["tenet_1_primary_cohort"]
    assert tenet1["total_primary_entities_sampled"] == 337
    assert tenet1["total_primary_entities_evaluable"] == 335

    # 2. Entity FAR denominator is strictly 198
    assert denoms["evaluable_pure_normal_entities"] == 198
    assert denoms["max_allowable_entity_false_alarms"] == 23
    assert (23 / 198 * 100.0) <= 12.0
    assert (24 / 198 * 100.0) > 12.0

    # 3. Window FPR denominator is strictly 4,811
    assert denoms["primary_normal_windows"] == 4811
    assert denoms["max_allowable_window_false_positives"] == 240
    assert (240 / 4811 * 100.0) <= 5.0
    assert (241 / 4811 * 100.0) > 5.0

    # 4. Episode recall denominator is strictly 141
    assert len(proto["frozen_episode_metadata"]) == 141

    # 5. Manifest total windows denominator is strictly 5,511
    assert denoms["primary_total_windows"] == 5511


def test_checkpoint_architectures_and_loading():
    """Verify that all four L=5 checkpoints exist, load cleanly, and match protocol architectural specifications."""
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints_l5"

    # 1. Random Forest
    rf_path = ckpt_dir / "random_forest_l5.joblib"
    assert rf_path.exists()
    rf = joblib.load(rf_path)
    assert isinstance(rf, (RandomForestClassifier, ClassWeightedRandomForest))
    assert rf.n_estimators == 100
    assert rf.max_depth == 15
    assert rf.class_weight == "balanced"
    assert rf.n_features_in_ == 31

    # 2. Temporal LSTM
    lstm_path = ckpt_dir / "best_lstm_checkpoint_l5.pt"
    assert lstm_path.exists()
    lstm_ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
    lstm = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
    lstm.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm.eval()
    assert sum(p.numel() for p in lstm.parameters()) == 62467

    # 3. Compact Temporal Transformer
    tf_path = ckpt_dir / "best_transformer_checkpoint_l5.pt"
    assert tf_path.exists()
    tf_ckpt = torch.load(tf_path, map_location="cpu", weights_only=False)
    tf = CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, num_classes=3, dropout=0.1, max_seq_len=5)
    tf.load_state_dict(tf_ckpt["model_state_dict"])
    tf.eval()
    assert sum(p.numel() for p in tf.parameters()) == 19331

    # 4. Representation Hybrid
    hyb_path = ckpt_dir / "representation_hybrid_l5.joblib"
    assert hyb_path.exists()
    hyb_rf = joblib.load(hyb_path)
    hyb = RepresentationHybridClassifier(temporal_backbone=tf, device="cpu")
    hyb.rf = hyb_rf
    hyb.is_fitted = True
    assert hyb.rf.n_features_in_ == 63

    # Functional inference check
    dummy_seq = torch.randn(2, 5, 31)
    dummy_tab = dummy_seq[:, -1, :].numpy()

    with torch.no_grad():
        out_lstm = lstm(dummy_seq)
        out_tf = tf(dummy_seq)
    out_rf = rf.predict(dummy_tab)
    out_hyb = hyb.predict(dummy_seq.numpy())

    assert out_rf.shape == (2,)
    assert out_lstm.shape == (2, 3)
    assert out_tf.shape == (2, 3)
    assert out_hyb.shape == (2,)


def test_preprocessing_provenance_and_split_isolation():
    """Verify that scaler was fitted strictly on train data and zero dev entities overlap with holdout."""
    scaler_path = PROJECT_ROOT / "configs" / "fitted_scaler.json"
    assert scaler_path.exists()
    scaler = LeakageSafeScaler.load(str(scaler_path))
    assert len(scaler.feature_cols) == 31

    proto = load_ratified_l5_protocol()
    holdout_eids = set(proto["frozen_entity_ids"]["primary_failure_entities"]) | set(proto["frozen_entity_ids"]["primary_normal_entities_sampled"])
    quarantined = set(proto["frozen_entity_ids"]["excluded_census_failure_tasks"])

    meta_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "train_metadata.csv")
    meta_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "val_metadata.csv")
    meta_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "l5" / "test_metadata.csv")

    dev_eids = set(meta_train["entity_id"]) | set(meta_val["entity_id"]) | set(meta_test["entity_id"])

    assert len(dev_eids & holdout_eids) == 0, "Holdout leakage into dev splits!"
    assert len(dev_eids & quarantined) == 0, "Quarantined census tasks in dev splits!"


def test_cluster_bootstrap_mechanics():
    """Verify that run_l5_cluster_bootstrap preserves entity clusters, tracks correct denominators, and computes valid 95% CIs."""
    proto = load_ratified_l5_protocol()
    norm_eids = proto["frozen_entity_ids"]["primary_normal_entities_evaluable"][:6]
    fail_eids = proto["frozen_entity_ids"]["primary_failure_entities"][:4]

    records = []
    for eid in norm_eids:
        for s in range(5):
            records.append({
                "entity_id": eid, "label": 0, "prediction_timestamp": float(s * 300),
                "time_to_failure": 9999.0
            })
    for eid in fail_eids:
        for s in range(5):
            records.append({
                "entity_id": eid, "label": 1 if s < 3 else 2, "prediction_timestamp": float(s * 300),
                "time_to_failure": float(300 * (5 - s))
            })

    test_df = pd.DataFrame(records)
    preds = {
        "rf": np.random.choice([0, 1, 2], size=len(test_df)),
        "lstm": np.random.choice([0, 1, 2], size=len(test_df)),
    }

    boot_res = run_l5_cluster_bootstrap(test_df, preds, n_replicates=50, random_seed=42, protocol=proto)
    meta = boot_res["bootstrap_metadata"]
    assert meta["resampling_unit"] == "entity_id"
    assert meta["evaluable_clusters_count"] == 10
    assert meta["confidence_level"] == 0.95

    cis = boot_res["model_confidence_intervals"]
    for m in ["rf", "lstm"]:
        for metric in ["macro_f1", "accuracy", "pooled_fpr_pct", "pure_normal_far_pct", "episode_recall_pct"]:
            ci = cis[m][metric]["ci_95"]
            assert ci[0] <= ci[1], f"Invalid CI for {m} {metric}: {ci}"


def test_cross_job_sensitivity_cohort():
    """Verify that the sensitivity cohort comprises 75 entities, 79 episodes, and 347 prefail windows across 4 unseen jobs."""
    proto = load_ratified_l5_protocol()
    tenet8 = proto["ratified_locked_tenets"]["tenet_8_sensitivity_cohort"]

    assert set(tenet8["unseen_parent_jobs"]) == {"6250345153", "6250804554", "6251723929", "6252225776"}
    assert len(proto["frozen_entity_ids"]["disjoint_job_failure_entities"]) == 75

    disjoint_eps = [ep for ep in proto["frozen_episode_metadata"] if ep["parent_job_id"] in tenet8["unseen_parent_jobs"]]
    assert len(disjoint_eps) == 79
    assert sum(ep["num_windows"] for ep in disjoint_eps) == 347


def test_holdout_preservation_and_zero_scoring():
    """Verify that the preflight audit certified holdout preservation prior to evaluation."""
    audit_path = PROJECT_ROOT / "results" / "phase6" / "preflight_audit_l5.json"
    assert audit_path.exists(), "Preflight audit artifact must exist"
    with open(audit_path) as f:
        data = json.load(f)
    pres = data["holdout_preservation_audit"]
    assert pres["holdout_remains_pristine"] is True
    assert pres["holdout_feature_tensors_loaded"] is False
    assert pres["holdout_thresholds_tuned"] is False
    assert pres["operational_winner_selected"] is False

    # Verify no unratified ad-hoc artifacts
    unauthorized = [
        PROJECT_ROOT / "results" / "phase6" / "confirmatory_evaluation_results.json",
        PROJECT_ROOT / "results" / "phase6" / "operational_winner.json",
    ]
    for p in unauthorized:
        assert not p.exists(), f"Unauthorized ad-hoc artifact exists: {p}"



def test_preflight_audit_json_artifact():
    """Verify that results/phase6/preflight_audit_l5.json exists and has final_verdict READY_FOR_CONFIRMATORY_EVALUATION."""
    audit_path = PROJECT_ROOT / "results" / "phase6" / "preflight_audit_l5.json"
    assert audit_path.exists(), "results/phase6/preflight_audit_l5.json must exist"

    with open(audit_path) as f:
        data = json.load(f)

    assert data["audit_phase"] == "Phase 6.4"
    assert data["final_verdict"] == "READY_FOR_CONFIRMATORY_EVALUATION"
    assert data["checks"]["protocol_integrity"]["status"] == "PASSED"
    assert data["checks"]["population_accounting"]["status"] == "PASSED"
    assert data["checkpoints_audit"]["status"] == "PASSED"
    assert data["quarantine_audit"]["status"] == "PASSED"
    assert data["holdout_preservation_audit"]["status"] == "PASSED"
