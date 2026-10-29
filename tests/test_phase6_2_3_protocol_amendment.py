"""Tests for Phase 6.2.3 — Formal Ratification of Protocol Amendment 2.1.

Verifies:
1. results/phase6/protocol_l5.json remains strictly unmodified (SHA256 verified).
2. results/phase6/protocol_l5_amendment_2_1.json is ratified with status RATIFIED_PRE_REGISTERED_FROZEN.
3. Primary manifest locks exactly 5,511 windows: 400 Critical, 300 Warning, 4,811 Normal.
4. Normal breakdown locks 4,515 pure-normal windows + 296 failing-cohort normal windows.
5. The 61 non-pre-failing census tasks are strictly quarantined and cannot enter evaluation.
6. Operational gates are locked to <= 23 false alarm entities and <= 240 false positive windows.
7. Downstream evaluation code (src/evaluation/confirmatory_evaluator_l5.py) enforces these rules dynamically.
8. Model selection enforces the operational gate filter and tie-breaker rules.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.evaluation.confirmatory_evaluator_l5 import (
    load_ratified_l5_protocol,
    validate_evaluation_manifest,
    evaluate_l5_operational_gates,
    select_operational_winner,
)

ORIGINAL_PROTOCOL_SHA256 = "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_original_protocol_preservation_and_sha256():
    """Verify results/phase6/protocol_l5.json is completely preserved as an immutable historical record."""
    orig_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5.json"
    assert orig_path.exists(), "Original protocol_l5.json must exist"

    current_sha256 = compute_sha256(orig_path)
    assert current_sha256 == ORIGINAL_PROTOCOL_SHA256, (
        f"Original protocol_l5.json was modified! Expected SHA256 {ORIGINAL_PROTOCOL_SHA256}, got {current_sha256}"
    )


def test_amendment_artifact_ratification_and_metadata():
    """Verify results/phase6/protocol_l5_amendment_2_1.json exists, is ratified, and contains all 10 locked tenets."""
    amend_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json"
    assert amend_path.exists(), "Ratified amendment artifact must exist"

    proto = load_ratified_l5_protocol(str(amend_path))

    assert proto["amendment_version"] == "2.1"
    assert proto["status"] == "RATIFIED_PRE_REGISTERED_FROZEN"
    assert proto["final_ratification_verdict"]["verdict"] == "AMENDMENT_RATIFIED_READY_FOR_L5_TRAINING"

    safeguards = proto["safeguards_complied"]
    assert safeguards["no_model_training_conducted"] is True
    assert safeguards["no_holdout_scoring_conducted"] is True
    assert safeguards["no_holdout_threshold_tuning"] is True
    assert safeguards["phase5_artifacts_unmodified"] is True
    assert safeguards["original_protocol_l5_preserved_unmodified"] is True
    assert safeguards["cohort_membership_and_labels_unchanged"] is True

    # Check 10 tenets exist
    tenets = proto["ratified_locked_tenets"]
    expected_tenets = [
        "tenet_1_primary_cohort",
        "tenet_2_primary_evaluation_windows",
        "tenet_3_normal_window_breakdown",
        "tenet_4_excluded_descriptive_census",
        "tenet_5_entity_far_gate",
        "tenet_6_window_fpr_gate",
        "tenet_7_disaggregated_reporting",
        "tenet_8_sensitivity_cohort",
        "tenet_9_bootstrap_resampling",
        "tenet_10_model_selection_and_ranking_rule",
    ]
    for t in expected_tenets:
        assert t in tenets, f"Missing {t} in ratified_locked_tenets"


def test_locked_denominators_and_reconstruction():
    """Reconstruct window counts and denominators from the ratified manifest and entity metadata."""
    proto = load_ratified_l5_protocol()
    denoms = proto["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    # Load metadata files to reconstruct counts independently
    with open(PROJECT_ROOT / "scratch" / "failing_entities_metadata.json") as f:
        f_meta = json.load(f)

    # Reconstruct failure cohort prefail and normal windows
    total_prefail = sum(item["total_prefail_windows"] for item in f_meta)
    total_critical = sum(item["critical_windows"] for item in f_meta)
    total_warning = sum(item["warning_windows"] for item in f_meta)

    assert total_critical == denoms["critical_windows"] == 400
    assert total_warning == denoms["warning_windows"] == 300
    assert total_prefail == denoms["prefail_windows"] == 700

    # Normal window breakdown
    pure_normal = denoms["pure_normal_windows"]
    failing_normal = denoms["failure_cohort_normal_windows"]
    pooled_normal = denoms["primary_normal_windows"]

    assert pure_normal == 4515
    assert failing_normal == 296
    assert pure_normal + failing_normal == pooled_normal == 4811

    # Total primary windows
    assert pooled_normal + total_prefail == denoms["primary_total_windows"] == 5511

    # Entities
    assert denoms["evaluable_pure_normal_entities"] == 198
    assert denoms["evaluable_failure_entities"] == 137
    assert len(proto["frozen_entity_ids"]["primary_failure_entities"]) == 137
    assert len(proto["frozen_entity_ids"]["primary_normal_entities_sampled"]) == 200
    assert len(proto["frozen_entity_ids"]["primary_normal_entities_evaluable"]) == 198
    assert len(proto["frozen_entity_ids"]["primary_normal_entities_excluded"]) == 2


def test_quarantine_of_61_excluded_census_tasks():
    """Verify that the 61 non-pre-failing tasks are strictly quarantined and cannot enter evaluation."""
    proto = load_ratified_l5_protocol()
    excluded_tasks = proto["frozen_entity_ids"]["excluded_census_failure_tasks"]
    primary_failure = set(proto["frozen_entity_ids"]["primary_failure_entities"])
    primary_normal = set(proto["frozen_entity_ids"]["primary_normal_entities_sampled"])

    assert len(excluded_tasks) == 61
    assert len(set(excluded_tasks) & primary_failure) == 0, "Excluded tasks overlap with primary failure entities!"
    assert len(set(excluded_tasks) & primary_normal) == 0, "Excluded tasks overlap with primary normal entities!"

    # Verify that validate_evaluation_manifest raises ValueError if an excluded task is present
    eval_norm_eids = proto["frozen_entity_ids"]["primary_normal_entities_evaluable"]
    # Create mock metadata with an excluded task injected
    mock_data = []
    # Add an excluded task
    mock_data.append({"entity_id": excluded_tasks[0], "label": 0, "prediction_timestamp": 300.0, "time_to_failure": np.nan})
    mock_df = pd.DataFrame(mock_data)

    with pytest.raises(ValueError, match="FATAL: Detected 1 excluded census tasks"):
        validate_evaluation_manifest(mock_df, proto)


def test_manifest_validation_on_exact_primary_population():
    """Verify that validate_evaluation_manifest passes on a correctly formed primary manifest."""
    proto = load_ratified_l5_protocol()
    denoms = proto["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    fail_eids = proto["frozen_entity_ids"]["primary_failure_entities"]
    norm_eids = proto["frozen_entity_ids"]["primary_normal_entities_evaluable"]

    records = []
    # 400 Critical windows across failure entities
    for i in range(400):
        eid = fail_eids[i % len(fail_eids)]
        records.append({"entity_id": eid, "label": 2, "prediction_timestamp": float(i * 300), "time_to_failure": 600.0})
    # 300 Warning windows across failure entities
    for i in range(300):
        eid = fail_eids[i % len(fail_eids)]
        records.append({"entity_id": eid, "label": 1, "prediction_timestamp": float(i * 300), "time_to_failure": 1500.0})
    # 296 Normal windows from failure cohort
    for i in range(296):
        eid = fail_eids[i % len(fail_eids)]
        records.append({"entity_id": eid, "label": 0, "prediction_timestamp": float(i * 300), "time_to_failure": 3000.0})
    # 4,515 Pure-normal windows across 198 normal entities
    for i in range(4515):
        eid = norm_eids[i % len(norm_eids)]
        records.append({"entity_id": eid, "label": 0, "prediction_timestamp": float(i * 300), "time_to_failure": np.nan})

    mock_df = pd.DataFrame(records)
    res = validate_evaluation_manifest(mock_df, proto)
    assert res["is_valid"] is True
    assert res["total_windows"] == 5511
    assert res["critical_windows"] == 400
    assert res["warning_windows"] == 300
    assert res["normal_windows"] == 4811
    assert res["pure_normal_windows"] == 4515
    assert res["failing_cohort_normal_windows"] == 296
    assert res["evaluable_failure_entities"] == 137
    assert res["evaluable_pure_normal_entities"] == 198


def test_operational_gates_boundary_conditions():
    """Verify the exact mathematical cutoff for Entity FAR (<= 12.0%) and Window FPR (<= 5.0%)."""
    proto = load_ratified_l5_protocol()
    denoms = proto["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    fail_eids = proto["frozen_entity_ids"]["primary_failure_entities"]
    norm_eids = proto["frozen_entity_ids"]["primary_normal_entities_evaluable"]

    # Build primary manifest dataframe
    records = []
    for i in range(400):
        records.append({"entity_id": fail_eids[i % len(fail_eids)], "label": 2})
    for i in range(300):
        records.append({"entity_id": fail_eids[i % len(fail_eids)], "label": 1})
    for i in range(296):
        records.append({"entity_id": fail_eids[i % len(fail_eids)], "label": 0})
    for i in range(4515):
        records.append({"entity_id": norm_eids[i % len(norm_eids)], "label": 0})

    df = pd.DataFrame(records)

    # CASE A: Exactly at the boundary (23 false alarm entities, 240 false positive windows)
    # Total normal windows = 4,811 (indices 700 to 5510)
    y_pred_pass = np.zeros(len(df), dtype=int)
    # Set alerts on windows belonging to first 23 pure-normal entities
    alerted_entities = set(norm_eids[:23])
    # Distribute 240 false positive alerts across normal windows
    fp_count = 0
    for idx, row in df.iterrows():
        if row["label"] == 0 and fp_count < 240:
            if row["entity_id"] in alerted_entities:
                y_pred_pass[idx] = 1
                fp_count += 1
            elif len(alerted_entities) == 23 and fp_count < 240:
                # If we need more window FPs without exceeding 23 entities, place on existing alerted entities
                if row["entity_id"] in alerted_entities:
                    y_pred_pass[idx] = 1
                    fp_count += 1

    # Fill remaining up to 240 FPs if needed on those 23 entities
    normal_indices_alerted = df[(df["label"] == 0) & (df["entity_id"].isin(alerted_entities))].index
    y_pred_pass[normal_indices_alerted[:240]] = 1

    gates_res = evaluate_l5_operational_gates(df, y_pred_pass, proto)
    assert gates_res["entity_far"]["false_alarm_entities"] == 23
    assert gates_res["entity_far"]["gate_passed"] is True
    assert gates_res["entity_far"]["rate_pct"] == pytest.approx(23 / 198 * 100, 0.001)

    assert gates_res["window_fpr"]["pooled_primary"]["false_positive_windows"] == 240
    assert gates_res["window_fpr"]["pooled_primary"]["gate_passed"] is True
    assert gates_res["window_fpr"]["pooled_primary"]["rate_pct"] == pytest.approx(240 / 4811 * 100, 0.001)

    assert gates_res["operational_gates_verdict"]["passed_both_gates"] is True

    # CASE B: Exceeding FAR boundary by 1 entity (24 entities)
    y_pred_fail_far = y_pred_pass.copy()
    extra_entity = norm_eids[23]
    extra_idx = df[(df["label"] == 0) & (df["entity_id"] == extra_entity)].index[0]
    y_pred_fail_far[extra_idx] = 1
    # remove one from an already alerted entity to keep window FPs at 240
    # or let window FPs be 240
    gates_fail_far = evaluate_l5_operational_gates(df, y_pred_fail_far, proto)
    assert gates_fail_far["entity_far"]["false_alarm_entities"] == 24
    assert gates_fail_far["entity_far"]["gate_passed"] is False
    assert gates_fail_far["operational_gates_verdict"]["passed_both_gates"] is False

    # CASE C: Exceeding FPR boundary by 1 window (241 windows)
    y_pred_fail_fpr = y_pred_pass.copy()
    unalerted_normal_idx = df[(df["label"] == 0) & (y_pred_fail_fpr == 0) & (df["entity_id"].isin(alerted_entities))].index[0]
    y_pred_fail_fpr[unalerted_normal_idx] = 1
    gates_fail_fpr = evaluate_l5_operational_gates(df, y_pred_fail_fpr, proto)
    assert gates_fail_fpr["window_fpr"]["pooled_primary"]["false_positive_windows"] == 241
    assert gates_fail_fpr["window_fpr"]["pooled_primary"]["gate_passed"] is False
    assert gates_fail_fpr["operational_gates_verdict"]["passed_both_gates"] is False


def test_model_selection_and_ranking_rule():
    """Verify operational winner selection and fallback to NO_OPERATIONAL_WINNER."""
    # Scenario 1: All candidates fail operational gates -> NO OPERATIONAL WINNER
    candidates_all_failed = [
        {
            "model_name": "Random Forest",
            "episode_recall": 80.0,
            "macro_f1": 0.65,
            "operational_gates": {
                "entity_far": {"gate_passed": False, "false_alarm_entities": 25, "rate_pct": 12.6},
                "window_fpr": {"pooled_primary": {"gate_passed": True, "false_positive_windows": 200, "rate_pct": 4.1}},
                "operational_gates_verdict": {"passed_both_gates": False}
            }
        },
        {
            "model_name": "Temporal LSTM",
            "episode_recall": 85.0,
            "macro_f1": 0.70,
            "operational_gates": {
                "entity_far": {"gate_passed": True, "false_alarm_entities": 20, "rate_pct": 10.1},
                "window_fpr": {"pooled_primary": {"gate_passed": False, "false_positive_windows": 260, "rate_pct": 5.4}},
                "operational_gates_verdict": {"passed_both_gates": False}
            }
        }
    ]
    res1 = select_operational_winner(candidates_all_failed)
    assert res1["winner_selected"] is False
    assert res1["verdict"] == "NO_OPERATIONAL_WINNER"
    assert len(res1["disqualified_candidates"]) == 2

    # Scenario 2: Two candidates pass, ranked by episode_recall, then macro_f1
    candidates_mixed = [
        {
            "model_name": "Model_A",
            "episode_recall": 75.0,
            "macro_f1": 0.72,
            "operational_gates": {
                "entity_far": {"gate_passed": True, "false_alarm_entities": 18, "rate_pct": 9.1},
                "window_fpr": {"pooled_primary": {"gate_passed": True, "false_positive_windows": 180, "rate_pct": 3.7}},
                "operational_gates_verdict": {"passed_both_gates": True}
            }
        },
        {
            "model_name": "Model_B",
            "episode_recall": 82.0,
            "macro_f1": 0.68,
            "operational_gates": {
                "entity_far": {"gate_passed": True, "false_alarm_entities": 22, "rate_pct": 11.1},
                "window_fpr": {"pooled_primary": {"gate_passed": True, "false_positive_windows": 220, "rate_pct": 4.5}},
                "operational_gates_verdict": {"passed_both_gates": True}
            }
        },
        {
            "model_name": "Model_C_Disqualified",
            "episode_recall": 90.0,
            "macro_f1": 0.80,
            "operational_gates": {
                "entity_far": {"gate_passed": False, "false_alarm_entities": 30, "rate_pct": 15.1},
                "window_fpr": {"pooled_primary": {"gate_passed": False, "false_positive_windows": 300, "rate_pct": 6.2}},
                "operational_gates_verdict": {"passed_both_gates": False}
            }
        }
    ]
    res2 = select_operational_winner(candidates_mixed)
    assert res2["winner_selected"] is True
    assert res2["winner_model_name"] == "Model_B"  # 82.0% > 75.0%
    assert res2["ranked_eligible_models"] == ["Model_B", "Model_A"]
    assert len(res2["disqualified_candidates"]) == 1


def test_cross_job_sensitivity_cohort_preservation():
    """Verify that the 75-entity, 79-episode sensitivity analysis across 4 unseen jobs is preserved."""
    proto = load_ratified_l5_protocol()
    tenet8 = proto["ratified_locked_tenets"]["tenet_8_sensitivity_cohort"]

    assert tenet8["failure_entities_count"] == 75
    assert tenet8["failure_episodes_count"] == 79
    assert tenet8["prefail_windows_count"] == 347
    assert set(tenet8["unseen_parent_jobs"]) == {"6250345153", "6250804554", "6251723929", "6252225776"}

    disjoint_eids = proto["frozen_entity_ids"]["disjoint_job_failure_entities"]
    assert len(disjoint_eids) == 75
    for eid in disjoint_eids:
        job_id = eid.split("_")[0]
        assert job_id in tenet8["unseen_parent_jobs"]


def test_bootstrap_cluster_level_invariant():
    """Verify that bootstrap resampling is locked strictly at entity_id level."""
    proto = load_ratified_l5_protocol()
    tenet9 = proto["ratified_locked_tenets"]["tenet_9_bootstrap_resampling"]

    assert tenet9["resampling_unit"] == "entity_id"
    assert tenet9["total_clusters"] == 337
    assert tenet9["independent_failure_clusters"] == 137
    assert tenet9["independent_normal_clusters"] == 200
    assert "Treating episodes as independent clusters is prohibited" in tenet9["clustering_rule"]
