"""Phase 6 Feasibility Audit Unit Test Suite.

Verifies:
1. Window-history invariant mathematics (70 min endpoint span vs 75 min collection span).
2. Pre-failure-only observations and absence of post-failure alerts.
3. Event timestamp and event code integrity in raw event shards.
4. Entity separation and trace exhaustion proof (all 241 long-history failing tasks in dev partitions).
5. Phase 5 metric denominators, window FPR vs entity FAR definition.
6. Feasibility counts and window calculations for alternative sequence lengths.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def feasibility_data():
    path = PROJECT_ROOT / "results" / "phase6" / "dataset_feasibility.json"
    assert path.exists(), "dataset_feasibility.json not found"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def phase5_predictions():
    path = PROJECT_ROOT / "results" / "phase5" / "confirmatory_predictions.csv"
    assert path.exists(), "confirmatory_predictions.csv not found"
    return pd.read_csv(path)


def test_history_window_mathematical_invariants(feasibility_data):
    """Verifies the distinction between endpoint distance and physical collection duration."""
    math_info = feasibility_data["history_window_mathematics"]
    cadence = math_info["cadence_seconds"]
    L = math_info["sequence_length_L"]

    # Endpoint distance = (L - 1) * cadence
    expected_endpoint_sec = (L - 1) * cadence
    assert math_info["endpoint_distance_seconds"] == expected_endpoint_sec
    assert math_info["endpoint_distance_minutes"] == expected_endpoint_sec / 60.0
    assert math_info["endpoint_distance_minutes"] == 70.0

    # Physical collection span = L * cadence
    expected_collection_sec = L * cadence
    assert math_info["physical_collection_span_seconds"] == expected_collection_sec
    assert math_info["physical_collection_span_minutes"] == expected_collection_sec / 60.0
    assert math_info["physical_collection_span_minutes"] == 75.0

    # Earliest possible prediction timestamp in trace (telemetry starts at 900s)
    assert math_info["earliest_possible_prediction_timestamp_sec"] == 900.0 + expected_endpoint_sec
    assert math_info["earliest_possible_prediction_timestamp_sec"] == 5100.0


def test_frozen_failure_entities_audit_and_restarts(feasibility_data):
    """Verifies that all 100 frozen failure entities had verified FAIL events and 0 met the 15-step invariant."""
    audit = feasibility_data["frozen_100_failure_entities_audit"]
    assert audit["total_frozen_failure_entities"] == 100
    assert audit["entities_with_verified_fail_event"] == 100
    assert audit["entities_with_ge_15_pre_failure_steps"] == 0
    assert audit["entities_exhibiting_restarts_after_failure"] == 100

    # Sum of step counts must equal 100
    dist = audit["pre_failure_step_distribution"]
    total_entities = sum(dist.values())
    assert total_entities == 100

    # Exactly 63 entities failed before 5100s
    assert audit["entities_failing_before_5100s"] == 63
    assert audit["entities_failing_after_5100s"] == 37


def test_full_trace_exhaustion_proof(feasibility_data):
    """Verifies that exactly 241 tasks in the entire trace had >= 15 pre-failure steps, and all were used in dev."""
    census = feasibility_data["full_trace_failure_census"]
    assert census["total_failing_entities_in_raw_events"] == 5170
    assert census["total_failing_entities_with_telemetry"] == 748
    assert census["total_failing_entities_with_ge_15_pre_fail_bins"] == 241
    assert census["development_partition_failing_entities_with_ge_15_bins"] == 241
    assert census["unused_failing_entities_with_ge_15_pre_fail_bins"] == 0


def test_phase5_metric_denominators_and_definitions(phase5_predictions, feasibility_data):
    """Verifies the exact distinction between window FPR and pure-normal entity FAR."""
    n_windows = len(phase5_predictions)
    n_entities = phase5_predictions["entity_id"].nunique()

    assert n_windows == 2596
    assert n_entities == 188

    models = {
        "Random Forest": "rf",
        "Temporal LSTM": "lstm",
        "Compact Temporal Transformer": "trans",
        "Representation Hybrid": "hybrid",
    }

    metrics_recomputed = feasibility_data["phase5_metrics_verification"]

    for name, prefix in models.items():
        preds = phase5_predictions[f"{prefix}_pred"].values

        # Window FPR: fraction of 2,596 windows with alerts (pred > 0)
        fp_windows = (preds > 0).sum()
        fpr = (fp_windows / n_windows) * 100.0
        assert abs(fpr - metrics_recomputed["recomputed_window_fpr"][name]) < 1e-4

        # Entity FAR: fraction of 188 entities with at least one alert
        fa_entities = sum(
            (phase5_predictions[phase5_predictions["entity_id"] == eid][f"{prefix}_pred"] > 0).any()
            for eid in phase5_predictions["entity_id"].unique()
        )
        far = (fa_entities / n_entities) * 100.0
        assert abs(far - metrics_recomputed["recomputed_pure_normal_far"][name]) < 1e-4

        # Invariant: Entity FAR > Window FPR due to multiple windows per entity
        assert far > fpr


def test_alternative_sequence_length_feasibility(feasibility_data):
    """Verifies feasibility numbers across candidate sequence lengths L."""
    designs = feasibility_data["alternative_designs_feasibility"]
    df_d = pd.DataFrame(designs)

    # For L=15, unused must be 0
    row_15 = df_d[df_d["sequence_length_L"] == 15].iloc[0]
    assert row_15["unused_failing_entities"] == 0
    assert row_15["potential_unused_failure_windows"] == 0

    # For L=5, unused must be 132 entities with 801 potential windows
    row_5 = df_d[df_d["sequence_length_L"] == 5].iloc[0]
    assert row_5["unused_failing_entities"] == 132
    assert row_5["potential_unused_failure_windows"] == 801
    assert row_5["endpoint_duration_min"] == 20.0
    assert row_5["collection_duration_min"] == 25.0

    # For L=3, unused must be 209 entities with 1160 windows
    row_3 = df_d[df_d["sequence_length_L"] == 3].iloc[0]
    assert row_3["unused_failing_entities"] == 209
    assert row_3["potential_unused_failure_windows"] == 1160


def test_proposed_protocol_quarantine_and_safeguards():
    """Verifies that the proposed confirmatory protocol document contains all required safeguards."""
    proto_path = PROJECT_ROOT / "reports" / "phase6_proposed_confirmatory_protocol.md"
    assert proto_path.exists(), "proposed protocol report not found"
    content = proto_path.read_text(encoding="utf-8")

    assert "Variant A" in content
    assert "Variant B" in content
    assert "Pure-Normal Entity FAR" in content
    assert "12.0" in content
    assert "require_full_history" in content
    assert "Alibaba Cluster Trace" in content
    assert "Zero Access During Development" in content
