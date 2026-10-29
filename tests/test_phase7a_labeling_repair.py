"""Phase 7A Regression and Data Integrity Tests for Target Labeling.

Verifies strict temporal censoring, post-failure handling, repeated failures,
exact horizon boundaries, and data-level invariants under Protocol Amendment 2.1.
"""

import pytest
import numpy as np
import pandas as pd

from src.ingestion.schema import ClusterEventTypeV2, ClusterEventTypeV3, FailureLabel
from src.labeling.target_generator import TargetLabelGenerator


class TestPhase7ALabelingRepair:
    """Comprehensive regression test suite for Phase 7A labeling repair."""

    @pytest.fixture
    def generator_2011(self):
        return TargetLabelGenerator(
            critical_horizon_sec=900.0,
            warning_horizon_sec=1800.0,
            trace_version="2011",
        )

    @pytest.fixture
    def generator_2019(self):
        return TargetLabelGenerator(
            critical_horizon_sec=900.0,
            warning_horizon_sec=1800.0,
            trace_version="2019",
        )

    # 1. Valid normal execution with sufficient follow-up
    def test_scenario_01_valid_normal_sufficient_followup(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame(columns=["event_type", "time_sec"])
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 1800.0)
        assert lbl == FailureLabel.NORMAL
        assert tau is None

    # 2. Valid normal FINISH event within 30 min
    def test_scenario_02_valid_normal_finish_event(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FINISH)], "time_sec": [t + 600.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 600.0)
        assert lbl == FailureLabel.NORMAL
        assert tau is None

    # 3. Failure within critical horizon
    def test_scenario_03_failure_within_critical_horizon(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 300.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 300.0)
        assert lbl == FailureLabel.CRITICAL
        assert np.isclose(tau, 300.0)

    # 4. Failure exactly at critical boundary (900.0s)
    def test_scenario_04_failure_exactly_at_critical_boundary(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 900.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 900.0)
        assert lbl == FailureLabel.CRITICAL
        assert np.isclose(tau, 900.0)

    # 5. Failure in warning interval (1200.0s)
    def test_scenario_05_failure_in_warning_interval(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 1200.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 1200.0)
        assert lbl == FailureLabel.WARNING
        assert np.isclose(tau, 1200.0)

    # 6. Failure exactly at warning boundary (1800.0s)
    def test_scenario_06_failure_exactly_at_warning_boundary(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 1800.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 1800.0)
        assert lbl == FailureLabel.WARNING
        assert np.isclose(tau, 1800.0)

    # 7. Failure just beyond warning horizon (1800.1s -> NORMAL)
    def test_scenario_07_failure_just_beyond_warning_horizon(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 1800.1]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 1800.1)
        assert lbl == FailureLabel.NORMAL
        assert np.isclose(tau, 1800.1)

    # 8. Insufficient future telemetry (< 1800s, no terminal event -> UNKNOWN)
    def test_scenario_08_insufficient_future_telemetry(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame(columns=["event_type", "time_sec"])
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 600.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 9. Telemetry after a failure (must be strictly censored as UNKNOWN)
    def test_scenario_09_telemetry_after_failure_strictly_censored(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t - 500.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 2000.0)
        # Protocol Amendment 2.1 line 201: post-failure step -> label -1 (strictly excluded)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 10. Task with repeated failures (step after earlier failure)
    def test_scenario_10_repeated_failures_step_after_prior_failure(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({
            "event_type": [int(ClusterEventTypeV2.FAIL), int(ClusterEventTypeV2.FAIL)],
            "time_sec": [t - 500.0, t + 600.0]
        })
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 600.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 11. Window between two failure events
    def test_scenario_11_window_between_two_failures_strictly_censored(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({
            "event_type": [int(ClusterEventTypeV2.FAIL), int(ClusterEventTypeV2.FAIL)],
            "time_sec": [t - 1000.0, t + 3000.0]
        })
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 3000.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 12. Failure occurring exactly at prediction timestamp t (tau = 0.0s -> UNKNOWN)
    def test_scenario_12_failure_exactly_at_prediction_timestamp(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 2000.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 13. Telemetry gaps / cancellation within 30m (KILL -> UNKNOWN)
    def test_scenario_13_user_cancellation_within_30m(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.KILL)], "time_sec": [t + 600.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 600.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 14. Missing or malformed prediction timestamp (NaN -> UNKNOWN)
    def test_scenario_14_missing_or_nan_timestamp(self, generator_2011):
        events = pd.DataFrame(columns=["event_type", "time_sec"])
        lbl, tau = generator_2011.derive_label_for_task(np.nan, events, 2000.0)
        assert lbl == FailureLabel.UNKNOWN
        assert tau is None

    # 15. Existing valid pre-failure cases retain original labels
    def test_scenario_15_clean_prefailure_retains_label(self, generator_2011):
        t = 1000.0
        events = pd.DataFrame({"event_type": [int(ClusterEventTypeV2.FAIL)], "time_sec": [t + 700.0]})
        lbl, tau = generator_2011.derive_label_for_task(t, events, t + 700.0)
        assert lbl == FailureLabel.CRITICAL
        assert np.isclose(tau, 700.0)

    # 16. Integration: WindowGenerator strictly excludes UNKNOWN endpoints
    def test_scenario_16_window_generator_excludes_unknown_endpoints(self, generator_2011):
        from src.labeling.window_generator import WindowGenerator
        
        # Build 10 steps of telemetry: t = 300, 600, ..., 3000
        timestamps = np.arange(300.0, 3300.0, 300.0)
        telem_df = pd.DataFrame({
            "entity_id": ["task_test"] * len(timestamps),
            "end_time_sec": timestamps,
            "feat_1": np.ones(len(timestamps), dtype=np.float32),
            "feat_2": np.zeros(len(timestamps), dtype=np.float32),
        })
        
        # Failure occurs at t = 1600.0 (between step 5 at 1500 and step 6 at 1800)
        fail_events = pd.DataFrame({
            "entity_id": ["task_test"],
            "event_type": [int(ClusterEventTypeV2.FAIL)],
            "time_sec": [1600.0],
        })
        
        labeled_df = generator_2011.label_telemetry_windows(telem_df, fail_events)
        
        # Steps 1-5 (300 to 1500) are pre-failure
        # Step 5 (1500): tau = 100s <= 900s -> CRITICAL (2)
        # Steps 6-10 (1800 to 3000): time_sec > 1600 -> UNKNOWN (-1)
        assert labeled_df.loc[labeled_df["end_time_sec"] == 1500.0, "label"].iloc[0] == FailureLabel.CRITICAL
        assert (labeled_df.loc[labeled_df["end_time_sec"] >= 1800.0, "label"] == FailureLabel.UNKNOWN).all()
        
        # Run WindowGenerator with exclude_unknown=True
        win_gen = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
        X, y, meta_df = win_gen.build_dataset_tensors(labeled_df, ["feat_1", "feat_2"])
        
        # Only step 5 can generate a window (steps 1-5 = 5 steps, label=2)
        # Post-failure steps 6-10 are excluded
        assert len(X) == 1
        assert len(y) == 1
        assert len(meta_df) == 1
        assert y[0] == 2
        assert meta_df["prediction_timestamp"].iloc[0] == 1500.0
        assert meta_df["label"].iloc[0] == 2

    # 17. Integration: No generated sequence contains telemetry at or after first failure
    def test_scenario_17_no_sequence_contains_post_failure_telemetry(self, generator_2011):
        from src.labeling.window_generator import WindowGenerator
        
        # Construct entity with restart and 2 failures
        timestamps = np.arange(300.0, 6300.0, 300.0) # 20 steps
        telem_df = pd.DataFrame({
            "entity_id": ["task_multi"] * len(timestamps),
            "end_time_sec": timestamps,
            "feat_1": np.arange(len(timestamps), dtype=np.float32),
        })
        
        # Failure 1 at 2000s, Failure 2 at 5000s
        fail_events = pd.DataFrame({
            "entity_id": ["task_multi", "task_multi"],
            "event_type": [int(ClusterEventTypeV2.FAIL), int(ClusterEventTypeV2.FAIL)],
            "time_sec": [2000.0, 5000.0],
        })
        
        labeled_df = generator_2011.label_telemetry_windows(telem_df, fail_events)
        
        win_gen = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
        X, y, meta_df = win_gen.build_dataset_tensors(labeled_df, ["feat_1"])
        
        # All steps at or after 2000s are labeled UNKNOWN (-1)
        # Pre-failure steps: 300, 600, 900, 1200, 1500, 1800 (6 steps)
        # Valid L=5 windows can only end at 1500 and 1800
        assert len(X) == 2
        assert len(y) == 2
        assert len(meta_df) == 2
        
        # Assert that EVERY timestamp across ALL steps in generated tensors is strictly < 2000.0s
        for w_idx in range(len(X)):
            pred_t = meta_df["prediction_timestamp"].iloc[w_idx]
            assert pred_t < 2000.0
            # Check trailing span
            t_start = pred_t - 4 * 300.0
            assert t_start >= 300.0
            assert pred_t <= 1800.0

    # 18. Integration: Feature and label arrays remain aligned
    def test_scenario_18_feature_and_label_arrays_remain_aligned(self, generator_2011):
        from src.labeling.window_generator import WindowGenerator
        
        timestamps = np.arange(300.0, 3300.0, 300.0)
        telem_df = pd.DataFrame({
            "entity_id": ["task_align"] * len(timestamps),
            "end_time_sec": timestamps,
            "feat_a": np.arange(len(timestamps), dtype=np.float32),
            "feat_b": np.arange(len(timestamps), dtype=np.float32) * 2.0,
        })
        
        fail_events = pd.DataFrame({
            "entity_id": ["task_align"],
            "event_type": [int(ClusterEventTypeV2.FAIL)],
            "time_sec": [2200.0],
        })
        
        labeled_df = generator_2011.label_telemetry_windows(telem_df, fail_events)
        win_gen = WindowGenerator(sequence_length=5, expected_interval_sec=300.0, require_full_history=True, exclude_unknown=True)
        X, y, meta_df = win_gen.build_dataset_tensors(labeled_df, ["feat_a", "feat_b"])
        
        # Dimensionality alignment
        assert X.shape[0] == len(y) == len(meta_df)
        assert X.shape[1] == 5
        assert X.shape[2] == 2
        
        # Value alignment: the last step in X should match the feature value at pred_t
        for i in range(len(X)):
            pred_t = meta_df["prediction_timestamp"].iloc[i]
            row = telem_df[telem_df["end_time_sec"] == pred_t].iloc[0]
            assert np.isclose(X[i, -1, 0], row["feat_a"])
            assert np.isclose(X[i, -1, 1], row["feat_b"])
            assert y[i] == meta_df["label"].iloc[i]
