"""Unit tests for target label generation, exact horizon boundaries, and censoring."""

import pytest
import numpy as np
import pandas as pd

from src.ingestion.schema import ClusterEventTypeV2, ClusterEventTypeV3, FailureLabel
from src.labeling.target_generator import TargetLabelGenerator


def test_exact_failure_horizon_boundaries():
    generator = TargetLabelGenerator(
        critical_horizon_sec=900.0,  # 15 min
        warning_horizon_sec=1800.0, # 30 min
        trace_version="2011"
    )
    t_pred = 1000.0

    # 1. Failure at tau = 900.0s (exactly 15 min) -> CRITICAL
    events_crit = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.FAIL)],
        "time_sec": [t_pred + 900.0],
    })
    label, tau = generator.derive_label_for_task(t_pred, events_crit, t_pred)
    assert label == FailureLabel.CRITICAL
    assert np.isclose(tau, 900.0)

    # 2. Failure at tau = 900.1s (just over 15 min) -> WARNING
    events_warn1 = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.FAIL)],
        "time_sec": [t_pred + 900.1],
    })
    label, tau = generator.derive_label_for_task(t_pred, events_warn1, t_pred)
    assert label == FailureLabel.WARNING
    assert np.isclose(tau, 900.1)

    # 3. Failure at tau = 1800.0s (exactly 30 min) -> WARNING
    events_warn2 = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.FAIL)],
        "time_sec": [t_pred + 1800.0],
    })
    label, tau = generator.derive_label_for_task(t_pred, events_warn2, t_pred)
    assert label == FailureLabel.WARNING
    assert np.isclose(tau, 1800.0)

    # 4. Failure at tau = 1800.1s (just over 30 min) -> NORMAL
    events_norm = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.FAIL)],
        "time_sec": [t_pred + 1800.1],
    })
    label, tau = generator.derive_label_for_task(t_pred, events_norm, t_pred)
    assert label == FailureLabel.NORMAL
    assert np.isclose(tau, 1800.1)


def test_normal_completion_and_cancellation():
    generator = TargetLabelGenerator(trace_version="2011")
    t_pred = 1000.0

    # Normal FINISH at t + 600s (< 30 min) -> NORMAL
    events_finish = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.FINISH)],
        "time_sec": [t_pred + 600.0],
    })
    label, _ = generator.derive_label_for_task(t_pred, events_finish, t_pred + 600.0)
    assert label == FailureLabel.NORMAL

    # User KILL cancellation at t + 600s (< 30 min) -> UNKNOWN (censored)
    events_kill = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV2.KILL)],
        "time_sec": [t_pred + 600.0],
    })
    label, _ = generator.derive_label_for_task(t_pred, events_kill, t_pred + 600.0)
    assert label == FailureLabel.UNKNOWN

    # Truncated observation (telemetry stops at t + 300s with no terminal event) -> UNKNOWN
    events_none = pd.DataFrame(columns=["event_type", "time_sec"])
    label, _ = generator.derive_label_for_task(t_pred, events_none, t_pred + 300.0)
    assert label == FailureLabel.UNKNOWN

    # Telemetry extends past t + 1800s (>= 30 min) with no events -> NORMAL
    label, _ = generator.derive_label_for_task(t_pred, events_none, t_pred + 2000.0)
    assert label == FailureLabel.NORMAL


def test_version_2019_event_classifier():
    # In 2019 (v3), FAIL is code 5 (not 3)
    generator_v3 = TargetLabelGenerator(trace_version="2019")
    t_pred = 500.0

    events_v3_fail = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV3.FAIL)], # 5
        "time_sec": [t_pred + 600.0],
    })
    label, tau = generator_v3.derive_label_for_task(t_pred, events_v3_fail, t_pred)
    assert label == FailureLabel.CRITICAL

    # Verify that in 2019, code 3 is SCHEDULE, not FAIL!
    events_v3_sched = pd.DataFrame({
        "event_type": [int(ClusterEventTypeV3.SCHEDULE)], # 3
        "time_sec": [t_pred + 600.0],
    })
    label, _ = generator_v3.derive_label_for_task(t_pred, events_v3_sched, t_pred)
    assert label != FailureLabel.CRITICAL
