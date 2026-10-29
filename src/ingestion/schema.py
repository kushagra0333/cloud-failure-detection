"""Data schema and event enumeration definitions for cloud cluster traces."""

from enum import IntEnum
from typing import Dict, List, Optional
import pandas as pd


class ClusterEventTypeV2(IntEnum):
    """Google Cluster Trace 2011 (v2.1) lifecycle event enumerations (Reiss et al. 2011)."""
    SUBMIT = 0          # Task eligible for scheduling
    SCHEDULE = 1        # Task assigned/scheduled to machine
    EVICT = 2           # Task descheduled due to resource overcommit / preemption
    FAIL = 3            # Task terminated abnormally due to internal execution error
    FINISH = 4          # Task terminated normally and successfully
    KILL = 5            # Task explicitly cancelled by user/parent/dependency
    LOST = 6            # Task disappeared or record missing
    UPDATE_PENDING = 7  # Resource request updated while pending
    UPDATE_RUNNING = 8  # Resource request updated while running

    @classmethod
    def is_abnormal_failure(cls, event_code: int) -> bool:
        return event_code == cls.FAIL

    @classmethod
    def is_normal_completion(cls, event_code: int) -> bool:
        return event_code == cls.FINISH

    @classmethod
    def is_cancellation(cls, event_code: int) -> bool:
        return event_code == cls.KILL

    @classmethod
    def is_preemption_eviction(cls, event_code: int) -> bool:
        return event_code == cls.EVICT


class ClusterEventTypeV3(IntEnum):
    """Google Cluster Trace 2019 (v3) lifecycle event enumerations.
    
    Verified from official protobuf: clusterdata_trace_format_v3.proto (Wilkes 2020).
    Note that QUEUE and ENABLE were introduced, shifting SCHEDULE to 3, EVICT to 4,
    FAIL to 5, and FINISH to 6.
    """
    SUBMIT = 0          # Submitted to scheduler
    QUEUE = 1           # Marked not eligible by batch scheduler
    ENABLE = 2          # Became eligible for scheduling
    SCHEDULE = 3        # Started running
    EVICT = 4           # Descheduled due to higher priority or overcommitment
    FAIL = 5            # Descheduled due to execution failure
    FINISH = 6          # Completed normally
    KILL = 7            # Cancelled by user or dependency died
    LOST = 8            # Terminated with missing transition data
    UPDATE_PENDING = 9  # Updated while pending
    UPDATE_RUNNING = 10 # Updated while running

    @classmethod
    def is_abnormal_failure(cls, event_code: int) -> bool:
        return event_code == cls.FAIL

    @classmethod
    def is_normal_completion(cls, event_code: int) -> bool:
        return event_code == cls.FINISH

    @classmethod
    def is_cancellation(cls, event_code: int) -> bool:
        return event_code == cls.KILL

    @classmethod
    def is_preemption_eviction(cls, event_code: int) -> bool:
        return event_code == cls.EVICT


# Default alias corresponds to the verified raw dataset version in data/raw (ClusterData 2011-2)
ClusterEventType = ClusterEventTypeV2


def get_event_classifier(version: str = "2011"):
    """Returns the version-appropriate event enumeration class."""
    if "2019" in str(version) or "v3" in str(version).lower():
        return ClusterEventTypeV3
    return ClusterEventTypeV2


class FailureLabel(IntEnum):
    """Target classes for early failure prediction.
    
    H_critical = 15 minutes (0 < tau <= 15 min)
    H_warning  = 30 minutes (15 < tau <= 30 min)
    NORMAL     = No qualifying failure within (t, t + 30 min]
    UNKNOWN    = Insufficient future observation (< 30 min and no terminal event)
    """
    UNKNOWN = -1
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2


# Standard required columns for telemetry time-series
TELEMETRY_REQUIRED_COLUMNS = [
    "entity_id",
    "timestamp",  # Unix timestamp in seconds or elapsed seconds
    "cpu_utilization",
    "memory_utilization",
]

# Standard optional/supplementary telemetry columns
TELEMETRY_OPTIONAL_COLUMNS = [
    "assigned_memory",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "disk_io_time",
]

# Standard required columns for lifecycle event logs
LIFECYCLE_REQUIRED_COLUMNS = [
    "entity_id",
    "timestamp",
    "event_type",
]


def get_expected_dtypes() -> Dict[str, str]:
    """Expected pandas dtypes for telemetry validation."""
    return {
        "entity_id": "object",
        "timestamp": "float64",
        "cpu_utilization": "float64",
        "memory_utilization": "float64",
        "assigned_memory": "float64",
        "cycles_per_instruction": "float64",
        "memory_accesses_per_instruction": "float64",
        "disk_io_time": "float64",
        "event_type": "int64",
    }
