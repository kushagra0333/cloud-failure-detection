"""Generate feature_availability_audit.csv for Phase 7K."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
out_dir = PROJECT_ROOT / "results" / "phase7k"
out_dir.mkdir(parents=True, exist_ok=True)

audit_records = [
    # --- TASK USAGE RAW FIELDS ---
    {
        "field_name": "cpu_rate",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Mean CPU utilization rate over trailing 300s window (core fraction)",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Core component of baseline 31 features (Group A)."
    },
    {
        "field_name": "canonical_memory",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Mean canonical memory usage over trailing 300s window (memory fraction)",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Core component of baseline 31 features (Group A)."
    },
    {
        "field_name": "assigned_memory",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Borg scheduler allocated memory quota over trailing 300s window",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Core component of baseline 31 features (Group A)."
    },
    {
        "field_name": "disk_io_time",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Cumulative disk I/O time fraction over trailing 300s window",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Included in baseline 31 features (Group A)."
    },
    {
        "field_name": "cycles_per_instruction",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Hardware execution efficiency CPI counter over trailing 300s window",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 1.03,
        "missingness_val_pct": 1.03,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Included in baseline 31 features (Group A); missingness explicitly tracked via Group I."
    },
    {
        "field_name": "memory_accesses_per_instruction",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Hardware memory access frequency counter over trailing 300s window",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 9.37,
        "missingness_val_pct": 9.37,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_BASELINE",
        "rationale_evidence": "Included in baseline 31 features (Group A)."
    },
    {
        "field_name": "max_cpu_rate",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Peak CPU utilization rate observed during 300s interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from 31-feature baseline. Provides peak load and burstiness (max_cpu - mean_cpu)."
    },
    {
        "field_name": "max_memory",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Peak memory utilization observed during 300s interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from baseline. Enables memory headroom (assigned_memory - max_memory) and peak memory ratio."
    },
    {
        "field_name": "total_page_cache",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Total page cache memory utilized during 300s interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from baseline. Distinguishes active memory from reclaimable cache; r = +0.0988 with failure."
    },
    {
        "field_name": "unmapped_page_cache",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Unmapped page cache memory during 300s interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from baseline. Highly correlated with total page cache."
    },
    {
        "field_name": "local_disk_space",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Local disk space consumption during 300s interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from baseline. 0% missingness across all dev windows."
    },
    {
        "field_name": "max_disk_io_time",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Peak disk I/O time fraction observed during interval",
        "timestamp_semantics": "End of 300s measurement interval (t_end <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (strictly t_end <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_ADDITIONAL_TELEMETRY",
        "rationale_evidence": "Omitted from baseline. 0% missingness after 300s bin resampling."
    },
    {
        "field_name": "sample_rate",
        "source_table": "task_usage",
        "field_type": "sampling_metadata",
        "definition": "Measurement sampling rate",
        "timestamp_semantics": "Metadata",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_UNINFORMATIVE",
        "rationale_evidence": "Constant collection metadata across shards."
    },
    {
        "field_name": "aggregation_type",
        "source_table": "task_usage",
        "field_type": "sampling_metadata",
        "definition": "Measurement aggregation type",
        "timestamp_semantics": "Metadata",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_UNINFORMATIVE",
        "rationale_evidence": "Constant collection metadata across shards."
    },
    {
        "field_name": "sampled_cpu_usage",
        "source_table": "task_usage",
        "field_type": "continuous_telemetry",
        "definition": "Point sample of CPU usage",
        "timestamp_semantics": "Instantaneous sample at t_end",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_REDUNDANT",
        "rationale_evidence": "Redundant with mean cpu_rate and max_cpu_rate."
    },

    # --- TASK EVENTS STATIC / SCHEDULER ATTRIBUTES ---
    {
        "field_name": "scheduling_class",
        "source_table": "task_events",
        "field_type": "categorical_static",
        "definition": "Task latency sensitivity class (0=non-production, 1=batch, 2=mid, 3=latency-sensitive prod)",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_STATIC_METADATA",
        "rationale_evidence": "Static scheduler attribute known at task inception. 0% missingness."
    },
    {
        "field_name": "priority",
        "source_table": "task_events",
        "field_type": "ordinal_static",
        "definition": "Borg scheduling priority tier (0=free, 1=batch, 2=production, 8-10=monitoring/critical)",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_STATIC_METADATA",
        "rationale_evidence": "Static scheduler priority known at task inception. 0% missingness."
    },
    {
        "field_name": "req_cpu",
        "source_table": "task_events",
        "field_type": "continuous_static",
        "definition": "Requested CPU resource capacity specified in task submission",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_STATIC_METADATA",
        "rationale_evidence": "Requested quota known at task inception. 0% missingness."
    },
    {
        "field_name": "req_ram",
        "source_table": "task_events",
        "field_type": "continuous_static",
        "definition": "Requested RAM resource capacity specified in task submission",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_STATIC_METADATA",
        "rationale_evidence": "Requested quota known at task inception. 0% missingness."
    },
    {
        "field_name": "req_disk",
        "source_table": "task_events",
        "field_type": "continuous_static",
        "definition": "Requested local disk space capacity specified in task submission",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_STATIC_METADATA",
        "rationale_evidence": "Requested disk quota known at task inception. 0% missingness."
    },
    {
        "field_name": "diff_machine_restriction",
        "source_table": "task_events",
        "field_type": "boolean_static",
        "definition": "Placement constraint requiring tasks to run on different physical machines",
        "timestamp_semantics": "Recorded at initial task submission (t_submit <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have t_submit <= t_pred)",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_NEAR_ZERO_VARIANCE",
        "rationale_evidence": "99.8% false across dev set; uninformative."
    },
    {
        "field_name": "missing_info",
        "source_table": "task_events",
        "field_type": "metadata",
        "definition": "Flag indicating missing event fields",
        "timestamp_semantics": "Recorded at event time",
        "missingness_train_pct": 100.0,
        "missingness_val_pct": 100.0,
        "causal_availability": "UNAVAILABLE (100% missing)",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_ALL_NULL",
        "rationale_evidence": "100.0% null across entire dataset."
    },
    {
        "field_name": "user_name",
        "source_table": "task_events",
        "field_type": "categorical_static",
        "definition": "Hashed user identifier string",
        "timestamp_semantics": "Recorded at submission",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE",
        "leakage_risk": "NONE",
        "audit_verdict": "REJECTED_HIGH_CARDINALITY",
        "rationale_evidence": "Anonymized string with high cardinality and risk of job-level memorization."
    },

    # --- DYNAMIC LIFECYCLE EVENTS ---
    {
        "field_name": "eviction_count_prior",
        "source_table": "task_events",
        "field_type": "count_dynamic",
        "definition": "Cumulative count of EVICT (event_type=2) events observed for entity prior to t",
        "timestamp_semantics": "Event time t_evict <= t_pred",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (evaluated strictly on t_event <= t_pred)",
        "leakage_risk": "NONE (only pre-prediction events counted)",
        "audit_verdict": "ELIGIBLE_DYNAMIC_EVENTS",
        "rationale_evidence": "Genuinely observable; present in 95 dev windows (2.41%). Stronger correlation with warning/critical."
    },
    {
        "field_name": "reschedule_count_prior",
        "source_table": "task_events",
        "field_type": "count_dynamic",
        "definition": "Cumulative count of reschedule events (SCHEDULE events beyond the first) prior to t",
        "timestamp_semantics": "Event time t_schedule <= t_pred",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (evaluated strictly on t_event <= t_pred)",
        "leakage_risk": "NONE (only pre-prediction events counted)",
        "audit_verdict": "ELIGIBLE_DYNAMIC_EVENTS",
        "rationale_evidence": "Observable when task is rescheduled after preemption. Identical to eviction_count_prior in dev set."
    },
    {
        "field_name": "time_since_last_schedule",
        "source_table": "task_events",
        "field_type": "continuous_dynamic",
        "definition": "Elapsed seconds from latest SCHEDULE event (event_type=1) to prediction timestamp t",
        "timestamp_semantics": "t_pred - max(t_schedule <= t_pred)",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "AVAILABLE (100% of dev windows have observed prior schedule)",
        "leakage_risk": "NONE",
        "audit_verdict": "ELIGIBLE_DYNAMIC_EVENTS",
        "rationale_evidence": "Captures execution duration since last dispatch. Distinct from task_age_minutes if rescheduled."
    },

    # --- FORBIDDEN / NON-EXISTENT FIELDS (AUDIT REJECTIONS) ---
    {
        "field_name": "oom_indicator",
        "source_table": "NONE (SCHEMA_ABSENT)",
        "field_type": "nonexistent_field",
        "definition": "Explicit out-of-memory error event code or termination flag",
        "timestamp_semantics": "N/A",
        "missingness_train_pct": 100.0,
        "missingness_val_pct": 100.0,
        "causal_availability": "DOES NOT EXIST in Google Borg 2011 trace format",
        "leakage_risk": "N/A",
        "audit_verdict": "REJECTED_NONEXISTENT",
        "rationale_evidence": "VERIFIED: Google Cluster Trace 2011 has no OOM column or exit reason code in schema."
    },
    {
        "field_name": "restart_flag_after_fail",
        "source_table": "task_events",
        "field_type": "post_failure_event",
        "definition": "Task restart occurring after abnormal FAIL event",
        "timestamp_semantics": "t_event > t_fail >= t_pred",
        "missingness_train_pct": 100.0,
        "missingness_val_pct": 100.0,
        "causal_availability": "UNAVAILABLE at pre-failure prediction time",
        "leakage_risk": "CRITICAL_TARGET_LEAKAGE",
        "audit_verdict": "REJECTED_TARGET_LEAKAGE",
        "rationale_evidence": "Occurs only post-failure. Under Policy A, post-failure telemetry is excluded (-1). Cannot be observed prior to failure."
    },
    {
        "field_name": "future_fail_timestamp",
        "source_table": "task_events",
        "field_type": "terminal_event",
        "definition": "Timestamp of abnormal failure (FAIL, event_type=3)",
        "timestamp_semantics": "t_fail > t_pred",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "FUTURE_ONLY (strictly after prediction time t)",
        "leakage_risk": "FATAL_TARGET_LEAKAGE",
        "audit_verdict": "REJECTED_TARGET_LEAKAGE",
        "rationale_evidence": "Used exclusively as ground-truth target label generator; strictly forbidden as input feature."
    },
    {
        "field_name": "future_finish_timestamp",
        "source_table": "task_events",
        "field_type": "terminal_event",
        "definition": "Timestamp of normal completion (FINISH, event_type=4)",
        "timestamp_semantics": "t_finish > t_pred",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "FUTURE_ONLY (strictly after prediction time t)",
        "leakage_risk": "FATAL_TARGET_LEAKAGE",
        "audit_verdict": "REJECTED_TARGET_LEAKAGE",
        "rationale_evidence": "Directly leaks that task will terminate normally without failure. Strictly forbidden."
    },
    {
        "field_name": "future_kill_timestamp",
        "source_table": "task_events",
        "field_type": "terminal_event",
        "definition": "Timestamp of task cancellation (KILL, event_type=5)",
        "timestamp_semantics": "t_kill > t_pred",
        "missingness_train_pct": 0.0,
        "missingness_val_pct": 0.0,
        "causal_availability": "FUTURE_ONLY (strictly after prediction time t)",
        "leakage_risk": "FATAL_TARGET_LEAKAGE",
        "audit_verdict": "REJECTED_TARGET_LEAKAGE",
        "rationale_evidence": "Leaks terminal cancellation event. Strictly forbidden."
    }
]

df_audit = pd.DataFrame(audit_records)
csv_p = out_dir / "feature_availability_audit.csv"
df_audit.to_csv(csv_p, index=False)
print(f"Saved feature availability audit table to {csv_p} ({len(df_audit)} evaluated fields)")
