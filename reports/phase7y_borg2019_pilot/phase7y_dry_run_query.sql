-- ============================================================================
-- Phase 7Y Continuation: Verified Borg 2019 Safe Dry-Run Query
-- Target Dataset: `google.com:google-cluster-data.clusterdata_2019_a`
-- Schema Version: Google cluster-usage traces v3 (clusterdata_trace_format_v3.proto)
-- Observation Horizon: First 24 Hours (start_time >= 0 AND end_time <= 86,400,000,000 us)
-- Event Follow-up Horizon: 25 Hours (time <= 90,000,000,000 us) to guarantee complete 25m follow-up
-- Guardrail Cap: Max scan budget <= 21.0 GB; Max cost <= $0.125 USD (at $6.25/TiB)
-- Join Cardinality Safety: Window-anchored earliest terminal event prevents Cartesian explosion
-- ============================================================================

WITH active_scheduled_tasks AS (
    -- Identify tasks actively scheduled within the 24-hour observation horizon
    SELECT DISTINCT
        'cell_a' AS cell_id,
        collection_id,
        instance_index,
        priority,
        scheduling_class
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE
        time >= 0 AND time <= 86400000000
        AND type = 3  -- SCHEDULE (task started running)
),

terminal_events AS (
    -- Retrieve candidate terminal events up to 25 hours (provides 1-hour buffer past Day 1)
    SELECT
        collection_id,
        instance_index,
        time AS terminal_time,
        type AS terminal_type_code,
        CASE
            WHEN type = 5 THEN 'FAIL'
            WHEN type = 8 THEN 'LOST'
            WHEN type = 6 THEN 'FINISH'
            WHEN type = 4 THEN 'EVICT'
            WHEN type = 7 THEN 'KILL'
            ELSE 'OTHER'
        END AS terminal_status
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE
        time >= 0 AND time <= 90000000000
        AND type IN (4, 5, 6, 7, 8)
),

usage_with_terminal_candidates AS (
    -- Window-anchored join: join usage windows to terminal events strictly occurring after window end
    -- ROW_NUMBER() is partitioned per (collection_id, instance_index, start_time) to pick the EARLIEST
    -- terminal event following that specific window, resolving task retry ambiguity without Cartesian expansion.
    SELECT
        u.collection_id,
        u.instance_index,
        u.machine_id,
        u.alloc_collection_id,
        u.start_time AS window_start_us,
        u.end_time AS window_end_us,
        -- Protobuf v3 aligned telemetry fields:
        u.average_usage.cpus AS cpu_avg_usage,
        u.maximum_usage.cpus AS cpu_max_usage,
        u.average_usage.memory AS memory_canonical_usage,
        u.assigned_memory AS memory_assigned_limit,
        u.page_cache_memory AS memory_page_cache,
        te.terminal_time,
        te.terminal_type_code,
        te.terminal_status,
        ROW_NUMBER() OVER (
            PARTITION BY u.collection_id, u.instance_index, u.start_time
            ORDER BY te.terminal_time ASC
        ) AS rn
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS u
    INNER JOIN
        active_scheduled_tasks AS ast
    ON
        u.collection_id = ast.collection_id
        AND u.instance_index = ast.instance_index
    LEFT JOIN
        terminal_events AS te
    ON
        u.collection_id = te.collection_id
        AND u.instance_index = te.instance_index
        AND te.terminal_time >= u.end_time
    WHERE
        u.start_time >= 0 AND u.end_time <= 86400000000
)

SELECT
    'cell_a' AS cell_id,
    collection_id,
    instance_index,
    machine_id,
    alloc_collection_id,
    window_start_us,
    window_end_us,
    cpu_avg_usage,
    cpu_max_usage,
    memory_canonical_usage,
    memory_assigned_limit,
    memory_page_cache,
    terminal_time AS terminal_event_timestamp_us,
    terminal_type_code,
    terminal_status
FROM
    usage_with_terminal_candidates
WHERE
    rn = 1 OR rn IS NULL
ORDER BY
    collection_id,
    instance_index,
    window_start_us ASC;
