-- ============================================================================
-- Phase 7Y Optimization Candidate A: Minimal Leaf Projection Reduction
-- Target Dataset: `google.com:google-cluster-data.clusterdata_2019_a`
-- Optimization Strategy: Strip all non-essential columns from both tables.
-- Dropped from usage: machine_id, alloc_collection_id, cpu_max_usage, assigned_memory, page_cache_memory.
-- Dropped from events: priority, scheduling_class.
-- Retained leaf fields:
--   instance_usage: collection_id, instance_index, start_time, end_time, average_usage.cpus, average_usage.memory (6 cols)
--   instance_events: collection_id, instance_index, time, type (4 cols)
-- ============================================================================

WITH active_scheduled_tasks AS (
    SELECT DISTINCT
        collection_id,
        instance_index
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE
        time >= 0 AND time <= 86400000000
        AND type = 3  -- SCHEDULE
),

terminal_events AS (
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

usage_with_terminal AS (
    SELECT
        u.collection_id,
        u.instance_index,
        u.start_time AS window_start_us,
        u.end_time AS window_end_us,
        u.average_usage.cpus AS cpu_avg_usage,
        u.average_usage.memory AS memory_canonical_usage,
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
    collection_id,
    instance_index,
    window_start_us,
    window_end_us,
    cpu_avg_usage,
    memory_canonical_usage,
    terminal_time AS terminal_event_timestamp_us,
    terminal_type_code,
    terminal_status
FROM
    usage_with_terminal
WHERE
    rn = 1 OR rn IS NULL
ORDER BY
    collection_id,
    instance_index,
    window_start_us ASC;

