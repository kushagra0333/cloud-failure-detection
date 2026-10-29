-- ============================================================================
-- Phase 7Y Optimization Candidate C: Bounded Workload Cohort & Micro-Horizon
-- Target Dataset: `google.com:google-cluster-data.clusterdata_2019_a`
-- Optimization Strategy: Restrict workload scope to an explicit bounded cohort
-- (sample of collection IDs) and a 1-hour micro-pilot observation horizon (3,600s = 3.6e9 us).
-- Designed to test whether row-level predicates reduce scanned bytes on unpartitioned tables.
-- ============================================================================

WITH active_scheduled_tasks AS (
    SELECT DISTINCT
        collection_id,
        instance_index
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE
        time >= 0 AND time <= 3600000000  -- 1-hour micro horizon
        AND type = 3
        AND collection_id IN (1001, 1002, 1003, 1004, 1005)  -- Bounded cohort filter
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
        time >= 0 AND time <= 5100000000  -- 1h + 25m follow-up horizon
        AND type IN (4, 5, 6, 7, 8)
        AND collection_id IN (1001, 1002, 1003, 1004, 1005)
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
        u.start_time >= 0 AND u.end_time <= 3600000000
        AND u.collection_id IN (1001, 1002, 1003, 1004, 1005)
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

