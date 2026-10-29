-- ============================================================================
-- Phase 7X: Google Borg 2019 Pilot Extraction Query Plan
-- Dataset: google.com:google-cluster-data.clusterdata_2019_a
-- Target Scope: Cell 'a', First 24 Hours (0 to 86,400,000,000 microseconds)
-- Budget Constraint: Max scan < 20 GB; Execution cost < $0.10 USD
-- ============================================================================

-- STEP 1: Extract Task Lifecycle Event History for Filtered Time Horizon
WITH target_instances AS (
    SELECT DISTINCT
        'cell_a' AS cell_id,
        collection_id,
        instance_index,
        priority,
        scheduling_class
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE
        time >= 0 AND time <= 86400000000  -- First 24 Hours
        AND type = 3                       -- SCHEDULE (task started running)
),

terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        time AS terminal_time,
        type AS terminal_event_type,
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
        time >= 0 AND time <= 90000000000  -- Include 1-hour lookahead for terminal state
        AND type IN (4, 5, 6, 7, 8)
)

-- STEP 2: Extract Bounded 5-Minute Usage Windows Aligned with Tasks
SELECT
    'cell_a' AS cell_id,
    u.collection_id,
    u.instance_index,
    u.machine_id,
    u.start_time AS window_start_us,
    u.end_time AS window_end_us,
    u.cpu_usage.average AS cpu_avg_usage,
    u.cpu_usage.maximum AS cpu_max_usage,
    u.memory_usage.canonical_memory_usage AS memory_canonical_usage,
    u.memory_usage.assigned_memory_usage AS memory_assigned_usage,
    t.terminal_time AS failure_timestamp_us,
    t.terminal_event_type,
    t.terminal_status
FROM
    `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS u
INNER JOIN
    target_instances AS ti
ON
    u.collection_id = ti.collection_id
    AND u.instance_index = ti.instance_index
LEFT JOIN
    terminal_events AS t
ON
    u.collection_id = t.collection_id
    AND u.instance_index = t.instance_index
WHERE
    u.start_time >= 0 AND u.end_time <= 86400000000
ORDER BY
    u.collection_id,
    u.instance_index,
    u.start_time ASC;
