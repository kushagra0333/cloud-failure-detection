-- ============================================================================
-- Phase 7Y: Verified Google Borg 2019 Bounded Pilot Extraction Query
-- Target Dataset: `google.com:google-cluster-data.clusterdata_2019_a`
-- Schema Version: Google cluster-usage traces v3 (Protobuf aligned)
-- Time Horizon: First 24 Hours (0 to 86,400,000,000 microseconds)
-- Guardrail Cap: Max scan budget 25 GB; Max cost $0.125 USD
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
        AND type = 3  -- SCHEDULE
),

terminal_events AS (
    -- Retrieve subsequent terminal event for each instance attempt
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
        time >= 0 AND time <= 90000000000  -- 1-hour follow-up window (3600s)
        AND type IN (4, 5, 6, 7, 8)
),

earliest_terminal AS (
    -- Resolve join ambiguity: select earliest terminal event following observation
    SELECT
        collection_id,
        instance_index,
        terminal_time,
        terminal_type_code,
        terminal_status,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index 
            ORDER BY terminal_time ASC
        ) AS rn
    FROM
        terminal_events
)

SELECT
    'cell_a' AS cell_id,
    u.collection_id,
    u.instance_index,
    u.machine_id,
    u.alloc_collection_id,
    u.start_time AS window_start_us,
    u.end_time AS window_end_us,
    -- Official v3 protobuf schema field references:
    u.average_usage.cpus AS cpu_avg_usage,
    u.maximum_usage.cpus AS cpu_max_usage,
    u.average_usage.memory AS memory_canonical_usage,
    u.assigned_memory AS memory_assigned_limit,
    u.page_cache_memory AS memory_page_cache,
    et.terminal_time AS terminal_event_timestamp_us,
    et.terminal_type_code,
    et.terminal_status
FROM
    `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS u
INNER JOIN
    active_scheduled_tasks AS ast
ON
    u.collection_id = ast.collection_id
    AND u.instance_index = ast.instance_index
LEFT JOIN
    earliest_terminal AS et
ON
    u.collection_id = et.collection_id
    AND u.instance_index = et.instance_index
    AND et.rn = 1
    AND et.terminal_time >= u.end_time
WHERE
    u.start_time >= 0 AND u.end_time <= 86400000000
ORDER BY
    u.collection_id,
    u.instance_index,
    u.start_time ASC;
