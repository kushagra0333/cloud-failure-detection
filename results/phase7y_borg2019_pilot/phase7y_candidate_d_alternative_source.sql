-- ============================================================================
-- Phase 7Y Optimization Candidate D: Alternative Access Strategy (Job-Level Failure Log)
-- Target Dataset: `google.com:google-cluster-data.clusterdata_2019_a`
-- Optimization Strategy: Alternative access strategy evaluating collection_events
-- for job-level failure prediction without scanning the 2.2TB instance_usage table.
-- Researches whether a bounded event-level pilot can satisfy the 21 GB scan cap.
-- ============================================================================

WITH job_submissions AS (
    SELECT
        collection_id,
        parent_collection_id,
        user,
        collection_name,
        time AS submit_time_us
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.collection_events`
    WHERE
        time >= 0 AND time <= 86400000000
        AND type = 0  -- SUBMIT
),

job_terminal_events AS (
    SELECT
        collection_id,
        time AS terminal_time_us,
        type AS terminal_type_code,
        CASE
            WHEN type = 5 THEN 'FAIL'
            WHEN type = 6 THEN 'FINISH'
            WHEN type = 7 THEN 'KILL'
            WHEN type = 8 THEN 'LOST'
            ELSE 'OTHER'
        END AS terminal_status
    FROM
        `google.com:google-cluster-data.clusterdata_2019_a.collection_events`
    WHERE
        time >= 0 AND time <= 90000000000
        AND type IN (5, 6, 7, 8)
),

earliest_job_terminal AS (
    SELECT
        collection_id,
        terminal_time_us,
        terminal_type_code,
        terminal_status,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id
            ORDER BY terminal_time_us ASC
        ) AS rn
    FROM
        job_terminal_events
)

SELECT
    s.collection_id,
    s.parent_collection_id,
    s.user,
    s.collection_name,
    s.submit_time_us,
    t.terminal_time_us,
    t.terminal_type_code,
    t.terminal_status
FROM
    job_submissions AS s
LEFT JOIN
    earliest_job_terminal AS t
ON
    s.collection_id = t.collection_id
    AND t.rn = 1
    AND t.terminal_time_us >= s.submit_time_us
ORDER BY
    s.collection_id ASC;

