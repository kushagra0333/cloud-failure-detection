from google.cloud import bigquery

client = bigquery.Client(project="canvas-advice-464215-n1")
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

q_events = """
SELECT collection_id, instance_index, time, type
FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
WHERE time >= 0 AND time <= 86400000000
"""
j1 = client.query(q_events, job_config=job_config)
print(f"instance_events scan: {j1.total_bytes_processed:,} bytes ({j1.total_bytes_processed / (1024**3):.4f} GiB)")

q_usage = """
SELECT collection_id, instance_index, start_time, end_time,
       average_usage.cpus, maximum_usage.cpus, average_usage.memory, assigned_memory, page_cache_memory
FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
WHERE start_time >= 0 AND end_time <= 86400000000
"""
j2 = client.query(q_usage, job_config=job_config)
print(f"instance_usage scan: {j2.total_bytes_processed:,} bytes ({j2.total_bytes_processed / (1024**3):.4f} GiB)")

