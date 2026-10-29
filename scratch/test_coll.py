from google.cloud import bigquery

client = bigquery.Client()
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

q = """
SELECT time, type, collection_id, parent_collection_id, user, collection_name
FROM `google.com:google-cluster-data.clusterdata_2019_a.collection_events`
WHERE time >= 0 AND time <= 86400000000
"""
j = client.query(q, job_config=job_config)
print(f"collection_events query bytes: {j.total_bytes_processed:,} bytes ({j.total_bytes_processed / (1024**3):.4f} GiB)")

