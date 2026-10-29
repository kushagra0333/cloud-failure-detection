from google.cloud import bigquery

client = bigquery.Client()
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

def probe_cols(table_name, cols):
    print("=" * 60)
    print("Table:", table_name)
    for c in cols:
        q = f"SELECT {c} FROM `{table_name}`"
        j = client.query(q, job_config=job_config)
        b = j.total_bytes_processed
        print(f"  {c:35s}: {b:,} bytes ({b / (1024**3):.2f} GiB)")

usage_cols = [
    "collection_id",
    "instance_index",
    "machine_id",
    "alloc_collection_id",
    "start_time",
    "end_time",
    "average_usage.cpus",
    "maximum_usage.cpus",
    "average_usage.memory",
    "assigned_memory",
    "page_cache_memory",
]
probe_cols("google.com:google-cluster-data.clusterdata_2019_a.instance_usage", usage_cols)

event_cols = [
    "time",
    "type",
    "collection_id",
    "instance_index",
    "priority",
    "scheduling_class",
]
probe_cols("google.com:google-cluster-data.clusterdata_2019_a.instance_events", event_cols)

