import pandas as pd
import numpy as np

cand_df = pd.read_csv('results/phase7a_4/candidate_a_job_grouped_split.csv')
cand_eids = set(cand_df['entity_id'].unique())
USAGE_COLS = [
    'start_time', 'end_time', 'job_id', 'task_index', 'machine_id',
    'cpu_rate', 'canonical_memory', 'assigned_memory', 'unmapped_page_cache',
    'total_page_cache', 'max_memory', 'disk_io_time', 'local_disk_space',
    'max_cpu_rate', 'max_disk_io_time', 'cycles_per_instruction',
    'memory_accesses_per_instruction', 'sample_rate', 'aggregation_type',
    'sampled_cpu_usage'
]
usage_shards = ['data/raw/sample_task_usage.csv.gz', 'data/raw/sample_task_usage_part1.csv.gz']
chunks = []
for s in usage_shards:
    for chunk in pd.read_csv(s, header=None, names=USAGE_COLS, chunksize=500000):
        chunk['entity_id'] = chunk['job_id'].astype(str) + '_' + chunk['task_index'].astype(str)
        m = chunk[chunk['entity_id'].isin(cand_eids)]
        if not m.empty:
            chunks.append(m[['entity_id', 'cycles_per_instruction', 'memory_accesses_per_instruction', 'disk_io_time']])
df = pd.concat(chunks, ignore_index=True)
print('Total rows:', len(df))
print('CPI isna:', df['cycles_per_instruction'].isna().sum(), f"pct: {df['cycles_per_instruction'].isna().mean()*100:.2f}%")
print('MAI isna:', df['memory_accesses_per_instruction'].isna().sum(), f"pct: {df['memory_accesses_per_instruction'].isna().mean()*100:.2f}%")
print('Disk IO isna:', df['disk_io_time'].isna().sum(), f"pct: {df['disk_io_time'].isna().mean()*100:.2f}%")
cpi = df['cycles_per_instruction'].dropna()
print('CPI stats: min', cpi.min(), 'max', cpi.max(), 'mean', cpi.mean(), 'median', cpi.median())
