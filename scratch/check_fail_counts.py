import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

with open("results/phase4/confirmatory_protocol.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["failure_entities"])
print(f"Total frozen fail eids: {len(fail_eids)}")

# Check raw usage for these fail eids
from scripts.extract_confirmatory_data import EVENT_COLS, USAGE_COLS

usage_chunks = []
for s_path in protocol["source_usage_shards"]:
    p = Path(s_path)
    if p.exists():
        for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(fail_eids)]
            if not matched.empty:
                usage_chunks.append(matched)

if usage_chunks:
    df_fail_usage = pd.concat(usage_chunks, ignore_index=True)
    print(f"Total fail telemetry rows: {len(df_fail_usage)}")
    print(f"Entities in usage: {df_fail_usage['entity_id'].nunique()}")
    counts = df_fail_usage.groupby("entity_id").size()
    print("Telemetry step counts per fail entity summary:")
    print(counts.describe())
    print("Counts >= 15:", (counts >= 15).sum())
    print("Counts < 15:", (counts < 15).sum())
else:
    print("No fail usage found!")
