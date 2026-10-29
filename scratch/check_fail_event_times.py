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
from scripts.extract_confirmatory_data import EVENT_COLS

event_dfs = []
for s_path in protocol["source_event_shards"]:
    p = Path(s_path)
    if p.exists():
        df = pd.read_csv(p, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(fail_eids)]
        if not matched.empty:
            event_dfs.append(matched)
events_raw = pd.concat(event_dfs, ignore_index=True)
events_raw["time_sec"] = events_raw["timestamp"] / 1e6

f_events = events_raw[events_raw["event_type"] == 3]
min_fail = f_events.groupby("entity_id")["time_sec"].min()
print("Failure timestamp statistics across 100 failure entities:")
print(min_fail.describe())
print(f"Entities failing at or before 5100s: {(min_fail <= 5100.0).sum()} / 100")
print(f"Entities failing after 5100s: {(min_fail > 5100.0).sum()} / 100")
if (min_fail > 5100.0).sum() > 0:
    print(min_fail[min_fail > 5100.0])
