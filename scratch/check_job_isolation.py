import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Load dev partitions
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")

dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])
dev_job_ids = set(eid.split("_")[0] for eid in dev_eids)

print(f"Dev unique entities: {len(dev_eids)}")
print(f"Dev unique job_ids: {len(dev_job_ids)}")

# Load the 137 unused failing entities that produce pre-failure windows with L=5
from scratch.audit_l5_failure_cohort import prefail_windows_5

candidate_eids = set(prefail_windows_5["entity_id"].unique())
candidate_job_ids = set(eid.split("_")[0] for eid in candidate_eids)

print(f"\nCandidate holdout failing entities: {len(candidate_eids)}")
print(f"Candidate holdout unique job_ids: {len(candidate_job_ids)}")

# Entity-level intersection
shared_entities = candidate_eids & dev_eids
print(f"Shared entities (entity_id intersection): {len(shared_entities)}")

# Job-level intersection
shared_jobs = candidate_job_ids & dev_job_ids
print(f"Shared job_ids (job_id intersection): {len(shared_jobs)}")
if shared_jobs:
    print(f"Sample shared job_ids: {list(shared_jobs)[:10]}")
    # How many candidate entities belong to shared jobs?
    cand_in_shared_jobs = [eid for eid in candidate_eids if eid.split("_")[0] in shared_jobs]
    print(f"Candidate entities belonging to shared jobs: {len(cand_in_shared_jobs)} / {len(candidate_eids)}")
    # How many candidate entities belong to COMPLETELY DISJOINT jobs?
    cand_disjoint_jobs = [eid for eid in candidate_eids if eid.split("_")[0] not in shared_jobs]
    print(f"Candidate entities belonging to COMPLETELY DISJOINT jobs: {len(cand_disjoint_jobs)}")
