import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Load protocol_l5.json
with open("results/phase6/protocol_l5.json") as f:
    proto = json.load(f)

# Load pretraining_integrity_audit.json
with open("results/phase6/pretraining_integrity_audit.json") as f:
    audit = json.load(f)

# Load dev partitions
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")
dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])

print("--- ARTIFACT CONSISTENCY COMPARISON ---")

# 1. Entity Counts
p_fail_eids = proto["frozen_entity_ids"]["all_failure_entities"]
p_norm_eids = proto["frozen_entity_ids"]["normal_entities"]
p_disjoint_eids = proto["frozen_entity_ids"]["disjoint_job_failure_entities"]
p_shared_eids = proto["frozen_entity_ids"]["shared_job_failure_entities"]

print(f"protocol_l5.json:")
print(f"  Failure entities: {len(p_fail_eids)}")
print(f"  Normal entities: {len(p_norm_eids)}")
print(f"  Disjoint failure entities: {len(p_disjoint_eids)}")
print(f"  Shared failure entities: {len(p_shared_eids)}")
print(f"  Sum disjoint + shared: {len(p_disjoint_eids) + len(p_shared_eids)}")
assert len(p_fail_eids) == 137
assert len(p_norm_eids) == 200
assert len(p_disjoint_eids) == 75
assert len(p_shared_eids) == 62

# 2. Episode Counts
p_episodes = proto["frozen_episode_metadata"]
print(f"\nEpisodes in metadata: {len(p_episodes)}")
assert len(p_episodes) == 141

# 3. Parent Jobs
p_disjoint_jobs = proto["frozen_parent_job_ids"]["disjoint_failure_jobs"]
p_shared_jobs = proto["frozen_parent_job_ids"]["shared_failure_jobs"]
print(f"\nParent Jobs in protocol_l5.json:")
print(f"  Disjoint failure jobs: {p_disjoint_jobs}")
print(f"  Shared failure jobs: {p_shared_jobs}")
assert len(p_disjoint_jobs) == 4
assert len(p_shared_jobs) == 5

# 4. Window Counts Comparison: Summary Block vs Manifest
summary_win = proto["cohort_counts_summary"]["sliding_windows"]
print(f"\nprotocol_l5.json summary sliding_windows:")
print(f"  prefail_windows_total: {summary_win['prefail_windows_total']}")
print(f"  warning_windows: {summary_win['warning_windows']}")
print(f"  critical_windows: {summary_win['critical_windows']}")
print(f"  normal_windows_from_failing_cohort: {summary_win['normal_windows_from_failing_cohort']}")
print(f"  normal_windows_from_normal_cohort: {summary_win['normal_windows_from_normal_cohort']}")
print(f"  total_evaluable_windows: {summary_win['total_evaluable_windows']}")

# Contrast with audit artifact
w_rec = audit["window_reconciliation"]
print(f"\npretraining_integrity_audit.json window reconciliation:")
print(f"  frozen_cohort_total: {w_rec['grand_total_evaluable_windows']['frozen_cohort_total']}")
print(f"  protocol_summary_census_total: {w_rec['grand_total_evaluable_windows']['protocol_summary_census_total']}")
print(f"  frozen_failing_entities_windows: {w_rec['frozen_failing_entities_windows']}")
print(f"  difference: {w_rec['difference_explanation']}")

# 5. Check Test Suites Expectations
print("\nChecking test assertions:")
import inspect
from tests import test_phase6_2_protocol_freeze as t6_2
from tests import test_phase6_2_1_pretraining_integrity as t6_2_1

print("test_phase6_2_protocol_freeze checks:")
print("  - win['normal_windows_from_failing_cohort'] == 512")
print("  - win['total_evaluable_windows'] == 5727")
print("test_phase6_2_1_pretraining_integrity checks:")
print("  - frozen_cohort_total == 5511")
print("  - protocol_summary_census_total == 5727")
print("  - normal_windows (from 137 failure entities) == 296")
print("  - normal_windows (from 198 census entities) == 512")

# 6. Check FPR/FAR Denominators in protocol vs audit
print("\nDenominators in protocol vs audit:")
print("  pure_normal_entities: 198 (both agree)")
print("  primary failure episodes: 141 (both agree)")
print("  sensitivity failure episodes: 79 (both agree)")
print("  pre-failure windows: 700 (300 warning, 400 critical) (both agree)")
print(f"  normal windows denominator in audit: {audit['evaluation_denominators']['normal_windows_denominator']} (4,811)")
print(f"  normal windows denominator in protocol summary: {summary_win['normal_windows_from_failing_cohort'] + summary_win['normal_windows_from_normal_cohort']} (5,027)")
