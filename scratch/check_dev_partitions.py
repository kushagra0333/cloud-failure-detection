import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Load dev metadata
tr = pd.read_csv("data/processed/train_metadata.csv")
va = pd.read_csv("data/processed/val_metadata.csv")
te = pd.read_csv("data/processed/test_metadata.csv")

print(f"Train rows: {len(tr)}, entities: {tr['entity_id'].nunique()}, failing entities: {tr[tr['label'] > 0]['entity_id'].nunique()}")
print(f"Val rows: {len(va)}, entities: {va['entity_id'].nunique()}, failing entities: {va[va['label'] > 0]['entity_id'].nunique()}")
print(f"Test rows: {len(te)}, entities: {te['entity_id'].nunique()}, failing entities: {te[te['label'] > 0]['entity_id'].nunique()}")

dev_entities = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])
print(f"Total unique entities across train/val/test: {len(dev_entities)}")

failing_dev_entities = set(tr[tr["label"] > 0]["entity_id"]) | set(va[va["label"] > 0]["entity_id"]) | set(te[te["label"] > 0]["entity_id"])
print(f"Total failing entities with valid pre-failure windows in train/val/test: {len(failing_dev_entities)}")

# Check chronological partition if any
chrono_files = list(Path("data/processed/chronological").glob("*.csv"))
print(f"Chronological files: {chrono_files}")
for cf in chrono_files:
    df_c = pd.read_csv(cf)
    print(f"  {cf.name}: rows={len(df_c)}, entities={df_c['entity_id'].nunique()}")

# Where did the 241 number in Phase 6 come from?
# Let's inspect scripts/audit_phase6_feasibility.py line 56-75!
