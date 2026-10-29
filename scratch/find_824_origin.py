import glob
import json
import os
import pandas as pd

results = []

# Check candidate_a_job_grouped_split.csv
df_split = pd.read_csv('results/phase7a_4/candidate_a_job_grouped_split.csv')
print("Candidate A Split CSV assigned_split:")
print(df_split['assigned_split'].value_counts())
print("\nCandidate A Split CSV original_split:")
print(df_split['original_split'].value_counts())

# Check if 824 appears in any column or grouping
for col in df_split.columns:
    if 824 in df_split[col].values:
        print(f"824 found in candidate_a column {col}")

# Check test_metadata.csv
df_meta = pd.read_csv('data/processed/phase7c/test_metadata.csv')
print(f"\ntest_metadata.csv rows: {len(df_meta)}")
print(f"test_metadata unique entities: {df_meta['entity_id'].nunique()}")
print(f"test_metadata unique parent_jobs: {df_meta['parent_job_id'].nunique()}")

# Check all json files for 824
for p in glob.glob('results/**/*.json', recursive=True) + glob.glob('configs/**/*.json', recursive=True):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
            if '824' in content:
                # find where
                data = json.loads(content)
                # check if 824 is a count or int
                def find_int_824(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            find_int_824(v, f"{path}.{k}")
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            find_int_824(v, f"{path}[{i}]")
                    elif obj == 824:
                        print(f"Match 824 exactly in {p} at {path}")
                find_int_824(data)
    except Exception:
        pass
