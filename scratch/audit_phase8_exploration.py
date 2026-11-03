import pandas as pd
import numpy as np
from pathlib import Path

def explore():
    data_dir = Path("data/processed/phase7c")
    train_meta = pd.read_csv(data_dir / "train_metadata.csv")
    val_meta = pd.read_csv(data_dir / "val_metadata.csv")
    test_meta = pd.read_csv(data_dir / "test_metadata.csv")

    print("--- Phase 7C Candidate A Splits ---")
    print(f"Train: {len(train_meta)} windows, {train_meta['entity_id'].nunique()} entities, {train_meta['parent_job_id'].nunique()} parent jobs")
    print(f"Val:   {len(val_meta)} windows, {val_meta['entity_id'].nunique()} entities, {val_meta['parent_job_id'].nunique()} parent jobs")
    print(f"Test:  {len(test_meta)} windows, {test_meta['entity_id'].nunique()} entities, {test_meta['parent_job_id'].nunique()} parent jobs")

    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print("\nClass counts (0=Normal, 1=Warning, 2=Critical):")
    print("Train:", np.bincount(y_train), f"Percentages: {np.round(np.bincount(y_train)/len(y_train)*100, 2)}")
    print("Val:  ", np.bincount(y_val), f"Percentages: {np.round(np.bincount(y_val)/len(y_val)*100, 2)}")
    print("Test: ", np.bincount(y_test), f"Percentages: {np.round(np.bincount(y_test)/len(y_test)*100, 2)}")

    # Check entity overlap
    train_entities = set(train_meta["entity_id"])
    val_entities = set(val_meta["entity_id"])
    test_entities = set(test_meta["entity_id"])

    train_jobs = set(train_meta["parent_job_id"])
    val_jobs = set(val_meta["parent_job_id"])
    test_jobs = set(test_meta["parent_job_id"])

    print("\nOverlap Analysis:")
    print("Entity overlap Train & Val:", len(train_entities & val_entities))
    print("Entity overlap Train & Test:", len(train_entities & test_entities))
    print("Entity overlap Val & Test:", len(val_entities & test_entities))
    print("Job overlap Train & Val:", len(train_jobs & val_jobs))
    print("Job overlap Train & Test:", len(train_jobs & test_jobs))
    print("Job overlap Val & Test:", len(val_jobs & test_jobs))

    # Check temporal ranges
    print("\nTemporal Prediction Timestamp Ranges:")
    for name, df in [("Train", train_meta), ("Val", val_meta), ("Test", test_meta)]:
        t_min = df["prediction_timestamp"].min()
        t_max = df["prediction_timestamp"].max()
        print(f"{name}: min={t_min}, max={t_max}")

if __name__ == "__main__":
    explore()
