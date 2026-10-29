"""Phase 1 Verification Script: Confirms environment, modular architecture, and dataset audit."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import numpy as np
import pandas as pd
import sklearn

from src.ingestion.schema import ClusterEventType, FailureLabel
from src.ingestion.validator import TimeSeriesValidator


def main():
    print("=" * 70)
    print("PHASE 1 VERIFICATION: DATASET AUDIT AND PROJECT FOUNDATION")
    print("=" * 70)

    # 1. Environment Verification
    print("\n[1] Environment & Dependencies:")
    print(f"  - Python Executable: {sys.executable}")
    print(f"  - Python Version: {sys.version.split()[0]}")
    print(f"  - PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - Pandas: {pd.__version__}")
    print(f"  - Scikit-Learn: {sklearn.__version__}")
    print(f"  - PyYAML: {yaml.__version__}")

    # 2. Project Directory Layout Verification
    print("\n[2] Project Directory Structure:")
    required_dirs = [
        "configs",
        "src/ingestion",
        "src/features",
        "src/labeling",
        "src/splitting",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/inference",
        "tests",
        "reports",
        "data/raw",
        "data/processed",
        "data/synthetic_benchmark",
    ]
    for d in required_dirs:
        p = Path(d)
        status = "OK" if p.is_dir() else "MISSING"
        print(f"  - {d:30s} : [{status}]")
        assert p.is_dir(), f"Missing required directory: {d}"

    # 3. Master Config Verification
    print("\n[3] Master Configuration:")
    config_path = Path("configs/default_config.yaml")
    assert config_path.is_file(), "default_config.yaml missing!"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"  - Project: {config['project']['name']} v{config['project']['version']}")
    print(f"  - Sampling: Delta_t = {config['time_series']['observation_interval_minutes']} min, L = {config['time_series']['sequence_length']} steps")
    print(f"  - Horizons: Critical = {config['labeling']['critical_horizon_minutes']} min, Warning = {config['labeling']['warning_horizon_minutes']} min")
    print(f"  - Splitting: {config['splitting']['strategy']} ({config['splitting']['train_ratio']}/{config['splitting']['val_ratio']}/{config['splitting']['test_ratio']})")

    # 4. Schema & Enum Semantics Verification
    print("\n[4] Schema & Lifecycle Event Enumerations:")
    print(f"  - FAIL enum: {int(ClusterEventType.FAIL)} (Abnormal failure: {ClusterEventType.is_abnormal_failure(3)})")
    print(f"  - FINISH enum: {int(ClusterEventType.FINISH)} (Normal completion: {ClusterEventType.is_normal_completion(4)})")
    print(f"  - EVICT enum: {int(ClusterEventType.EVICT)} (Preemption/Eviction: {ClusterEventType.is_preemption_eviction(2)})")
    print(f"  - KILL enum: {int(ClusterEventType.KILL)} (Cancellation: {ClusterEventType.is_cancellation(5)})")
    print(f"  - Target labels: NORMAL={int(FailureLabel.NORMAL)}, WARNING={int(FailureLabel.WARNING)}, CRITICAL={int(FailureLabel.CRITICAL)}, UNKNOWN={int(FailureLabel.UNKNOWN)}")

    # 5. Dataset Audit Report Verification
    print("\n[5] Dataset Audit Report:")
    audit_path = Path("reports/dataset_audit.md")
    assert audit_path.is_file(), "reports/dataset_audit.md missing!"
    audit_size = audit_path.stat().st_size
    print(f"  - reports/dataset_audit.md : [{audit_size} bytes verified]")

    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION SUCCESSFUL: ALL COMPLETION GATES PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
