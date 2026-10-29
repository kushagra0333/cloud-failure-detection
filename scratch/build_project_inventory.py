"""Assembles the complete, verified project inventory for Phase 6.8 Project Discovery.
Generates results/project_discovery/project_inventory.json.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

inventory: Dict[str, Any] = {
    "project_name": "Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing",
    "discovery_date": "2026-10-27",
    "auditor": "Independent ML Scientific Auditor & Pair Programmer",
    "codebase_modules": {},
    "scripts": {},
    "model_checkpoints": {},
    "configuration_files": {},
    "processed_datasets": {},
    "raw_datasets": {},
    "results_artifacts": {},
    "reports": {},
    "test_suites": {},
}

# 1. Codebase modules in src/
src_dir = PROJECT_ROOT / "src"
for p in sorted(src_dir.rglob("*.py")):
    if "__pycache__" in str(p):
        continue
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    inventory["codebase_modules"][rel_path] = {
        "file_path": rel_path,
        "line_count": len(p.read_text(encoding="utf-8", errors="replace").splitlines()),
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

# 2. Scripts in scripts/
scripts_dir = PROJECT_ROOT / "scripts"
for p in sorted(scripts_dir.glob("*.py")):
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    inventory["scripts"][rel_path] = {
        "file_path": rel_path,
        "line_count": len(p.read_text(encoding="utf-8", errors="replace").splitlines()),
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

# 3. Model Checkpoints
for ckpt_folder in ["checkpoints_l5", "checkpoints"]:
    ckpt_dir = PROJECT_ROOT / "models" / ckpt_folder
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("*")):
            if p.is_file():
                rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                info = {
                    "file_path": rel_path,
                    "file_size_bytes": p.stat().st_size,
                    "sha256": compute_sha256(p),
                    "model_family": ckpt_folder,
                }
                if p.suffix == ".pt":
                    try:
                        ckpt = torch.load(p, map_location="cpu", weights_only=False)
                        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                            sd = ckpt["model_state_dict"]
                            info["parameter_count"] = sum(v.numel() for v in sd.values())
                            info["best_epoch"] = ckpt.get("epoch")
                            info["best_metric"] = ckpt.get("best_metric")
                    except Exception as e:
                        info["load_note"] = str(e)
                inventory["model_checkpoints"][rel_path] = info

# 4. Configuration files
configs_dir = PROJECT_ROOT / "configs"
for p in sorted(configs_dir.glob("*")):
    if p.is_file():
        rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        inventory["configuration_files"][rel_path] = {
            "file_path": rel_path,
            "file_size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p),
        }

# 5. Processed datasets
proc_dir = PROJECT_ROOT / "data" / "processed"
for p in sorted(proc_dir.rglob("*.npy")):
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    arr = np.load(p, mmap_mode="r")
    inventory["processed_datasets"][rel_path] = {
        "file_path": rel_path,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

# 6. Raw datasets
raw_dir = PROJECT_ROOT / "data" / "raw"
for p in sorted(raw_dir.glob("*.csv.gz")):
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    inventory["raw_datasets"][rel_path] = {
        "file_path": rel_path,
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

# 7. Reports
reports_dir = PROJECT_ROOT / "reports"
for p in sorted(reports_dir.glob("*.md")):
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    inventory["reports"][rel_path] = {
        "file_path": rel_path,
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

# 8. Results
results_dir = PROJECT_ROOT / "results"
for p in sorted(results_dir.rglob("*")):
    if p.is_file() and "project_discovery" not in str(p):
        rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        inventory["results_artifacts"][rel_path] = {
            "file_path": rel_path,
            "file_size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p),
        }

# 9. Test suites
tests_dir = PROJECT_ROOT / "tests"
for p in sorted(tests_dir.glob("*.py")):
    rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    inventory["test_suites"][rel_path] = {
        "file_path": rel_path,
        "line_count": len(p.read_text(encoding="utf-8", errors="replace").splitlines()),
        "file_size_bytes": p.stat().st_size,
        "sha256": compute_sha256(p),
    }

out_p = PROJECT_ROOT / "results" / "project_discovery" / "project_inventory.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2)

print(f"Project inventory successfully written to {out_p}")
print(f"Total modules: {len(inventory['codebase_modules'])}")
print(f"Total scripts: {len(inventory['scripts'])}")
print(f"Total checkpoints: {len(inventory['model_checkpoints'])}")
print(f"Total test suites: {len(inventory['test_suites'])}")
print(f"Total reports: {len(inventory['reports'])}")
print(f"Total results: {len(inventory['results_artifacts'])}")
