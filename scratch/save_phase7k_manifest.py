"""Generate cryptographic manifest for results/phase7k/."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
phase_dir = PROJECT_ROOT / "results" / "phase7k"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = [
    "feature_availability_audit.csv",
    "phase7k_ablation_protocol.json",
    "phase7k_feature_audit_report.md"
]

manifest = {
    "manifest_id": "PHASE-7K-FEATURE-AUDIT-MANIFEST",
    "timestamp_iso": "2026-10-28T23:02:30+05:30",
    "phase": "Phase 7K — Feature Availability Audit and Predeclared Ablation Protocol",
    "status": "SEALED",
    "training_conducted": False,
    "candidate_a_test_accessed": False,
    "phase6_holdout_accessed": False,
    "files": {}
}

for fname in files_to_hash:
    fpath = phase_dir / fname
    if fpath.exists():
        manifest["files"][fname] = {
            "path": f"results/phase7k/{fname}",
            "size_bytes": fpath.stat().st_size,
            "sha256": compute_sha256(fpath)
        }

manifest_p = phase_dir / "phase7k_manifest.json"
with open(manifest_p, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved Phase 7K manifest to {manifest_p}")
