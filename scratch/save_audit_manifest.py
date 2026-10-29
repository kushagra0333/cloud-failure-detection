"""Generate cryptographic manifest for results/phase7j_audit/."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
audit_dir = PROJECT_ROOT / "results" / "phase7j_audit"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = [
    "independent_threshold_metrics.csv",
    "independent_gate_summary.csv",
    "metric_discrepancy_register.csv",
    "phase7j_audit_integrity.json",
    "phase7j_audit_report.md"
]

manifest = {
    "manifest_id": "PHASE-7J-A-AUDIT-MANIFEST",
    "timestamp_iso": "2026-10-28T22:47:30+05:30",
    "phase": "Phase 7J-A — Independent Metric Reproduction and Artifact Consistency Audit",
    "status": "SEALED",
    "executive_verdict": "NO_OPERATIONAL_WINNER_CONFIRMED",
    "files": {}
}

for fname in files_to_hash:
    fpath = audit_dir / fname
    if fpath.exists():
        manifest["files"][fname] = {
            "path": f"results/phase7j_audit/{fname}",
            "size_bytes": fpath.stat().st_size,
            "sha256": compute_sha256(fpath)
        }
    else:
        print(f"Warning: {fname} not found!")

manifest_p = audit_dir / "phase7j_audit_manifest.json"
with open(manifest_p, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved manifest to {manifest_p}")
