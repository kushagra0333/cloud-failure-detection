"""Generate results/phase7i_audit/manifest.json."""
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
audit_dir = PROJECT_ROOT / "results" / "phase7i_audit"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "manifest_version": "1.0",
    "phase": "Phase 7I-A — Training Collapse, Label Distribution, and Metric Audit",
    "timestamp_iso": "2026-10-28T21:55:00+05:30",
    "target_phase": "Phase 7I — Controlled Loss-Weighting Experiment",
    "safeguards_verification": {
        "candidate_a_test_accessed": False,
        "phase6_confirmatory_holdout_accessed": False,
        "existing_artifacts_overwritten": False,
        "new_models_trained_for_selection": False,
        "confirmatory_testing_authorized": False
    },
    "artifacts": {}
}

for p in sorted(audit_dir.glob("*")):
    if p.is_file() and p.name != "manifest.json":
        manifest["artifacts"][p.name] = {
            "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p)
        }

with open(audit_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Successfully generated {audit_dir / 'manifest.json'}")
print("Artifacts tracked:")
for name, meta in manifest["artifacts"].items():
    print(f"  {name}: {meta['size_bytes']} bytes, SHA-256: {meta['sha256'][:16]}...")
