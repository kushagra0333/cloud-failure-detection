"""Update phase7j_manifest.json to include all artifacts."""
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
results_dir = PROJECT_ROOT / "results" / "phase7j"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "manifest_version": "1.0",
    "phase": "Phase 7J — Validation-Only Decision Threshold and Operating-Point Investigation",
    "timestamp_iso": "2026-10-28T22:15:00+05:30",
    "protocol_id": "PROTOCOL-PHASE-7J-01",
    "safeguards_verification": {
        "candidate_a_test_accessed": False,
        "phase6_confirmatory_holdout_accessed": False,
        "existing_artifacts_overwritten": False,
        "new_models_trained": False,
        "confirmatory_testing_authorized": False
    },
    "summary_findings": {
        "policyA_passing_operating_points": 0,
        "policyB_passing_operating_points": 0,
        "operational_winner": None,
        "verdict": "NO OPERATIONAL WINNER"
    },
    "artifacts": {}
}

for p in sorted(results_dir.glob("*")):
    if p.is_file() and p.name != "phase7j_manifest.json":
        manifest["artifacts"][p.name] = {
            "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p)
        }

with open(results_dir / "phase7j_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Successfully updated {results_dir / 'phase7j_manifest.json'}")
for name, meta in manifest["artifacts"].items():
    print(f"  {name}: {meta['size_bytes']} bytes, SHA-256: {meta['sha256'][:16]}...")
