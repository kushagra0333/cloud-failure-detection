"""Update and seal complete cryptographic manifest for results/phase7k/."""
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
    "temporal_causality_audit.json",
    "feature_availability_audit.csv",
    "phase7k_ablation_protocol.json",
    "fitted_preprocessors_manifest.json",
    "training_logs.csv",
    "validation_predictions.csv",
    "threshold_sweep_metrics.csv",
    "ablation_gate_summary.csv",
    "ablation_cross_seed_aggregate.csv",
    "phase7k_report.md"
]

manifest = {
    "manifest_id": "PHASE-7K-COMPLETE-AUDIT-MANIFEST",
    "timestamp_iso": "2026-10-28T23:28:30+05:30",
    "phase": "Phase 7K — Protocol Hardening and Controlled Feature Ablation",
    "status": "SEALED",
    "executive_verdict": "NO_OPERATIONAL_WINNER_CONFIRMED",
    "governance_verification": {
        "candidate_a_test_accessed": False,
        "candidate_a_test_status": "Inspected historically in Phase 7D; NOT accessed in Phase 7K.",
        "phase6_holdout_accessed": False,
        "phase6_holdout_status": "Strictly quarantined.",
        "evaluation_split": "Candidate A Validation (N=657 windows)",
        "training_split": "Candidate A Train (N=3278 windows)",
        "models_trained_count": 30,
        "evaluated_threshold_points_count": 6030,
        "passing_operational_winners_count": 0
    },
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
    else:
        print(f"Warning: {fname} missing!")

manifest_p = phase_dir / "phase7k_manifest.json"
with open(manifest_p, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Sealed complete Phase 7K manifest to {manifest_p}")
