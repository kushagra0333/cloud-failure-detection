import json
import hashlib
import shutil
from pathlib import Path

res_dir = Path("results/phase7s_relative_ablation")
rep_dir = Path("reports/phase7s_relative_ablation")
rep_dir.mkdir(parents=True, exist_ok=True)

with open(res_dir / "phase7s_manifest.json", "r") as f:
    manifest = json.load(f)

deliverables = manifest["deliverables"]
print(f"Checking {len(deliverables)} deliverables...")

mismatches = []
for fname, entry in deliverables.items():
    p = res_dir / fname
    if not p.exists():
        mismatches.append(f"{fname}: not found in results")
        continue
    actual_size = p.stat().st_size
    actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    if entry["sha256"] != actual_hash:
        mismatches.append(f"{fname}: sha256 mismatch")
    if entry["bytes"] != actual_size:
        mismatches.append(f"{fname}: byte mismatch {entry['bytes']} vs {actual_size}")
    
    # Mirror to reports
    shutil.copy2(p, rep_dir / fname)

print("Mismatches:", mismatches)
missing_rep = [fname for fname in deliverables if not (rep_dir / fname).exists()]
print("Missing in reports:", missing_rep)

if not mismatches and not missing_rep:
    print("ALL 23 DELIVERABLES PERFECTLY VERIFIED AND MIRRORED!")
