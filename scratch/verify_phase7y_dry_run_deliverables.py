import hashlib
import json
from pathlib import Path

results_dir = Path("results/phase7y_borg2019_pilot")
reports_dir = Path("reports/phase7y_borg2019_pilot")

manifest_file = results_dir / "phase7y_dry_run_manifest.json"
with open(manifest_file, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("Verifying new deliverables in results and reports:")
all_valid = True
for name, meta in manifest["new_deliverables"].items():
    rf = results_dir / name
    pf = reports_dir / name
    
    if not rf.is_file():
        print(f"MISSING in results: {name}")
        all_valid = False
        continue
    if not pf.is_file():
        print(f"MISSING in reports: {name}")
        all_valid = False
        continue
        
    r_bytes = rf.stat().st_size
    p_bytes = pf.stat().st_size
    r_hash = hashlib.sha256(rf.read_bytes()).hexdigest()
    p_hash = hashlib.sha256(pf.read_bytes()).hexdigest()
    
    match_manifest = (r_bytes == meta["bytes"] and r_hash == meta["sha256"])
    match_mirror = (r_bytes == p_bytes and r_hash == p_hash)
    
    status = "OK" if (match_manifest and match_mirror) else "MISMATCH"
    if status != "OK":
        all_valid = False
    print(f"  {name:40s} : bytes={r_bytes} sha256={r_hash[:12]}... status={status}")

print(f"\nTotal verified: {len(manifest['new_deliverables'])} | All valid: {all_valid}")

# Also verify phase7y_dry_run_manifest.json mirror
manifest_r_bytes = manifest_file.stat().st_size
manifest_p_bytes = (reports_dir / "phase7y_dry_run_manifest.json").stat().st_size
manifest_r_hash = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
manifest_p_hash = hashlib.sha256((reports_dir / "phase7y_dry_run_manifest.json").read_bytes()).hexdigest()
print(f"phase7y_dry_run_manifest.json mirror match: {manifest_r_bytes == manifest_p_bytes and manifest_r_hash == manifest_p_hash} (SHA256: {manifest_r_hash})")
