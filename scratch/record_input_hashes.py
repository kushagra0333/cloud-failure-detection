import hashlib
import json
from pathlib import Path

root = Path(".")
files = [
    root / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv",
    root / "results" / "phase7b_1" / "phase7b_evaluation_protocol_v2.json",
    root / "configs" / "phase7c" / "fitted_scaler_phase7c.json",
    root / "configs" / "phase7c" / "feature_registry_phase7c.json",
] + sorted((root / "data" / "processed" / "phase7c").glob("*"))

hashes = {}
for p in files:
    if p.is_file():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[str(p.as_posix())] = h
        print(f"{p.as_posix()}: {h}")

with open("results/phase7d_preflight/input_hashes.json", "w") as f:
    json.dump(hashes, f, indent=2)
