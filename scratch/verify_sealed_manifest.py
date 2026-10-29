import json
import hashlib
from pathlib import Path

p = Path("results/phase7r_score_forensics/phase7r_manifest.json")
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)

for fname, meta in m["deliverables"].items():
    fpath = Path("results/phase7r_score_forensics") / fname
    assert fpath.exists(), f"Missing {fname}"
    sz = fpath.stat().st_size
    expected_sz = meta["bytes"]
    assert sz == expected_sz, f"Size mismatch for {fname}: {sz} vs {expected_sz}"
    with open(fpath, "rb") as fp:
        h = hashlib.sha256(fp.read()).hexdigest()
    expected_h = meta["sha256"]
    assert h == expected_h, f"Hash mismatch for {fname}: {h} vs {expected_h}"

print(f"SUCCESS: ALL {len(m['deliverables'])} DELIVERABLES IN MANIFEST VERIFIED 100% MATCH!")
