import time
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(".").resolve()
import sys
sys.path.insert(0, str(root))

from scripts.reconstruct_phase7c_raw_features import (
    reconstruct_raw_features,
    compute_sha256
)

print("Testing reconstruction of L=12 and L=5 raw features...")
# We can check how reconstruct_phase7c_raw_features is implemented.
# Let's inspect reconstruct_phase7c_raw_features main block
