import glob
import numpy as np
import os
import pandas as pd

print("Checking NPY headers:")
for p in sorted(glob.glob('data/**/*.npy', recursive=True) + glob.glob('results/**/*.npy', recursive=True)):
    try:
        with open(p, 'rb') as f:
            major, minor = np.lib.format.read_magic(f)
            if major == 1:
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            if 824 in shape or shape[0] in [824, 1025, 657, 3278, 4960]:
                print(f"{p}: shape={shape}, dtype={dtype}")
    except Exception as e:
        pass

print("\nChecking CSV line counts:")
for p in sorted(glob.glob('data/**/*.csv', recursive=True) + glob.glob('results/**/*.csv', recursive=True)):
    try:
        # count lines without loading data
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            n_lines = sum(1 for _ in f)
        if n_lines in [824, 825, 1025, 1026, 657, 658, 3278, 3279, 4960, 4961]:
            print(f"{p}: lines={n_lines}")
    except Exception as e:
        pass
