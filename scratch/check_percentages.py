import pandas as pd
import numpy as np

train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
val = pd.read_csv("data/processed/phase7c/val_metadata.csv")

print("Checking train subsets:")
# Any parent job filter?
# Any time filter?
# Let's check:
print("Total train windows:", len(train))
print("Train labels:\n", train["label"].value_counts())
print("\nValidation labels:\n", val["label"].value_counts())

# What is 2797 / 3278?
print("2797 / 3278 =", 2797 / 3278)
print("244 / 3278 =", 244 / 3278)
print("237 / 3278 =", 237 / 3278)

# What is 560 / 657?
print("560 / 657 =", 560 / 657)
print("48 / 657 =", 48 / 657)
print("49 / 657 =", 49 / 657)

# What is 3044 / 3278?
print("3044 / 3278 =", 3044 / 3278)
print("118 / 3278 =", 118 / 3278)
print("116 / 3278 =", 116 / 3278)

# Notice:
# 2797 / 3278 = 85.326% -> 85.3%
# 560 / 657 = 85.236% -> 85.2% (or 85.3% rounded)
# 3044 / 3278 = 92.86% -> 92.9%
