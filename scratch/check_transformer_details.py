import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

p7b_pred = pd.read_csv("results/phase7b/test_predictions.csv")
p7d_pred = pd.read_csv("results/phase7d/test_predictions.csv")

print("=== TRANSFORMER CLASS 2 PREDICTION INDICES ===")
idx_7b_c2 = np.where(p7b_pred["y_pred_compact_transformer"] == 2)[0]
idx_7d_c2 = np.where(p7d_pred["y_pred_compact_transformer"] == 2)[0]
print("7B Transformer predicted class 2 on indices:", idx_7b_c2)
print("7D Transformer predicted class 2 on indices:", idx_7d_c2)

print("\n7B Transformer rows with pred == 2:")
print(p7b_pred.iloc[idx_7b_c2][["entity_id", "prediction_timestamp", "y_true", "y_pred_compact_transformer", "prob_compact_transformer_c0", "prob_compact_transformer_c1", "prob_compact_transformer_c2"]])

print("\n7D Transformer rows with pred == 2:")
print(p7d_pred.iloc[idx_7d_c2][["entity_id", "prediction_timestamp", "y_true", "y_pred_compact_transformer", "prob_compact_transformer_c0", "prob_compact_transformer_c1", "prob_compact_transformer_c2"]])

# Check where the 6 disagreement indices are for Transformer
diff_idx = np.where(p7b_pred["y_pred_compact_transformer"] != p7d_pred["y_pred_compact_transformer"])[0]
print("\nIndices where 7B and 7D Transformer differ in discrete predictions:", diff_idx)
if len(diff_idx) > 0:
    for i in diff_idx:
        print(f"Index {i}: 7B pred={p7b_pred['y_pred_compact_transformer'].iloc[i]}, 7D pred={p7d_pred['y_pred_compact_transformer'].iloc[i]}, true={p7b_pred['y_true'].iloc[i]}")
        print(f"   7B probs: c0={p7b_pred['prob_compact_transformer_c0'].iloc[i]:.4f}, c1={p7b_pred['prob_compact_transformer_c1'].iloc[i]:.4f}, c2={p7b_pred['prob_compact_transformer_c2'].iloc[i]:.4f}")
        print(f"   7D probs: c0={p7d_pred['prob_compact_transformer_c0'].iloc[i]:.4f}, c1={p7d_pred['prob_compact_transformer_c1'].iloc[i]:.4f}, c2={p7d_pred['prob_compact_transformer_c2'].iloc[i]:.4f}")
