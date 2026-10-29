import pandas as pd
import numpy as np

b = pd.read_csv("results/phase7b/test_predictions.csv")
d = pd.read_csv("results/phase7d/test_predictions.csv")

# Sort both by entity_id, prediction_timestamp
b = b.sort_values(["entity_id", "prediction_timestamp"]).reset_index(drop=True)
d = d.sort_values(["entity_id", "prediction_timestamp"]).reset_index(drop=True)

assert (b["entity_id"] == d["entity_id"]).all()
assert (b["prediction_timestamp"] == d["prediction_timestamp"]).all()
assert (b["y_true"] == d["y_true"]).all()

print("ALIGNED COMPARISON (7B vs 7D on identical windows):")
for m in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
    pred_b = b[f"y_pred_{m}"].values
    pred_d = d[f"y_pred_{m}"].values
    agree = (pred_b == pred_d).mean() * 100.0
    print(f"\nModel: {m}")
    print(f"  Prediction agreement: {agree:.2f}% ({(pred_b == pred_d).sum()}/{len(pred_b)})")
    
    prob_b = b[[f"prob_{m}_c0", f"prob_{m}_c1", f"prob_{m}_c2"]].values
    prob_d = d[[f"prob_{m}_c0", f"prob_{m}_c1", f"prob_{m}_c2"]].values
    max_p_diff = np.max(np.abs(prob_b - prob_d))
    mean_p_diff = np.mean(np.abs(prob_b - prob_d))
    print(f"  Max absolute prob diff: {max_p_diff:.6e}")
    print(f"  Mean absolute prob diff: {mean_p_diff:.6e}")
    
    b_counts = pd.Series(pred_b).value_counts().to_dict()
    d_counts = pd.Series(pred_d).value_counts().to_dict()
    print(f"  7B counts: {b_counts}")
    print(f"  7D counts: {d_counts}")
    
    # Are discrete predictions 100% identical?
    if (pred_b == pred_d).all():
        print(f"  --> DISCRETE PREDICTIONS ARE 100% IDENTICAL!")
    else:
        diff_idx = np.where(pred_b != pred_d)[0]
        print(f"  --> DISCRETE PREDICTIONS DIFFER on {len(diff_idx)} rows:")
        for idx in diff_idx[:5]:
            print(f"      eid={b.iloc[idx]['entity_id']}, time={b.iloc[idx]['prediction_timestamp']}, true={b.iloc[idx]['y_true']}, 7B={pred_b[idx]}, 7D={pred_d[idx]}")
