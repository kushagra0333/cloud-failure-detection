"""Generates aligned comparative error analysis artifacts between Random Forest and LSTM."""

from pathlib import Path
import pandas as pd
import numpy as np

rf_df = pd.read_csv("results/phase3a/test_predictions_random_forest.csv")
lstm_df = pd.read_csv("results/phase3b/test_predictions_lstm.csv")

assert len(rf_df) == len(lstm_df) == 1003
assert (rf_df["entity_id"] == lstm_df["entity_id"]).all()
assert (rf_df["prediction_timestamp"] == lstm_df["prediction_timestamp"]).all()

out_df = rf_df[["entity_id", "prediction_timestamp", "label", "time_to_failure"]].copy()
out_df["rf_predicted_label"] = rf_df["predicted_label"]
out_df["rf_prob_normal"] = rf_df["prob_normal"]
out_df["rf_prob_warning"] = rf_df["prob_warning"]
out_df["rf_prob_critical"] = rf_df["prob_critical"]

out_df["lstm_predicted_label"] = lstm_df["predicted_label"]
out_df["lstm_prob_normal"] = lstm_df["prob_normal"]
out_df["lstm_prob_warning"] = lstm_df["prob_warning"]
out_df["lstm_prob_critical"] = lstm_df["prob_critical"]

out_df["rf_correct"] = out_df["rf_predicted_label"] == out_df["label"]
out_df["lstm_correct"] = out_df["lstm_predicted_label"] == out_df["label"]
out_df["agreement"] = out_df["rf_predicted_label"] == out_df["lstm_predicted_label"]

out_dir = Path("results/phase3c")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "comparative_error_analysis.csv"
out_df.to_csv(out_csv, index=False)
print(f"Saved comparative error analysis table to {out_csv} ({len(out_df):,} rows)")
