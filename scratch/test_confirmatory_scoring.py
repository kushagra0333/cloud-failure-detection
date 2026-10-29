import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import ClassWeightedRandomForest
from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer
from src.models.hybrid import RepresentationHybridClassifier

# Load training data to initialize RF and Hybrid
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

# Load confirmatory data
X_conf = np.load("data/processed/confirmatory/X_confirmatory.npy")
y_conf = np.load("data/processed/confirmatory/y_confirmatory.npy")
meta_conf = pd.read_csv("data/processed/confirmatory/confirmatory_metadata.csv")

print(f"Confirmatory data shape: X={X_conf.shape}, y={y_conf.shape}, meta={len(meta_conf)}")
print("True label counts in confirmatory data:", pd.Series(y_conf).value_counts().to_dict())

# 1. Random Forest
rf = ClassWeightedRandomForest(strategy="final_step", random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_conf)
rf_probs = rf.predict_proba(X_conf)
print("\n[Random Forest]")
print("Pred counts:", pd.Series(rf_preds).value_counts().to_dict())
print("FPR (alerts on normal windows):", (rf_preds > 0).mean() * 100)

# 2. LSTM
device = torch.device("cpu")
lstm = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2)
ckpt_l = torch.load("models/checkpoints/best_lstm_checkpoint.pt", map_location=device)
lstm.load_state_dict(ckpt_l["model_state_dict"])
lstm.eval()

with torch.no_grad():
    t_conf = torch.tensor(X_conf, dtype=torch.float32)
    logits_l = lstm(t_conf)
    probs_l = torch.softmax(logits_l, dim=-1).numpy()
    preds_l = np.argmax(probs_l, axis=-1)

print("\n[Temporal LSTM]")
print("Pred counts:", pd.Series(preds_l).value_counts().to_dict())
print("FPR (alerts on normal windows):", (preds_l > 0).mean() * 100)

# 3. Compact Transformer
trans = CompactTemporalTransformer(input_dim=31, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3, dropout=0.2)
ckpt_t = torch.load("models/checkpoints/best_transformer_checkpoint.pt", map_location=device)
trans.load_state_dict(ckpt_t["model_state_dict"])
trans.eval()

with torch.no_grad():
    logits_t = trans(t_conf)
    probs_t = torch.softmax(logits_t, dim=-1).numpy()
    preds_t = np.argmax(probs_t, axis=-1)

print("\n[Compact Transformer]")
print("Pred counts:", pd.Series(preds_t).value_counts().to_dict())
print("FPR (alerts on normal windows):", (preds_t > 0).mean() * 100)

# 4. Representation Hybrid
hybrid = RepresentationHybridClassifier(temporal_backbone=trans, device=device, random_state=42)
hybrid.fit(X_train, y_train)
hybrid_preds = hybrid.predict(X_conf)
hybrid_probs = hybrid.predict_proba(X_conf)

print("\n[Representation Hybrid]")
print("Pred counts:", pd.Series(hybrid_preds).value_counts().to_dict())
print("FPR (alerts on normal windows):", (hybrid_preds > 0).mean() * 100)
