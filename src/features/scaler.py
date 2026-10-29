"""Leakage-safe standardization and imputation fitted strictly on training data."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd


class LeakageSafeScaler:
    """Standardizes continuous features using training set statistics only.
    
    Adheres strictly to Master Engineering Prompt Section 2.4 and 2.5:
    mu_j = (1/n_train) * sum x_{r,j}
    s_j  = sqrt((1/n_train) * sum (x_{r,j} - mu_j)^2)
    z_j  = (x_j - mu_j) / (s_j + epsilon)
    
    Median imputation is fitted strictly on the training set:
    a_j = median(x_{train, j})
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.medians: Dict[str, float] = {}
        self.feature_cols: List[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "LeakageSafeScaler":
        """Fits means, standard deviations, and median imputers on training data only."""
        self.feature_cols = list(feature_cols)
        self.means = {}
        self.stds = {}
        self.medians = {}

        for col in self.feature_cols:
            if col not in train_df.columns:
                raise KeyError(f"Feature column '{col}' missing from training DataFrame.")

            series = train_df[col].dropna()
            if len(series) == 0:
                med = 0.0
                mean = 0.0
                std = 1.0
            else:
                med = float(series.median())
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                # Safe handling for zero variance / constant features
                if std < self.epsilon or np.isnan(std):
                    std = 1.0

            self.medians[col] = med
            self.means[col] = mean
            self.stds[col] = std

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies training medians and standardization without fitting or leaking statistics."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before calling transform.")

        out_df = df.copy()
        for col in self.feature_cols:
            if col not in out_df.columns:
                continue

            # 1. Median imputation using training median
            col_vals = out_df[col].fillna(self.medians[col]).values.astype(float)

            # 2. Standardization using training mean and std
            mean_val = self.means[col]
            std_val = self.stds[col]
            scaled = (col_vals - mean_val) / (std_val + self.epsilon)

            out_df[col] = scaled

        return out_df

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit on training data and return scaled training DataFrame."""
        return self.fit(train_df, feature_cols).transform(train_df)

    def save(self, file_path: Path) -> None:
        """Serializes fitted statistics to JSON."""
        state = {
            "feature_cols": self.feature_cols,
            "means": self.means,
            "stds": self.stds,
            "medians": self.medians,
            "epsilon": self.epsilon,
            "is_fitted": self._is_fitted,
        }
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "LeakageSafeScaler":
        """Loads fitted scaler from JSON artifact."""
        with open(file_path, "r") as f:
            state = json.load(f)
        scaler = cls(epsilon=state.get("epsilon", 1e-6))
        scaler.feature_cols = state["feature_cols"]
        scaler.means = state["means"]
        scaler.stds = state["stds"]
        scaler.medians = state["medians"]
        scaler._is_fitted = state["is_fitted"]
        return scaler
