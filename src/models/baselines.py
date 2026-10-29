"""Baseline model wrappers and sequence-to-vector adapters for early failure prediction.

Implements Phase 3A Baseline Models:
- Majority-Class Classifier
- Class-Weighted Logistic Regression
- Class-Weighted Random Forest

Provides explicit, defensible conversion strategies from (N, L, F) sequences to fixed-length vectors.
"""

from typing import Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def extract_feature_vector(
    X_seq: np.ndarray,
    strategy: Literal["final_step", "summary", "flattened"] = "final_step",
) -> np.ndarray:
    """Converts 3D sequence tensors (N, L, F) into 2D feature matrices (N, D).
    
    Strategies:
    - 'final_step': Selects the final observed timestep X[:, -1, :] (shape N x F).
      Defensible rationale: Trailing causal features (lags, rolling stats, OLS slope,
      exceedance fractions) are already computed in the feature pipeline and summarized
      at timestep L=15. Preserves parsimony (31 features) and avoids collinearity.
    - 'summary': Concatenates final step, temporal mean, and temporal std across L (shape N x 3F).
    - 'flattened': Flattens all L timesteps into a single vector (shape N x (L*F)).
      Caveat: Produces 465 features, creating collinearity and over-parameterization on small classes.
    """
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array (N, L, F), got shape {X_seq.shape}")

    N, L, F = X_seq.shape

    if strategy == "final_step":
        return X_seq[:, -1, :].copy()
    elif strategy == "summary":
        final_step = X_seq[:, -1, :]
        mean_step = np.mean(X_seq, axis=1)
        std_step = np.std(X_seq, axis=1)
        return np.hstack([final_step, mean_step, std_step])
    elif strategy == "flattened":
        return X_seq.reshape(N, L * F).copy()
    else:
        raise ValueError(f"Unknown sequence extraction strategy: {strategy}")


class MajorityClassBaseline:
    """Majority-class naive baseline that predicts the dominant training class (NORMAL = 0)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
        self.is_fitted = False
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassBaseline":
        X_vec = extract_feature_vector(X, strategy="final_step") if X.ndim == 3 else X
        self.clf.fit(X_vec, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_vec = extract_feature_vector(X, strategy="final_step") if X.ndim == 3 else X
        return self.clf.predict(X_vec)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 3-class probability distribution."""
        X_vec = extract_feature_vector(X, strategy="final_step") if X.ndim == 3 else X
        N = len(X_vec)
        # DummyClassifier with most_frequent outputs 1.0 for dominant class (0)
        raw_probs = self.clf.predict_proba(X_vec)
        if raw_probs.shape[1] == 1:
            # Expand to (N, 3)
            probs = np.zeros((N, 3), dtype=np.float64)
            dominant_class = self.clf.classes_[0]
            probs[:, dominant_class] = 1.0
            return probs
        return raw_probs


class ClassWeightedLogisticRegression:
    """Multinomial logistic regression with inverse class weights and L2 regularization."""

    def __init__(
        self,
        strategy: Literal["final_step", "summary", "flattened"] = "final_step",
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassWeightedLogisticRegression":
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        return self.clf.predict(X_vec)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        return self.clf.predict_proba(X_vec)


class ClassWeightedRandomForest:
    """Random Forest classifier with class-balanced subsampling and restrained depth."""

    def __init__(
        self,
        strategy: Literal["final_step", "summary", "flattened"] = "final_step",
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.strategy = strategy
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted = False
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassWeightedRandomForest":
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        return self.clf.predict(X_vec)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_vec = extract_feature_vector(X, strategy=self.strategy) if X.ndim == 3 else X
        return self.clf.predict_proba(X_vec)
