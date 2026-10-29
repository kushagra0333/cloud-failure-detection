"""Sliding window sequence generator constructing strictly chronological model inputs."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch

from src.ingestion.schema import FailureLabel


class WindowGenerator:
    """Constructs 3D sequence tensors X in R^{N x L x F} and label vectors y in {0, 1, 2}^N.
    
    Adheres strictly to Master Engineering Prompt Section 2.1 and 2.7:
    - Sequence length L = 15 observations (75 minutes at 5-minute sampling).
    - Window contains strictly trailing observations up to prediction time t.
    - Excludes UNKNOWN labels from training/evaluation sets.
    """

    def __init__(
        self,
        sequence_length: int = 15,
        expected_interval_sec: float = 300.0,
        allowed_jitter_sec: float = 30.0,
        require_full_history: bool = True,
        exclude_unknown: bool = True,
    ):
        self.sequence_length = sequence_length
        self.expected_interval_sec = expected_interval_sec
        self.allowed_jitter_sec = allowed_jitter_sec
        self.require_full_history = require_full_history
        self.exclude_unknown = exclude_unknown

    def is_valid_contiguous_history(self, timestamps: np.ndarray) -> bool:
        """Validates that timestamps within a candidate history window are contiguous without large gaps or sub-interval distortions."""
        if len(timestamps) < 2:
            return True
        deltas = np.diff(timestamps)
        # Check that consecutive observation step interval is approximately expected_interval_sec (nominal 300s)
        # Minimum allowed delta (240s) prevents sub-interval fragments (1s, 2s fragments) from being treated as 5-minute steps
        min_allowed_delta = max(1.0, self.expected_interval_sec - 60.0)
        max_allowed_delta = 1.5 * self.expected_interval_sec
        if not (np.all(deltas >= min_allowed_delta) and np.all(deltas <= max_allowed_delta)):
            return False

        # If full sequence is evaluated, verify total elapsed history span is approximately 75 minutes (nominally 4200s between step 0 and step 14)
        if len(timestamps) == self.sequence_length:
            total_duration = timestamps[-1] - timestamps[0]
            nominal_duration = (self.sequence_length - 1) * self.expected_interval_sec
            min_duration = nominal_duration * 0.90  # at least 3780s (63 min)
            max_duration = nominal_duration * 1.50  # at most 6300s (105 min)
            if not (min_duration <= total_duration <= max_duration):
                return False

        return True

    def create_windows_for_entity(
        self,
        entity_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Constructs valid sliding windows for a single entity."""
        df = entity_df.sort_values("end_time_sec").reset_index(drop=True)
        num_rows = len(df)
        windows = []

        if num_rows < (self.sequence_length if self.require_full_history else 1):
            return windows

        timestamps = df["end_time_sec"].values
        feature_mat = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values
        ttfs = df["time_to_failure_sec"].values
        entity_id = df["entity_id"].iloc[0]

        start_idx = self.sequence_length - 1 if self.require_full_history else 0

        for pred_idx in range(start_idx, num_rows):
            target_label = labels[pred_idx]
            if self.exclude_unknown and target_label == FailureLabel.UNKNOWN:
                continue

            window_start_idx = max(0, pred_idx - self.sequence_length + 1)
            history_features = feature_mat[window_start_idx : pred_idx + 1]
            history_times = timestamps[window_start_idx : pred_idx + 1]

            # Validate contiguity of the window
            if not self.is_valid_contiguous_history(history_times):
                continue

            # Left pad if require_full_history is False and len < sequence_length
            actual_len = len(history_features)
            if actual_len < self.sequence_length:
                pad_len = self.sequence_length - actual_len
                pad_features = np.zeros((pad_len, len(feature_cols)), dtype=np.float32)
                window_tensor = np.vstack([pad_features, history_features])
                mask = np.concatenate([np.zeros(pad_len, dtype=np.float32), np.ones(actual_len, dtype=np.float32)])
            else:
                window_tensor = history_features
                mask = np.ones(self.sequence_length, dtype=np.float32)

            pred_time = float(timestamps[pred_idx])
            ttf = float(ttfs[pred_idx]) if not np.isnan(ttfs[pred_idx]) else None

            windows.append({
                "features": window_tensor,  # Shape (L, F)
                "mask": mask,              # Shape (L,)
                "label": int(target_label),
                "entity_id": str(entity_id),
                "prediction_timestamp": pred_time,
                "time_to_failure": ttf,
            })

        return windows

    def build_dataset_tensors(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Processes an entire dataset across all entities into 3D tensors and metadata.
        
        Returns:
            X: Array of shape (N, L, F)
            y: Array of shape (N,)
            metadata_df: DataFrame with prediction metadata per example
        """
        all_windows = []
        for eid, group in df.groupby("entity_id", sort=False):
            ent_windows = self.create_windows_for_entity(group, feature_cols)
            all_windows.extend(ent_windows)

        if not all_windows:
            empty_X = np.empty((0, self.sequence_length, len(feature_cols)), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            empty_meta = pd.DataFrame(columns=["entity_id", "prediction_timestamp", "label", "time_to_failure"])
            return empty_X, empty_y, empty_meta

        X = np.stack([w["features"] for w in all_windows], axis=0).astype(np.float32)
        y = np.array([w["label"] for w in all_windows], dtype=np.int64)

        metadata_records = [{
            "entity_id": w["entity_id"],
            "prediction_timestamp": w["prediction_timestamp"],
            "label": w["label"],
            "time_to_failure": w["time_to_failure"],
        } for w in all_windows]

        metadata_df = pd.DataFrame(metadata_records)
        return X, y, metadata_df
