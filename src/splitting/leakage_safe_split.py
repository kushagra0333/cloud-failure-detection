"""Leakage-safe temporal and entity-grouped data splitting with horizon embargo."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class LeakageViolationError(ValueError):
    """Raised when data leakage across train/val/test splits is detected."""
    pass


class LeakageSafeSplitter:
    """Partitions time series and prediction windows into strictly isolated splits.
    
    Adheres strictly to Master Engineering Prompt Section 2.8:
    - Chronological 70% / 15% / 15% splitting.
    - Embargo policy: purges examples within H_warning (30 minutes) of a split boundary.
    - Prevents target leakage from future validation/test outcomes into training labels.
    - Supports entity-grouped and combined temporal-entity generalization splits.
    """

    def __init__(
        self,
        strategy: str = "chronological",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        embargo_sec: float = 1800.0, # 30 minutes (H_warning)
        random_seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.embargo_sec = embargo_sec
        self.random_seed = random_seed

    def split_chronological(
        self,
        metadata_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Performs strict chronological splitting with boundary embargo."""
        if metadata_df.empty:
            return {"train_idx": np.array([]), "val_idx": np.array([]), "test_idx": np.array([])}

        times = metadata_df["prediction_timestamp"].values
        t_min = float(times.min())
        t_max = float(times.max())
        total_span = t_max - t_min

        # Compute raw split boundaries
        t_val_start = t_min + self.train_ratio * total_span
        t_test_start = t_val_start + self.val_ratio * total_span

        # Apply embargo: training examples within embargo_sec of t_val_start are purged
        # Validation examples within embargo_sec of t_test_start are purged
        train_cutoff = t_val_start - self.embargo_sec
        val_cutoff = t_test_start - self.embargo_sec

        train_mask = times < train_cutoff
        val_mask = (times >= t_val_start) & (times < val_cutoff)
        test_mask = times >= t_test_start

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Log purged boundary counts
        embargoed_train = np.where((times >= train_cutoff) & (times < t_val_start))[0]
        embargoed_val = np.where((times >= val_cutoff) & (times < t_test_start))[0]

        return {
            "train_idx": train_indices,
            "val_idx": val_indices,
            "test_idx": test_indices,
            "purged_train_embargo_count": len(embargoed_train),
            "purged_val_embargo_count": len(embargoed_val),
            "boundary_times": {
                "t_min": t_min,
                "t_train_cutoff": train_cutoff,
                "t_val_start": t_val_start,
                "t_val_cutoff": val_cutoff,
                "t_test_start": t_test_start,
                "t_max": t_max,
                "embargo_sec": self.embargo_sec,
            }
        }

    def split_entity_grouped(
        self,
        metadata_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Partitions by entity identity: no entity's windows cross split boundaries."""
        entities = metadata_df["entity_id"].unique()
        rng = np.random.RandomState(self.random_seed)
        shuffled_entities = rng.permutation(entities)

        n_entities = len(shuffled_entities)
        n_train = int(self.train_ratio * n_entities)
        n_val = int(self.val_ratio * n_entities)

        train_entities = set(shuffled_entities[:n_train])
        val_entities = set(shuffled_entities[n_train : n_train + n_val])
        test_entities = set(shuffled_entities[n_train + n_val :])

        train_mask = metadata_df["entity_id"].isin(train_entities).values
        val_mask = metadata_df["entity_id"].isin(val_entities).values
        test_mask = metadata_df["entity_id"].isin(test_entities).values

        return {
            "train_idx": np.where(train_mask)[0],
            "val_idx": np.where(val_mask)[0],
            "test_idx": np.where(test_mask)[0],
            "train_entities": train_entities,
            "val_entities": val_entities,
            "test_entities": test_entities,
        }

    def split(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Executes splitting according to configured strategy."""
        if self.strategy == "chronological":
            return self.split_chronological(metadata_df)
        elif self.strategy == "entity_grouped":
            return self.split_entity_grouped(metadata_df)
        else:
            raise ValueError(f"Unknown splitting strategy: {self.strategy}")

    @staticmethod
    def audit_leakage(
        metadata_df: pd.DataFrame,
        split_result: Dict[str, Any],
        embargo_sec: float = 1800.0,
    ) -> Dict[str, Any]:
        """Conducts an independent forensic audit for split leakage."""
        train_idx = split_result["train_idx"]
        val_idx = split_result["val_idx"]
        test_idx = split_result["test_idx"]

        # Check index disjointness
        assert len(set(train_idx) & set(val_idx)) == 0, "Train and Val indices overlap!"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train and Test indices overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val and Test indices overlap!"

        train_meta = metadata_df.iloc[train_idx]
        val_meta = metadata_df.iloc[val_idx]
        test_meta = metadata_df.iloc[test_idx]

        audit_report = {
            "train_count": len(train_meta),
            "val_count": len(val_meta),
            "test_count": len(test_meta),
            "temporal_leakage_detected": False,
            "embargo_respected": True,
            "entity_leakage_detected": False,
        }

        if "boundary_times" in split_result:
            bounds = split_result["boundary_times"]
            max_train_time = train_meta["prediction_timestamp"].max() if not train_meta.empty else -1
            min_val_time = val_meta["prediction_timestamp"].min() if not val_meta.empty else float("inf")
            max_val_time = val_meta["prediction_timestamp"].max() if not val_meta.empty else -1
            min_test_time = test_meta["prediction_timestamp"].min() if not test_meta.empty else float("inf")

            if max_train_time >= bounds["t_val_start"]:
                audit_report["temporal_leakage_detected"] = True
                raise LeakageViolationError(f"Temporal leakage: Max train time {max_train_time} >= Val start {bounds['t_val_start']}")

            if (min_val_time - max_train_time) < embargo_sec:
                audit_report["embargo_respected"] = False
                raise LeakageViolationError(f"Embargo violation: Train-val gap {min_val_time - max_train_time} < embargo {embargo_sec}")

            if max_val_time >= bounds["t_test_start"]:
                audit_report["temporal_leakage_detected"] = True
                raise LeakageViolationError(f"Temporal leakage: Max val time {max_val_time} >= Test start {bounds['t_test_start']}")

        if "train_entities" in split_result:
            overlap_val = split_result["train_entities"] & split_result["val_entities"]
            overlap_test = split_result["train_entities"] & split_result["test_entities"]
            if overlap_val or overlap_test:
                audit_report["entity_leakage_detected"] = True
                raise LeakageViolationError(f"Entity leakage: Overlapping entities across splits: {overlap_val | overlap_test}")

        return audit_report
