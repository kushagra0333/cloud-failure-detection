"""Data validation algorithms for cloud monitoring time series and lifecycle events."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.ingestion.schema import (
    TELEMETRY_REQUIRED_COLUMNS,
    LIFECYCLE_REQUIRED_COLUMNS,
    ClusterEventType,
)


class SchemaValidationError(ValueError):
    """Raised when incoming dataset does not adhere to mandatory schema requirements."""
    pass


class TimeSeriesValidator:
    """Validates schema, timestamps, gaps, duplicates, missingness, and outliers.
    
    Adheres strictly to the Master Engineering Prompt Section 2.2 validation algorithms.
    """

    def __init__(
        self,
        expected_interval_sec: float = 300.0,  # 5 minutes
        allowed_jitter_sec: float = 15.0,
        cpu_upper_bound: float = 1.0,  # Or 100.0 if percentage
        memory_upper_bound: float = 1.0,
    ):
        self.expected_interval_sec = expected_interval_sec
        self.allowed_jitter_sec = allowed_jitter_sec
        self.cpu_upper_bound = cpu_upper_bound
        self.memory_upper_bound = memory_upper_bound

    def validate_schema(
        self,
        df: pd.DataFrame,
        is_lifecycle: bool = False,
    ) -> None:
        """Section 2.2.A: Schema validation.
        
        Validates required columns and basic types. Fails with actionable errors.
        """
        required_cols = (
            LIFECYCLE_REQUIRED_COLUMNS if is_lifecycle else TELEMETRY_REQUIRED_COLUMNS
        )
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"Validation Failed: Missing required columns: {missing_cols}. "
                f"Present columns: {list(df.columns)}"
            )

        if not pd.api.types.is_numeric_dtype(df["timestamp"]):
            raise SchemaValidationError(
                f"Timestamp column must be numeric (seconds/epoch). Found dtype: {df['timestamp'].dtype}"
            )

        if not is_lifecycle:
            for col in ["cpu_utilization", "memory_utilization"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise SchemaValidationError(
                        f"Metric column '{col}' must be numeric. Found dtype: {df[col].dtype}"
                    )
        else:
            if not pd.api.types.is_integer_dtype(df["event_type"]):
                raise SchemaValidationError(
                    f"Lifecycle 'event_type' must be integer enumeration. Found dtype: {df['event_type'].dtype}"
                )

    def detect_and_resolve_duplicates(
        self,
        df: pd.DataFrame,
        resolution: str = "mean",
    ) -> Tuple[pd.DataFrame, int]:
        """Section 2.2.B: Duplicate detection and resolution.
        
        Checks duplicate (entity_id, timestamp) pairs and aggregates them deterministically.
        """
        initial_count = len(df)
        dup_mask = df.duplicated(subset=["entity_id", "timestamp"], keep=False)
        num_duplicates = int(dup_mask.sum())

        if num_duplicates == 0:
            return df.copy(), 0

        # Group and aggregate numerical fields, take first/last for categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_cols and c not in ["entity_id", "timestamp"]]

        agg_dict = {c: resolution for c in numeric_cols if c not in ["timestamp"]}
        for c in non_numeric:
            agg_dict[c] = "first"

        resolved_df = (
            df.groupby(["entity_id", "timestamp"], as_index=False)
            .agg(agg_dict)
            .reset_index(drop=True)
        )
        return resolved_df, num_duplicates

    def validate_timestamps_and_gaps(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Section 2.2.C: Timestamp validation.
        
        Sorts chronologically, measures delta_t = t_t - t_{t-1}, reports gaps and out-of-order records.
        """
        report = {
            "total_entities": 0,
            "out_of_order_entities": 0,
            "total_observations": len(df),
            "expected_interval_sec": self.expected_interval_sec,
            "gap_count": 0,
            "irregular_interval_count": 0,
            "delta_t_stats": {},
        }

        all_deltas = []
        entities = df["entity_id"].unique()
        report["total_entities"] = len(entities)

        for entity_id in entities:
            entity_data = df[df["entity_id"] == entity_id]
            timestamps = entity_data["timestamp"].values

            # Check if sorted
            if not np.all(np.diff(timestamps) >= 0):
                report["out_of_order_entities"] += 1

            if len(timestamps) > 1:
                deltas = np.diff(np.sort(timestamps))
                all_deltas.extend(deltas)

                # Flag irregular gaps (> 1.5 * expected interval)
                gaps = deltas[deltas > 1.5 * self.expected_interval_sec]
                report["gap_count"] += len(gaps)

                # Flag sub-interval jitter
                irregular = deltas[np.abs(deltas - self.expected_interval_sec) > self.allowed_jitter_sec]
                report["irregular_interval_count"] += len(irregular)

        if all_deltas:
            all_deltas_arr = np.array(all_deltas)
            report["delta_t_stats"] = {
                "min": float(np.min(all_deltas_arr)),
                "median": float(np.median(all_deltas_arr)),
                "mean": float(np.mean(all_deltas_arr)),
                "max": float(np.max(all_deltas_arr)),
                "std": float(np.std(all_deltas_arr)),
            }

        return report

    def analyze_missingness(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Section 2.2.D: Missing-value analysis.
        
        Calculates r_j = missing_j / total_observations for each feature.
        """
        if features is None:
            features = [c for c in df.columns if c not in ["entity_id", "timestamp"]]

        total_obs = len(df)
        if total_obs == 0:
            return {f: 0.0 for f in features}

        return {
            feat: float(df[feat].isna().sum() / total_obs)
            for feat in features
            if feat in df.columns
        }

    def analyze_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Section 2.2.E: Outlier analysis using Tukey's IQR and domain physical bounds.
        
        IQR = Q3 - Q1
        Lower = Q1 - 1.5 * IQR
        Upper = Q3 + 1.5 * IQR
        
        Flags observations without deleting them.
        """
        if columns is None:
            columns = [c for c in ["cpu_utilization", "memory_utilization"] if c in df.columns]

        report = {}
        for col in columns:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            iqr_outliers = vals[(vals < lower_bound) | (vals > upper_bound)]
            
            # Physical domain bounds check
            domain_viol_lower = vals[vals < 0.0]
            domain_viol_upper = vals[vals > self.cpu_upper_bound] if "cpu" in col else vals[vals > self.memory_upper_bound]

            report[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "tukey_lower": lower_bound,
                "tukey_upper": upper_bound,
                "iqr_outlier_count": len(iqr_outliers),
                "iqr_outlier_rate": float(len(iqr_outliers) / len(vals)),
                "physical_negative_count": len(domain_viol_lower),
                "physical_exceed_upper_count": len(domain_viol_upper),
            }

        return report
