"""Feature engineering engine for time-series cloud workload telemetry."""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.features.registry import FeatureRegistry


class FeatureEngineer:
    """Computes trailing, leakage-free mathematical features on cloud monitoring time series.
    
    Adheres strictly to Master Engineering Prompt Section 2.3.
    Features are computed strictly per entity using trailing windows.
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None, epsilon: float = 1e-5):
        self.registry = registry or FeatureRegistry()
        self.epsilon = epsilon

    @staticmethod
    def _calculate_ols_slope(series: pd.Series, window: int = 6) -> pd.Series:
        """Calculates least-squares trend slope over trailing w observations."""
        def _slope(window_vals):
            n = len(window_vals)
            if n < 3:
                return 0.0
            valid_mask = ~np.isnan(window_vals)
            if valid_mask.sum() < 3:
                return 0.0
            k = np.arange(n)
            x = k[valid_mask]
            y = window_vals[valid_mask]
            x_bar = float(np.mean(x))
            y_bar = float(np.mean(y))
            d = float(np.sum((x - x_bar) ** 2))
            return float(np.sum((x - x_bar) * (y - y_bar)) / (d + 1e-9)) if d > 0 else 0.0

        return series.rolling(window=window, min_periods=3).apply(_slope, raw=True).fillna(0.0)

    def transform(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms entire telemetry dataframe grouped strictly by entity_id using vectorized ops."""
        if "entity_id" not in telemetry_df.columns:
            raise KeyError("telemetry_df must contain 'entity_id'.")

        # Sort strictly by entity and time
        df = telemetry_df.sort_values(["entity_id", "end_time_sec"]).reset_index(drop=True)
        grp = df.groupby("entity_id", sort=False)

        # Group A — Raw Resource Utilization
        df["cpu_utilization"] = df["cpu_rate"].astype(float)
        df["memory_utilization"] = df["canonical_memory"].astype(float)
        df["assigned_memory"] = df["assigned_memory"].astype(float)
        if "disk_io_time" in df.columns:
            df["disk_io_time"] = df["disk_io_time"].fillna(0.0).astype(float)
        else:
            df["disk_io_time"] = 0.0

        if "cycles_per_instruction" in df.columns:
            df["cycles_per_instruction"] = df["cycles_per_instruction"].astype(float)
        else:
            df["cycles_per_instruction"] = np.nan

        if "memory_accesses_per_instruction" in df.columns:
            df["memory_accesses_per_instruction"] = df["memory_accesses_per_instruction"].astype(float)
        else:
            df["memory_accesses_per_instruction"] = np.nan

        # Group B — Lag Features
        for k in [1, 2, 3, 6]:
            df[f"cpu_utilization_lag_{k}"] = grp["cpu_utilization"].shift(k)
            df[f"memory_utilization_lag_{k}"] = grp["memory_utilization"].shift(k)

        # Group C — First Differences & Relative Changes
        cpu_shift1 = grp["cpu_utilization"].shift(1)
        df["cpu_diff_1"] = df["cpu_utilization"] - cpu_shift1
        df["cpu_rel_change_1"] = (df["cpu_utilization"] - cpu_shift1) / (cpu_shift1.abs() + self.epsilon)

        mem_shift1 = grp["memory_utilization"].shift(1)
        df["memory_diff_1"] = df["memory_utilization"] - mem_shift1

        # Group D — Trailing Rolling Statistics (w in {3, 6})
        for w in [3, 6]:
            df[f"cpu_rolling_mean_{w}"] = grp["cpu_utilization"].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            df[f"cpu_rolling_std_{w}"] = grp["cpu_utilization"].transform(
                lambda s: s.rolling(w, min_periods=1).std(ddof=0)
            ).fillna(0.0)
            df[f"cpu_rolling_range_{w}"] = (
                grp["cpu_utilization"].transform(lambda s: s.rolling(w, min_periods=1).max()) -
                grp["cpu_utilization"].transform(lambda s: s.rolling(w, min_periods=1).min())
            ).fillna(0.0)

            df[f"memory_rolling_mean_{w}"] = grp["memory_utilization"].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )

        # Group E — Trend and Slope (w=6)
        df["cpu_trend_slope_6"] = grp["cpu_utilization"].transform(
            lambda s: self._calculate_ols_slope(s, window=6)
        )

        # Group F — Threshold Exceedance
        df["cpu_exceed_85"] = (df["cpu_utilization"] > 0.85).astype(float)
        df["cpu_exceed_fraction_6"] = grp["cpu_exceed_85"].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )

        # Group G — Resource Utilization Relative to Allocation
        df["memory_to_assigned_ratio"] = (
            df["memory_utilization"] / (df["assigned_memory"] + self.epsilon)
        ).clip(0.0, 5.0)

        # Group H — Task Age and Lifecycle
        task_starts = grp["start_time_sec"].transform("min")
        df["task_age_minutes"] = (df["end_time_sec"] - task_starts) / 60.0

        # Group I — Missingness Tracking
        cpi_missing = df["cycles_per_instruction"].isna().astype(float)
        df["cpi_missing_fraction_6"] = cpi_missing.groupby(df["entity_id"]).transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )

        return df
