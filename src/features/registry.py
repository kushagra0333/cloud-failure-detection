"""Feature Registry defining mathematical formulas, source columns, units, and leakage policies."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class FeatureDefinition:
    """Formal definition of an engineered feature."""
    name: str
    group: str
    formula: str
    source_columns: List[str]
    units: str
    missing_policy: str
    availability_time: str
    leakage_risk: str
    description: str


class FeatureRegistry:
    """Central registry of all validated mathematical features.
    
    Adheres strictly to Master Engineering Prompt Section 2.3.
    """

    def __init__(self):
        self._registry: Dict[str, FeatureDefinition] = {}
        self._register_all_features()

    def _register(self, feat: FeatureDefinition):
        self._registry[feat.name] = feat

    def _register_all_features(self):
        # Group A — Raw Resource Utilization
        self._register(FeatureDefinition(
            name="cpu_utilization",
            group="Group A: Raw Utilization",
            formula="u_t^{cpu} = cpu_rate_t",
            source_columns=["cpu_rate"],
            units="Core Fraction [0.0 - 1.0]",
            missing_policy="Forward fill within entity (max 2 intervals), then training median",
            availability_time="<= t (observed in [t - Delta t, t])",
            leakage_risk="NONE",
            description="Mean CPU utilization over the trailing 5-minute measurement interval"
        ))
        self._register(FeatureDefinition(
            name="memory_utilization",
            group="Group A: Raw Utilization",
            formula="u_t^{mem} = canonical_memory_t",
            source_columns=["canonical_memory"],
            units="Normalized Memory Fraction [0.0 - 1.0]",
            missing_policy="Forward fill within entity (max 2 intervals), then training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Canonical page memory consumption over trailing 5-minute interval"
        ))
        self._register(FeatureDefinition(
            name="assigned_memory",
            group="Group A: Raw Utilization",
            formula="a_t^{mem} = assigned_memory_t",
            source_columns=["assigned_memory"],
            units="Normalized Memory Fraction [0.0 - 1.0]",
            missing_policy="Forward fill within entity, then training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Assigned memory capacity allocated by Borg scheduler"
        ))
        self._register(FeatureDefinition(
            name="disk_io_time",
            group="Group A: Raw Utilization",
            formula="d_t^{io} = disk_io_time_t",
            source_columns=["disk_io_time"],
            units="Normalized I/O Time Fraction [0.0 - 1.0]",
            missing_policy="Fill 0.0 if omitted by hardware counter or training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Cumulative disk I/O time spent during measurement window"
        ))
        self._register(FeatureDefinition(
            name="cycles_per_instruction",
            group="Group A: Raw Utilization",
            formula="cpi_t = cycles_per_instruction_t",
            source_columns=["cycles_per_instruction"],
            units="Cycles / Instruction (ratio)",
            missing_policy="Preserve missingness; flag via missingness indicator, then training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Hardware performance monitoring counter for CPI execution efficiency"
        ))
        self._register(FeatureDefinition(
            name="memory_accesses_per_instruction",
            group="Group A: Raw Utilization",
            formula="mai_t = memory_accesses_per_instruction_t",
            source_columns=["memory_accesses_per_instruction"],
            units="Accesses / Instruction (ratio)",
            missing_policy="Preserve missingness; flag via missingness indicator, then training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Hardware performance monitoring counter for memory access frequency"
        ))

        # Group B — Lag Features (k in {1, 2, 3, 6})
        for k in [1, 2, 3, 6]:
            lag_min = k * 5
            self._register(FeatureDefinition(
                name=f"cpu_utilization_lag_{k}",
                group="Group B: Lags",
                formula=f"lag_{k}(u_t^{{cpu}}) = u_{{t - {k}}}^{{cpu}}",
                source_columns=["cpu_rate"],
                units="Core Fraction [0.0 - 1.0]",
                missing_policy="Strictly trailing; NaN if entity history < k steps, imputed via training median",
                availability_time=f"<= t - {lag_min} min",
                leakage_risk="NONE",
                description=f"CPU utilization lagged by {k} steps ({lag_min} minutes)"
            ))
            self._register(FeatureDefinition(
                name=f"memory_utilization_lag_{k}",
                group="Group B: Lags",
                formula=f"lag_{k}(u_t^{{mem}}) = u_{{t - {k}}}^{{mem}}",
                source_columns=["canonical_memory"],
                units="Normalized Memory Fraction [0.0 - 1.0]",
                missing_policy="Strictly trailing; NaN if entity history < k steps, imputed via training median",
                availability_time=f"<= t - {lag_min} min",
                leakage_risk="NONE",
                description=f"Memory utilization lagged by {k} steps ({lag_min} minutes)"
            ))

        # Group C — First Differences and Relative Changes
        self._register(FeatureDefinition(
            name="cpu_diff_1",
            group="Group C: Differences",
            formula="Delta u_t^{cpu} = u_t^{cpu} - u_{t-1}^{cpu}",
            source_columns=["cpu_rate"],
            units="Delta Core Fraction / 5 min",
            missing_policy="NaN if history < 2 steps, imputed via training median (nominal 0.0)",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Absolute 1-step change in CPU utilization"
        ))
        self._register(FeatureDefinition(
            name="cpu_rel_change_1",
            group="Group C: Differences",
            formula="r_t^{cpu} = (u_t^{cpu} - u_{t-1}^{cpu}) / (|u_{t-1}^{cpu}| + epsilon), epsilon=1e-5",
            source_columns=["cpu_rate"],
            units="Dimensionless Relative Change",
            missing_policy="NaN if history < 2 steps, imputed via training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Relative 1-step rate of change in CPU utilization with zero-denominator protection"
        ))
        self._register(FeatureDefinition(
            name="memory_diff_1",
            group="Group C: Differences",
            formula="Delta u_t^{mem} = u_t^{mem} - u_{t-1}^{mem}",
            source_columns=["canonical_memory"],
            units="Delta Memory Fraction / 5 min",
            missing_policy="NaN if history < 2 steps, imputed via training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Absolute 1-step change in memory utilization"
        ))

        # Group D — Trailing Rolling Statistics (w in {3, 6})
        for w in [3, 6]:
            w_min = w * 5
            self._register(FeatureDefinition(
                name=f"cpu_rolling_mean_{w}",
                group="Group D: Rolling Statistics",
                formula=f"mu_t^{{cpu, {w}}} = (1/{w}) * sum_{{k=0}}^{{{w-1}}} u_{{t-k}}^{{cpu}}",
                source_columns=["cpu_rate"],
                units="Core Fraction",
                missing_policy=f"Requires min_periods=2 valid steps in trailing window, else training median",
                availability_time="<= t",
                leakage_risk="NONE",
                description=f"Trailing rolling mean CPU utilization over {w_min} minutes"
            ))
            self._register(FeatureDefinition(
                name=f"cpu_rolling_std_{w}",
                group="Group D: Rolling Statistics",
                formula=f"sigma_t^{{cpu, {w}}} = sqrt((1/{w}) * sum_{{k=0}}^{{{w-1}}} (u_{{t-k}}^{{cpu}} - mu_t^{{cpu, {w}}})^2)",
                source_columns=["cpu_rate"],
                units="Core Fraction (std dev)",
                missing_policy="Requires min_periods=2; single-point std is 0.0",
                availability_time="<= t",
                leakage_risk="NONE",
                description=f"Trailing rolling standard deviation of CPU over {w_min} minutes"
            ))
            self._register(FeatureDefinition(
                name=f"cpu_rolling_range_{w}",
                group="Group D: Rolling Statistics",
                formula=f"R_t^{{cpu, {w}}} = max_{{0<=k<{w}}} u_{{t-k}} - min_{{0<=k<{w}}} u_{{t-k}}",
                source_columns=["cpu_rate"],
                units="Core Fraction",
                missing_policy="Requires min_periods=2, else training median",
                availability_time="<= t",
                leakage_risk="NONE",
                description=f"Trailing rolling range (max - min) of CPU over {w_min} minutes"
            ))
            self._register(FeatureDefinition(
                name=f"memory_rolling_mean_{w}",
                group="Group D: Rolling Statistics",
                formula=f"mu_t^{{mem, {w}}} = (1/{w}) * sum_{{k=0}}^{{{w-1}}} u_{{t-k}}^{{mem}}",
                source_columns=["canonical_memory"],
                units="Memory Fraction",
                missing_policy="Requires min_periods=2, else training median",
                availability_time="<= t",
                leakage_risk="NONE",
                description=f"Trailing rolling mean memory utilization over {w_min} minutes"
            ))

        # Group E — Trend and Slope (Least squares slope over w=6)
        self._register(FeatureDefinition(
            name="cpu_trend_slope_6",
            group="Group E: Trend & Slope",
            formula="b_t^{cpu, 6} = sum_k (k - k_bar)(u_k - u_bar) / sum_k (k - k_bar)^2, k in [0, 5]",
            source_columns=["cpu_rate"],
            units="Core Fraction change per 5-minute step",
            missing_policy="Requires at least 3 valid observations in window, else training median",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Ordinary least-squares linear trend slope of CPU over last 30 minutes"
        ))

        # Group F — Threshold Exceedance
        self._register(FeatureDefinition(
            name="cpu_exceed_85",
            group="Group F: Threshold Exceedance",
            formula="e_t^{cpu} = 1(u_t^{cpu} > 0.85)",
            source_columns=["cpu_rate"],
            units="Binary Indicator {0, 1}",
            missing_policy="0 if missing",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Instantaneous threshold indicator flagging high CPU load (> 85% core)"
        ))
        self._register(FeatureDefinition(
            name="cpu_exceed_fraction_6",
            group="Group F: Threshold Exceedance",
            formula="q_t^{cpu, 6} = (1/6) * sum_{k=0}^5 1(u_{t-k}^{cpu} > 0.85)",
            source_columns=["cpu_rate"],
            units="Fraction [0.0 - 1.0]",
            missing_policy="Computed over available steps in window",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Fraction of the trailing 30 minutes spent exceeding 85% CPU core load"
        ))

        # Group G — Resource Utilization Relative to Allocation
        self._register(FeatureDefinition(
            name="memory_to_assigned_ratio",
            group="Group G: Allocation Ratios",
            formula="rho_t = canonical_memory_t / (assigned_memory_t + epsilon), epsilon=1e-5",
            source_columns=["canonical_memory", "assigned_memory"],
            units="Utilization / Allocation Ratio",
            missing_policy="Capped at 5.0 to handle anomalies; training median if unassigned",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Ratio of actual memory consumption to scheduler-assigned allocation"
        ))

        # Group H — Task Age and Lifecycle
        self._register(FeatureDefinition(
            name="task_age_minutes",
            group="Group H: Task Age",
            formula="age_{i,t} = (t - start_time_i) / 60.0",
            source_columns=["start_time"],
            units="Minutes",
            missing_policy="Non-negative; computed from first observed interval of entity",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Elapsed execution age of the task instance up to prediction timestamp t"
        ))

        # Group I — Missingness Tracking
        self._register(FeatureDefinition(
            name="cpi_missing_fraction_6",
            group="Group I: Observation Quality",
            formula="m_{t,cpi}^6 = (1/6) * sum_{k=0}^5 1(cpi_{t-k} is missing)",
            source_columns=["cycles_per_instruction"],
            units="Missingness Fraction [0.0 - 1.0]",
            missing_policy="Computed directly from missing status",
            availability_time="<= t",
            leakage_risk="NONE",
            description="Fraction of CPI performance counter measurements missing in trailing 30 min"
        ))

    def get_feature(self, name: str) -> FeatureDefinition:
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' not found in FeatureRegistry.")
        return self._registry[name]

    def list_features(self) -> List[FeatureDefinition]:
        return list(self._registry.values())

    def get_feature_names(self) -> List[str]:
        return list(self._registry.keys())

    def to_markdown_table(self) -> str:
        """Export registry as a GitHub-flavored Markdown table for documentation."""
        headers = [
            "Feature Name", "Group", "Mathematical Formula",
            "Source Columns", "Units", "Availability Time", "Leakage Risk"
        ]
        rows = [
            f"| `{f.name}` | {f.group} | ${f.formula}$ | `{', '.join(f.source_columns)}` | {f.units} | {f.availability_time} | {f.leakage_risk} |"
            for f in self._registry.values()
        ]
        table = (
            "| " + " | ".join(headers) + " |\n"
            + "| " + " | ".join(["---"] * len(headers)) + " |\n"
            + "\n".join(rows)
        )
        return table
