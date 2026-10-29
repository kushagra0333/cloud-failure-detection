# Feature Registry: Mathematical Formulations, Provenance, and Leakage Audit

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Document ID:** `reports/feature_registry.md`  
**Phase:** Phase 2 — Data Pipeline, Feature Engineering, and Leakage-Safe Splits  
**Implementation Source:** [`src/features/registry.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/registry.py)  
**Total Registered Features:** 31 engineered features across 9 functional groups  

---

## 1. Overview and Engineering Methodology

All features are computed in strict compliance with the project's causal availability policy:
$$\mathcal{F}_t = f(\{x_s : s \le t\})$$
where $t$ denotes the **prediction time**, defined as the exact ending timestamp of the latest fully observed 5-minute telemetry measurement window:
$$t = \text{end\_time}$$

No terminal-event information, future observations ($s > t$), or target-derived fields are permitted in $\mathcal{F}_t$. All imputers and standardizers are fitted strictly on the training partition ($D_{\text{train}}$) and applied downstream to validation and test partitions without parameter leakage.

---

## 2. Feature Registry Table

| Feature Name | Functional Group | Mathematical Formula | Source Columns | Units | Availability Time | Missingness Handling Policy | Leakage Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_utilization` | Group A: Raw Utilization | $u_t^{\text{cpu}} = \text{cpu\_rate}_t$ | `cpu_rate` | Core Fraction $[0.0 - 1.0]$ | $\le t$ | Forward fill (max 2 steps), then training median | NONE |
| `memory_utilization` | Group A: Raw Utilization | $u_t^{\text{mem}} = \text{canonical\_memory}_t$ | `canonical_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t$ | Forward fill (max 2 steps), then training median | NONE |
| `assigned_memory` | Group A: Raw Utilization | $a_t^{\text{mem}} = \text{assigned\_memory}_t$ | `assigned_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t$ | Forward fill within entity, then training median | NONE |
| `disk_io_time` | Group A: Raw Utilization | $d_t^{\text{io}} = \text{disk\_io\_time}_t$ | `disk_io_time` | Time Fraction $[0.0 - 1.0]$ | $\le t$ | Filled with 0.0 or training median | NONE |
| `cycles_per_instruction` | Group A: Raw Utilization | $\text{cpi}_t = \text{cpi}_t$ | `cycles_per_instruction` | Cycles / Instruction (Ratio) | $\le t$ | Preserved; flagged via indicator, imputed via train median | NONE |
| `memory_accesses_per_instruction` | Group A: Raw Utilization | $\text{mai}_t = \text{mai}_t$ | `memory_accesses_per_instruction` | Accesses / Instruction (Ratio) | $\le t$ | Preserved; flagged via indicator, imputed via train median | NONE |
| `cpu_utilization_lag_1` | Group B: Lags | $\text{lag}_1(u_t^{\text{cpu}}) = u_{t-1}^{\text{cpu}}$ | `cpu_rate` | Core Fraction $[0.0 - 1.0]$ | $\le t - 5\text{ min}$ | NaN if history $< 1$, imputed via training median | NONE |
| `memory_utilization_lag_1` | Group B: Lags | $\text{lag}_1(u_t^{\text{mem}}) = u_{t-1}^{\text{mem}}$ | `canonical_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t - 5\text{ min}$ | NaN if history $< 1$, imputed via training median | NONE |
| `cpu_utilization_lag_2` | Group B: Lags | $\text{lag}_2(u_t^{\text{cpu}}) = u_{t-2}^{\text{cpu}}$ | `cpu_rate` | Core Fraction $[0.0 - 1.0]$ | $\le t - 10\text{ min}$ | NaN if history $< 2$, imputed via training median | NONE |
| `memory_utilization_lag_2` | Group B: Lags | $\text{lag}_2(u_t^{\text{mem}}) = u_{t-2}^{\text{mem}}$ | `canonical_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t - 10\text{ min}$ | NaN if history $< 2$, imputed via training median | NONE |
| `cpu_utilization_lag_3` | Group B: Lags | $\text{lag}_3(u_t^{\text{cpu}}) = u_{t-3}^{\text{cpu}}$ | `cpu_rate` | Core Fraction $[0.0 - 1.0]$ | $\le t - 15\text{ min}$ | NaN if history $< 3$, imputed via training median | NONE |
| `memory_utilization_lag_3` | Group B: Lags | $\text{lag}_3(u_t^{\text{mem}}) = u_{t-3}^{\text{mem}}$ | `canonical_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t - 15\text{ min}$ | NaN if history $< 3$, imputed via training median | NONE |
| `cpu_utilization_lag_6` | Group B: Lags | $\text{lag}_6(u_t^{\text{cpu}}) = u_{t-6}^{\text{cpu}}$ | `cpu_rate` | Core Fraction $[0.0 - 1.0]$ | $\le t - 30\text{ min}$ | NaN if history $< 6$, imputed via training median | NONE |
| `memory_utilization_lag_6` | Group B: Lags | $\text{lag}_6(u_t^{\text{mem}}) = u_{t-6}^{\text{mem}}$ | `canonical_memory` | Memory Fraction $[0.0 - 1.0]$ | $\le t - 30\text{ min}$ | NaN if history $< 6$, imputed via training median | NONE |
| `cpu_diff_1` | Group C: Differences | $\Delta u_t^{\text{cpu}} = u_t^{\text{cpu}} - u_{t-1}^{\text{cpu}}$ | `cpu_rate` | $\Delta \text{Core Fraction} / 5\text{ min}$ | $\le t$ | NaN if history $< 2$, imputed via training median | NONE |
| `cpu_rel_change_1` | Group C: Differences | $r_t^{\text{cpu}} = \frac{u_t^{\text{cpu}} - u_{t-1}^{\text{cpu}}}{\vert u_{t-1}^{\text{cpu}}\vert + 10^{-5}}$ | `cpu_rate` | Dimensionless Ratio | $\le t$ | Zero-denominator protected ($10^{-5}$), imputed via median | NONE |
| `memory_diff_1` | Group C: Differences | $\Delta u_t^{\text{mem}} = u_t^{\text{mem}} - u_{t-1}^{\text{mem}}$ | `canonical_memory` | $\Delta \text{Memory Fraction} / 5\text{ min}$ | $\le t$ | NaN if history $< 2$, imputed via training median | NONE |
| `cpu_rolling_mean_3` | Group D: Rolling Stats | $\mu_t^{\text{cpu}, 3} = \frac{1}{3} \sum_{k=0}^2 u_{t-k}^{\text{cpu}}$ | `cpu_rate` | Core Fraction | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `cpu_rolling_std_3` | Group D: Rolling Stats | $\sigma_t^{\text{cpu}, 3} = \sqrt{\frac{1}{3}\sum (u_{t-k}^{\text{cpu}} - \mu)^2}$ | `cpu_rate` | Core Fraction (Std) | $\le t$ | Requires $\ge 2$ steps; single point is 0.0 | NONE |
| `cpu_rolling_range_3` | Group D: Rolling Stats | $R_t^{\text{cpu}, 3} = \max u_{t-k} - \min u_{t-k}$ | `cpu_rate` | Core Fraction (Range) | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `memory_rolling_mean_3` | Group D: Rolling Stats | $\mu_t^{\text{mem}, 3} = \frac{1}{3} \sum_{k=0}^2 u_{t-k}^{\text{mem}}$ | `canonical_memory` | Memory Fraction | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `cpu_rolling_mean_6` | Group D: Rolling Stats | $\mu_t^{\text{cpu}, 6} = \frac{1}{6} \sum_{k=0}^5 u_{t-k}^{\text{cpu}}$ | `cpu_rate` | Core Fraction | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `cpu_rolling_std_6` | Group D: Rolling Stats | $\sigma_t^{\text{cpu}, 6} = \sqrt{\frac{1}{6}\sum (u_{t-k}^{\text{cpu}} - \mu)^2}$ | `cpu_rate` | Core Fraction (Std) | $\le t$ | Requires $\ge 2$ steps, else 0.0 | NONE |
| `cpu_rolling_range_6` | Group D: Rolling Stats | $R_t^{\text{cpu}, 6} = \max u_{t-k} - \min u_{t-k}$ | `cpu_rate` | Core Fraction (Range) | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `memory_rolling_mean_6` | Group D: Rolling Stats | $\mu_t^{\text{mem}, 6} = \frac{1}{6} \sum_{k=0}^5 u_{t-k}^{\text{mem}}$ | `canonical_memory` | Memory Fraction | $\le t$ | Requires $\ge 2$ steps, else training median | NONE |
| `cpu_trend_slope_6` | Group E: Trend & Slope | $b_t^{\text{cpu}, 6} = \frac{\sum (k - \bar{k})(u_k - \bar{u})}{\sum (k - \bar{k})^2}$ | `cpu_rate` | Core Fraction / 5 min | $\le t$ | Requires $\ge 3$ steps, else training median | NONE |
| `cpu_exceed_85` | Group F: Threshold | $\mathbf{1}(u_t^{\text{cpu}} > 0.85)$ | `cpu_rate` | Binary Indicator $\{0, 1\}$ | $\le t$ | 0 if missing | NONE |
| `cpu_exceed_fraction_6` | Group F: Threshold | $\frac{1}{6}\sum_{k=0}^5 \mathbf{1}(u_{t-k}^{\text{cpu}} > 0.85)$ | `cpu_rate` | Fraction $[0.0 - 1.0]$ | $\le t$ | Computed over available steps in window | NONE |
| `memory_to_assigned_ratio` | Group G: Allocation Ratio | $\rho_t = \frac{u_t^{\text{mem}}}{a_t^{\text{mem}} + 10^{-5}}$ | `canonical_memory`, `assigned_memory` | Utilization / Allocation Ratio | $\le t$ | Clipped at 5.0 to guard extreme anomalies | NONE |
| `task_age_minutes` | Group H: Task Age | $\text{age}_{i,t} = \frac{t - t_{i,\text{start}}}{60.0}$ | `start_time` | Minutes | $\le t$ | Computed from entity first observed interval | NONE |
| `cpi_missing_fraction_6` | Group I: Observation Quality | $m_{t,\text{cpi}}^6 = \frac{1}{6}\sum_{k=0}^5 \mathbf{1}(\text{cpi}_{t-k} \text{ missing})$ | `cycles_per_instruction` | Fraction $[0.0 - 1.0]$ | $\le t$ | Directly indicates hardware counter availability | NONE |

---

## 3. Provenance and Scientific Justification

1. **Group A (Raw Telemetry):** Sourced from Google Borg measurement counters. Sourced columns represent mean consumption over 300-second windows.
2. **Group B (Trailing Lags):** Captures temporal dynamics across 5, 10, 15, and 30-minute horizons ($k \in \{1, 2, 3, 6\}$).
3. **Group C (Differences):** Measures first-order velocity of load increases, critical for detecting rapid memory leaks or CPU spin-loops prior to task crashes.
4. **Group D (Rolling Statistics):** Captures short-term (15 min) and medium-term (30 min) baseline drift and variance. Elevated standard deviation often signals thrashing before failure.
5. **Group E (Ordinary Least Squares Slope):** Linear trend coefficient calculated analytically over a 6-step window. Differentiates monotonic resource exhaustion from transient spikes.
6. **Group F (Threshold Exceedance):** Flags sustained high load ($>85\%$ CPU) which Google Borg traces link to throttling and scheduling contention.
7. **Group G (Allocation Ratios):** Normalizes actual usage by assigned quota. When $\rho_t \to 1.0$, tasks face out-of-memory or Borg throttling.
8. **Group H (Task Age):** Measures elapsed task execution duration. Early infant mortality vs. long-running task exhaustion exhibit distinct failure signatures.
9. **Group I (Counter Quality):** Tracks CPI missingness explicitly as an informative feature, preventing model bias from unobserved hardware counters without discarding valid CPU/memory records.
