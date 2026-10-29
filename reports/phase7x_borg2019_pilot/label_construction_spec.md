# Borg 2019 Failure Prediction Target Construction Specification

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Governing Standard**: Causal Failure-Proximity Windowing Protocol  
**Source Dataset**: Google Borg ClusterData 2019  

---

## 1. Terminal Event Taxonomy & Target Scope

In Borg 2019, instances terminate under several distinct conditions recorded in `instance_events.type`:
- **Primary Failure Event (`FAIL`, Type = 5)**:
  - Definition: The process or container crashed, raised an unhandled exception, exited with a non-zero code, or was killed by the OS kernel (e.g. out-of-memory OOM kill).
  - Role: **Primary Ground-Truth Failure Event ($t_{\text{fail}}$)**.
- **Sensitivity Failure Event (`LOST`, Type = 8)**:
  - Definition: Machine heartbeat failed or node unreachable; task presumed dead.
  - Role: **Sensitivity Target Only** (evaluated in secondary robustness checks, not conflated with confirmed application crashes).
- **Non-Failure Terminal Events**:
  - `FINISH` (Type = 6): Normal, successful job completion.
  - `KILL` (Type = 7): User-initiated cancellation or administrative cleanup.
  - `EVICT` (Type = 4): Scheduler-driven preemption to accommodate higher-priority work.
  - Restriction: Under no circumstances are `FINISH`, `KILL`, or `EVICT` labeled as failures.

---

## 2. Multi-Class Temporal Proximity Target Formulation

Let $W$ be an observation window with observation end timestamp $t_{\text{window\_end}}$. Let $t_{\text{fail}}$ be the earliest subsequent timestamp of a primary `FAIL` event for that instance. The time-to-failure lead time is:
$$\Delta t = t_{\text{fail}} - t_{\text{window\_end}}$$

The target class $y \in \{0, 1, 2\}$ is assigned strictly as follows:

| Target Class | Class Name | Mathematical Horizon Condition ($\mu	ext{s}$) | Human Time Condition | Semantic Operational Meaning |
| :---: | :---: | :---: | :---: | :--- |
| **Class 2** | **CRITICAL** | $0 < \Delta t \le 300,000,000$ | $0 < \Delta t \le 5\text{ min}$ | Imminent failure; immediate intervention or state checkpoint required. |
| **Class 1** | **WARNING** | $600,000,000 < \Delta t \le 1,500,000,000$ | $10 < \Delta t \le 25\text{ min}$ | Early degradation; proactive mitigation, load shedding, or live migration viable. |
| **Class 0** | **NORMAL** | $\Delta t > 1,500,000,000$ OR Pure-Normal | $\Delta t > 25\text{ min}$ OR No Failure | Stable execution; no intervention needed. |
| **EXCLUDED** | **Ambiguous Buffer** | $300,000,000 < \Delta t \le 600,000,000$ | $5 < \Delta t \le 10\text{ min}$ | Transition buffer interval; **strictly excluded** from training and evaluation. |
| **EXCLUDED** | **Post-Event Leakage** | $\Delta t \le 0$ ($t_{\text{window\_end}} \ge t_{\text{fail}}$) | Window at or after failure | Post-failure telemetry; **strictly censored** and removed from dataset. |

---

## 3. Strict Anti-Leakage Safeguards
1. **Window Anchoring**: The observation window is strictly anchored to telemetry received prior to $t_{\text{window\_end}}$.
2. **Buffer Exclusion**: Observations falling in the 5-to-10 minute buffer window are explicitly dropped to prevent label boundary confusion between Warning and Critical.
3. **No Event-Flag Copying**: Row-level event flags from future timestamps are never propagated backwards as static features.
