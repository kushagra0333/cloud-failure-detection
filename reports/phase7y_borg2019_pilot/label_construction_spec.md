# Borg 2019 Causal Failure-Proximity Label Specification

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Governing Principle**: Strict Backward-Looking Temporal Alignment & Anti-Leakage  

---

## 1. Mathematical Target Definition

Let an observation window $W$ have observation end timestamp $t_{\text{cutoff}} = W_{\text{end}}$.  
Let $t_{\text{fail}}$ be the earliest subsequent timestamp of a primary `FAIL` event (`instance_events.type = 5`) for that specific instance.  
The forward lead time is:
$$\Delta t = t_{\text{fail}} - t_{\text{cutoff}}$$

The target class $y \in \{0, 1, 2\}$ is assigned strictly as:

$$\begin{cases}
y = 2 \quad (\text{CRITICAL}) & \text{if } 0 < \Delta t \le 300,000,000 \ \mu\text{s} \quad (0 < \Delta t \le 5\text{ minutes}) \y = 1 \quad (\text{WARNING}) & \text{if } 600,000,000 < \Delta t \le 1,500,000,000 \ \mu\text{s} \quad (10 < \Delta t \le 25\text{ minutes}) \y = 0 \quad (\text{NORMAL}) & \text{if } \Delta t > 1,500,000,000 \ \mu\text{s} \quad (\Delta t > 25\text{ minutes}) \text{ OR task ended with } \text{FINISH} \\text{EXCLUDED} & \text{if } 300,000,000 < \Delta t \le 600,000,000 \ \mu\text{s} \quad (5 < \Delta t \le 10\text{ minutes buffer}) \\text{EXCLUDED} & \text{if } \Delta t \le 0 \quad (t_{\text{cutoff}} \ge t_{\text{fail}} \text{ post-event observation})
\end{cases}$$

---

## 2. Right-Censoring and Follow-Up Truncation Rules

1. **Truncation Guard**:
   - If a 24-hour pilot slice ends at $T_{\text{max}} = 86,400,000,000 \ \mu\text{s}$, any prediction window whose cutoff $t_{\text{cutoff}} > T_{\text{max}} - 1,500,000,000 \ \mu\text{s}$ lacks the full 25-minute follow-up window.
   - **Rule**: If the event history is truncated and the task has not completed (`FINISH`), the window **must NOT be assumed to be NORMAL**. It is classified as **RIGHT_CENSORED** and excluded from supervised evaluation.
2. **Terminal Event Priority**:
   - If an instance experiences normal completion (`FINISH`, type = 6) before 25 minutes elapse, it is confirmed **NORMAL (0)**.
   - If an instance is cancelled (`KILL`, type = 7) or preempted (`EVICT`, type = 4), it is excluded as an unobservable competing risk.
3. **Sensitivity Analysis for `LOST`**:
   - `LOST` (type = 8) events are tracked in a secondary sensitivity target column (`y_lost`), never merged into primary software crash labels (`FAIL`).
