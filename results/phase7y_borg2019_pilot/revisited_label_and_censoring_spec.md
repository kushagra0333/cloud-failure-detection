# Revisited Causal Labeling and Right-Censoring Specification (Phase 7Y Continuation)

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Governing Rule**: Complete 25-Minute Follow-Up, Anti-Leakage Exclusion, and Right-Censoring Elimination  
**Audit Timestamp**: 2026-10-29T10:40:04.919673+00:00  

---

## 1. Mathematical Ground Truth Formulation

For any candidate observation window $W$ ending at $t_{\text{cutoff}} = W_{\text{end}}$:
Let $t_{\text{fail}}$ be the timestamp of the earliest subsequent `FAIL` event (`instance_events.type = 5`) for that specific task instance.
The forward lead time is defined as:
$$\Delta t = t_{\text{fail}} - t_{\text{cutoff}}$$

### Three-State Failure-Proximity Classes

- **Class 2 (CRITICAL)**: $0 < \Delta t \le 300\times 10^6 \ \mu\text{s}$ ($0 < \Delta t \le 5$ minutes). Imminent software crash requiring immediate mitigation.
- **Class 1 (WARNING)**: $600\times 10^6 < \Delta t \le 1,500\times 10^6 \ \mu\text{s}$ ($10 < \Delta t \le 25$ minutes). Early warning window allowing orderly checkpointing or preemptive migration.
- **Ambiguous Buffer (STRICTLY EXCLUDED)**: $300\times 10^6 < \Delta t \le 600\times 10^6 \ \mu\text{s}$ ($5 < \Delta t \le 10$ minutes). Omitted from all training and evaluation to prevent transition-boundary noise.
- **Class 0 (NORMAL)**: $\Delta t > 1,500\times 10^6 \ \mu\text{s}$ ($>25$ minutes) with confirmed 25-minute follow-up OR confirmed clean task termination via `FINISH` (`type = 6`).

---

## 2. Right-Censoring and Complete Follow-Up Protocol

1. **Complete 25-Minute Follow-Up Guarantee**:
   - Every retained Normal observation ($y=0$) MUST have verified execution telemetry or lifecycle event logs extending at least 25 minutes ($1,500\times 10^6 \ \mu\text{s}$) past $t_{\text{cutoff}}$.
   - If a trace terminates before 25 minutes elapse and no terminal event is logged, the window **CANNOT** be classified as Normal.

2. **Right-Censoring Truncation Guard**:
   - In Borg 2019 Day 1 pilot data ($T_{\text{max}} = 86,400\times 10^6 \ \mu\text{s}$), any observation window ending after:
     $$t_{\text{cutoff}} > T_{\text{max}} - 1,500\times 10^6 \ \mu\text{s} = 84,900\times 10^6 \ \mu\text{s} \quad (23.583\text{ hours})$$
     lacks the required 25-minute follow-up window.
   - **Protocol**: If such a window has not experienced a terminal event prior to $T_{\text{max}}$, it is flagged as `RIGHT_CENSORED` and dropped from supervised training and evaluation datasets.

3. **Post-Event Observation Prohibition**:
   - Any observation where $t_{\text{cutoff}} \ge t_{\text{fail}}$ is strictly excluded. Telemetry collected during or after failure represents post-mortem or retry artifact, not pre-failure predictive signal.

4. **Source Event Type Taxonomy (Google ClusterData 2019 v3)**:
   - `0: SUBMIT` — Submitted to cluster scheduler.
   - `1: QUEUE` — Marked ineligible by batch scheduler.
   - `2: ENABLE` — Transitioned to eligible for scheduling.
   - `3: SCHEDULE` — Task instance assigned to machine and started running.
   - `4: EVICT` — Preempted due to priority or resource overcommit (competing risk, excluded from failure).
   - `5: FAIL` — Abnormal software/execution termination (**Primary Target Ground Truth**).
   - `6: FINISH` — Normal successful task completion (**Normal Anchor Ground Truth**).
   - `7: KILL` — Cancelled by user or workflow manager (competing risk, excluded from failure).
   - `8: LOST` — Unreachable node/instance failure (**Secondary Sensitivity Target Ground Truth**).
   - `9: UPDATE_PENDING`, `10: UPDATE_RUNNING` — In-place resource adjustments (non-terminal).
