# Label Semantics and Temporal Proximity Audit

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Governing Protocol**: Phase 7B Evaluation Protocol v2.1  
**Target Definition**:
- **Class 0 (NORMAL)**: Entity window is greater than 30 minutes away from any failure event, or belongs to a pure-normal execution.
- **Class 1 (WARNING)**: Entity window is in the pre-failure proximity interval: **$10 < (t_{\text{fail}} - t_{\text{window}}) \le 25$ minutes**.
- **Class 2 (CRITICAL)**: Entity window is in the imminent pre-failure proximity interval: **$0 < (t_{\text{fail}} - t_{\text{window}}) \le 5$ minutes**.

---

## 1. The Fundamental Difference: Proximity Windows vs Event Indicators

A critical methodological pitfall in failure prediction research is conflating:
1. **A Point-in-Time Failure Event Indicator**: A discrete flag indicating that at timestamp $T_{\text{event}}$, a task crashed, failed, or was evicted.
2. **A Temporal Failure Proximity Label**: A multi-step causal labeling rule that marks earlier telemetry windows as Warning or Critical based on their backward lead time to a future failure.

None of the external datasets (Google Borg 2019, Alibaba 2018, ClusterWise, LANL CFDR) provide native 3-class failure proximity labels.
- In Google Borg 2019, `task_events` contains `event_type = 3 (FAIL)` or `event_type = 7 (LOST)`.
- In Alibaba 2018, `batch_task.csv` and `batch_instance.csv` contain `status = 'Failed'`.
- In ClusterWise, `node_state` contains state transitions like `down` or `error`.

---

## 2. Protocol Requirements for Constructing Valid Failure Proximity Labels

To derive valid proximity labels from an external dataset without temporal leakage or artificial label distortion:
1. **Unambiguous Failure Event Identification**:
   - Must distinguish catastrophic, unhandled process deaths (e.g. out-of-memory kill, hardware crash, SIGKILL) from normal completion (`FINISH`), cancellation (`CANCEL`), or planned preemption (`EVICT`).
2. **Causal Lead-Time Anchor ($t_{\text{fail}}$)**:
   - The failure timestamp must be taken strictly from the earliest failure event timestamp of the task instance.
3. **Censoring and Negative-Window Policy**:
   - Telemetry recorded after $t_{\text{fail}}$ must be strictly censored and excluded to prevent post-failure leakage.
   - Normal windows (Class 0) must be drawn either from entities that never fail (pure-normal entities) or from pre-failure entities at least 30 minutes prior to failure ($t_{\text{fail}} - t > 1800\text{s}$).
4. **Buffer Window Exclusion**:
   - The interval between Warning and Critical ($5 < t_{\text{fail}} - t \le 10\text{ minutes}$) must either be explicitly defined or censored to prevent boundary classification ambiguity.
5. **Why Frozen Models Cannot Be Evaluated on Naive Labels**:
   - The frozen Phase 7S, 7T, and 7U models were trained against specific logit margin objectives and probability thresholds calibrated on Borg 2011 label dynamics. Evaluating them on raw point-in-time failure flags without a faithful 3-class proximity pipeline produces invalid metrics.
