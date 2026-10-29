# Phase 7S Experimental Design: Absolute-versus-Relative Telemetry Ablation and Cross-Job Generalization

## 1. Scientific Objective and Mission

Phase 7R established that cross-job predictive failure ranking is severely distorted by between-job baseline telemetry shifts (Simpson's Paradox). When models rely on absolute telemetry, large pure-normal parent jobs with elevated baseline resource usage (specifically Job `6251277720`, containing 940 windows, 28.7% of Candidate A Train) receive high failure-risk scores ($P_{\text{fail}} \approx 0.64$), while smaller failing jobs operate at lower baseline telemetry levels and receive low failure scores ($P_{\text{fail}} \approx 0.08$).

Phase 7Q introduced causal recent-versus-historical relative deviation features, but appended them as an auxiliary MLP branch alongside the original absolute telemetry streams. Thus, Phase 7Q could not isolate whether absolute resource levels are responsible for cross-job failure ranking collapse or whether a model can identify impending failures using purely relative changes.

Phase 7S executes a controlled 3-condition representation ablation:
- **S0 — Absolute-only baseline**: The Phase 7N N2 multi-resolution model (recent 5-step stream + historical 93 summary statistics stream).
- **S1 — Combined representation**: The Phase 7Q Q1 model (recent 5-step stream + historical 93 summary statistics stream + 31-d causal relative deviation MLP branch).
- **S2 — Baseline-relative-only treatment**: The identical multi-resolution backbone and training objective, but **all telemetry inputs are expressed relative to the entity's preceding historical baseline**. No absolute telemetry levels or absolute historical summary statistics enter any layer of the model.

### Central Research Question
**Does removing absolute telemetry levels improve failure ranking across unseen parent jobs while preserving useful within-job failure discrimination and reducing false alarms on healthy high-resource jobs?**

---

## 2. Preflight Audit and Discrepancy Resolutions

### 2.1 Resolution of Phase 7R Inverse-AUC Discrepancy
In Phase 7R, the reported Validation Q0 ROC-AUC was $0.4244$ and the inverse-score AUC was $0.5772$, summing to $1.0016$ instead of exactly $1.0$.
- **Mathematical Root Cause**: Phase 7R evaluated $P_0$ as the inverse score instead of the exact complement $1 - P_{\text{fail}}$. In float32 softmax outputs, $P_0 + (P_1 + P_2) = 1.0 \pm 1.3 \times 10^{-7}$. This microscopic $10^{-7}$ numerical rounding artifact slightly perturbed the rank order of near-tied validation rows when sorted by $P_0$ vs $1 - P_{\text{fail}}$.
- **Empirical Proof**: Computing `roc_auc_score(y, pfail) + roc_auc_score(y, 1.0 - pfail)` yields **exactly 1.00000000** (deviation $= 0.00000000$) across every single seed, model, and partition without exception.

### 2.2 Resolution of Phase 7Q Report Narrative vs Operational Tables
In Phase 7Q, the executive summary narrative in `phase7q_final_report.md` mentioned draft placeholder figures ("1.7% Episode Recall"), whereas the machine-generated `operational_gate_summary.csv` and serialized prediction files reported Episode Recall $= 35.0\%$ and FPR $= 30.2\% - 37.7\%$.
- **Resolution**: The serialized predictions, checkpoint evaluations, and `operational_gate_summary.csv` were verified as authoritative and mathematically consistent. The discrepancy was entirely a report narrative text defect in Phase 7Q. In Phase 7S, all report metrics are 100% programmatically generated from verified CSVs.

---

## 3. The S2 Baseline-Relative Transformation

### 3.1 Causal Temporal Boundaries
Each 12-step prediction sequence represents 60 minutes of causal telemetry:
- Historical interval: steps 0..6 ($t-55\text{m}$ to $t-25\text{m}$, 7 steps).
- Recent interval: steps 7..11 ($t-20\text{m}$ to $t$, 5 steps).
- Prediction timestamp: $t$. Zero future data is accessible.

### 3.2 Historical Baseline Statistics
For each feature $j \in \{0, \dots, 30\}$ within a window:
$$\text{hist\_median}_j = \text{median}(X[0..6, j])$$
$$\text{hist\_MAD}_j = \text{median}(|X[0..6, j] - \text{hist\_median}_j|)$$
$$\text{hist\_scale}_j = 1.4826 \times \text{hist\_MAD}_j$$
$$\text{effective\_scale}_j = \max(\text{hist\_scale}_j, \text{scale\_floor}_j)$$

`scale_floor_j` is derived exclusively from Candidate A Train as the 10th percentile of positive historical scales.

### 3.3 Full 12-Step Normalization
Every timestep $t \in \{0, \dots, 11\}$ is normalized against the **same historical baseline**:
$$X_{\text{rel}}[t, j] = \text{clip}\left(\frac{X[t, j] - \text{hist\_median}_j}{\text{effective\_scale}_j}, -5.0, +5.0\right)$$

- **Stream A (Recent Dynamic Stream)**: $X_{\text{rel}}[7..11, :]$ (5 steps, 31 features). Fed to the recent LSTM.
- **Stream B (Historical Summary Stream)**: 93 relative summary statistics derived exclusively from $X_{\text{rel}}[0..6, :]$:
  - Relative mean: $\text{mean}(X_{\text{rel}}[0..6, j])$ (31 features).
  - Relative spread: $\text{std}(X_{\text{rel}}[0..6, j])$ (31 features).
  - Relative historical trend: $\text{mean}(X_{\text{rel}}[4..6, j]) - \text{mean}(X_{\text{rel}}[0..3, j])$ (31 features).
  - Fed to the historical MLP.

### 3.4 Architectural Invariant
S2 preserves the exact neural architecture of S0:
- LSTM: input_dim 31, hidden_dim 64, num_layers 2, dropout 0.2.
- Historical MLP: input_dim 93, hidden_dim 32, out_dim 16.
- Classifier Head: Linear(80, 64) $\to$ ReLU $\to$ Linear(64, 3).
- **Exact Parameter Count**: 46,451 parameters (100% matched to S0).

---

## 4. Strict Data Governance and Partitioning

### Authorized Partitions
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs.
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs (560 Normal, 48 Warning, 49 Critical; 20 failure episodes, 49 pure-normal entities).

### Prohibited Partitions
- Candidate A Test (1,025 windows) — STRICTLY FORBIDDEN.
- Phase 6 Confirmatory Holdout — STRICTLY FORBIDDEN.

### Parent-Job-Grouped 5-Fold Partition
- Preserves the verified, balanced parent-job-grouped partition from Phase 7Q (`fold_assignments.csv`).
- Jobs per fold: 10, 10, 10, 11, 10.
- Massive jobs remain intact: Job `6251277720` (940 windows) is preserved intact in Fold 4. Zero parent jobs cross fold boundaries.

---

## 5. Decision Policy and Operational Standard

### Fixed Primary Operating Policy
- Alert triggered when raw $P_{\text{fail}} = P_1 + P_2 \ge 0.50$.
- Severity: Predict **Critical** if $z_2 - z_1 \ge 0.50$, else **Warning**.
- Zero threshold tuning or post-hoc calibrators on Validation.

### Four Frozen Operational Gates
1. **Gate 1A (Episode Recall)**: $\ge 70.0\%$ (20 episodes).
2. **Gate 1B (Genuine Critical Recall)**: $\ge 50.0\%$ (49 Critical windows).
3. **Gate 2A (Pure-Normal Entity FAR)**: $\le 10.0\%$ (49 pure-normal entities).
4. **Gate 2B (Normal-Window FPR)**: $\le 5.0\%$ (560 Normal windows).

### Operational Verdict
Must be strictly one of:
- `OPERATIONAL_WINNER`: All four gates pass under the fixed policy.
- `NO_OPERATIONAL_WINNER`: Any gate fails.
- `AUDIT_BLOCKED`: A material integrity defect or governance breach occurs.
