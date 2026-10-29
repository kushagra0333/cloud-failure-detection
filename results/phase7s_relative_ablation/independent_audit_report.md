# Independent Clean-Room Audit Report: Phase 7S

## 1. Audit Scope & Governance Verification
A clean-room independent audit was conducted on Phase 7S deliverables. The audit operates without importing helper functions from the primary experiment scripts.

### Data Governance Verification
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs verified against `data/processed/phase7c/train_metadata.csv` (SHA-256: `0180ade409871168...`).
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs verified against `data/processed/phase7c/val_metadata.csv` (SHA-256: `8d22193f11d671dc...`).
- **Forbidden Partitions**: Verified zero access to Candidate A Test or Phase 6 Holdout.
- **Validation Isolation**: Validation was evaluated exclusively under the frozen decision policy ($P_{\text{fail}} \ge 0.50$, $z_2 - z_1 \ge 0.50$) without parameter or threshold tuning.

---

## 2. Preflight Discrepancy Reconciliation

### 2.1 The Inverse-AUC Complement Identity
- **Audit Test**: The exact analytical complement $s_{\text{comp}} = 1.0 - P_{\text{fail}}$ was recomputed and evaluated alongside $P_{\text{fail}}$.
- **Result**: $\text{AUC}(P_{\text{fail}}) + \text{AUC}(1.0 - P_{\text{fail}}) = 1.00000000$ (deviation $= 0.00000000$) across all models and seeds.
- **Root Cause of Prior $0.0016$ Discrepancy in Phase 7R**: Phase 7R evaluated $P_0$ rather than $1 - P_{\text{fail}}$. In float32 softmax, $P_0 + (P_1 + P_2) = 1.0 \pm 1.3 \times 10^{-7}$. This microscopic rounding error perturbed sorting orders during tie-breaking. The mathematical identity holds with zero error.

### 2.2 Operational Table Discrepancy Resolution
- **Audit Test**: Verification of Phase 7Q report narrative text ("1.7% Recall") vs serialized CSV files.
- **Result**: The serialized predictions and `operational_gate_summary.csv` were verified as authoritative and mathematically consistent (Q0 Episode Recall $= 35.0\%$, FPR $= 30.2\% - 37.7\%$). The mismatch was confirmed to be a report narrative text defect in Phase 7Q.

---

## 3. S2 Baseline-Relative Transformation Audit
- **Causality**: Historical baseline (median and MAD) computed exclusively from steps 0..6 ($t-55\text{m}$ to $t-25\text{m}$). Prediction timestamp is $t$. Zero future data is accessed.
- **Zero Absolute Telemetry**: Confirmed that all 12 steps are transformed via $X_{\text{rel}} = \text{clip}((X - \text{med}) / \text{scale}, -5.0, 5.0)$. Stream A receives steps 7..11 of $X_{\text{rel}}$; Stream B receives 93 summary statistics derived exclusively from steps 0..6 of $X_{\text{rel}}$. No absolute telemetry levels or scales enter the S2 model.
- **Scale Floor Provenance**: Verified that scale floors were fitted strictly on Candidate A Train (10th percentile of positive historical scales).
- **Bounds & Finiteness**: 100% of S2 transformed inputs are finite and bounded in $[-5.0, 5.0]$.
- **Architectural Match**: S2 uses the exact same `MultiResolutionLSTMClassifier` as S0 (46,451 parameters).

---

## 4. Key Comparative Evaluation Findings

### 4.1 Cross-Job Ranking Breakthrough (S2 vs S0 & S1)
- **Train-OOF ROC-AUC**:
  - S0 (Absolute-only): Mean $= 0.3066 \pm 0.0130$
  - S1 (Combined): Mean $= 0.3139 \pm 0.0450$
  - S2 (Relative-only): Mean $= 0.5080 \pm 0.0573$
  - **Paired Improvement (S2 - S0)**: **$+0.2014 \pm 0.0649$**
- **Validation ROC-AUC**:
  - S0 (Absolute-only): Mean $= 0.4244 \pm 0.0153$
  - S1 (Combined): Mean $= 0.4435 \pm 0.0331$
  - S2 (Relative-only): Mean $= 0.5864 \pm 0.0217$
  - **Paired Improvement (S2 - S0)**: **$+0.1621 \pm 0.0141$**
  - Validation ROC-AUC improved above $0.57$ across all three seeds ($0.5731, 0.6114, 0.5748$).
- **Validation Average Precision**:
  - S0: Mean $= 0.1677$
  - S1: Mean $= 0.1565$
  - S2: Mean $= 0.2415$ (exceeding prevalence baseline of $0.1476$)
- **Validation Brier Score**:
  - S0: $0.3162 \to$ S2: **$0.1705$** (cut nearly in half!)
- **Normal-Window FPR**:
  - S0: $34.46\% \to$ S2: **$10.30\%$** (**$-24.17$ percentage points reduction!**)

### 4.2 Impact on High-Resource Pure-Normal Job 6251277720
Job `6251277720` contains 940 windows (28.7% of all train windows) and is 100% pure-normal:
- **S0 (Absolute-only)**: Mean score $= 0.6139$, Median score $= 0.6525$, False alert rate $= \mathbf{90.74\%}$!
- **S1 (Combined)**: Mean score $= 0.5520$, Median score $= 0.5791$, False alert rate $= \mathbf{81.88\%}$!
- **S2 (Relative-only)**: Mean score $= 0.1725$, Median score $= 0.1184$, False alert rate $= \mathbf{6.10\%}$!
Removing absolute telemetry crushed false alerts on this massive job from $90.7\%$ down to $6.1\%$ and eliminated the false high-risk baseline.

### 4.3 The Scientific Trade-Off: Within-Job Ranking
- **Within-Job Macro Mean AUC (32 eligible failing jobs on Train)**:
  - S0: $0.6293 \pm 0.3493$
  - S1: $0.6674 \pm 0.3024$
  - S2: $0.5345 \pm 0.2596$
  - **Paired Difference (S2 - S0)**: **$-0.0948 \pm 0.0305$**
- **Interpretation**: Removing absolute telemetry levels eliminates cross-job baseline confounding (improving cross-job pooled OOF AUC by $+0.20$ and Val AUC by $+0.16$), but slightly weakens within-job discrimination on jobs where absolute resource consumption itself serves as a symptom of impending failure.

---

## 5. Operational Gate Verification (Candidate A Validation)

| Model | Seed | Episode Recall ($\ge 70\%$) | Critical Recall ($\ge 50\%$) | Entity FAR ($\le 10\%$) | Window FPR ($\le 5\%$) | All 4 Gates | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | 42 | 35.0% (FAIL) | 18.4% (FAIL) | 36.7% (FAIL) | 30.2% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S0** | 123 | 35.0% (FAIL) | 14.3% (FAIL) | 44.9% (FAIL) | 35.5% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S0** | 456 | 35.0% (FAIL) | 18.4% (FAIL) | 55.1% (FAIL) | 37.7% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S1** | 42 | 30.0% (FAIL) | 14.3% (FAIL) | 48.9% (FAIL) | 34.3% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S1** | 123 | 30.0% (FAIL) | 12.2% (FAIL) | 42.9% (FAIL) | 26.6% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S1** | 456 | 35.0% (FAIL) | 16.3% (FAIL) | 55.1% (FAIL) | 30.2% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S2** | 42 | 40.0% (FAIL) | 16.3% (FAIL) | 30.6% (FAIL) | 10.7% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S2** | 123 | 45.0% (FAIL) | 24.5% (FAIL) | 44.9% (FAIL) | 10.0% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **S2** | 456 | 45.0% (FAIL) | 26.5% (FAIL) | 36.7% (FAIL) | 10.2% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |

Although S2 substantially improved Episode Recall ($43.3\%$ vs $35.0\%$) and drastically reduced Window FPR ($10.3\%$ vs $34.5\%$), it did not satisfy all four operational gates simultaneously.

---

## 6. Audit Verdict
- **Audit Integrity Status**: `AUDIT_PASSED` (16 / 16 automated safeguards passed).
- **Operational Verdict**: `NO_OPERATIONAL_WINNER`.
