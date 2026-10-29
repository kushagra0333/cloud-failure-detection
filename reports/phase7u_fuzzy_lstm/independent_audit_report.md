# Independent Clean-Room Audit Report: Phase 7U

## 1. Audit Scope & Governance Verification
A clean-room independent audit was conducted on Phase 7U deliverables. The audit operates without importing helper functions or inference classes from the primary experiment script.

### Data Governance Verification
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs verified against `data/processed/phase7c/train_metadata.csv` (SHA-256: `0180ade409871168...`).
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs verified against `data/processed/phase7c/val_metadata.csv` (SHA-256: `8d22193f11d671dc...`).
- **Forbidden Partitions**: Verified zero access to Candidate A Test or Phase 6 Holdout.
- **Validation Isolation**: Input membership function breakpoints ($q_{20}, q_{50}, q_{80}$) and operating threshold $\tau$ were derived strictly from Train-OOF predictions before evaluating Candidate A Validation.

---

## 2. Preflight Discrepancy Reconciliation

### 2.1 Train-OOF Episode Recall Exactness
- **Observation**: Phase 7T reported $0.25\%$ episode recall at $\tau = 0.765$.
- **Finding**: Exactly verified. On Train-OOF (131 failing episodes), Seed 123 detected exactly 1 episode ($1 / 131 = 0.7634\%$), while Seeds 42 and 456 detected 0 ($0 / 131 = 0.0\%$). The mean across 3 seeds is $(0.7634\% + 0.0\% + 0.0\%) / 3 = 0.2545\% \approx 0.25\%$.
- **Conclusion**: The reported number is an exact arithmetic percentage, NOT a decimal fraction reporting defect.

### 2.2 Brier Score vs Constant Prevalence
- **Observation**: Phase 7T reported hybrid Brier score $0.1558$ as 'best calibration'.
- **Finding**: While $0.1558$ is substantially lower than $S_0$ ($0.3162$) and $S_2$ ($0.1705$), it is higher (worse) than a trivial constant-prevalence baseline ($0.1258$).
- **Conclusion**: The hybrid is uncalibrated and must not be described as a calibrated probability predictor.

---

## 3. Fuzzy Membership Functions and Inference Engine Audit
- **Breakpoints Derived from Train-OOF (Pooled Across Seeds)**:
  - $R_{\text{rel}}$ ($S_2$): $20^{\text{th}} = 0.0923$, $50^{\text{th}} = 0.1821$, $80^{\text{th}} = 0.3230$ (Strictly increasing: True)
  - $R_{\text{abs}}$ ($S_0$): $20^{\text{th}} = 0.0968$, $50^{\text{th}} = 0.2667$, $80^{\text{th}} = 0.6248$ (Strictly increasing: True)
- **Synthetic Unit Tests**: All 9 unit tests passed. Monotonicity confirmed. Relative-primary safeguard confirmed (LOW relative risk cannot produce HIGH output regardless of absolute risk).
- **Inference Verification**: Vectorized Mamdani Min-Max Centroid inference independently re-implemented and verified with zero numerical difference on all 3,278 Train-OOF rows and 657 Validation rows.

---

## 4. Train-OOF Threshold Selection Audit
- **Grid**: 201 candidate thresholds from $0.000$ to $1.000$ in increments of $0.005$.
- **Specificity Constraints**: $\text{Pure-Normal Entity FAR} \le 10.0\%$ AND $\text{Normal-Window FPR} \le 5.0\%$ across all 3 seeds.
- **Eligible Thresholds**: Exactly **28 thresholds** in $[0.865, 1.000]$, all producing 0 alerts due to the maximum defuzzified output ceiling ($\approx 0.8603$).
- **Tie-Breaking Result**: Under Step D, $\tau = 1.0000$ is selected as the conservative tie-break. No threshold with positive alerts satisfied the specificity gates on Train-OOF.
- **Policy Sealed**: Verified that `frozen_operating_policy.json` was sealed prior to inspecting Validation outcomes.

---

## 5. Primary Comparative Findings

### 5.1 Recovery of Within-Job Failure Discrimination
- **Within-Job Macro Mean AUC (32 eligible failing parent jobs on Train)**:
  - $S_0$ (Absolute-only): $0.6293 \pm 0.3493$
  - $S_2$ (Relative-only): $0.5345 \pm 0.2596$
  - **Fuzzy Treatment**: **$0.5607 \pm 0.2829$**
  - **Recovery over $S_2$**: **$+0.0262 \pm 0.0194$** (+2.62 percentage points across all 3 seeds).

### 5.2 Failure to Suppress False Alarms on High-Resource Job 6251277720
On the massive pure-normal parent job `6251277720` (940 windows, 28.7% of Train):
- $S_0$ False Alert Rate (at $\tau = 0.50$): **$90.74\%$**
- $S_2$ False Alert Rate (at $\tau = 0.50$): **$6.10\%$**
- Phase 7T Hybrid (80/20) False Alert Rate (at $\tau = 0.50$): **$7.30\%$**
- **Fuzzy Treatment False Alert Rate (at $\tau = 0.50$)**: **$43.69\%$**
- **Root Cause Identified**: In Job `6251277720`, benign resource variations place $R_{\text{rel}}$ in MEDIUM while heavy baseline load places $R_{\text{abs}}$ in HIGH. Rule 6 (`IF R_rel is MEDIUM AND R_abs is HIGH THEN Output is HIGH`) repeatedly fires, triggering false alarms on $43.7\%$ of windows.
- **Scientific Takeaway**: The linear 80/20 hybrid is vastly superior to the fuzzy rule base in suppressing high-resource false alarms because it strictly bounds the absolute risk contribution to 20%, whereas Rule 6 elevates moderate relative deviations directly to HIGH.

### 5.3 Validation Ranking & Operational Rates (Fixed Diagnostic $\tau = 0.50$)
- **Validation ROC-AUC**:
  - $S_0$: $0.4244 \pm 0.0154$
  - $S_2$: $0.5864 \pm 0.0217$
  - Phase 7T Hybrid: $0.5619 \pm 0.0189$
  - **Fuzzy Treatment**: **$0.5827 \pm 0.0515$** (reaching $0.6289$ on Seed 123).
- **Validation Episode Recall (at $\tau = 0.50$)**:
  - $S_0$: $35.00\%$
  - $S_2$: $43.33\%$
  - Phase 7T Hybrid: $56.67\%$
  - **Fuzzy Treatment**: **$66.67\% \pm 5.77\%$** (detecting 14 / 20 episodes on Seeds 42 and 456!).
- **Validation Normal-Window FPR (at $\tau = 0.50$)**:
  - $S_2$: $10.30\%$
  - Phase 7T Hybrid: $9.52\%$
  - **Fuzzy Treatment**: **$17.44\% \pm 2.08\%$** (elevated false alarms due to Rule 6 firing).

---

## 6. Operational Gate Verification (Candidate A Validation)

| Model Condition | Seed | Episode Recall ($\ge 70\%$) | Critical Recall ($\ge 50\%$) | Entity FAR ($\le 10\%$) | Window FPR ($\le 5\%$) | All 4 Gates | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fuzzy Primary ($\tau = 1.000$)** | 42 | 0.0% (0/20) [FAIL] | 0.0% (0/49) [FAIL] | 0.0% (0/49) [PASS] | 0.0% (0/560) [PASS] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Fuzzy Primary ($\tau = 1.000$)** | 123 | 0.0% (0/20) [FAIL] | 0.0% (0/49) [FAIL] | 0.0% (0/49) [PASS] | 0.0% (0/560) [PASS] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Fuzzy Primary ($\tau = 1.000$)** | 456 | 0.0% (0/20) [FAIL] | 0.0% (0/49) [FAIL] | 0.0% (0/49) [PASS] | 0.0% (0/560) [PASS] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Fuzzy Diagnostic ($\tau = 0.500$)**| 42 | 70.0% (14/20) [PASS] | 20.4% (10/49) [FAIL] | 44.9% (22/49) [FAIL] | 17.9% (100/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Fuzzy Diagnostic ($\tau = 0.500$)**| 123 | 60.0% (12/20) [FAIL] | 30.6% (15/49) [FAIL] | 55.1% (27/49) [FAIL] | 15.2% (85/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Fuzzy Diagnostic ($\tau = 0.500$)**| 456 | 70.0% (14/20) [PASS] | 30.6% (15/49) [FAIL] | 44.9% (22/49) [FAIL] | 19.3% (108/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |

Although the diagnostic policy achieved high Episode Recall ($66.67\%$, reaching $70.0\%$ on two seeds), Normal-Window FPR ($17.44\%$) and Pure-Normal Entity FAR ($48.30\%$) failed the required operating limits.

---

## 7. Audit Verdict
- **Audit Integrity Status**: `AUDIT_PASSED` (16 / 16 automated safeguards passed).
- **Operational Verdict**: `NO_OPERATIONAL_WINNER`.
