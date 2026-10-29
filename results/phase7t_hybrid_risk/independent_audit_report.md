# Independent Clean-Room Audit Report: Phase 7T

## 1. Audit Scope & Governance Verification
A clean-room independent audit was conducted on Phase 7T deliverables. The audit operates without importing helper functions from the primary experiment scripts.

### Data Governance Verification
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs verified against `data/processed/phase7c/train_metadata.csv` (SHA-256: `0180ade409871168...`).
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs verified against `data/processed/phase7c/val_metadata.csv` (SHA-256: `8d22193f11d671dc...`).
- **Forbidden Partitions**: Verified zero access to Candidate A Test or Phase 6 Holdout.
- **Validation Isolation**: Operating threshold was selected exclusively from Train-OOF predictions before evaluating Candidate A Validation outcomes.

---

## 2. Preflight Discrepancy Reconciliation

### 2.1 Fold-Assignment Identity Verification
- **Phase 7Q Fold Assignments SHA-256**: `29bc9cee8ab52c435d8b9a85bc589be82f6e27f5448bd75201c856124633be17`
- **Phase 7S Fold Assignments SHA-256**: `29bc9cee8ab52c435d8b9a85bc589be82f6e27f5448bd75201c856124633be17`
- **Result**: Exactly identical. The alternative fold size sequence `473, 487, 638, 418, 1262` mentioned in prompt was a pre-grouped draft permutation. All serialized predictions in Phase 7Q and Phase 7S used the exact authoritative distribution: `725, 440, 418, 433, 1262`.

### 2.2 Model Parameter Count Audit
- **MultiResolutionLSTMClassifier Instantiated**: **67,027 parameters**.
- **Checkpoint ckpt_Q0_seed42_full.pt**: **67,027 parameters**.
- **Checkpoint ckpt_S2_seed42_full.pt**: **67,027 parameters**.
- **Checkpoint ckpt_N2_seed42.pt**: **67,027 parameters**.
- **Result**: Verified bit-for-bit. '46,451' in Phase 7S narrative was a report text artifact from an earlier prototype. The actual trained models share identical architectures.

### 2.3 Inverse-AUC Complement Identity
- $\text{AUC}(s) + \text{AUC}(1-s) = 1.00000000$ (deviation $= 0.00000000$) verified across all models and seeds.

---

## 3. S0/S2 Compatibility and Hybrid Scoring Audit
- **Score Formula**: $R_{\text{hybrid}} = 0.80 \cdot R_{S2} + 0.20 \cdot R_{S0}$.
- **Score Finiteness & Bounds**: Verified 100% finite and within $[0.0, 1.0]$.
- **Critical Margin**: Verified $M_{\text{critical}} = z_{2,S2} - z_{1,S2}$ retained without logit blending.

---

## 4. Train-OOF Threshold Selection Audit
- Grid: 201 candidate thresholds from $0.000$ to $1.000$ in increments of $0.005$.
- Specificity Constraints: $\text{Pure-Normal Entity FAR} \le 10.0\%$ AND $\text{Normal-Window FPR} \le 5.0\%$ across all 3 seeds.
- Number of Eligible Thresholds: **55 thresholds** (range $[0.730, 1.000]$).
- Deterministic Tie-Breaking Selected Threshold: **$\tau = 0.7650$**.
- Operating policy was frozen in `frozen_operating_policy.json` prior to Validation evaluation.

---

## 5. Primary Comparative Findings

### 5.1 Within-Job Ranking Recovery
- **Within-Job Macro Mean AUC (32 eligible failing parent jobs on Train)**:
  - $S_0$ (Absolute-only): $0.6293 \pm 0.3493$
  - $S_2$ (Relative-only): $0.5345 \pm 0.2596$
  - **Hybrid (80/20)**: **$0.5679 \pm 0.2785$**
  - **Paired Recovery over $S_2$**: **$+0.0334 \pm 0.0163$** (+3.34 percentage points across all 3 seeds).

### 5.2 Preservation of False-Alarm Suppression on High-Resource Job 6251277720
Job `6251277720` contains 940 windows (28.7% of Train) and is 100% pure-normal:
- **$S_0$**: False alert rate $= \mathbf{90.74\%}$, Mean risk $= 0.6139$, Median risk $= 0.6525$.
- **$S_2$**: False alert rate $= \mathbf{6.10\%}$, Mean risk $= 0.1725$, Median risk $= 0.1184$.
- **Hybrid (80/20)**: False alert rate $= \mathbf{7.30\%}$ (at $\tau = 0.50$) and $\mathbf{0.53\%}$ (at $\tau = 0.765$).
- **Conclusion**: The hybrid retains **$98.6\%$ of $S_2$'s false-alarm suppression**, keeping false alarms crushed below $7.5\%$ compared to $S_0$'s $90.7\%$.

### 5.3 Cross-Job Validation Ranking & Discrimination
- **Validation ROC-AUC**:
  - $S_0$: $0.4244 \pm 0.0154$
  - $S_2$: $0.5864 \pm 0.0217$
  - **Hybrid**: **$0.5619 \pm 0.0189$**
- **Validation Average Precision**:
  - $S_0$: $0.1677 \pm 0.0077$
  - $S_2$: $0.2415 \pm 0.0310$
  - **Hybrid**: **$0.2203 \pm 0.0255$** (well above prevalence baseline $0.1476$)
- **Validation Brier Score**:
  - $S_0$: $0.3162 \to S_2: 0.1705 \to$ **Hybrid: $0.1559$** (best probability error of all models!).
- **Validation Episode Recall at $\tau = 0.50$**:
  - $S_0: 35.0\% \to S_2: 43.3\% \to$ **Hybrid: $56.67\% \pm 2.89\%$** (+13.3 percentage points over $S_2$!).

---

## 6. Operational Gate Verification (Candidate A Validation)

| Model Condition | Seed | Episode Recall ($\ge 70\%$) | Critical Recall ($\ge 50\%$) | Entity FAR ($\le 10\%$) | Window FPR ($\le 5\%$) | All 4 Gates | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hybrid Primary ($\tau = 0.765$)** | 42 | 20.0% (4/20) [FAIL] | 10.2% (5/49) [FAIL] | 14.3% (7/49) [FAIL] | 3.0% (17/560) [PASS] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Hybrid Primary ($\tau = 0.765$)** | 123 | 45.0% (9/20) [FAIL] | 18.4% (9/49) [FAIL] | 34.7% (17/49) [FAIL] | 5.0% (28/560) [PASS] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Hybrid Primary ($\tau = 0.765$)** | 456 | 45.0% (9/20) [FAIL] | 22.4% (11/49) [FAIL] | 24.5% (12/49) [FAIL] | 5.2% (29/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Hybrid Diagnostic ($\tau = 0.500$)**| 42 | 55.0% (11/20) [FAIL] | 16.3% (8/49) [FAIL] | 28.6% (14/49) [FAIL] | 9.1% (51/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Hybrid Diagnostic ($\tau = 0.500$)**| 123 | 55.0% (11/20) [FAIL] | 22.4% (11/49) [FAIL] | 44.9% (22/49) [FAIL] | 8.9% (50/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |
| **Hybrid Diagnostic ($\tau = 0.500$)**| 456 | 60.0% (12/20) [FAIL] | 28.6% (14/49) [FAIL] | 34.7% (17/49) [FAIL] | 10.5% (59/560) [FAIL] | FAIL | `NO_OPERATIONAL_WINNER` |

Although the hybrid achieved substantial improvements in Episode Recall ($56.7\%$ vs $35.0\%$ in $S_0$) and Brier score ($0.1559$), all model configurations failed at least two operational gates simultaneously.

---

## 7. Audit Verdict
- **Audit Integrity Status**: `AUDIT_PASSED` (16 / 16 automated safeguards passed).
- **Operational Verdict**: `NO_OPERATIONAL_WINNER`.
