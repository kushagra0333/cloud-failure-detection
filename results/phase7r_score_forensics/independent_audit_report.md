# Independent Clean-Room Audit Report: Phase 7R

## 1. Audit Scope and Governance Verification
An independent, clean-room audit was conducted on Phase 7R deliverables and prior phase artifacts. The audit did not import any helper functions from primary experiment scripts.

### Governance Allowlist Verification
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs. Verified against `data/processed/phase7c/train_metadata.csv` (SHA-256: `0180ade409871168...`).
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs. Verified against `data/processed/phase7c/val_metadata.csv` (SHA-256: `8d22193f11d671dc...`).
- **Candidate A Test & Phase 6 Holdout**: Verified completely untouched; no process attempted access.
- **Validation Tuning Isolation**: Verified that Validation was evaluated strictly as frozen development evidence.

---

## 2. Row Alignment, Label Semantics, and Score Construction Audit
- **Label Alignment**: Exact equality between `data/processed/phase7c/y_train.npy` and prediction files `y_true`: **0 mismatches / 3,278 rows**.
- **Entity ID Alignment**: Exact equality between metadata `entity_id` and prediction files: **0 mismatches / 3,278 rows**.
- **Timestamp Alignment**: Max timestamp discrepancy between split metadata and predictions: **0.000000 seconds**.
- **Horizon Verification**:
  - Critical windows ($y=2$): TTF strictly in $(0, 900	ext456]$ (min: 1.0s, max: 899.0s).
  - Warning windows ($y=1$): TTF strictly in $(900, 1800	ext456]$ (min: 901.0s, max: 1798.9s).
  - Normal pre-failure windows ($y=0$): TTF strictly $> 1800	ext456$ (min: 1801.0s).
- **Probability Arithmetic**:
  - $P_0 + P_1 + P_2 = 1.0$: Max error across all models, seeds, and cohorts = **1.57e-07**.
  - $P_{\text{fail}} = P_1 + P_2$: Max error = **1.60e-07**.
  - $P_{\text{fail}} = 1 - P_0$: Max error = **1.60e-07**.
  - Mismatched probability rows ($>10^{-5}$): **0 rows**.

---

## 3. Discrepancy Reconciliation and Root Cause Analysis

### Question 1: Why is failure ROC-AUC < 0.50 across held-out parent jobs?
**Finding**: Failure ROC-AUC below 0.50 is NOT caused by a software bug, inverted class mapping, transposed probability columns, or training loss mis-specification.
- In-sample training on Candidate A Train achieves **ROC-AUC of 0.8511 to 0.8915** with mean failure probability of **0.38 - 0.40** on failures vs **0.14 - 0.16** on Normal windows.
- Within individual parent jobs on Candidate A Train (where both Normal and Failure windows exist), the **unweighted mean within-job ROC-AUC is 0.6405** (and up to **1.0000** on individual jobs on Validation).
- However, between different parent jobs, telemetry baselines differ drastically. In particular:
  - Job `6251277720` contains **940 windows** (28.7% of all train windows!) and is **100% pure-normal**. Its telemetry profile causes models to predict an average failure probability of **0.6362** on its normal windows.
  - Failing jobs (e.g. Job `6251995937`, 283 windows, 81.3% failures) have lower baseline telemetry, resulting in mean failure probability of **0.0761**.
- When windows from heterogeneous jobs are pooled into a single cross-job ranking, the 940 normal windows with $P_{\text{fail}} \approx 0.64$ rank above failure windows with $P_{\text{fail}} \approx 0.08$. This is a classic textbook instance of **Simpson's Paradox / Cross-Job Baseline Confounding**.

### Question 2: Why did Phase 7P Platt calibration have negative slopes?
**Finding**: Because Fold 0 in Phase 7P held out Job `6251414911` (643 windows, 99.1% normal) whose true labels were 0 but received higher scores than smaller failing jobs. Fitting a logistic regression on pooled OOF predictions with cross-job confounding produced a negative slope ($a < 0$), which inverted the score ranking ($AUC_{\text{cal}} = 1 - AUC_{\text{raw}}$).

### Question 3: Why was Phase 7P calibrated Brier equal to constant prevalence?
**Finding**: Platt scaling merely centered all probabilities around the training failure base rate ($14.67\%$). On Validation (base rate $14.76\%$), a trivial predictor assigning $p=0.1476$ to all rows achieves Brier $= p(1-p) = 0.125844$. Phase 7P calibrated Brier was $0.125955$, showing zero individual discriminatory signal over a static prior.

### Question 4: Does Logistic Regression suffer from the same cross-job transfer failure?
**Finding**: Yes. A simple linear Logistic Regression trained on recent 5-step features achieved a Train-OOF ROC-AUC of **0.2479**! This proves that cross-job baseline shift affects both linear and deep sequence models when trained on absolute telemetry.

---

## 4. Operational Gate Audit (Candidate A Validation)

| Model | Seed | Episode Recall ($\ge 70\%$) | Critical Recall ($\ge 50\%$) | Entity FAR ($\le 10\%$) | Window FPR ($\le 5\%$) | All 4 Gates | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q0 Baseline** | 42 | 35.0% (FAIL) | 18.4% (FAIL) | 36.7% (FAIL) | 30.2% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Q0 Baseline** | 123 | 35.0% (FAIL) | 14.3% (FAIL) | 44.9% (FAIL) | 35.5% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Q0 Baseline** | 456 | 35.0% (FAIL) | 18.4% (FAIL) | 55.1% (FAIL) | 37.7% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Q1 Treatment** | 42 | 30.0% (FAIL) | 14.3% (FAIL) | 49.0% (FAIL) | 34.3% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Q1 Treatment** | 123 | 30.0% (FAIL) | 12.2% (FAIL) | 42.9% (FAIL) | 26.6% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Q1 Treatment** | 456 | 35.0% (FAIL) | 16.3% (FAIL) | 55.1% (FAIL) | 30.2% (FAIL) | FAIL | `NO_OPERATIONAL_WINNER` |
| **Logistic Regression** | 42 | 10.0% (FAIL) | 0.0% (FAIL) | 8.2% (PASS) | 1.6% (PASS) | FAIL | `NO_OPERATIONAL_WINNER` |

All configurations fail the operational gates. Zero configurations qualify as an operational winner.

---

## 5. Audit Verdict
- **Integrity Status**: `AUDIT_PASSED` (Zero integrity defects, 100% code and data verified).
- **Operational Verdict**: `NO_OPERATIONAL_WINNER`.
