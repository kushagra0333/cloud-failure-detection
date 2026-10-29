# Phase 7N: Independent Clean-Room Audit Report

**Project Root:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7N Audit  
**Date:** October 29, 2026  
**Auditor:** Antigravity Independent ML Audit Subsystem  
**Scope:** Independent Verification of Model Outputs, Probabilities, Loss Invariants, Gates, and Scientific Reproducibility  
**Audit Outcome:** `AUDIT_PASSED_EXACT_REPRODUCTION`  
**Operational Verdict:** `NO_OPERATIONAL_WINNER`  

---

## 1. Audit Summary & Scope

An independent clean-room audit was conducted on all Phase 7N experimental artifacts without importing metric functions from the primary experiment runner:
1. **Data Governance & Isolation:**
   - Candidate A Split Hash: `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946` bit-for-bit verified.
   - Cohort Sizes: Exactly 3,278 Train windows and 657 Validation windows.
   - Label Invariants: Exactly 560 Normal (Class 0), 48 Warning (Class 1), 49 Critical (Class 2).
   - Operational Units: 20 failure episodes, 49 pure-normal entities, 4 pure-normal parent jobs.
   - Prohibited Data: Candidate A Test and Phase 6 Holdout were never accessed.
2. **Numerical Reproduction:**
   - Evaluated 12 model runs (N0, N1, N2, N3 across seeds 42, 123, 456) across the predeclared primary policy, secondary argmax, and 201 threshold points (2412 total points).
   - Maximum absolute reproduction discrepancy across all continuous metrics, gate booleans, and confusion matrices: **$0.00 \times 10^0$** (bit-for-bit exact match).
3. **Operational Gate Verdict:**
   - Under the predeclared primary evaluation policy ($	heta = 0.50, \text{logit}_2 - \text{logit}_1 \ge 0.50$), simultaneous gate passage count is **0 / 12**.
   - Across all 2412 sweep points, simultaneous gate passage count is **0 / 2412**.
   - Operational verdict **`NO_OPERATIONAL_WINNER`** is independently verified.

---

## 2. Checkpoint & Checksum Registry

| Run Tag | Condition | Seed | Best Epoch | Best Val Loss | Checkpoint SHA-256 (prefix) |
|:---|:---|:---:|:---:|:---:|:---:|
| `N0_seed42` | Reproduced Phase 7M Baseline | 42 | 1 | 0.7482 | `b3cace6fb46f` |
| `N0_seed123` | Reproduced Phase 7M Baseline | 123 | 1 | 0.6729 | `3cb11d5f2957` |
| `N0_seed456` | Reproduced Phase 7M Baseline | 456 | 1 | 0.7533 | `2ea375129114` |
| `N1_seed42` | Critical-Weight Sensitivity | 42 | 1 | 0.8188 | `cf4cc7c536ea` |
| `N1_seed123` | Critical-Weight Sensitivity | 123 | 1 | 0.7454 | `1eb398d26b7d` |
| `N1_seed456` | Critical-Weight Sensitivity | 456 | 1 | 0.8400 | `8051a088b772` |
| `N2_seed42` | Asymmetric Critical Margin Loss | 42 | 1 | 0.7702 | `6372f581825e` |
| `N2_seed123` | Asymmetric Critical Margin Loss | 123 | 1 | 0.6974 | `1544d933cc76` |
| `N2_seed456` | Asymmetric Critical Margin Loss | 456 | 1 | 0.7935 | `169a91b5fe91` |
| `N3_seed42` | Multiclass Focal Loss | 42 | 1 | 0.7366 | `e698f72f5742` |
| `N3_seed123` | Multiclass Focal Loss | 123 | 1 | 0.7312 | `9609bc75c914` |
| `N3_seed456` | Multiclass Focal Loss | 456 | 1 | 0.7794 | `1618c15920d7` |

---

## 3. Discrepancy Register

All 12 evaluated runs and all preflight reconciliation items were audited against saved artifacts. Zero numerical discrepancies were detected.
Audit Status: **`PASSED_EXACT_MATCH`**.
