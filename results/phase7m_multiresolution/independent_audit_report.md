# Phase 7M: Independent Clean-Room Audit Report

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7M Audit  
**Date:** October 29, 2026  
**Auditor:** Antigravity Independent ML Audit Subsystem  
**Audit Scope:** Independent Verification of Model Outputs, Probabilities, Gates, and Scientific Invariants  
**Audit Outcome:** `AUDIT_PASSED_EXACT_REPRODUCTION`  
**Operational Winner:** `NO OPERATIONAL WINNER` (Reaffirmed)  

---

## 1. Audit Summary

An independent, clean-room audit was conducted on all Phase 7M experimental outputs without importing metric-calculation functions from the training script:
1. **Data Governance & Integrity:**
   - Candidate A Split Hash: `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946` verified bit-for-bit.
   - Cohort Sizes: Exactly 3,278 Train windows and 657 Validation windows.
   - Label Invariant: Exactly 560 Normal, 48 Warning, 49 Critical.
   - Stream Separation: Stream A ($t-20\text{m}$ to $t$) and Stream B ($t-55\text{m}$ to $t-25\text{m}$) verified completely non-overlapping and causal.
   - Train-Only Scaling: Stream B StandardScaler parameters verified fitted exclusively on Candidate A Train; zero validation observations were used during fitting.
   - Restricted Data: Candidate A Test and Phase 6 Holdout were never accessed.
2. **Numerical Reproduction:**
   - Evaluated 18 model configurations (6 M0 baseline, 6 M1 control, 6 M2 treatment) across 201 thresholds ($3,618$ evaluation points).
   - Maximum absolute reproduction discrepancy across all continuous metrics, gate booleans, and confusion matrices: **$0.00 \times 10^0$** (bit-for-bit exact match).
3. **Operational Gate Verdict:**
   - Across all 3,618 threshold evaluation points, **simultaneous gate passage count is exactly 0**.
   - No model passes all four operational gates simultaneously.
   - The operational verdict **`NO OPERATIONAL WINNER`** is independently verified.

---

## 2. Model & Checkpoint Verification

| Model Run Tag | Architecture | Sequence Span | Param Count | Best Epoch | Best Val Loss | Checkpoint SHA-256 (prefix) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_cond_B_seed42` | Recent-Only Control (LSTM) | $L=5$ (25m) | 62,467 | 1 | 0.7573 | `a10a4ad1dcf5` |
| `M1_cond_B_seed123`| Recent-Only Control (LSTM) | $L=5$ (25m) | 62,467 | 1 | 0.7391 | `c91db92be337` |
| `M1_cond_B_seed456`| Recent-Only Control (LSTM) | $L=5$ (25m) | 62,467 | 1 | 0.7780 | `9c9878426682` |
| `M2_cond_B_seed42` | Multi-Resolution (Dual-Stream) | $5\text{m} + 35\text{m}$ | 67,027 | 1 | 0.7482 | `d2f07a663cdf` |
| `M2_cond_B_seed123`| Multi-Resolution (Dual-Stream) | $5\text{m} + 35\text{m}$ | 67,027 | 1 | 0.6729 | `e81a1279ddbb` |
| `M2_cond_B_seed456`| Multi-Resolution (Dual-Stream) | $5\text{m} + 35\text{m}$ | 67,027 | 1 | 0.7533 | `8482b4d88b62` |

---

## 3. Discrepancy Register

All 18 evaluated runs were audited against saved artifacts. Zero numerical discrepancies were detected.
Status: **`PASSED_EXACT_MATCH`**.
